"""Record one SO-101 leader-to-follower trial episode in local LeRobot format.

Control follows the pinned practical wrapper at 60 Hz. Unique wrist-camera frames
are timestamp-matched to the nearest preceding control observation. Video uses a
nominal 30 FPS timeline; the sidecar preserves measured host timing and gaps.
The LeRobot action is the follower-coordinate action returned by the actual send call.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import os
import signal
import queue
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from so101_paths import ROOT
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = tuple(name + ".pos" for name in JOINTS)
CAMERA_VID = int(os.environ.get('SO101_CAMERA_VID','0x32E6'),0)
CAMERA_PID = int(os.environ.get('SO101_CAMERA_PID','0x9251'),0)
CAMERA_FPS = 30
CONTROL_FPS = 60
CAMERA_SIZE = (640, 480)
QUEUE_LIMIT = 12
MIN_EPISODE_FRAMES = 30


def load_follow_module():
    path = ROOT / "scripts" / "follow-practical.py"
    spec = importlib.util.spec_from_file_location("so101_follow_practical", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load follow wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_camera():
    from cv2_enumerate_cameras import enumerate_cameras
    import cv2
    from lerobot.cameras.opencv import OpenCVCameraConfig

    devices = enumerate_cameras(cv2.CAP_DSHOW)
    report = [
        {
            key: getattr(camera, key, None)
            for key in ("index", "name", "path", "vid", "pid", "backend")
        }
        for camera in devices
    ]
    selected = [camera for camera in devices if camera.vid == CAMERA_VID and camera.pid == CAMERA_PID]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one wrist camera VID/PID {CAMERA_VID:04x}:{CAMERA_PID:04x}; found {len(selected)}: {report}")
    camera = selected[0]
    config = OpenCVCameraConfig(
        index_or_path=camera.index,
        backend=camera.backend,
        fps=CAMERA_FPS,
        width=CAMERA_SIZE[0],
        height=CAMERA_SIZE[1],
        warmup_s=1,
    )
    selected_info = {
        "index_or_path": camera.index,
        "backend": int(camera.backend),
        "fps": CAMERA_FPS,
        "width": CAMERA_SIZE[0],
        "height": CAMERA_SIZE[1],
        "vid": CAMERA_VID,
        "pid": CAMERA_PID,
        "name": camera.name,
    }
    return config, report, selected_info


def json_values(values):
    return {key: float(value) for key, value in values.items()}


def feature_contract(features):
    contract = {}
    for key, value in features.items():
        if key not in ("action", "observation.state", "observation.images.wrist_camera"):
            continue
        contract[key] = {}
        for field in ("dtype", "shape", "names"):
            item = value.get(field)
            contract[key][field] = list(item) if isinstance(item, (tuple, list)) else item
    return contract


class CheckedImageWriter:
    """Write on our bounded dataset worker; propagate PNG errors to that worker."""

    def save_image(self, image, fpath, compress_level=1):
        from lerobot.datasets.image_writer import image_array_to_pil_image
        image_array_to_pil_image(image).save(fpath, compress_level=compress_level)

    def wait_until_done(self):
        pass  # save_image is synchronous on EpisodeWriter's thread.

    def stop(self):
        pass


def read_operator_keys():
    import msvcrt
    keys = []
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue
        keys.append(key)
    return keys


def operator_stop_requested():
    import msvcrt
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key.lower() == "q":
            return True
        if key in ("\x00", "\xe0") and msvcrt.kbhit():
            msvcrt.getwch()
    return False


def read_camera_frame(camera, timeout_ms=500):
    # Pinned 0.6.1 API: pair the image and its host receipt timestamp under one lock.
    if camera.thread is None or not camera.thread.is_alive():
        raise RuntimeError("Wrist camera native read thread is not running")
    if not camera.new_frame_event.wait(timeout_ms / 1000.0):
        raise TimeoutError("No new wrist camera frame")
    with camera.frame_lock:
        image, received = camera.latest_frame, camera.latest_timestamp
        camera.new_frame_event.clear()
        if image is None or received is None or image.shape != (CAMERA_SIZE[1], CAMERA_SIZE[0], 3):
            raise RuntimeError("Invalid wrist camera frame or host timestamp")
        return image.copy(), float(received)


def select_control_sample(samples, camera_time):
    for sample in reversed(samples):
        if sample["observation_time"] <= camera_time:
            if camera_time - sample["observation_time"] > 0.1:
                raise RuntimeError("No recent follower observation for wrist frame")
            if sample["action_time"] - camera_time > 0.1:
                raise RuntimeError("Follower action is too late relative to wrist frame")
            return sample
    return None  # A first camera frame may precede the first complete control sample.


class EpisodeWriter:
    """One bounded queue and one worker; no hidden unbounded image queue."""

    def __init__(self, dataset, task, emit):
        self.dataset, self.task, self.emit = dataset, task, emit
        self.queue = queue.Queue(maxsize=QUEUE_LIMIT)
        self.stop_event = threading.Event()
        self.errors = []
        self.frame_count = 0
        self.first_camera_time = self.last_camera_time = None
        self.first_frame_id = self.last_frame_id = None
        self.max_camera_interval = 0.0
        self.thread = threading.Thread(target=self.run, name="so101-dataset-writer", daemon=True)
        self.thread.start()

    def submit(self, item):
        if self.stop_event.is_set() or self.errors:
            raise RuntimeError("Dataset writer is stopping or failed")
        try:
            self.queue.put_nowait(item)
        except queue.Full as exc:
            self.errors.append("Dataset frame queue full; episode must be discarded")
            raise RuntimeError(self.errors[-1]) from exc

    def run(self):
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if self.errors:
                    continue
                observation = {**item["observation"], "wrist_camera": item["image"]}
                obs_frame = build_dataset_frame(self.dataset.features, observation, prefix=OBS_STR)
                action_frame = build_dataset_frame(self.dataset.features, item["sent_action"], prefix=ACTION)
                self.dataset.add_frame({**obs_frame, **action_frame, "task": self.task})
                i = self.frame_count
                if self.first_camera_time is None:
                    self.first_camera_time = item["camera_time"]
                    self.first_frame_id = item["camera_frame_id"]
                dt = None if self.last_camera_time is None else item["camera_time"] - self.last_camera_time
                skipped = 0 if self.last_frame_id is None else item["camera_frame_id"] - self.last_frame_id - 1
                self.last_camera_time, self.last_frame_id = item["camera_time"], item["camera_frame_id"]
                if dt is not None:
                    self.max_camera_interval = max(self.max_camera_interval, dt)
                self.emit({
                    "event": "episode_frame", "frame_index": i,
                    "dataset_timestamp_s": i / CAMERA_FPS,
                    "camera_frame_id": item["camera_frame_id"],
                    "camera_host_receive_monotonic": item["camera_time"],
                    "camera_interval_s": dt,
                    "reader_frames_skipped_since_previous_sample": skipped,
                    "camera_elapsed_s": item["camera_time"] - self.first_camera_time,
                    "timeline_deviation_ms": (item["camera_time"] - self.first_camera_time - i / CAMERA_FPS) * 1000,
                    "state_read_started_monotonic": item["observation_started"],
                    "state_observation_monotonic": item["observation_time"],
                    "action_send_started_monotonic": item["action_started"],
                    "sent_action_monotonic": item["action_time"],
                    "camera_to_state_ms": (item["observation_time"] - item["camera_time"]) * 1000,
                    "camera_to_action_ms": (item["action_time"] - item["camera_time"]) * 1000,
                    "follower_observation": item["observation"],
                    "leader_action": item["leader_action"],
                    "mapped_target": item["mapped_target"],
                    "bounded_target": item["bounded_target"],
                    "requested_action": item["requested_action"],
                    "sent_action_follower_coordinates": item["sent_action"],
                    "saturated_joints": item["saturated_joints"], "task": self.task,
                })
                self.frame_count += 1
            except BaseException as exc:
                self.errors.append(repr(exc))
            finally:
                self.queue.task_done()

    def stop(self, timeout=10):
        self.stop_event.set()  # Never block putting a sentinel into a full queue.
        self.thread.join(timeout)
        if self.thread.is_alive():
            if "Dataset writer stop timed out" not in self.errors:
                self.errors.append("Dataset writer stop timed out")
            return False
        if self.queue.unfinished_tasks:
            self.errors.append("Dataset writer exited with unfinished frames")
        return True

    def timing_summary(self):
        span = 0.0 if self.first_camera_time is None else self.last_camera_time - self.first_camera_time
        return {
            "sampled_camera_fps_measured": (self.frame_count - 1) / span if span > 0 else None,
            "sampled_camera_span_s": span,
            "dataset_span_s": max(0, self.frame_count - 1) / CAMERA_FPS,
            "timeline_deviation_ms": (span - max(0, self.frame_count - 1) / CAMERA_FPS) * 1000,
            "max_sampled_camera_interval_ms": self.max_camera_interval * 1000,
            "reader_frames_skipped": 0 if self.first_frame_id is None else self.last_frame_id - self.first_frame_id + 1 - self.frame_count,
        }


def open_dataset(dataset_id, follower, dataset_root=None):
    from lerobot.datasets import (
        LeRobotDataset,
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.processor import make_default_processors

    dataset_root = Path(dataset_root) if dataset_root is not None else ROOT / "datasets" / dataset_id
    repo_id = "local/" + dataset_id
    teleop_processor, _, observation_processor = make_default_processors()
    observation_features = dict(follower.observation_features)
    observation_features["wrist_camera"] = (CAMERA_SIZE[1], CAMERA_SIZE[0], 3)
    features = {
        **aggregate_pipeline_dataset_features(
            pipeline=teleop_processor,
            initial_features=create_initial_features(action=follower.action_features),
            use_videos=True,
        ),
        **aggregate_pipeline_dataset_features(
            pipeline=observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=True,
        ),
    }
    from lerobot.configs.video import RGBEncoderConfig
    writer_options = {
        "rgb_encoder": RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=23, g=2),
        "image_writer_processes": 0,
        "image_writer_threads": 0,
        "streaming_encoding": False,
    }
    if (dataset_root / ".recording-pending.json").exists():
        raise RuntimeError(f"Previous dataset session did not finalize; preserve and review: {dataset_root}")
    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        expected, actual = feature_contract(features), feature_contract(info["features"])
        if expected != actual or info["fps"] != CAMERA_FPS or info.get("robot_type") != "so101_follower":
            raise RuntimeError("Existing dataset schema, fps or robot_type differs; choose another dataset-id")
        # Force a local metadata load first: incomplete metadata must never trigger Hub fallback.
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        metadata = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
        metadata.root, metadata.repo_id = dataset_root, repo_id
        metadata._load_metadata()
        dataset = LeRobotDataset.resume(
            repo_id, root=dataset_root, batch_encoding_size=1,
            encoder_queue_maxsize=30, **writer_options,
        )
    else:
        if dataset_root.exists():
            raise RuntimeError(f"Dataset directory exists without info.json; preserve it and choose another id: {dataset_root}")
        # LeRobotMetadata.create itself creates root with exist_ok=False.
        dataset_root.parent.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset.create(
            repo_id, CAMERA_FPS, root=dataset_root, robot_type="so101_follower",
            features=features, use_videos=True, batch_encoding_size=1,
            encoder_queue_maxsize=30, **writer_options,
        )
    dataset.writer.image_writer = CheckedImageWriter()
    # This entry saves exactly one episode per process. Close numeric Parquet
    # before video encoding, so interrupted encoding cannot strand its footer.
    original_save_data = dataset.writer._save_episode_data

    def save_complete_numeric_data(episode_buffer):
        metadata = original_save_data(episode_buffer)
        dataset.writer.close_writer()
        return metadata

    dataset.writer._save_episode_data = save_complete_numeric_data
    return dataset, dataset_root, repo_id, features


def write_episode_view(dataset_root, episode_root, episode_index):
    """Provide per-episode files alongside the canonical v3 dataset, after finalize."""
    import shutil
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    dataset_root = Path(dataset_root)
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    rows = [row for path in sorted((dataset_root / "meta/episodes").rglob("*.parquet"))
            for row in pq.read_table(path).to_pylist()]
    row = next(row for row in rows if row["episode_index"] == episode_index)
    directory = Path(episode_root) / f"episode_{episode_index:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    data_source = dataset_root / info["data_path"].format(
        chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
    table = pq.read_table(data_source)
    table = table.filter(pc.equal(table["episode_index"], episode_index))
    if table.num_rows != row["length"]:
        raise RuntimeError("Episode view row count differs from canonical metadata")
    pq.write_table(table, directory / "data.parquet")
    references = {}
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        prefix = f"videos/{key}/"
        source = dataset_root / info["video_path"].format(
            video_key=key, chunk_index=row[prefix + "chunk_index"], file_index=row[prefix + "file_index"])
        if os.name == "nt":
            # Renamed encoder temp files retain their restrictive Windows DACL.
            # Inherit the destination directory ACL before exposing an episode link.
            import subprocess
            subprocess.run(["icacls", str(source), "/reset"], check=True, capture_output=True)
        references[key] = {"source": str(source), "from_timestamp": row[prefix + "from_timestamp"],
                           "to_timestamp": row[prefix + "to_timestamp"]}
        sharing = [r for r in rows if r.get(prefix + "chunk_index") == row[prefix + "chunk_index"]
                   and r.get(prefix + "file_index") == row[prefix + "file_index"]]
        if len(sharing) == 1:
            target = directory / (key.removeprefix("observation.images.") + ".mp4")
            try:
                os.link(source, target)  # Same-volume view; do not encode video again.
            except OSError:
                shutil.copy2(source, target)
            references[key]["episode_file"] = str(target)
    record = {"canonical_dataset": str(dataset_root), "format": info["codebase_version"],
              "episode_index": episode_index, "frames": row["length"], "fps": info["fps"],
              "tasks": row["tasks"], "videos": references,
              "note": "Canonical training dataset is the parent lerobot directory; this folder is an episode view."}
    (directory / "episode.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return directory


class EpisodeBatch:
    """Save episodes off the control thread while one teleop session stays active."""

    def __init__(self, args, follower, template):
        self.args, self.follower = args, follower
        self.template = json.loads(json.dumps(template))
        self.console = sys.stdout
        self.state = 'preparing'
        self.current = None
        self.worker = None
        self.error = None
        self.closing = threading.Event()
        self.control_active = False
        self.saved_this_session = 0
        self.saved_count = self._saved_count()
        self.progress_path = args.dataset_root.parent / 'batch-progress.json'
        self._prepare()

    def _say(self, message):
        self.console.write(message + '\n')
        self.console.flush()

    def _saved_count(self):
        info = self.args.dataset_root / 'meta' / 'info.json'
        return json.loads(info.read_text(encoding='utf-8'))['total_episodes'] if info.exists() else 0

    def _progress(self, status):
        value = {'task_id': self.args.dataset_id.removesuffix('_formal'), 'phase': 'formal',
                 'target_saved_episodes': self.args.target_total, 'saved_episodes': self.saved_count,
                 'status': status, 'control_remains_active': self.control_active and not self.closing.is_set(),
                 'updated': datetime.now().astimezone().isoformat()}
        temp = self.progress_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temp, self.progress_path)

    def _prepare(self):
        if self.saved_count >= self.args.target_total:
            self.current = None
            self.state = 'target_reached'
            self._progress(self.state)
            self._say('目标数量已保存。主从仍跟随；请托稳从臂后按X退出并释放电机。')
            return
        dataset, root, repo_id, features = open_dataset(self.args.dataset_id, self.follower, self.args.dataset_root)
        stamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
        stem = self.args.log_root / f'record-so101-{stamp}'
        outcome = json.loads(json.dumps(self.template))
        outcome.update(started=datetime.now().astimezone().isoformat(), status='ready',
                       dataset_root=str(root), repo_id=repo_id, episode_index=dataset.meta.total_episodes,
                       feature_contract=feature_contract(features), continuous_batch=True,
                       motor_enable_attempted=self.control_active, cleanup={'follower': 'kept active for reset' if self.control_active else 'not enabled', 'leader': 'teleop remains active' if self.control_active else 'not enabled'},
                       episode_saved=False, episode_commit_complete=False, frames_saved=0)
        context = {'dataset': dataset, 'root': root, 'outcome': outcome, 'summary': stem.with_suffix('.json'),
                   'log_path': stem.with_suffix('.jsonl'), 'log': stem.with_suffix('.jsonl').open('w', encoding='utf-8', buffering=1),
                   'start': None, 'closed': False, 'last_flush': time.perf_counter(),
                   'timing': {'count': 0, 'first': None, 'last': None, 'max_interval_ms': 0.0, 'intervals_over_35ms': 0}}
        context['emit'] = lambda row: self._emit(context, row)
        context['pending'] = root / '.recording-pending.json'
        context['pending'].write_text(json.dumps({'log': str(context['summary']), 'episode_index': dataset.meta.total_episodes}), encoding='utf-8')
        context['writer'] = EpisodeWriter(dataset, self.args.task, context['emit'])
        self.current = context
        self.state = 'ready'
        self._progress('ready_for_next_episode')
        self._say(f'READY：已保存{self.saved_count}/{self.args.target_total}。用主臂复位并重摆螺母；就位后按Enter开始下一条。复位不录制。')

    def _emit(self, context, row):
        now = time.perf_counter()
        context['log'].write(json.dumps({'time_s': time.time(), 'monotonic_s': now, **row}, separators=(',', ':')) + '\n')
        if row.get('event') != 'episode_frame' or now-context['last_flush'] >= .5:
            context['log'].flush()
            context['last_flush'] = now

    @property
    def recording(self):
        return self.state == 'recording'

    @property
    def started_at(self):
        return self.current['start'] if self.recording else float('inf')

    def start(self):
        if self.state != 'ready':
            self._say('请等待SAVED和READY提示，再按Enter开始；当前未开始新一条。')
            return
        self.current['outcome'].update(motor_enable_attempted=True, cleanup={'follower': 'kept active for reset', 'leader': 'teleop remains active'})
        self.current['start'] = time.perf_counter()
        self.current['outcome']['status'] = 'active'
        self.current['emit']({'event': 'recording_active', 'started_monotonic': self.current['start']})
        self.state = 'recording'
        self._progress('recording')
        self._say(f'RECORDING：第{self.saved_count+1}/{self.args.target_total}条。Q仅结束本条并保存；X退出控制。')

    def check(self):
        if self.error:
            raise RuntimeError('Continuous episode save failed: ' + self.error)
        if self.state in ('ready', 'recording'):
            writer = self.current['writer']
            if writer.errors or not writer.thread.is_alive():
                raise RuntimeError('Dataset writer failed: ' + str(writer.errors))

    def handle_key(self, key):
        if key.lower() == 'x' or key == '\x1b':
            raise KeyboardInterrupt
        if key in ('\r', '\n'):
            self.start()
        elif key.lower() == 'q' and self.recording:
            # Change state on the control thread before the next submit.
            self.state = 'saving'
            context = self.current
            self._say('SAVING：本条录制已停止，主从继续跟随。请等待SAVED；可用主臂复位。')
            self.worker = threading.Thread(target=self._save_and_prepare, args=(context,), name='so101-episode-save')
            self.worker.start()

    def note_control(self, sample):
        if not self.recording:
            return
        timing = self.current['timing']
        action_time = sample['action_time']
        if timing['first'] is None:
            timing['first'] = action_time
        if timing['last'] is not None:
            interval = (action_time-timing['last'])*1000
            timing['max_interval_ms'] = max(timing['max_interval_ms'], interval)
            timing['intervals_over_35ms'] += int(interval > 35)
        timing['last'] = action_time
        timing['count'] += 1

    def submit(self, item):
        if self.recording and item['camera_time'] >= self.started_at and item['observation_time'] >= self.started_at:
            self.current['writer'].submit(item)

    def _commit(self, context, status, usable):
        dataset, writer, outcome = context['dataset'], context['writer'], context['outcome']
        try:
            if not writer.stop():
                raise RuntimeError('Episode writer did not stop; preserve pending marker')
            if writer.errors:
                raise RuntimeError('Episode writer error: ' + str(writer.errors))
            count = writer.frame_count
            outcome.update(status=status, frames_written=count, sampling_timing=writer.timing_summary())
            timing = dict(context['timing'])
            span = timing['last']-timing['first'] if timing['first'] is not None else 0
            timing['hz_measured'] = (timing['count']-1)/span if span else None
            outcome['control_timing'] = timing
            if usable and count >= MIN_EPISODE_FRAMES:
                outcome['status'] = 'saving_episode'
                context['emit']({'event': 'saving_episode', 'frames': count})
                os.fsync(context['log'].fileno())
                context['summary'].write_text(json.dumps(outcome, indent=2), encoding='utf-8')
                dataset.save_episode()
                outcome['episode_saved'] = True
            else:
                dataset.clear_episode_buffer()
                outcome['status'] = 'stopped_without_usable_episode'
            dataset.finalize()
            outcome['episode_commit_complete'] = outcome['episode_saved']
            outcome['frames_saved'] = count if outcome['episode_saved'] else 0
            context['pending'].unlink(missing_ok=True)
            outcome['status'] = status if outcome['episode_saved'] else 'stopped_without_usable_episode'
        except BaseException as exc:
            outcome.update(status='failed', error=repr(exc))
            # Do not erase the pending marker after a failed save/finalize.
            try:
                dataset.finalize()
            except BaseException as final_exc:
                outcome['dataset_finalize_error'] = repr(final_exc)
            raise
        finally:
            outcome['ended'] = datetime.now().astimezone().isoformat()
            context['emit']({'event': 'session_summary', 'summary': outcome})
            context['log'].close()
            context['closed'] = True
            context['summary'].write_text(json.dumps(outcome, indent=2), encoding='utf-8')
        if outcome['episode_commit_complete'] and self.args.episode_root is not None:
            import shutil
            try:
                episode_dir = write_episode_view(context['root'], self.args.episode_root, outcome['episode_index'])
                outcome['episode_directory'] = str(episode_dir)
                context['summary'].write_text(json.dumps(outcome, indent=2), encoding='utf-8')
                shutil.copy2(context['log_path'], episode_dir / 'alignment.jsonl')
                shutil.copy2(context['summary'], episode_dir / 'recording_summary.json')
            except BaseException as exc:
                outcome['episode_view_error'] = repr(exc)
                context['summary'].write_text(json.dumps(outcome, indent=2), encoding='utf-8')
                raise
        return bool(outcome['episode_commit_complete'])

    def _save_and_prepare(self, context):
        try:
            saved = self._commit(context, 'completed', True)
            self.saved_count = self._saved_count()
            self.saved_this_session += int(saved)
            self._progress('episode_saved' if saved else 'episode_too_short')
            if saved:
                control_note = '控制已退出。' if self.closing.is_set() else '主从继续跟随。'
                self._say(f'SAVED：已保存{self.saved_count}/{self.args.target_total}；{control_note}')
            else:
                self._say('本条不足1秒，未保存；已保存数据保持不变。')
            if not self.closing.is_set():
                self._prepare()
        except BaseException as exc:
            self.error = repr(exc)
            self.state = 'failed'
            try:
                self.saved_count = self._saved_count()
                self._progress('failed')
            except BaseException:
                pass

    def finish(self, status):
        self.closing.set()
        # Motors must already be released by main's cleanup before waiting here.
        if self.worker is not None:
            self.worker.join()
        context = self.current
        if context is not None and not context['closed']:
            context['outcome']['cleanup'] = {'follower': 'control stopped; see batch summary', 'leader': 'control stopped; see batch summary'}
            saved = self._commit(context, status, self.state == 'recording' and status != 'failed')
            self.saved_this_session += int(saved)
            self.saved_count = self._saved_count()
        if self.error:
            raise RuntimeError(self.error)
        self._progress('stopped')


def main():
    parser = argparse.ArgumentParser(description="Record corrected-coordinate SO-101 episodes, optionally keeping teleoperation active between episodes.")
    parser.add_argument("--task", required=True, help="Natural-language task instruction stored with the episode")
    parser.add_argument("--dataset-id", default="so101_trials", help="Local dataset folder name; repeated runs append episodes")
    parser.add_argument("--dataset-root", type=Path, help="Explicit canonical v3 dataset path")
    parser.add_argument("--episode-root", type=Path, help="Per-episode views alongside the canonical dataset")
    parser.add_argument("--log-root", type=Path, help="Directory for recording session diagnostics")
    parser.add_argument("--target-total", type=int, help="Keep following between episodes until this saved total; X exits control")
    args = parser.parse_args()
    if args.target_total is not None and (not 1 <= args.target_total <= 500 or not all((args.dataset_root, args.episode_root, args.log_root))):
        parser.error("--target-total requires explicit dataset/episode/log roots and a total from 1 to 500")
    for path in (args.dataset_root, args.episode_root, args.log_root):
        if path is not None and (not path.is_absolute() or not path.resolve().is_relative_to(ROOT.resolve())):
            parser.error("Recording paths must be absolute paths under the release workspace")
    if not args.task.strip():
        parser.error("--task cannot be empty")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.dataset_id):
        parser.error("--dataset-id must be 1-64 lowercase letters, digits, '_' or '-' and start with a letter/digit")

    os.environ["HF_HUB_OFFLINE"] = "1"  # This entry only appends local data.
    follow = load_follow_module()
    cfg = follow.load_config()
    leader, follower = follow.make_devices(cfg)
    camera_config, camera_inventory, camera_info = choose_camera()
    from lerobot.cameras.opencv import OpenCVCamera
    camera = OpenCVCamera(camera_config)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    log_stem = (args.log_root or ROOT / "logs") / (("batch-so101-" if args.target_total is not None else "record-so101-") + stamp)
    event_log_path = log_stem.with_suffix(".jsonl")
    summary_path = log_stem.with_suffix(".json")
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    event_log = event_log_path.open("w", encoding="utf-8", buffering=1)
    log_lock = threading.Lock()

    outcome = {
        "started": datetime.now().astimezone().isoformat(),
        "status": "checking",
        "task": args.task,
        "dataset_id": args.dataset_id,
        "dataset_root": str(args.dataset_root or ROOT / "datasets" / args.dataset_id),
        "control_fps_target": CONTROL_FPS,
        "dataset_fps": CAMERA_FPS,
        "camera": camera_info,
        "camera_inventory": camera_inventory,
        "lerobot_version": cfg["lerobot_version"],
        "coordinate_frame_version": cfg.get("coordinate_frame_version"),
        "calibration_sha256": cfg["calibration_sha256"],
        "mapping_offsets": cfg["offset_degrees"],
        "motor_enable_attempted": False,
        "cleanup": {},
        "frames_saved": 0,
        "episode_saved": False,
        "episode_commit_complete": False,
        "camera_timestamp_semantics": "host frame postprocess receipt; not hardware exposure",
    }

    batch = None
    camera_stop = threading.Event()
    recording_active = threading.Event()
    camera_lock = threading.Lock()
    camera_latest = {"frame_id": 0, "image": None, "received_monotonic": None}
    camera_errors = []
    writer = None
    camera_thread = None
    dataset = None
    pending_path = None
    mapping = None
    cleanup_failed = False
    control_timing = {"count": 0, "first": None, "last": None, "max_interval_ms": 0.0, "intervals_over_35ms": 0}
    last_flushed_at = [time.perf_counter()]

    def emit(row, force=False):
        now = time.perf_counter()
        record = {"time_s": time.time(), "monotonic_s": now, **row}
        with log_lock:
            event_log.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
            if force or now - last_flushed_at[0] >= 0.5 or row.get("event") != "episode_frame":
                event_log.flush()
                last_flushed_at[0] = now

    def camera_reader():
        with camera_lock:
            frame_id, last_received = camera_latest["frame_id"], camera_latest["received_monotonic"]
        while not camera_stop.is_set():
            try:
                image, received = read_camera_frame(camera)
                if received <= last_received:
                    continue
                frame_id += 1
                last_received = received
                with camera_lock:
                    camera_latest.update({"frame_id": frame_id, "image": image, "received_monotonic": received})
            except BaseException as exc:
                if not camera_stop.is_set():
                    camera_errors.append(repr(exc))
                return

    def stop_camera():
        camera_stop.set()
        if camera_thread is not None:
            camera_thread.join(timeout=2)
            if camera_thread.is_alive():
                camera_errors.append("Camera consumer did not stop within 2 seconds")

    def cleanup_buses():
        nonlocal cleanup_failed
        for role, device in (("follower", follower), ("leader", leader)):
            if not device.bus.is_connected:
                continue
            try:
                if outcome["motor_enable_attempted"]:
                    device.bus.disable_torque(num_retry=3)
                    torque = {name: device.bus.read("Torque_Enable", name, normalize=False, num_retry=3) for name in JOINTS}
                    outcome["cleanup"][role] = {"torque": torque}
                    if any(torque.values()):
                        raise RuntimeError("Torque release could not be verified")
                else:
                    outcome["cleanup"][role] = "closed without motion setup"
            except BaseException as exc:
                cleanup_failed = True
                outcome["cleanup"][role] = {"error": repr(exc)}
                print(f"Cleanup check failed for {role}: {exc}", flush=True)
            finally:
                try:
                    device.bus.disconnect(disable_torque=False)
                except BaseException as exc:
                    cleanup_failed = True
                    outcome["cleanup"][role + "_disconnect_error"] = repr(exc)
                    print(f"Closing {role}: {exc}", flush=True)

    try:
        camera.connect()
        probe_image, probe_received = read_camera_frame(camera, timeout_ms=1500)
        with camera_lock:
            camera_latest.update({"frame_id": 1, "image": probe_image, "received_monotonic": probe_received})
        camera_thread = threading.Thread(target=camera_reader, name="so101-wrist-camera", daemon=True)
        camera_thread.start()

        from serial.tools import list_ports
        ports = {port.serial_number: port.device for port in list_ports.comports()}
        for role, serial_number in cfg["usb_serials"].items():
            if ports.get(serial_number) != cfg["ports"][role]:
                raise RuntimeError(f"USB identity changed: {role}; expected {serial_number} at {cfg['ports'][role]}, found {ports.get(serial_number)}")

        for device in (leader, follower):
            device.bus.connect()
        preflight = follow.inspect_connected(cfg, leader, follower, allow_start_outside=True)
        emit({"event": "preflight", "snapshot": preflight}, force=True)
        if preflight["follower"].get("outside_start_limits"):
            print("起始姿态提示：" + json.dumps(preflight["follower"]["outside_start_limits"], ensure_ascii=False), flush=True)
            print("小幅越界允许准备；按Enter后从当前位置平滑接入3秒，随后仍使用原工作限位。", flush=True)
        if args.target_total is not None:
            batch = EpisodeBatch(args, follower, outcome)
            dataset_root = args.dataset_root
        else:
            dataset, dataset_root, repo_id, dataset_features = open_dataset(args.dataset_id, follower, args.dataset_root)
            outcome["dataset_root"] = str(dataset_root)
            outcome["repo_id"] = repo_id
            outcome["episode_index"] = dataset.meta.total_episodes
            pending_path = dataset_root / ".recording-pending.json"
            pending_path.write_text(json.dumps({"log": str(summary_path), "episode_index": outcome["episode_index"]}), encoding="utf-8")
            outcome["feature_contract"] = feature_contract(dataset_features)
        print(f"Preflight passed. Leader={cfg['ports']['leader']}; follower={cfg['ports']['follower']}; wrist camera={camera_info['index_or_path']}.", flush=True)
        print(f"Dataset: {dataset_root} | task: {args.task}", flush=True)
        print("Control target 60 Hz; dataset/video 30 Hz; action values are the follower goals actually returned by send_action.", flush=True)
        if batch:
            print("连续采集：Q结束当前条并后台保存，主从保持跟随；用主臂复位后Enter录下一条。托稳从臂后按X退出控制。", flush=True)
        else:
            print("Before starting, place the task in the wrist camera view and clear the arm path. Support the follower before pressing Q; stopping releases torque.", flush=True)
        if camera_errors:
            raise RuntimeError("Wrist camera read failed during preflight: " + camera_errors[-1])

        with contextlib.nullcontext():
            writer = None if batch else EpisodeWriter(dataset, args.task, emit)
            try:
                input("Press ENTER when ready to start the demonstration: ")
                if camera_errors:
                    raise RuntimeError("Wrist camera read failed before motor enable: " + camera_errors[-1])
                preflight = follow.inspect_connected(cfg, leader, follower, allow_start_outside=True)
                mapping = follow.WristMapping(cfg, follower)
                mapping.prepare_start(preflight["follower"]["normalized"])
                # Configuration can enable torque even when a later startup read fails.
                # Claim cleanup responsibility before the first configuration/goal write.
                outcome["motor_enable_attempted"] = True
                leader.configure()
                follower_raw = follower.bus.sync_read("Present_Position", normalize=False, num_retry=3)
                follower.bus.sync_write("Goal_Position", follower_raw, normalize=False)
                if follower.bus.sync_read("Goal_Position", normalize=False, num_retry=3) != follower_raw:
                    raise RuntimeError("Follower initial goal readback mismatch")
                follower.configure()
                from lerobot.robots.utils import ensure_safe_goal_position
                from lerobot.scripts.lerobot_teleoperate import teleop_loop
                _, robot_action_processor, robot_observation_processor = __import__("lerobot.processor", fromlist=["make_default_processors"]).make_default_processors()
                metrics = {}
                observation_cache = {"value": None, "read_at": None, "read_started": None}
                original_observe = follower.get_observation
                original_leader_action = leader.get_action
                original_send = follower.send_action
                control_samples = deque(maxlen=256)
                last_camera_frame_id = [0]
                original_stdout = sys.stdout

                def observe():
                    started = time.perf_counter()
                    observation = original_observe()
                    observation_cache.update({"value": observation, "read_at": time.perf_counter(), "read_started": started})
                    return observation

                def leader_action():
                    return original_leader_action()

                def process(item):
                    return mapping(item)

                def logged_send(action):
                    if batch:
                        for key in read_operator_keys():
                            batch.handle_key(key)
                        batch.check()
                    elif operator_stop_requested():
                        raise KeyboardInterrupt
                    if camera_errors:
                        raise RuntimeError("Wrist camera read failed: " + camera_errors[-1])
                    if writer is not None and (writer.errors or not writer.thread.is_alive()):
                        raise RuntimeError("Dataset writer failed: " + str(writer.errors))
                    begin = time.perf_counter()
                    refreshed = False
                    if cfg.get("reuse_current_observation"):
                        observation = observation_cache["value"]
                        age = time.perf_counter() - observation_cache["read_at"]
                        if age > cfg["observation_max_age_seconds"]:
                            started = time.perf_counter()
                            observation = original_observe()
                            observation_cache.update({"value": observation, "read_at": time.perf_counter(), "read_started": started})
                            action = mapping((mapping.pending["leader_action"], observation))
                            refreshed = True
                        if set(key for key in observation if key.endswith(".pos")) != set(KEYS):
                            raise RuntimeError("Incomplete current-frame follower observation")
                        goals = {name: (float(action[name + ".pos"]), float(observation[name + ".pos"])) for name in JOINTS}
                        safe = ensure_safe_goal_position(goals, cfg["max_relative_target"])
                        action_started = time.perf_counter()
                        sent = original_send({name + ".pos": float(safe[name]) for name in JOINTS})
                    else:
                        action_started = time.perf_counter()
                        sent = original_send(action)
                    action_time = time.perf_counter()
                    if control_timing["first"] is None:
                        control_timing["first"] = action_time
                    if control_timing["last"] is not None:
                        interval_ms = (action_time - control_timing["last"]) * 1000
                        control_timing["max_interval_ms"] = max(control_timing["max_interval_ms"], interval_ms)
                        control_timing["intervals_over_35ms"] += int(interval_ms > 35)
                    control_timing["last"] = action_time
                    control_timing["count"] += 1
                    observation_time = observation_cache["read_at"]
                    pending = dict(mapping.pending)
                    control_sample = {
                        "observation_time": observation_time,
                        "observation_started": observation_cache["read_started"],
                        "action_started": action_started,
                        "action_time": action_time,
                        "observation": {key: float(observation_cache["value"][key]) for key in KEYS},
                        "leader_action": json_values(pending["leader_action"]),
                        "mapped_target": json_values(pending["mapped_target"]),
                        "bounded_target": json_values(pending["bounded_target"]),
                        "requested_action": json_values(pending["requested_action"]),
                        "sent_action": json_values(sent),
                        "saturated_joints": list(pending["saturated_joints"]),
                    }
                    control_samples.append(control_sample)
                    if batch:
                        batch.note_control(control_sample)
                    with camera_lock:
                        camera_frame = dict(camera_latest)
                    capture_time = camera_frame["received_monotonic"]
                    if capture_time is None or time.perf_counter() - capture_time > 0.1:
                        raise RuntimeError("Wrist camera frame is stale")
                    is_recording = batch.recording if batch else recording_active.is_set()
                    if is_recording and camera_frame["frame_id"] != last_camera_frame_id[0]:
                        last_camera_frame_id[0] = camera_frame["frame_id"]
                        started_at = batch.started_at if batch else outcome["recording_started_monotonic"]
                        if capture_time >= started_at:
                            sample = select_control_sample(control_samples, capture_time)
                            if sample is not None:
                                # Preserve every observed unique camera frame. A second 30 Hz
                                # gate can discard frames merely because of arrival jitter.
                                item = {**sample, "image": camera_frame["image"], "camera_time": capture_time,
                                        "camera_frame_id": camera_frame["frame_id"]}
                                if batch:
                                    batch.submit(item)
                                else:
                                    writer.submit(item)
                    end = time.perf_counter()
                    metrics.update({"last_send_ms": (end - begin) * 1000.0, "feedback_refreshed": refreshed})
                    return sent

                follower.get_observation = observe
                leader.get_action = leader_action
                follower.send_action = logged_send
                from lerobot.processor import make_default_processors
                _, _, robot_observation_processor = make_default_processors()
                logging.getLogger().setLevel(logging.ERROR)
                outcome["status"] = "active"
                if batch:
                    batch.control_active = True
                    batch.start()
                else:
                    outcome["recording_started_monotonic"] = time.perf_counter()
                    recording_active.set()
                    emit({"event": "recording_active", "started_monotonic": outcome["recording_started_monotonic"]}, force=True)
                    print("ACTIVE: perform the task with the black leader. Support the follower, then press Q once to finish. Keep this window open until SAVED.", flush=True)
                with contextlib.redirect_stdout(follow.LoopStatusOutput(original_stdout)):
                    teleop_loop(
                        teleop=leader,
                        robot=follower,
                        fps=CONTROL_FPS,
                        teleop_action_processor=process,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        display_data=False,
                        duration=None,
                    )
                outcome["status"] = "completed"
            except KeyboardInterrupt:
                outcome["status"] = "stopped_by_user"
            except BaseException as exc:
                outcome["status"] = "failed"
                outcome["error"] = repr(exc)
                raise
    except KeyboardInterrupt:
        outcome["status"] = "stopped_by_user"
    except BaseException as exc:
        outcome["status"] = "failed"
        outcome["error"] = repr(exc)
        raise
    finally:
        # Further Ctrl+C must not interrupt torque release, queue drain or file closure.
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        recording_active.clear()
        cleanup_buses()  # Release/read back motor torque before waiting for any I/O.
        stop_camera()
        if camera.is_connected:
            try:
                camera.disconnect()
            except BaseException as exc:
                camera_errors.append(repr(exc))
        if batch:
            if cleanup_failed or camera_errors:
                outcome["status"] = "failed"
            try:
                batch.finish(outcome["status"])
            except BaseException as exc:
                outcome.update(status="failed", error=repr(exc))
            outcome.update(batch_saved_total=batch.saved_count, saved_episodes_this_session=batch.saved_this_session,
                           continuous_batch=True)
        else:
            writer_stopped = writer is None or writer.stop()
            writer_errors = [] if writer is None else writer.errors
            if camera_errors:
                outcome["camera_error"] = camera_errors
            if writer_errors:
                outcome["writer_error"] = writer_errors
            if cleanup_failed or camera_errors or writer_errors:
                outcome["status"] = "failed"
            count = 0 if writer is None else writer.frame_count
            outcome["frames_written"] = count
            if writer is not None:
                outcome["sampling_timing"] = writer.timing_summary()
            outcome["control_timing"] = control_timing
            span = None if control_timing["first"] is None else control_timing["last"] - control_timing["first"]
            outcome["control_timing"]["hz_measured"] = (control_timing["count"] - 1) / span if span else None
            if dataset is not None and writer_stopped:
                try:
                    usable = outcome["status"] in ("stopped_by_user", "completed") and count >= MIN_EPISODE_FRAMES
                    if usable:
                        print(f"STOPPED: motors released. Saving {count} frames; keep this window open and wait for SAVED.", flush=True)
                        emit({"event": "saving_episode", "frames": count}, force=True)
                        os.fsync(event_log.fileno())
                        summary_path.write_text(json.dumps({**outcome, "status": "saving_episode", "frames_written": count}, indent=2), encoding="utf-8")
                        dataset.save_episode()
                        outcome["episode_saved"] = True
                    else:
                        dataset.clear_episode_buffer()
                        if outcome["status"] in ("stopped_by_user", "completed"):
                            outcome["status"] = "stopped_without_usable_episode"
                    dataset.finalize()
                    outcome["episode_commit_complete"] = outcome["episode_saved"]
                    outcome["frames_saved"] = count if outcome["episode_saved"] else 0
                    if pending_path is not None:
                        pending_path.unlink(missing_ok=True)
                except BaseException as exc:
                    # Save is not transactional in 0.6.1; preserve partial output and marker.
                    outcome["status"] = "failed"
                    outcome["dataset_save_or_finalize_error"] = repr(exc)
                    try:
                        dataset.finalize()
                    except BaseException as finalize_exc:
                        outcome["dataset_finalize_error"] = repr(finalize_exc)
            elif dataset is not None:
                outcome["status"] = "failed"
                outcome["dataset_finalize_skipped"] = "Writer is still active; preserve pending marker for review"
        outcome["ended"] = datetime.now().astimezone().isoformat()
        outcome.pop("recording_started_monotonic", None)
        emit({"event": "session_summary", "summary": outcome}, force=True)
        event_log.close()
        summary_path.write_text(json.dumps(outcome, indent=2, ensure_ascii=True), encoding="utf-8")
        if outcome["episode_commit_complete"] and args.episode_root is not None:
            try:
                import shutil
                episode_dir = write_episode_view(dataset_root, args.episode_root, outcome["episode_index"])
                outcome["episode_directory"] = str(episode_dir)
                summary_path.write_text(json.dumps(outcome, indent=2, ensure_ascii=True), encoding="utf-8")
                shutil.copy2(event_log_path, episode_dir / "alignment.jsonl")
                shutil.copy2(summary_path, episode_dir / "recording_summary.json")
            except BaseException as exc:
                outcome["episode_view_error"] = repr(exc)
                summary_path.write_text(json.dumps(outcome, indent=2, ensure_ascii=True), encoding="utf-8")
                print(f"Dataset saved; episode view needs attention: {exc}", flush=True)
        print(f"Finished: {outcome['status']}. Summary: {summary_path}", flush=True)
        print(f"Detailed alignment log: {event_log_path}", flush=True)
        if outcome["episode_commit_complete"]:
            print("SAVED: episode files and metadata are complete; this window can now be closed.", flush=True)
        signal.signal(signal.SIGINT, previous_sigint)
    return 1 if outcome["status"] == "failed" or "episode_view_error" in outcome else 0


if __name__ == "__main__":
    raise SystemExit(main())

