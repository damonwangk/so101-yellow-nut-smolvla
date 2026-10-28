"""Read-only SO-101 sample -> DataLoader -> checkpoint preprocessor demonstration.

Run in the existing Windows environment, not on the documentation Mac:
    Set-Location D:\\so-101
    .\\.venv\\Scripts\\python.exe .\\explain_real_batch.py --output D:\\so-101\\docs\\batch-demo

No policy weights, policy forward, optimizer, camera, serial bus or robot object is
created. Only local data/tokenizer/processor files are read. Output goes into a
NEW directory selected by --output; the source dataset is never modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(r"D:\so-101")
DATASET = ROOT / "datasets/yellow_nut_to_brown_tray/formal/lerobot"
CHECKPOINT = ROOT / "outputs/train/yellow_nut_smolvla_20261019T110537/checkpoints/005000/pretrained_model"
TOKENIZER = ROOT / "models/SmolVLM2-500M-Video-Instruct"
ANCHORS = [0, 30, 60, 120, 180, 245, 440, 480]
CAMERA = "observation.images.wrist_camera"
KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shape_map(batch):
    import torch

    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, torch.Tensor)
        else {"type": type(value).__name__, **({"length": len(value)} if hasattr(value, "__len__") else {"value": value})}
        for key, value in batch.items()
    }


def build_fixture(dataset_root=DATASET, checkpoint=CHECKPOINT, tokenizer_root=TOKENIZER):
    """Return JSON-serializable fixture plus true decoded JPEG bytes, no writes."""
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoTokenizer

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    dataset_root, checkpoint, tokenizer_root = map(Path, (dataset_root, checkpoint, tokenizer_root))
    # Hash the real source files before and after. No source writer is used.
    native_paths = sorted(p for base in ("data", "meta", "videos") for p in (dataset_root / base).rglob("*") if p.is_file())
    hashes_before = {str(p.relative_to(dataset_root)).replace("\\", "/"): sha256(p) for p in native_paths}
    cfg = SmolVLAConfig.from_pretrained(checkpoint)
    cfg.device = "cpu"
    cfg.vlm_model_name = str(tokenizer_root)
    meta = LeRobotDatasetMetadata("local/yellow_nut_to_brown_tray_formal", root=dataset_root)
    deltas = resolve_delta_timestamps(cfg, meta)
    # These are the same factory arguments as the actual training reader. We
    # deliberately select one TRAIN episode so all eight anchors are easy to see.
    ds = LeRobotDataset(
        "local/yellow_nut_to_brown_tray_formal", root=dataset_root,
        episodes=[22], delta_timestamps=deltas,
        video_backend="pyav", return_uint8=True, image_transforms=None,
    )
    assert ds.num_frames == 490
    assert not ds.meta.has_language_columns
    batch = next(iter(DataLoader(Subset(ds, ANCHORS), batch_size=8, shuffle=False, num_workers=0)))
    raw = copy.deepcopy(batch)
    pre, _post = make_pre_post_processors(
        cfg, pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_root)},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    trainer_converted = copy.deepcopy(batch)
    # lerobot_train.py converts uint8 camera tensors BEFORE its preprocessor.
    # VISUAL:IDENTITY does not perform this pixel conversion for us.
    for camera_key in ds.meta.camera_keys:
        if trainer_converted[camera_key].dtype == torch.uint8:
            trainer_converted[camera_key] = trainer_converted[camera_key].float().div(255)
    processed = pre(copy.deepcopy(trainer_converted))
    # Call ONLY parameter-free preparation helpers. This tiny object contains
    # a config, no neural network and no loaded policy parameters.
    prepare_only = type("PreparationOnly", (), {"config": cfg})()
    images, image_masks = SmolVLAPolicy.prepare_images(prepare_only, processed)
    padded_state = SmolVLAPolicy.prepare_state(prepare_only, processed)
    padded_action = SmolVLAPolicy.prepare_action(prepare_only, processed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    tokens = processed[OBS_LANGUAGE_TOKENS]
    language_mask = processed[OBS_LANGUAGE_ATTENTION_MASK]
    image_bytes = {}
    samples = []
    for row, anchor in enumerate(ANCHORS):
        assert int(raw["frame_index"][row]) == anchor
        assert int(raw["episode_index"][row]) == 22
        img = raw[CAMERA][row]
        # Export exactly the RGB tensor returned by the training reader. JPEG
        # is only the browser preview; hashes/pixel probes describe source RGB.
        rgb = img.permute(1, 2, 0).numpy()
        assert rgb.dtype == np.uint8 and rgb.shape == (480, 640, 3)
        filename = f"episode22-frame{anchor:04d}.jpg"
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format="JPEG", quality=92, optimize=True)
        image_bytes[filename] = output.getvalue()
        state = raw["observation.state"][row].reshape(-1)
        norm_state = processed["observation.state"][row].reshape(-1)
        ids = tokens[row].tolist()
        mask = language_mask[row].tolist()
        sample = {
            "episode_index": 22, "frame_index": anchor,
            "index": int(raw["index"][row]), "timestamp": float(raw["timestamp"][row]),
            "task_index": int(raw["task_index"][row]), "task": raw["task"][row],
            "task_after_newline": raw["task"][row] if raw["task"][row].endswith("\n") else raw["task"][row] + "\n",
            "image": f"assets/batch/{filename}",
            "image_tensor_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "image_pixel_rgb_y240_x320": rgb[240, 320].tolist(),
            "image_tensor_minmax": [int(img.min()), int(img.max())],
            "state": state.tolist(), "actions": raw["action"][row].tolist(),
            "action_is_pad": raw["action_is_pad"][row].tolist(),
            "normalized_state": norm_state.tolist(),
            "normalized_actions": processed["action"][row].tolist(),
            "action_frame_indices": [min(anchor + t, 489) for t in range(50)],
            "valid_action_count": int((~raw["action_is_pad"][row]).sum()),
            "token_ids": ids, "token_attention_mask": mask,
            "token_pieces": tokenizer.convert_ids_to_tokens(ids),
            "token_decode_valid": tokenizer.decode([token for token, valid in zip(ids, mask) if valid], skip_special_tokens=False),
            "token_decode_padded": tokenizer.decode(ids, skip_special_tokens=False),
        }
        assert sample["valid_action_count"] == min(50, 490 - anchor)
        samples.append(sample)
    # Read original Parquet to cross-check the absolute row, not just our own
    # fixture serialization. This also proves all tail values repeat last frame.
    episode_path = dataset_root / meta.get_data_file_path(22)
    table = pq.read_table(episode_path).to_pydict()
    episode_rows = {int(frame): i for i, (ep, frame) in enumerate(zip(table["episode_index"], table["frame_index"])) if ep == 22}
    for sample in samples:
        row = episode_rows[sample["frame_index"]]
        assert np.array_equal(np.array(sample["state"], dtype=np.float32), np.array(table["observation.state"][row], dtype=np.float32))
        for t, frame in enumerate(sample["action_frame_indices"]):
            assert np.array_equal(np.array(sample["actions"][t], dtype=np.float32), np.array(table["action"][episode_rows[frame]], dtype=np.float32))
    hashes_after = {str(p.relative_to(dataset_root)).replace("\\", "/"): sha256(p) for p in native_paths}
    assert hashes_before == hashes_after
    # Saved processor statistics, rather than recomputing from all 50 episodes.
    from safetensors.torch import load_file
    norm_file = checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    norm_tensors = load_file(str(norm_file))
    stats = {key: value.tolist() for key, value in norm_tensors.items() if "observation.state" in key or "action" in key}
    fixture = {
        "schema_version": 1, "recorded_at": "2026-10-21",
        "provenance": {
            "kind": "Real official LeRobotDataset/DataLoader/checkpoint processor outputs; deterministic teaching batch",
            "source_dataset": str(dataset_root), "checkpoint": str(checkpoint),
            "source_episode": 22, "episode_length": 490, "anchors": ANCHORS,
            "source_native_file_count": len(native_paths),
            "source_hashes_unchanged": hashes_before == hashes_after,
            "source_hashes": hashes_before,
            "processor_sha256": sha256(checkpoint / "policy_preprocessor.json"),
            "normalizer_sha256": sha256(norm_file),
            "lerobot_version": importlib.metadata.version("lerobot"),
            "torch_version": torch.__version__, "video_backend": "pyav",
            "reader_return_uint8": True, "no_policy_weights_loaded": True,
            "no_policy_forward": True, "hardware_objects_created": 0,
            "not_a_replayed_historical_training_batch": True,
            "image_preview": "JPEG exported from the exact reader RGB tensor; lossy display derivative",
        },
        "joint_names": KEYS,
        "joint_units": ["degrees"] * 5 + ["percentage_points"],
        "fps": meta.fps, "action_window_length": cfg.chunk_size,
        "delta_timestamps": deltas,
        "samples": samples,
        "batch_shapes": shape_map(raw),
        "trainer_converted_shapes": shape_map(trainer_converted),
        "preprocessed_shapes": shape_map(processed),
        "model_ready_shapes": {
            "images[0]": {"shape": list(images[0].shape), "dtype": str(images[0].dtype), "minmax": [float(images[0].min()), float(images[0].max())]},
            "image_masks[0]": {"shape": list(image_masks[0].shape), "dtype": str(image_masks[0].dtype)},
            "state": {"shape": list(padded_state.shape), "dtype": str(padded_state.dtype)},
            "action": {"shape": list(padded_action.shape), "dtype": str(padded_action.dtype)},
        },
        "tokens": {
            "shape": list(tokens.shape), "dtype": str(tokens.dtype),
            "attention_mask_shape": list(language_mask.shape),
            "ids": tokens[0].tolist(), "attention_mask": language_mask[0].tolist(),
            "pieces": tokenizer.convert_ids_to_tokens(tokens[0].tolist()),
            "decoded_valid": samples[0]["token_decode_valid"],
            "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
        },
        "normalizer_checkpoint_tensors": stats,
        "pixels_example": {
            "raw_rgb_y240_x320": samples[0]["image_pixel_rgb_y240_x320"],
            "preprocessed_rgb_y240_x320": processed[CAMERA][0, :, 240, 320].tolist(),
            "model_ready_pad_rgb_y0_x0": images[0][0, :, 0, 0].tolist(),
        },
    }
    return fixture, image_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # A new artifact directory makes accidental overwrites obvious.
    if args.output.exists():
        raise SystemExit("Choose a NEW --output directory; existing output is preserved.")
    fixture, pictures = build_fixture(args.dataset, args.checkpoint, args.tokenizer)
    args.output.mkdir(parents=True)
    image_dir = args.output / "batch"
    image_dir.mkdir()
    for name, data in pictures.items():
        (image_dir / name).write_bytes(data)
    (args.output / "batch-real.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Dataset anchor frames:", ANCHORS)
    print("DataLoader batch:", json.dumps(fixture["batch_shapes"], ensure_ascii=False, indent=2))
    print("After checkpoint preprocessor:", json.dumps(fixture["preprocessed_shapes"], ensure_ascii=False, indent=2))
    print("After parameter-free SmolVLA prepare helpers:", json.dumps(fixture["model_ready_shapes"], indent=2))
    print("frame0 raw state:", fixture["samples"][0]["state"])
    print("frame0 normalized state:", fixture["samples"][0]["normalized_state"])
    print("frame480 valid/pad action positions:", fixture["samples"][-1]["valid_action_count"], sum(fixture["samples"][-1]["action_is_pad"]))
    print("Task token IDs:", fixture["tokens"]["ids"])
    print("Task valid decode:", repr(fixture["tokens"]["decoded_valid"]))
    print("Source unchanged:", fixture["provenance"]["source_hashes_unchanged"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
