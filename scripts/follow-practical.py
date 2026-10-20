"""Trial wrist correction using installed LeRobot drivers and official teleop_loop.
Run --offline-check without hardware, --read-only for live checks, or interactively.
"""
import argparse
import contextlib
import sys
import hashlib
import json
import logging
import math
import statistics
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from so101_paths import ROOT
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
KEYS = tuple(n + '.pos' for n in JOINTS)
def load_config():
    cfg = json.loads((ROOT/'configs/practical-follow.json').read_text(encoding='utf-8-sig'))
    if version('lerobot') != cfg['lerobot_version']:
        raise RuntimeError('LeRobot version changed; review mapping before starting')
    for role, expected in cfg['calibration_sha256'].items():
        path = ROOT/'calibration'/role/('win_'+role+'.json')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(role + ' calibration changed; correction must be reviewed')
    return cfg
def make_devices(cfg):
    from lerobot.teleoperators.so_leader.so_leader import SOLeader
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    leader = SOLeader(SOLeaderTeleopConfig(port=cfg['ports']['leader'], id='win_leader', calibration_dir=ROOT/'calibration/leader', use_degrees=True))
    follower = SOFollower(SOFollowerRobotConfig(port=cfg['ports']['follower'], id='win_follower', calibration_dir=ROOT/'calibration/follower', use_degrees=True, max_relative_target=None if cfg.get('reuse_current_observation') else cfg['max_relative_target']))
    return leader, follower
def normalize_raw(device, raw):
    values = device.bus._normalize({device.bus.motors[n].id: v for n,v in raw.items()})
    return {n+'.pos': float(values[device.bus.motors[n].id]) for n in raw}
class WristMapping:
    def __init__(self, cfg, follower):
        self.cfg = cfg
        self.follower = follower
        self.bounds = {}
        for n, (lo, hi) in cfg['follower_limits_raw'].items():
            # Keep one encoder count inside endpoints to tolerate integer truncation.
            a = normalize_raw(follower, {n:lo+1})[n+'.pos']
            b = normalize_raw(follower, {n:hi-1})[n+'.pos']
            self.bounds[n+'.pos'] = (a,b)
        self.previous = {}
        self.started = None
        self.initial = None
        self.pending = {}
        self.start_bounds = None
    def prepare_start(self, observation):
        obs = {k:float(observation[k]) for k in KEYS}
        if not all(math.isfinite(v) for v in obs.values()):
            raise ValueError('Non-finite follower starting position')
        bounds = {}
        for k, (lo, hi) in self.bounds.items():
            distance = max(lo-obs[k], obs[k]-hi, 0.0)
            # Total startup recovery differs from the per-cycle goal-error cap.
            # A small manual overshoot returns to original bounds over the 3s blend.
            cap = self.cfg['max_relative_target']
            cap = cap[k.removesuffix('.pos')] if isinstance(cap,dict) else cap
            if k != 'gripper.pos':
                cap = 10.0
            if distance > cap:
                raise RuntimeError(f'Starting pose exceeds smooth recovery margin: {k}; outside={distance:.3f}, margin={cap:.3f}')
            bounds[k] = (min(lo,obs[k]), max(hi,obs[k]))
        self.start_bounds = bounds
        self.initial = obs
    def nominal(self, action):
        if set(action) != set(KEYS) or not all(math.isfinite(float(v)) for v in action.values()):
            raise ValueError('Expected six finite joint positions')
        out = {k:float(v) for k,v in action.items()}
        for k, delta in self.cfg['offset_degrees'].items():
            out[k] += delta
        return out
    def bounded(self, action):
        return {k: max(self.bounds[k][0], min(self.bounds[k][1], float(v))) for k,v in action.items()}
    def __call__(self, item):
        leader, observation = item
        obs = {k:float(observation[k]) for k in KEYS}
        if not all(math.isfinite(v) for v in obs.values()):
            raise ValueError('Non-finite follower position')
        for role, current in [('leader',leader),('follower',obs)]:
            old = self.previous.get(role)
            for k in KEYS[:-1]:
                if old is not None and abs(current[k]-old[k]) > 180:
                    raise RuntimeError(f'{role} {k}: encoder discontinuity; previous={old[k]:.6f}, current={current[k]:.6f} degrees; no wrap-around command sent')
            self.previous[role] = dict(current)
        now = time.monotonic()
        if self.started is None:
            self.prepare_start(obs)
            self.started = now
        alpha = min(1.0, (now-self.started)/self.cfg['startup_blend_seconds'])
        active_bounds = self.start_bounds if alpha < 1.0 else self.bounds
        for k, (lo, hi) in active_bounds.items():
            tolerance = 1.0 if k=='gripper.pos' else 2.0
            if obs[k] < lo-tolerance or obs[k] > hi+tolerance:
                raise RuntimeError('Follower feedback outside recorded usable range: '+k)
        nominal = self.nominal(leader)
        target = self.bounded(nominal)
        # First goal holds the measured pose; the temporary starting envelope ends
        # after the existing 3-second blend. Normal targets retain original bounds.
        command = dict(target) if alpha >= 1.0 else {k:self.initial[k]+alpha*(target[k]-self.initial[k]) for k in KEYS}
        self.pending = {'leader_action':dict(leader),'follower_observation':obs,'mapped_target':nominal,'bounded_target':target,'requested_action':command,'startup_fraction':alpha,'saturated_joints':[k for k in KEYS if abs(nominal[k]-target[k])>1e-9]}
        return command
def offline_check(cfg, leader, follower):
    mapping = WristMapping(cfg, follower)
    refs = [json.loads((ROOT/'logs'/p).read_text(encoding='utf-8-sig')) for p in cfg['reference_sources']]
    residuals = []
    for s in refs:
        reference = {n+'.pos':v for n,v in s['arms']['leader']['normalized'].items()}
        raw = {k:v+cfg.get('reference_leader_coordinate_shifts_degrees',{}).get(k,0) for k,v in reference.items()}
        sent = mapping.bounded(mapping.nominal(raw))
        residuals.append(sent['wrist_flex.pos']-s['arms']['follower']['normalized']['wrist_flex'])
        assert all(abs(mapping.nominal(raw)[k]-reference[k])<1e-9 for k in KEYS if k!='wrist_flex.pos')
    assert abs(residuals[0]) < 1e-8, 'Wrist offset sign or units are incorrect'
    rows = 0
    clipped = {n:0 for n in JOINTS}
    max_step = 0.0
    last = None
    source = ROOT/'logs/identify-leader-joint-20261015T174441.jsonl'
    for line in source.read_text(encoding='utf-8-sig').splitlines():
        row=json.loads(line)
        source_raw={n:(v-cfg.get('offline_source_raw_shifts',{}).get(n,0))%4096 for n,v in row['raw'].items()}
        action=normalize_raw(leader,source_raw)
        nominal=mapping.nominal(action)
        bounded=mapping.bounded(nominal)
        ints=follower.bus._unnormalize({follower.bus.motors[n].id:bounded[n+'.pos'] for n in JOINTS})
        for n in JOINTS:
            lo,hi=cfg['follower_limits_raw'][n]
            assert lo <= ints[follower.bus.motors[n].id] <= hi,(n,ints)
            clipped[n] += int(abs(nominal[n+'.pos']-bounded[n+'.pos'])>1e-9)
        if last is not None:
            max_step=max(max_step,abs(bounded['wrist_flex.pos']-last))
        last=bounded['wrist_flex.pos'];rows+=1
    for wrist in (-10000.0,10000.0):
        candidate={k:0.0 for k in KEYS};candidate['wrist_flex.pos']=wrist
        bounded=mapping.bounded(mapping.nominal(candidate))
        q=follower.bus._unnormalize({4:bounded['wrist_flex.pos']})[4]
        assert cfg['follower_limits_raw']['wrist_flex'][0]<=q<=cfg['follower_limits_raw']['wrist_flex'][1]
    from lerobot.robots.utils import ensure_safe_goal_position
    limits=cfg['max_relative_target']
    for sign in (-1,1):
        goals={n:(50.0+sign*1000,50.0) for n in JOINTS}
        capped=ensure_safe_goal_position(goals,limits)
        for n in JOINTS:
            bound=limits[n] if isinstance(limits,dict) else limits
            assert abs(capped[n]-50.0-sign*bound)<1e-9
        assert abs(capped['gripper']-50.0)==2.0
    assert follower.config.max_relative_target is None if cfg.get('reuse_current_observation') else follower.config.max_relative_target==limits
    # Verify first-frame blending, end-of-blend correction and discontinuity rejection.
    raw={k:0.0 for k in KEYS};obs={k:(lo+hi)/2 for k,(lo,hi) in mapping.bounds.items()}
    first=mapping((raw,obs));assert first==obs
    mapping.started-=cfg['startup_blend_seconds']+1
    assert mapping((raw,obs))==mapping.bounded(mapping.nominal(raw))
    wrapped=dict(raw);wrapped['wrist_flex.pos']=359.0
    try:mapping((wrapped,obs))
    except RuntimeError:pass
    else:raise AssertionError('Encoder discontinuity must fail')
    out={'time':datetime.now().isoformat(),'status':'passed_offline_only','replay_rows':rows,'clipped_frames_by_joint':clipped,'wrist_residuals_degrees':residuals,'max_replayed_wrist_target_step_degrees':max_step,'hardware_opened':False,'scope':'Mapping sign, units, native integer conversion, absolute bounds, startup blend and encoder wrap rejection. No dynamic following validation.'}
    (ROOT/'logs/practical-follow-offline-check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
def inspect_connected(cfg, leader, follower, *, allow_start_outside=False):
    snapshot={}
    for role,d in [('leader',leader),('follower',follower)]:
        if d.bus.read_calibration()!=d.calibration:
            raise RuntimeError(role+' motor/file calibration mismatch')
        torque={n:d.bus.read('Torque_Enable',n,normalize=False) for n in JOINTS}
        if any(torque.values()):raise RuntimeError(role+' torque already enabled; stop other controller first')
        for n in JOINTS:
            if d.bus.read('Operating_Mode',n,normalize=False)!=0:raise RuntimeError('Unexpected operating mode: '+role+' '+n)
            if d.bus.read('Phase',n,normalize=False)&0x10:raise RuntimeError('Unexpected angle feedback mode: '+role+' '+n)
        raw=d.bus.sync_read('Present_Position',normalize=False)
        outside = {}
        if role=='follower':
            for n,p in raw.items():
                lo,hi=cfg['follower_limits_raw'][n]
                tolerance=7 if n=='gripper' else 23
                if p<lo-tolerance or p>hi+tolerance:
                    if not allow_start_outside:
                        raise RuntimeError('Follower starting position outside usable range: '+n)
                    outside[n] = {'raw':p, 'limits_raw':[lo,hi]}
        snapshot[role]={'raw':raw,'normalized':normalize_raw(d,raw),'torque':torque}
        if outside:
            snapshot[role]['outside_start_limits'] = outside
    return snapshot

class LoopStatusOutput:
    def __init__(self,target):self.target=target;self.last=-float('inf')
    def write(self,value):
        if value.startswith('Teleop loop time:'):
            now=time.perf_counter()
            if now-self.last>=1.0:
                self.target.write(value+'\n');self.target.flush();self.last=now
        elif value.strip() and not value.startswith('\x1b['):self.target.write(value)
        return len(value)
    def flush(self):self.target.flush()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline-check',action='store_true')
    parser.add_argument('--read-only',action='store_true')
    args=parser.parse_args()
    if args.offline_check and args.read_only:parser.error('Choose one mode')
    cfg=load_config();leader,follower=make_devices(cfg)
    if args.offline_check:
        offline_check(cfg,leader,follower);return
    from serial.tools import list_ports
    ports={p.serial_number:p.device for p in list_ports.comports()}
    for role,serial in cfg['usb_serials'].items():
        if ports.get(serial)!=cfg['ports'][role]:raise RuntimeError('USB identity changed: '+role)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    stem=ROOT/'logs'/('practical-follow-'+stamp)
    outcome={'started':datetime.now().isoformat(),'config':cfg,'status':'checking','motor_enable_attempted':False,'cleanup':{}}
    try:
        with stem.with_suffix('.jsonl').open('w',encoding='utf-8') as log:
            last_flush=[time.perf_counter()]
            def emit(row):
                now=time.perf_counter()
                log.write(json.dumps({'time_s':time.time(),'monotonic_s':now,**row})+'\n')
                if row.get('event')!='frame' or now-last_flush[0]>=cfg.get('log_flush_seconds',0.5):
                    log.flush();last_flush[0]=now
            for d in (leader,follower):d.bus.connect()
            snapshot=inspect_connected(cfg,leader,follower)
            emit({'event':'preflight','snapshot':snapshot})
            print('Preflight passed. Both arms torque OFF. Wrist correction: %.2f deg.'%cfg['offset_degrees']['wrist_flex.pos'],flush=True)
            print('Logs: '+str(stem)+'.jsonl',flush=True)
            if args.read_only:
                outcome['status']='read_only_check_passed';print('Read-only checks complete; no goals or torque settings written.',flush=True);return
            print('All six joints follow. Startup blends from the follower pose over 3 seconds.',flush=True)
            print('Clear the workspace, support the follower when stopping. Ctrl+C stops and releases torque.',flush=True)
            input('Press ENTER when physically ready to enable following: ')
            snapshot=inspect_connected(cfg,leader,follower)
            # Configure the leader with official code, then seed fresh follower goals while OFF.
            leader.configure()
            actual=follower.bus.sync_read('Present_Position',normalize=False)
            follower.bus.sync_write('Goal_Position',actual,normalize=False)
            if follower.bus.sync_read('Goal_Position',normalize=False)!=actual:
                raise RuntimeError('Initial goal readback mismatch')
            outcome['motor_enable_attempted']=True
            follower.configure()  # Official configuration enables torque when its context exits.
            processor=WristMapping(cfg,follower)
            from lerobot.robots.utils import ensure_safe_goal_position
            metrics={};cache={'observation':None,'read_at':None};last_log_ms=[0.0]
            original_observe=follower.get_observation
            original_leader_action=leader.get_action
            original_send=follower.send_action
            def observe():
                begin=time.perf_counter();obs=original_observe();end=time.perf_counter()
                metrics.clear();metrics.update({'frame_start':begin,'observation_ms':(end-begin)*1000})
                cache.update({'observation':obs,'read_at':end})
                return obs
            def leader_action():
                begin=time.perf_counter();action=original_leader_action()
                metrics['leader_ms']=(time.perf_counter()-begin)*1000
                return action
            def process(item):
                begin=time.perf_counter();action=processor(item)
                metrics['mapping_ms']=(time.perf_counter()-begin)*1000
                return action
            def logged_send(action):
                begin=time.perf_counter();refreshed=False
                if cfg.get('reuse_current_observation'):
                    obs=cache['observation'];age=time.perf_counter()-cache['read_at']
                    if age>cfg['observation_max_age_seconds']:
                        refresh_begin=time.perf_counter();obs=original_observe()
                        cache.update({'observation':obs,'read_at':time.perf_counter()})
                        metrics['refresh_ms']=(time.perf_counter()-refresh_begin)*1000
                        # Re-run existing finite/range/discontinuity checks on refreshed feedback.
                        action=processor((processor.pending['leader_action'],obs));refreshed=True
                    if set(k for k in obs if k.endswith('.pos'))!=set(KEYS):
                        raise RuntimeError('Incomplete current-frame follower observation')
                    age_ms=(time.perf_counter()-cache['read_at'])*1000
                    goals={n:(float(action[n+'.pos']),float(obs[n+'.pos'])) for n in JOINTS}
                    safe=ensure_safe_goal_position(goals,cfg['max_relative_target'])
                    sent=original_send({n+'.pos':v for n,v in safe.items()})
                else:
                    age_ms=None;sent=original_send(action)
                end=time.perf_counter()
                timing={k:v for k,v in metrics.items() if k!='frame_start'}
                timing.update({'send_ms':(end-begin)*1000,'work_before_log_ms':(end-metrics['frame_start'])*1000,'previous_log_ms':last_log_ms[0]})
                emit({'event':'frame',**processor.pending,'sent_action':sent,'timing_ms':timing,'limiter_feedback_age_ms':age_ms,'feedback_refreshed':refreshed})
                last_log_ms[0]=(time.perf_counter()-end)*1000
                return sent
            follower.get_observation=observe;leader.get_action=leader_action;follower.send_action=logged_send
            from lerobot.processor import make_default_processors
            from lerobot.scripts.lerobot_teleoperate import teleop_loop
            _,robot_action_processor,robot_observation_processor=make_default_processors()
            logging.getLogger().setLevel(logging.ERROR)
            outcome['status']='active';emit({'event':'active'})
            print('ACTIVE: move the leader. End-of-range targets are held at the boundary; no automatic return.',flush=True)
            with contextlib.redirect_stdout(LoopStatusOutput(sys.stdout)):
                teleop_loop(teleop=leader,robot=follower,fps=cfg['fps'],teleop_action_processor=process,robot_action_processor=robot_action_processor,robot_observation_processor=robot_observation_processor,display_data=False,duration=None)
    except KeyboardInterrupt:
        outcome['status']='stopped_by_user'
    except BaseException as exc:
        outcome['status']='failed';outcome['error']=repr(exc)
        raise
    finally:
        for role,d in [('follower',follower),('leader',leader)]:
            if not d.bus.is_connected:continue
            try:
                if outcome['motor_enable_attempted']:
                    d.bus.disable_torque()
                    torque={n:d.bus.read('Torque_Enable',n,normalize=False) for n in JOINTS}
                    outcome['cleanup'][role]={'torque':torque}
                    if any(torque.values()):raise RuntimeError('Torque release could not be verified')
                else:outcome['cleanup'][role]='closed without writes'
            except BaseException as exc:
                outcome['cleanup'][role]={'error':repr(exc)}
                print('Cleanup check failed for '+role+': '+repr(exc),flush=True)
            finally:
                try:d.bus.disconnect(disable_torque=False)
                except BaseException as exc:print('Closing '+role+': '+repr(exc),flush=True)
        outcome['ended']=datetime.now().isoformat()
        stem.with_suffix('.json').write_text(json.dumps(outcome,indent=2),encoding='utf-8')
        print('Finished: '+outcome['status']+'. Summary: '+str(stem)+'.json',flush=True)
if __name__=='__main__':main()
