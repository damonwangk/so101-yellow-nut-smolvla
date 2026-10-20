"""Fixed training-derived start; no hardware access or actuator commands here."""
import hashlib,json,math
from pathlib import Path
KEYS=tuple(j+'.pos' for j in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'))

def load_start_pose(path,cfg,bounds):
    profile=json.loads(Path(path).read_text(encoding='utf-8'))
    target=profile['target']
    if set(target)!=set(KEYS):raise ValueError('Expected six fixed-start joint targets')
    for key,value in target.items():
        if not math.isfinite(value) or not bounds[key][0]<=value<=bounds[key][1]:
            raise ValueError('Fixed-start target outside calibrated range: '+key)
    if profile['coordinate_frame_version']!=cfg['coordinate_frame_version']:
        raise ValueError('Fixed-start coordinate frame changed; regenerate the profile')
    current_hash=hashlib.sha256((Path(path).parent/'practical-follow.json').read_bytes()).hexdigest()
    if current_hash!=profile['control_config_sha256']:
        raise ValueError('Control config changed; regenerate the fixed-start profile')
    if profile.get('start_requires_second_enter') is not True:
        raise ValueError('Fixed start requires a separate local Enter before inference')
    return profile

class FixedStart:
    def __init__(self,profile):
        self.target=dict(profile['target'])
        self.tolerance=float(profile['body_tolerance_deg'])
        self.settle=float(profile['settle_seconds'])
        self.stall=float(profile['stall_seconds'])
        if not all(math.isfinite(v) and v>0 for v in (self.tolerance,self.settle,self.stall)):
            raise ValueError('Invalid fixed-start settling parameters')
        self.cancel()
    def cancel(self):
        self.phase='idle';self.began=None;self.stable_since=None
        self.marker_error=None;self.progress_at=None;self.deadline=None
        self.confirmed_at=None;self.arrived_at=None
    def begin(self,now):
        self.cancel();self.phase='moving';self.began=now
    def confirm(self,now):
        if self.phase!='at_start':return False
        self.phase='camera_wait';self.confirmed_at=now;return True
    def update(self,observation,now,frame_at=None):
        if self.phase not in ('moving','at_start','camera_wait'):return self.phase
        values=[float(observation[k]) for k in KEYS[:-1]]
        if not all(math.isfinite(v) for v in values):raise ValueError('Non-finite fixed-start observation')
        error=max(abs(v-self.target[k]) for k,v in zip(KEYS[:-1],values))
        # Only arm feedback determines arrival; an obstructed gripper cannot gate it.
        if self.phase in ('at_start','camera_wait') and error>self.tolerance:
            self.begin(now)
        if self.phase=='moving':
            if self.marker_error is None:
                self.marker_error=error;self.progress_at=now
                self.deadline=now+max(10.,error/15.+5.)
            if error<=self.tolerance:
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=self.settle:
                    self.phase='at_start';self.arrived_at=now
            else:
                self.stable_since=None
                if error<=self.marker_error-1.:
                    self.marker_error=error;self.progress_at=now
                if now-self.progress_at>self.stall:
                    raise RuntimeError('Fixed-start arm did not approach target for 3 seconds')
            if self.phase=='moving' and now>self.deadline:
                raise RuntimeError('Fixed-start positioning timed out')
        if self.phase=='camera_wait' and frame_at is not None and self.confirmed_at<frame_at<=now:
            self.phase='model'
        return self.phase
