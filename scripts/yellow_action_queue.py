"""Committed, smoothed policy segments with a shared, speed-retimed phase."""
import math

EXECUTION_HORIZON = 50
BLEND_STEPS = 5
POLICY_FPS = 30.


def sample(points, phase):
    phase = max(0., min(float(len(points)-1), phase))
    i = min(int(phase), len(points)-2)
    fraction = phase-i
    return [a+(b-a)*fraction for a,b in zip(points[i],points[i+1])]


def smooth(actions):
    # Smooth only the five arm joints. Keep gripper targets from the policy.
    # Symmetric 7-frame triangular filter: 0.1s on either side at 30 FPS.
    weights = (1.,2.,3.,4.,3.,2.,1.)
    return [[sum(w*actions[min(49,max(0,i+k-3))][j]
                 for k,w in enumerate(weights))/16. for j in range(5)]+[actions[i][5]]
            for i in range(50)]


class RetimedActionQueue:
    def __init__(self,execution_horizon=EXECUTION_HORIZON):
        if isinstance(execution_horizon,bool) or not isinstance(execution_horizon,int) or not 1<=execution_horizon<=50:
            raise ValueError('Execution horizon must be an integer from 1 to 50')
        self.execution_horizon=execution_horizon
        self.reset()
    def reset(self):
        self.progress=0.;self.phase=0.;self.base=0.;self.points=[]
        self.pending=None;self.born=None;self.last_observed=None
        self.installed_at=None;self.index=None;self.last_install=None
        self.source_offset=0.;self.proposed_step=0.;self.proposed_at=None
        self.progress_marker=0.;self.last_progress_at=None
    @property
    def completed(self):return int(self.progress+1e-8)
    def offer(self,actions,request_progress,observed_at,now):
        if len(actions)!=50 or any(len(a)!=6 or not all(math.isfinite(v) for v in a) for a in actions):
            raise ValueError('Expected 50x6 finite actions')
        if not all(math.isfinite(float(v)) for v in (request_progress,observed_at,now)):
            raise ValueError('Invalid prediction timing/progress')
        if request_progress>self.progress+1e-6:raise ValueError('Prediction progress is in the future')
        if now-observed_at>1.0:return False
        if self.last_observed is not None and observed_at<self.last_observed:return False
        self.pending=(smooth(actions),float(request_progress),observed_at)
        self.last_observed=observed_at
        return True
    def target(self,now,previous,dt,bounds=None):
        # Fresh predictions keep the watchdog alive; they cannot preempt the
        # checkpoint's committed segment. Q/reset clears all pending work.
        ready=not self.points or self.phase+1e-8>=min(self.execution_horizon,len(self.points)-1)
        if ready and self.pending is not None:
            actions,marker,born=self.pending;self.pending=None
            offset=max(0.,self.progress-marker)
            if now-born<=1.0 and offset<49.:
                old=self.points;old_phase=self.phase
                # Include the exact final target even with a fractional source
                # offset. Otherwise the last keyframe is silently discarded.
                future=[sample(actions,offset+i) for i in range(50-int(math.floor(offset)))]
                if bounds is not None:
                    future=[[max(lo,min(hi,v)) for v,(lo,hi) in zip(row,bounds)] for row in future]
                if old:
                    for i in range(min(BLEND_STEPS,len(future))):
                        at=old_phase+i+1
                        weight=(i+1)/BLEND_STEPS;prior=sample(old,at)
                        future[i]=[a+(b-a)*weight for a,b in zip(prior[:5],future[i][:5])]+[future[i][5]]
                self.points=[list(previous)]+future;self.phase=0.;self.base=self.progress
                self.source_offset=offset;self.born=born;self.installed_at=now
                if self.last_progress_at is None:self.last_progress_at=now
                self.last_install={'skipped_completed_steps':offset,'request_progress':marker,
                    'progress_at_install':self.progress,'observed_at':born,
                    'execution_horizon':self.execution_horizon,
                    'segment_intervals':len(self.points)-1,
                    'final_source_index':49,'blend_steps':BLEND_STEPS if old else 0}
        self.proposed_step=0.;self.proposed_at=now
        if not self.points or self.phase>=len(self.points)-1-1e-8:
            self.index=None;return None
        # Never cross a knot while scaling a segment: the guard's common scale
        # then corresponds exactly to the accepted phase. The gripper is sampled
        # at that accepted phase after the arm guard computes its scale.
        next_knot=math.floor(self.phase+1e-8)+1
        self.proposed_step=min(POLICY_FPS*min(max(dt,0.),1/60),next_knot-self.phase)
        self.index=min(49,int(self.source_offset+max(0.,next_knot-1)))
        return sample(self.points,self.phase+self.proposed_step)
    def gripper_target(self,common_scale):
        if not math.isfinite(common_scale) or not 0.<=common_scale<=1.+1e-9:
            raise ValueError('Invalid guarded phase scale')
        if not self.points:raise ValueError('No active policy segment')
        # This aligns action time only. No gripper slew/feedback/contact limit.
        return sample(self.points,self.phase+self.proposed_step*min(1.,common_scale))[5]
    def advance(self,common_scale):
        if not math.isfinite(common_scale) or not 0.<=common_scale<=1.+1e-9:
            raise ValueError('Invalid guarded phase scale')
        self.phase+=self.proposed_step*min(1.,common_scale)
        if abs(self.phase-round(self.phase))<1e-8:self.phase=float(round(self.phase))
        self.progress=self.base+self.phase;self.proposed_step=0.
        if self.progress-self.progress_marker>=1.-1e-8:
            self.progress_marker=self.progress;self.last_progress_at=self.proposed_at
    def stale_reason(self,now):
        if self.last_observed is not None and now-self.last_observed>1.5:
            return 'No fresh inference observation for 1.5 seconds'
        if self.last_progress_at is not None and now-self.last_progress_at>3.0:
            return 'Trajectory did not advance one policy interval within 3 seconds'
        return None
    def stale(self,now):return self.stale_reason(now) is not None
    def remaining(self):return max(0,int(math.ceil(len(self.points)-1-self.phase)))


def coordinated_target(previous,target,observation,caps,dt,speed=15.):
    """One common interpolation factor preserves all six joint proportions."""
    if not all(math.isfinite(float(v)) for values in (previous,target,observation) for v in values):raise ValueError('Non-finite control position')
    step=speed*min(max(dt,0.),1/60);delta=[t-p for p,t in zip(previous,target)];lower=0.;upper=1.
    for p,d,o,cap in zip(previous,delta,observation,caps):
        if abs(d)<=1e-12:
            if abs(p-o)>cap+1e-5:raise RuntimeError('Feedback left the safe slew envelope; hold')
            continue
        a,b=sorted(((o-cap-p)/d,(o+cap-p)/d))
        lower=max(lower,a);upper=min(upper,b,step/abs(d))
    if upper<lower-1e-9:raise RuntimeError('No velocity-limited target inside feedback limits; hold')
    alpha=max(0.,min(1.,upper));sent=[p+alpha*d for p,d in zip(previous,delta)]
    if any(abs(s-o)>cap+1e-5 for s,o,cap in zip(sent,observation,caps)):raise RuntimeError('Relative limit violation')
    return sent,alpha>=1.-1e-9,alpha
