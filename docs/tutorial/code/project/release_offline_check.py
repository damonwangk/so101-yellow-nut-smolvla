"""Release integrity + actual published controller inference; all robot I/O forbidden."""
import argparse,contextlib,importlib.util,json,pathlib,sys,queue
from unittest.mock import patch
from so101_paths import ROOT,CHECKPOINT,TOKENIZER

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def blocked(*a,**kw):raise AssertionError('Robot/camera access forbidden in offline verification')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--skip-model',action='store_true');args=parser.parse_args()
 import numpy as np,pyarrow.parquet as pq,cv2
 with contextlib.ExitStack() as stack:
  ports=stack.enter_context(patch('serial.Serial.open',side_effect=blocked))
  bus=stack.enter_context(patch('lerobot.motors.feetech.feetech.FeetechMotorsBus.connect',side_effect=blocked))
  cam=stack.enter_context(patch('lerobot.cameras.opencv.OpenCVCamera.connect',side_effect=blocked))
  m=load('released_model_test',ROOT/'scripts/test-yellow-smolvla-robot.py')
  cfg=m.follow.load_config();_,dev=m.follow.make_devices(cfg)
  home=json.loads((ROOT/'configs/yellow-model-start-pose.json').read_text(encoding='utf-8'));obs=home['target'];g=m.Guard(cfg,dev,obs);g.mapper.started=m.time.monotonic()-4
  from yellow_start_pose import load_start_pose
  load_start_pose(ROOT/'configs/yellow-model-start-pose.json',cfg,g.mapper.bounds)
  data=ROOT/'datasets/yellow_nut_to_brown_tray/formal/lerobot';info=json.loads((data/'meta/info.json').read_text(encoding='utf-8'))
  episodes=set();frames=0
  for p in (data/'data').rglob('*.parquet'):
   tab=pq.read_table(p,columns=['episode_index','action','observation.state']);episodes.update(tab['episode_index'].to_pylist());frames+=len(tab)
   for key in ['action','observation.state']:
    values=np.array(tab[key].to_pylist());assert values.shape[1]==6 and np.isfinite(values).all()
  assert len(episodes)==50 and frames==29570 and info['total_episodes']==50
  report={'status':'passed','dataset_episodes':len(episodes),'dataset_frames':frames,'model_inference_checked':False,'robot_io_opened':False}
  if not args.skip_model:
   class Stop:
    stopped=False
    def is_set(self):return self.stopped
    def set(self):self.stopped=True
   stop=Stop()
   image=cv2.cvtColor(cv2.imread(str(ROOT/'configs/yellow-model-start-reference.jpg')),cv2.COLOR_BGR2RGB)
   item={'image':image,'state':[obs[k] for k in m.KEYS],'epoch':1,'request_progress':0,'observed_at':0}
   class Requests:
    used=False
    def get(self,timeout=None):
     if self.used:raise queue.Empty()
     self.used=True;return item
   class Results:
    prediction=None;error=None
    def put(self,row,timeout=None):
     if row['event']=='prediction':self.prediction=row;stop.set()
     if row['event']=='worker_error':self.error=row['error'];stop.set()
   result=Results();m.worker(Requests(),result,stop)
   if result.error:raise RuntimeError(result.error)
   a=np.array(result.prediction['actions']);assert a.shape==(50,6) and np.isfinite(a).all()
   report.update(model_inference_checked=True,prediction_shape=list(a.shape),inference_seconds=result.prediction['inference_s'])
  report['forbidden_calls']={'serial':ports.call_count,'bus':bus.call_count,'camera':cam.call_count};assert not any(report['forbidden_calls'].values())
  (ROOT/'logs/release-offline-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
