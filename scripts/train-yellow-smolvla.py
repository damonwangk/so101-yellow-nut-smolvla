"""Formal single-yellow-nut SmolVLA training using the installed official LeRobot trainer."""
import argparse,copy,json,logging,math,os,re,statistics,sys,time,traceback
from datetime import datetime
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',PYTHONUTF8='1')
from so101_paths import ROOT

class ConsoleTee:
 """Mirror each live console write to a log without delaying flushes."""
 def __init__(self,console,log):self.console=console;self.log=log
 def write(self,text):
  if self.console is not None:self.console.write(text)
  self.log.write(text);self.log.flush();return len(text)
 def flush(self):
  if self.console is not None:self.console.flush()
  self.log.flush()
 def __getattr__(self,name):return getattr(self.console,name)

def print_training_event(row,total_steps):
 """Called inside the optimizer loop; metrics.jsonl keeps the full records."""
 stamp=row['time'][11:19];name=row['event']
 if name=='update':
  print(f"[{stamp}] TRAIN step={row['step']}/{total_steps} ({100*row['step']/total_steps:.1f}%) "
        f"loss={row['loss']:.6f} grad_norm={row['grad_norm']:.4f} lr={row['lr']:.6e} "
        f"gpu_peak={row['peak_allocated_mib']:.0f}MiB update={row['update_s']:.3f}s",flush=True)
 elif name=='validation':
  print(f"[{stamp}] VALID step={row['step']}/{total_steps} val_loss={row['eval_loss']:.6f}",flush=True)
 elif name=='checkpoint_saved':
  print(f"[{stamp}] SAVED step={row['step']}/{total_steps} path={row['path']}",flush=True)
 elif name=='formal_training_start':
  print(f"[{stamp}] START steps={total_steps} batch={row['batch_size']} resume={row['resume']}",flush=True)
 elif name=='formal_training_completed':
  print(f"[{stamp}] COMPLETED step={row['step']}/{total_steps}",flush=True)
 elif name=='stopped':
  print(f"[{stamp}] STOPPED {row['error']}",flush=True)
 else:print(json.dumps(row,ensure_ascii=True),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run-spec',type=Path,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 spec=json.loads(args.run_spec.read_text(encoding='utf-8'));logs=Path(spec['log_dir']);output=Path(spec['output_dir']);sys.argv=sys.argv[:1]
 if (logs/'early_stop_result.json').is_file():
  raise SystemExit('This run was stopped at the user-approved checkpoint. Keep its outputs; configure a new run for further training.')
 import msvcrt
 lock=(ROOT/'logs/yellow-smolvla-formal.lock').open('a+b');lock.write(b'0') if lock.tell()==0 else None;lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Formal training is already active; refusing a duplicate process.')
 # The training process prints directly and also retains stdout/stderr files.
 stamp=datetime.now().strftime('%Y%m%dT%H%M%S');paths={}
 for stream_name in ('stdout','stderr'):
  path=logs/f'{stream_name}-{stamp}.log';paths[stream_name]=str(path)
  setattr(sys,stream_name,ConsoleTee(getattr(sys,stream_name),path.open('a',encoding='utf-8',buffering=1)))
 paths['started']=datetime.now().astimezone().isoformat()
 (logs/'console_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
 previous=json.loads((logs/'training_status.json').read_text()) if args.resume and (logs/'training_status.json').exists() else {}
 state={'status':'starting','run_name':spec['run_name'],'step':0,'steps_total':spec['steps'],'batch_size':spec['batch_size'],'log_dir':str(logs),'output_dir':str(output),'pid':os.getpid(),'resume':args.resume,'training_loss_is_not_robot_success':True}
 state.update({k:previous[k] for k in ('best_validation_step','best_validation_loss','last_checkpoint_step','last_checkpoint') if k in previous})
 def write_state(**kw):
  state.update(kw,time=datetime.now().astimezone().isoformat());tmp=logs/'training_status.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(logs/'training_status.json')
 def event(name,**kw):
  row={'time':datetime.now().astimezone().isoformat(),'event':name,**kw}
  with (logs/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  print_training_event(row,spec['steps'])
 write_state()
 try:
  import numpy as np
  import pyarrow as pa
  import pyarrow.compute as pc
  import pyarrow.parquet as pq
  import torch
  from lerobot.configs.default import DatasetConfig,WandBConfig
  from lerobot.configs.train import TrainPipelineConfig
  from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
  import lerobot.scripts.lerobot_train as trainer
  torch.set_num_threads(8);torch.set_autocast_dtype('cuda',torch.bfloat16)
  base=ROOT/'models/smolvla_base';vlm=ROOT/'models/yellow_nut_smolvla_005000/tokenizer'
  if args.resume:
   pointer=output/'checkpoints/last_checkpoint.json'
   checkpoint=Path(json.loads(pointer.read_text())['path']) if pointer.is_file() else output/'checkpoints/last'
   if not (checkpoint/'pretrained_model/train_config.json').is_file():
    candidates=sorted(p for p in (output/'checkpoints').glob('[0-9]*') if (p/'training_state').is_dir() and (p/'pretrained_model/train_config.json').is_file())
    if not candidates:raise RuntimeError('No complete checkpoint available for resume.')
    checkpoint=candidates[-1]
   cfg_path=checkpoint/'pretrained_model/train_config.json';cfg=TrainPipelineConfig.from_pretrained(cfg_path,cli_args=[]);cfg.resume=True
   sys.argv.append('--config_path='+str(cfg_path))
   state['step']=json.loads((checkpoint/'training_state/training_step.json').read_text())['step']
  else:
   policy=SmolVLAConfig.from_pretrained(base);policy.pretrained_path=base;policy.input_features={};policy.output_features={};policy.vlm_model_name=str(vlm);policy.load_vlm_weights=False
   policy.device='cuda';policy.use_amp=True;policy.push_to_hub=False;policy.freeze_vision_encoder=True;policy.train_expert_only=True;policy.train_state_proj=True
   policy.optimizer_lr=spec['learning_rate'];policy.scheduler_warmup_steps=spec['warmup_steps'];policy.scheduler_decay_steps=spec['decay_steps'];policy.scheduler_decay_lr=spec['decay_lr']
   cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id=spec['dataset_id'],root=spec['dataset_root'],episodes=spec['train_episodes']+spec['validation_episodes'],eval_split=.1,video_backend='pyav',use_imagenet_stats=False),policy=policy,output_dir=output,job_name=spec['run_name'],batch_size=spec['batch_size'],num_workers=0,persistent_workers=False,steps=spec['steps'],log_freq=50,env_eval_freq=0,eval_steps=spec['eval_steps'],max_eval_samples=0,save_checkpoint=True,save_freq=spec['save_freq'],wandb=WandBConfig(enable=False),seed=spec['seed'])
  original_split=trainer.make_train_eval_datasets
  def split_with_train_stats(config):
   train,validation=original_split(config)
   assert train.episodes==spec['train_episodes'] and validation.episodes==spec['validation_episodes']
   assert set(train.episodes).isdisjoint(validation.episodes)
   # Read only state/action from the original Parquet; no dataset mutation.
   tab=pa.concat_tables([pq.read_table(p,columns=['episode_index','observation.state','action']) for p in sorted((Path(spec['dataset_root'])/'data').rglob('*.parquet'))])
   tab=tab.filter(pc.is_in(tab['episode_index'],value_set=pa.array(spec['train_episodes'],type=tab['episode_index'].type)))
   stats=copy.deepcopy(train.meta.stats);train_only={}
   for key in ('observation.state','action'):
    a=np.asarray(tab[key].to_pylist(),dtype=np.float64);st={'min':a.min(0).tolist(),'max':a.max(0).tolist(),'mean':a.mean(0).tolist(),'std':a.std(0).tolist(),'count':[len(a)]}
    for q in ('q01','q10','q50','q90','q99'):
     if q in stats[key]:st[q]=np.quantile(a,int(q[1:])/100,axis=0).tolist()
    stats[key]=st;train_only[key]=st
   train.meta.stats=stats
   (logs/'training_normalization_stats.json').write_text(json.dumps(train_only,indent=2),encoding='utf-8')
   logging.getLogger().addHandler(EvaluationLog())
   event('dataset_split',train_episodes=train.episodes,validation_episodes=validation.episodes,train_frames=train.num_frames,validation_frames=validation.num_frames,normalization='training episodes only',original_data_unchanged=True)
   return train,validation
  trainer.make_train_eval_datasets=split_with_train_stats
  original_processors=trainer.make_pre_post_processors
  def local_processors(*a,**kw):
   kw.setdefault('preprocessor_overrides',{})['tokenizer_processor']={'tokenizer_name':str(vlm)}
   return original_processors(*a,**kw)
  trainer.make_pre_post_processors=local_processors
  class EvaluationLog(logging.Handler):
   def emit(self,record):
    match=re.search(r'step (\d+): eval_loss=([\d.eE+-]+)',record.getMessage())
    if match:
     step=int(match[1]);loss=float(match[2]);assert math.isfinite(loss)
     event('validation',step=step,eval_loss=loss);write_state(last_validation_step=step,last_validation_loss=loss)
     previous=state.get('best_validation_loss',float('inf'))
     if loss<previous:write_state(best_validation_loss=loss,best_validation_step=step)
  original_update=trainer.update_policy;timings=[];count=state['step'];peak=0.0;tracked=None;seen=set();start=time.perf_counter()
  def monitored_update(*a,**kw):
   nonlocal count,peak,tracked
   metrics,model,batch=a[:3]
   batch_episodes=set(batch['episode_index'].detach().cpu().tolist());assert batch_episodes<=set(spec['train_episodes']);seen.update(batch_episodes)
   if tracked is None:
    name,param=next((n,p) for n,p in model.named_parameters() if p.requires_grad and n.endswith('action_out_proj.weight'));tracked=(name,param,param.detach().float().cpu().clone())
    event('first_batch',action_shape=list(batch['action'].shape),state_shape=list(batch['observation.state'].shape),images={k:list(v.shape) for k,v in batch.items() if k.startswith('observation.images')},trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),precision=str(torch.get_autocast_dtype('cuda')),gpu=torch.cuda.get_device_name())
   torch.cuda.synchronize();before=time.perf_counter();result=original_update(*a,**kw);torch.cuda.synchronize();duration=time.perf_counter()-before;count+=1;timings.append(duration);peak=max(peak,torch.cuda.max_memory_allocated()/1024**2)
   row={'step':count,'loss':float(result[0].loss.val),'grad_norm':float(result[0].grad_norm.val),'lr':float(result[0].lr.val),'update_s':duration,'peak_allocated_mib':peak,'reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
   if not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']):raise FloatingPointError(str(row))
   # Print in this update loop: first five updates, then every 25 steps.
   if count<=5 or count%25==0:
    median=statistics.median(timings[-50:]);write_state(status='running',**row,median_update_s=median,elapsed_process_s=time.perf_counter()-start,estimated_compute_remaining_s=median*(spec['steps']-count),training_episodes_seen=sorted(seen));event('update',**row,median_update_s=median)
   return result
  trainer.update_policy=monitored_update
  original_should_save=trainer.should_save_checkpoint
  trainer.should_save_checkpoint=lambda step,freq,total: step==spec['initial_checkpoint_step'] or original_should_save(step,freq,total)
  original_save=trainer.save_checkpoint
  def verified_save(*a,**kw):
   result=original_save(*a,**kw);directory=Path(kw['checkpoint_dir']);step=kw['step'];model_file=directory/'pretrained_model/model.safetensors'
   assert model_file.is_file() and (directory/'training_state').is_dir()
   event('checkpoint_saved',step=step,path=str(directory),model_bytes=model_file.stat().st_size);write_state(last_checkpoint_step=step,last_checkpoint=str(directory))
   if state.get('best_validation_step')==step:
    (output/'best_checkpoint.json').write_text(json.dumps({'step':step,'validation_loss':state['best_validation_loss'],'path':str(directory),'criterion':'held-out flow-matching loss; not physical success'},indent=2),encoding='utf-8')
   return result
  trainer.save_checkpoint=verified_save
  # A normal JSON pointer avoids Windows symlink privileges and keeps numbered checkpoints intact.
  def update_checkpoint_pointer(directory):
   pointer=Path(directory).parent/'last_checkpoint.json';pointer.write_text(json.dumps({'path':str(directory)},indent=2),encoding='utf-8');return pointer
  trainer.update_last_checkpoint=update_checkpoint_pointer
  event('formal_training_start',run_spec=str(args.run_spec),resume=args.resume,steps=spec['steps'],batch_size=cfg.batch_size)
  trainer.train(cfg)
  assert count==spec['steps'];delta=float((tracked[1].detach().float().cpu()-tracked[2]).abs().max());assert delta>0
  write_state(status='completed',step=count,weight_change_max_abs=delta,elapsed_process_s=time.perf_counter()-start);event('formal_training_completed',step=count,weight_change_max_abs=delta)
 except BaseException as exc:
  write_state(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));event('stopped',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
