"""Guarded SmolVLA checkpoint-5000 evaluation; local Enter is required for motion."""
from __future__ import annotations
import argparse, importlib.util, json, math, multiprocessing as mp, os, queue, sys, time, traceback
from datetime import datetime
from pathlib import Path
from yellow_action_queue import RetimedActionQueue,coordinated_target
from yellow_start_pose import FixedStart,load_start_pose
from so101_paths import ROOT
JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
KEYS=tuple(j+'.pos' for j in JOINTS)
TASK='Pick up the yellow nut and place it into the brown tray.'
from so101_paths import CHECKPOINT,TOKENIZER,require_local_hardware

def load_script(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def capture_policy_input(item):
    path=item.get('debug_image_path')
    if not path:return {}
    debug={'input_state':item['state']}
    try:
        import cv2
        if not cv2.imwrite(path,cv2.cvtColor(item['image'],cv2.COLOR_RGB2BGR)):raise OSError('Image write failed')
        debug['input_image']=path
    except Exception as exc:debug['input_capture_error']=repr(exc)
    return debug

follow=load_script('yellow_eval_follow','follow-practical.py')

class ModelMapping(follow.WristMapping):
    def nominal(self,action):
        # Dataset/model output already uses follower coordinates. Never add leader offsets.
        if set(action)!=set(KEYS) or not all(math.isfinite(float(v)) for v in action.values()):
            raise ValueError('Expected six finite follower-coordinate targets')
        return {k:float(action[k]) for k in KEYS}

class Guard:
    def __init__(self,cfg,follower,observation,model=True):
        self.cfg=cfg;self.mapper=(ModelMapping if model else follow.WristMapping)(cfg,follower)
        self.mapper.prepare_start(observation);self.previous=dict(observation);self.last_observation=None
    def command(self,target,observation,age,dt,*,policy_queue=None):
        if age>self.cfg['observation_max_age_seconds']:raise RuntimeError('Stale motor feedback; no goal sent')
        if self.last_observation is not None:
            for k in KEYS[:-1]:
                if abs(observation[k]-self.last_observation[k])>180:raise RuntimeError('Encoder discontinuity; no goal sent')
        self.last_observation=dict(observation)
        requested=self.mapper((target,observation))
        from lerobot.robots.utils import ensure_safe_goal_position
        values,reached,scale=coordinated_target(
            [self.previous[k] for k in KEYS[:-1]],[requested[k] for k in KEYS[:-1]],
            [observation[k] for k in KEYS[:-1]],[self.cfg['max_relative_target'][j] for j in JOINTS[:-1]],dt)
        sent=dict(zip(KEYS[:-1],values))
        low,high=self.mapper.bounds['gripper.pos']
        gripper_target=(policy_queue.gripper_target(scale) if policy_queue is not None
                        else self.mapper.pending['bounded_target']['gripper.pos'])
        sent['gripper.pos']=max(low,min(high,gripper_target))
        body_limits={j:self.cfg['max_relative_target'][j] for j in JOINTS[:-1]}
        checked=ensure_safe_goal_position({j:(sent[j+'.pos'],observation[j+'.pos']) for j in JOINTS[:-1]},body_limits)
        if any(abs(checked[j]-sent[j+'.pos'])>1e-5 for j in JOINTS[:-1]):raise RuntimeError('Final native limit check failed')
        self.previous=dict(sent)
        return sent,{'absolute_clipped':self.mapper.pending['saturated_joints'],'relative_or_speed_clipped':[k for k in KEYS[:-1] if abs(sent[k]-requested[k])>1e-6],'startup_fraction':self.mapper.pending['startup_fraction'],'target_reached':reached,'common_speed_scale':scale,'gripper_range_only':True,'gripper_policy_phase_aligned':policy_queue is not None}

def worker(requests,results,stop):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
    try:
        import numpy as np
        import torch
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        torch.set_num_threads(4)
        cfg=SmolVLAConfig.from_pretrained(CHECKPOINT);cfg.pretrained_path=CHECKPOINT;cfg.vlm_model_name=str(TOKENIZER);cfg.device='cuda';cfg.compile_model=False
        model=SmolVLAPolicy.from_pretrained(CHECKPOINT,config=cfg,strict=True,local_files_only=True).eval()
        pre,post=make_pre_post_processors(cfg,pretrained_path=str(CHECKPOINT),preprocessor_overrides={'device_processor':{'device':'cuda'},'tokenizer_processor':{'tokenizer_name':str(TOKENIZER)}})
        def predict(image,state):
            batch={'observation.images.wrist_camera':torch.from_numpy(image.copy()).permute(2,0,1).float().unsqueeze(0)/255,'observation.state':torch.tensor(state,dtype=torch.float32).unsqueeze(0),'task':[TASK]}
            batch=pre(batch);model.reset()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):action=model.predict_action_chunk(batch)
            action=post(action).float().cpu().numpy()[0]
            if action.shape!=(50,6) or not np.isfinite(action).all():raise ValueError('Invalid model action chunk')
            return action
        warm_times=[]
        for _ in range(3):
            before=time.perf_counter();predict(np.zeros((480,640,3),dtype=np.uint8),[0.0]*6);warm_times.append(time.perf_counter()-before)
        results.put({'event':'ready','warm_single_inference_s':warm_times,'gpu':torch.cuda.get_device_name(),
                     'chunk_size':cfg.chunk_size,'n_action_steps':cfg.n_action_steps},timeout=2)
        while not stop.is_set():
            try:item=requests.get(timeout=.1)
            except queue.Empty:continue
            started=time.perf_counter();action=predict(item['image'],item['state']);inference_s=time.perf_counter()-started
            debug=capture_policy_input(item)
            results.put({'event':'prediction','epoch':item['epoch'],'request_progress':item['request_progress'],'observed_at':item['observed_at'],'actions':action.tolist(),'inference_s':inference_s,**debug},timeout=2)
    except BaseException:
        try:results.put({'event':'worker_error','error':traceback.format_exc()},timeout=2)
        except Exception:pass

def offline_test(cfg):
    import numpy as np
    _,follower=follow.make_devices(cfg)
    path=ROOT/'logs/yellow_nut_smolvla_20261019T110537/offline_check_005000/predictions.npz'
    d=np.load(path);pred=d['prediction'];initial=d['observation'];checks=0;clipped=0
    for i in range(len(pred)):
        obs={k:float(initial[i,j]) for j,k in enumerate(KEYS)};guard=Guard(cfg,follower,obs)
        for tick in range(240):
            if tick:guard.mapper.started=time.monotonic()-tick/60
            target={k:float(pred[i,min(tick//2,49),j]) for j,k in enumerate(KEYS)}
            before=dict(guard.previous);sent,flags=guard.command(target,obs,0.005,1/60)
            for k in KEYS[:-1]:
                assert abs(sent[k]-obs[k])<=cfg['max_relative_target'][k.removesuffix('.pos')]+1e-5
                assert abs(sent[k]-before[k])<=.25001
            # Physical work bounds may be temporarily extended only for accepted initial poses.
            bounds=guard.mapper.bounds if tick>=180 else guard.mapper.start_bounds
            for k,(low,high) in bounds.items():assert low-1e-5<=sent[k]<=high+1e-5
            obs={k:obs[k]+.7*(sent[k]-obs[k]) for k in KEYS};checks+=1;clipped+=bool(flags['relative_or_speed_clipped'])
    good={k:float(initial[-1,j]) for j,k in enumerate(KEYS)}
    guard=Guard(cfg,follower,good)
    try:guard.command({**good,KEYS[0]:float('nan')},good,.005,1/60)
    except ValueError:pass
    else:raise AssertionError('NaN not rejected')
    try:Guard(cfg,follower,good).command(good,good,.1,1/60)
    except RuntimeError:pass
    else:raise AssertionError('Stale feedback not rejected')
    # No leader coordinate corrections enter the model guard.
    assert ModelMapping(cfg,follower).nominal(good)==good
    result={'status':'passed','simulated_control_cycles':checks,'cycles_limited':clipped,'nan_rejected':True,'stale_feedback_rejected':True,'no_leader_offsets':True,'gripper_mode':'range_only','hardware_opened':False}
    (ROOT/'logs/yellow-model-guard-offline.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline-check',action='store_true');args=parser.parse_args()
    cfg=follow.load_config()
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    if args.offline_check:
        raise SystemExit('Use OfflineCheck.cmd for the self-contained release check.')
    require_local_hardware()
    import msvcrt
    lock=(ROOT/'logs/yellow-model-eval.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another model test is active')
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');run=ROOT/'evaluations/yellow_nut_to_brown_tray'/('step005000-'+stamp);run.mkdir(parents=True)
    latest=ROOT/'logs/yellow-model-eval-latest.json';latest.write_text(json.dumps({'run_dir':str(run),'pid':os.getpid(),'checkpoint':str(CHECKPOINT)},indent=2),encoding='utf-8')
    log=(run/'control.jsonl').open('a',encoding='utf-8');last_flush=time.perf_counter();state={'status':'loading_model','pid':os.getpid(),'run_dir':str(run),'checkpoint':str(CHECKPOINT),'controller_version':'trajectory-phase-v7','motor_enable_attempted':False,'trials':0,'cleanup':{}}
    def status(value,**kw):
        state.update(status=value,time=datetime.now().astimezone().isoformat(),**kw)
        temp=run/'status.tmp';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(run/'status.json')
        print(value.upper()+(' '+json.dumps(kw) if kw else ''),flush=True)
    def emit(event,**kw):
        nonlocal last_flush
        now=time.perf_counter();log.write(json.dumps({'time':time.time(),'monotonic':now,'event':event,**kw})+'\n')
        if event!='control' or now-last_flush>.5:log.flush();last_flush=now
    context=mp.get_context('spawn');requests=context.Queue(maxsize=1);results=context.Queue(maxsize=2);stop=context.Event();process=context.Process(target=worker,args=(requests,results,stop),daemon=True)
    leader=follower=camera=None;enabled=False
    (run/'input_images').mkdir();request_id=0
    status('loading_model');process.start()
    try:
        print('正在加载第5000步模型，请稍候……',flush=True)
        deadline=time.perf_counter()+240
        while True:
            if not process.is_alive():raise RuntimeError('Inference process exited during loading')
            try:message=results.get(timeout=.1)
            except queue.Empty:
                if time.perf_counter()>deadline:raise TimeoutError('Model loading timed out')
                continue
            if message['event']=='worker_error':raise RuntimeError(message['error'])
            if message['event']=='ready':break
        if message.get('chunk_size')!=50:raise ValueError('Expected the saved 50-action checkpoint')
        action_queue=RetimedActionQueue(execution_horizon=message['n_action_steps'])
        emit('model_ready',**{k:v for k,v in message.items() if k!='event'})
        status('model_loaded',warm_single_inference_s=message['warm_single_inference_s'],
               chunk_size=message['chunk_size'],execution_horizon=action_queue.execution_horizon)
        recorder=load_script('yellow_eval_recorder','record-so101.py')
        from serial.tools import list_ports
        ports={p.serial_number:p.device for p in list_ports.comports()}
        for role,serial in cfg['usb_serials'].items():
            if ports.get(serial)!=cfg['ports'][role]:raise RuntimeError('USB identity changed: '+role)
        leader,follower=follow.make_devices(cfg)
        start_profile=load_start_pose(ROOT/'configs/yellow-model-start-pose.json',cfg,ModelMapping(cfg,follower).bounds)
        fixed_start=FixedStart(start_profile)
        for device in (leader,follower):device.bus.connect()
        snapshot=follow.inspect_connected(cfg,leader,follower,allow_start_outside=True)
        Guard(cfg,follower,snapshot['follower']['normalized'])
        camera_config,_,camera_info=recorder.choose_camera()
        from lerobot.cameras.opencv import OpenCVCamera
        import cv2
        camera=OpenCVCamera(camera_config);camera.connect();probe_image,probe_received=recorder.read_camera_frame(camera,1500)
        # Verify the new cross-process progress token and the real camera input
        # before any configure/goal write. This prediction is never executed.
        probe_observation=follower.get_observation();probe_time=time.perf_counter()
        requests.put({'epoch':0,'request_progress':0,'image':probe_image,'state':[probe_observation[k] for k in KEYS],'observed_at':min(probe_time,probe_received)},timeout=1)
        shadow=results.get(timeout=5)
        if shadow.get('event')!='prediction' or shadow.get('request_progress')!=0:raise RuntimeError('Startup inference check failed: '+str(shadow))
        cv2.imwrite(str(run/'ready_camera.jpg'),cv2.cvtColor(probe_image,cv2.COLOR_RGB2BGR))
        state['startup_shadow_passed']=True
        emit('startup_shadow_prediction',inference_s=shadow['inference_s'],prediction_shape=[len(shadow['actions']),len(shadow['actions'][0])],motor_enable_attempted=False)
        mode='ready';epoch=0;pending=False;pending_at=0;guard=None;hold=None;started=0;last_tick=time.perf_counter();last_display=0;last_report=time.perf_counter();report_cycles=0;preview_enabled=True
        title='SO101 yellow nut / model 5000 / Q hold / R reset / X exit'
        recorder.read_operator_keys()  # Ignore keys typed while the model was loading.
        status('ready',camera=camera_info,fixed_start_target=fixed_start.target,start_pose_source=start_profile['source'],start_pose_reference=start_profile['reference_image'])
        print('模型加载完成。请按一次 Enter 复位到固定起始位置。',flush=True)
        while True:
            tick=time.perf_counter();tick_precise=time.perf_counter();dt=tick_precise-last_tick;last_tick=tick_precise
            keys=recorder.read_operator_keys()
            if preview_enabled and tick-last_display>=1/30:
                with camera.frame_lock:image=camera.latest_frame;received=camera.latest_timestamp
                if image is not None:
                    try:
                        display=cv2.cvtColor(image,cv2.COLOR_RGB2BGR);cv2.putText(display,mode.upper()+' | '+({'ready':'ENTER 1: return to start','startup':'Returning to start...','homing':'Returning to start...','at_start':'ENTER 2: start inference','camera_wait':'Waiting for fresh image...','model':'Inference running','hold':'ENTER 1: return to start','reset':'ENTER 1: return to start'}.get(mode,''))+' | Q hold  X exit',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2);cv2.imshow(title,display)
                        key=cv2.waitKey(1)&255
                        if key!=255:keys.append(chr(key))
                    except cv2.error:
                        preview_enabled=False
                        emit('preview_disabled',reason='OpenCV GUI unavailable; live camera input remains active')
                        print('Camera input is live. Preview is unavailable; use keys in THIS terminal.',flush=True)
                last_display=tick
            if any(k.lower()=='x' for k in keys):break
            observation=follower.get_observation();observed_at=time.perf_counter()
            enter_pressed=any(k in ('\r','\n') for k in keys)
            if mode=='at_start' and enter_pressed:
                if fixed_start.confirm(time.perf_counter()):
                    mode='camera_wait';epoch+=1;action_queue.reset()
                    emit('start_pose_confirmed',trial=state['trials'],epoch=epoch)
                    status('camera_wait',trial=state['trials'])
            elif mode not in ('model','startup','homing','at_start','camera_wait') and enter_pressed:
                if not enabled:
                    snapshot=follow.inspect_connected(cfg,leader,follower,allow_start_outside=True)
                    Guard(cfg,follower,snapshot['follower']['normalized'])
                    state['motor_enable_attempted']=True
                    leader.configure();raw=follower.bus.sync_read('Present_Position',normalize=False,num_retry=3);follower.bus.sync_write('Goal_Position',raw,normalize=False)
                    if follower.bus.sync_read('Goal_Position',normalize=False,num_retry=3)!=raw:raise RuntimeError('Initial goal readback mismatch')
                    follower.configure();enabled=True
                    observation=follower.get_observation();observed_at=time.perf_counter()
                epoch+=1;action_queue.reset();fixed_start.cancel();guard=Guard(cfg,follower,observation);hold=dict(observation);mode='startup';started=time.perf_counter();state['trials']+=1
                emit('trial_started',trial=state['trials'],epoch=epoch);status('startup',trial=state['trials'])
                print('正在复位，请稍候……',flush=True)
            if enabled and any(k.lower() in ('q','s','f') for k in keys):
                result=next(({'s':'success','f':'failure'}[k.lower()] for k in keys if k.lower() in ('s','f')),'unlabelled')
                mode='hold';epoch+=1;action_queue.reset();fixed_start.cancel();hold=dict(observation);guard=Guard(cfg,follower,observation);emit('trial_stopped',trial=state['trials'],result=result);status('hold',result=result)
            if enabled and any(k.lower()=='r' for k in keys):
                mode='reset';epoch+=1;action_queue.reset();fixed_start.cancel();guard=Guard(cfg,follower,observation,model=False);status('reset');emit('reset_started')
            while True:
                try:message=results.get_nowait()
                except queue.Empty:break
                pending=False
                if message['event']=='worker_error':
                    mode='hold';epoch+=1;action_queue.reset();fixed_start.cancel();hold=dict(observation);guard=Guard(cfg,follower,observation);status('hold',error=message['error']);continue
                if message['event']=='prediction':
                    now=time.perf_counter()
                    if mode=='model' and message['epoch']==epoch:
                        accepted=action_queue.offer(message['actions'],message['request_progress'],message['observed_at'],now)
                        emit('prediction',**{k:v for k,v in message.items() if k!='event'},arrival_age_s=now-message['observed_at'],accepted=accepted)
            if enabled:
                if mode=='startup' and time.perf_counter()-started>=cfg['startup_blend_seconds'] and guard.mapper.started is not None and time.monotonic()-guard.mapper.started>=cfg['startup_blend_seconds']:
                    fixed_start.begin(time.perf_counter());mode='homing';action_queue.reset()
                    emit('startup_complete');emit('homing_started',trial=state['trials'],target=fixed_start.target,source=start_profile['source'])
                    status('homing',trial=state['trials'])
                if mode=='model' and (time.perf_counter()-started>45 or not process.is_alive()):
                    mode='hold';epoch+=1;action_queue.reset();fixed_start.cancel();hold=dict(observation);guard=Guard(cfg,follower,observation);status('hold',reason='45 second limit or inference worker stopped')
                with camera.frame_lock:image=camera.latest_frame;received=camera.latest_timestamp
                camera_age=time.perf_counter()-float(received or 0)
                if mode in ('homing','at_start','camera_wait'):
                    try:
                        if camera_age>.5 or not process.is_alive():raise RuntimeError('Camera or inference worker unavailable during fixed start')
                        phase=fixed_start.update(observation,time.perf_counter(),float(received or 0))
                        next_mode={'moving':'homing','at_start':'at_start','camera_wait':'camera_wait','model':'model'}[phase]
                        if next_mode!=mode:
                            mode=next_mode
                            if mode=='at_start':
                                emit('homing_complete',trial=state['trials'],observation=observation,target=fixed_start.target)
                                status('at_start',trial=state['trials'])
                                print('复位完成。请摆好黄色螺母并移开手，然后按第二次 Enter 开始推理。',flush=True)
                            elif mode=='model':
                                started=time.perf_counter();action_queue.reset()
                                emit('model_started',trial=state['trials'],epoch=epoch,frame_received=received)
                                status('model',trial=state['trials'])
                            else:status(mode,trial=state['trials'])
                    except (ValueError,RuntimeError) as exc:
                        mode='hold';epoch+=1;action_queue.reset();fixed_start.cancel();hold=dict(observation);guard=Guard(cfg,follower,observation)
                        status('hold',reason=str(exc))
                queue_fault=action_queue.stale_reason(time.perf_counter()) if mode=='model' else None
                if mode=='model' and (camera_age>.5 or (pending and time.perf_counter()-pending_at>1.5) or queue_fault):
                    mode='hold';epoch+=1;action_queue.reset();fixed_start.cancel();hold=dict(observation);guard=Guard(cfg,follower,observation);status('hold',reason=queue_fault or 'Camera or inference timeout')
                target=dict(guard.previous)
                if mode=='model':
                    if not pending and camera_age<.1 and image is not None:
                        try:
                            requests.put_nowait({'debug_image_path':str(run/'input_images'/f'request_{request_id:06d}.jpg'),'epoch':epoch,'image':image.copy(),'state':[observation[k] for k in KEYS],'request_progress':action_queue.progress,'observed_at':min(observed_at,float(received))});pending=True;pending_at=time.perf_counter();request_id+=1
                        except queue.Full:pass
                    waypoint=action_queue.target(time.perf_counter(),[guard.previous[k] for k in KEYS],dt,[guard.mapper.bounds[k] for k in KEYS])
                    if action_queue.last_install is not None:
                        emit('queue_install',**action_queue.last_install);action_queue.last_install=None
                    if waypoint is not None:target=dict(zip(KEYS,waypoint))
                elif mode in ('homing','at_start','camera_wait'):target=fixed_start.target
                elif mode=='startup':target=hold
                elif mode=='reset':target=leader.get_action()
                elif mode=='hold':target=hold
                age=time.perf_counter()-observed_at
                if age>cfg['observation_max_age_seconds']:observation=follower.get_observation();observed_at=time.perf_counter();age=0
                if mode=='hold':guard.mapper.started=time.monotonic()  # Freeze the accepted pose instead of progressing a startup blend.
                try:command,flags=guard.command(target,observation,age,dt,policy_queue=action_queue if mode=='model' and waypoint is not None else None)
                except (ValueError,RuntimeError) as exc:
                    mode='hold';epoch+=1;action_queue.reset();fixed_start.cancel();hold=dict(observation);guard=Guard(cfg,follower,observation);guard.mapper.started=time.monotonic();status('hold',error=str(exc));command,flags=guard.command(hold,observation,0,1/60)
                sent=follower.send_action(command);report_cycles+=1
                action_index=action_queue.index
                if mode=='model':action_queue.advance(flags['common_speed_scale'])
                emit('control',trial=state['trials'],mode=mode,observation=observation,requested=target,sent=sent,feedback_age_ms=age*1000,action_index=action_index,queue_progress=action_queue.progress,queue_remaining=action_queue.remaining(),**flags)
                if tick-last_report>1:
                    print(f"{mode.upper()} trial={state['trials']} Hz={report_cycles/max(tick-last_report,.001):.1f} camera_age={camera_age:.3f}s pending={pending} clipped={len(flags['relative_or_speed_clipped'])} Q=HOLD",flush=True);last_report=tick;report_cycles=0
            elapsed=time.perf_counter()-tick_precise
            if elapsed<1/60:time.sleep(1/60-elapsed)
        status('operator_exit')
    except KeyboardInterrupt:status('operator_exit')
    except BaseException as exc:
        status('failed',error=repr(exc));emit('error',traceback=traceback.format_exc())
        if state['motor_enable_attempted']:
            print('FAULT: last goal remains held. Support follower, then press ENTER to release torque.',flush=True)
            try:input()
            except (KeyboardInterrupt,EOFError):pass
        raise
    finally:
        for role,device in (('follower',follower),('leader',leader)):
            if device is None or not device.bus.is_connected:continue
            try:
                if state['motor_enable_attempted']:
                    device.bus.disable_torque(num_retry=3);torque={j:device.bus.read('Torque_Enable',j,normalize=False,num_retry=3) for j in JOINTS};state['cleanup'][role]=torque
                    if any(torque.values()):raise RuntimeError('Torque release not verified')
                else:state['cleanup'][role]='closed without motor setup'
            except BaseException as exc:state['cleanup'][role]={'error':repr(exc)};print('CLEANUP '+role+': '+str(exc),flush=True)
            finally:device.bus.disconnect(disable_torque=False)
        if camera is not None and camera.is_connected:camera.disconnect()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:pass
        stop.set();process.join(timeout=3)
        if process.is_alive():process.terminate();process.join(timeout=2)
        state['ended']=datetime.now().astimezone().isoformat();(run/'summary.json').write_text(json.dumps(state,indent=2),encoding='utf-8');log.close()
        print('Finished. '+str(run),flush=True)

if __name__=='__main__':mp.freeze_support();main()
