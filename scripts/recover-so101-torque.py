"""Release leftover torque only after a local operator confirms supporting the arms."""
import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

from so101_paths import ROOT


def release_torque(devices, joints):
    report={'roles':{},'all_off_verified':False,'errors':[]}
    try:
        for role,device in devices:
            device.bus.connect()
        for role,device in devices:
            states={}
            for name in joints:
                try:
                    device.bus.disable_torque(motors=name,num_retry=3)
                except Exception as exc:
                    report['errors'].append(f'{role} {name} release: {exc!r}')
                try:
                    states[name]=int(device.bus.read('Torque_Enable',name,normalize=False,num_retry=3))
                except Exception as exc:
                    states[name]=None
                    report['errors'].append(f'{role} {name} readback: {exc!r}')
            report['roles'][role]=states
        report['all_off_verified']=len(report['roles'])==len(devices) and all(v==0 for states in report['roles'].values() for v in states.values())
    except Exception as exc:
        report['errors'].append(repr(exc))
    finally:
        for role,device in devices:
            if device.bus.is_connected:
                try:
                    device.bus.disconnect(disable_torque=False)
                except Exception as exc:
                    report['errors'].append(f'{role} disconnect: {exc!r}')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-confirmed-release',action='store_true')
    args=parser.parse_args(argv)
    if not args.operator_confirmed_release:
        print('请从正式采集窗口执行恢复：先托稳机械臂，再按提示输入R确认。未连接硬件。',flush=True)
        return 2
    spec=importlib.util.spec_from_file_location('so101_follow_recovery',ROOT/'scripts/follow-practical.py')
    follow=importlib.util.module_from_spec(spec);spec.loader.exec_module(follow)
    cfg=follow.load_config()
    from serial.tools import list_ports
    ports={p.serial_number:p.device for p in list_ports.comports()}
    for role,serial in cfg['usb_serials'].items():
        if ports.get(serial)!=cfg['ports'][role]:
            raise RuntimeError('USB identity differs; recovery did not write motors: '+role)
    leader,follower=follow.make_devices(cfg)
    report=release_torque([('follower',follower),('leader',leader)],follow.JOINTS)
    report.update(time=datetime.now().astimezone().isoformat(),operator_confirmed=True,goal_positions_written=False,calibration_written=False)
    path=ROOT/'logs'/('torque-recovery-'+datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if not report['all_off_verified'] or report['errors']:
        print('恢复未完全确认，请继续支撑机械臂。日志：'+str(path),flush=True)
        return 1
    print('RECOVERED：主从臂12个电机力矩均已释放。请摆好姿态；采集仍会等待你按Enter。',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
