"""Import an existing verified robot profile. Never connects to motors."""
import argparse,hashlib,json,shutil
from pathlib import Path
from so101_paths import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--from-existing',type=Path);a=p.parse_args()
    source=(a.from_existing or Path(input('Existing reviewed SO-101 workspace path: ').strip().strip('"'))).resolve()
    if source==ROOT:raise SystemExit('Use an existing calibrated workspace, not this release folder.')
    cfg_path=source/'configs/practical-follow.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    relative=['configs/practical-follow.json','calibration/leader/win_leader.json','calibration/follower/win_follower.json']
    for role in ('leader','follower'):
        path=source/f'calibration/{role}/win_{role}.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['calibration_sha256'][role]:raise SystemExit('Source calibration hash mismatch: '+role)
    for rel in relative:
        dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/rel,dest)
    home=ROOT/'configs/yellow-model-start-pose.json'
    profile=json.loads(home.read_text(encoding='utf-8'));profile['control_config_sha256']=hashlib.sha256((ROOT/relative[0]).read_bytes()).hexdigest();home.write_text(json.dumps(profile,indent=2),encoding='utf-8')
    hashes={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in relative}
    (ROOT/'configs/local-hardware-confirmed.json').write_text(json.dumps({'source':'operator-selected reviewed local workspace','sha256':hashes},indent=2),encoding='utf-8')
    print('Profile imported. Review camera IDs, mapping, work bounds and clear return path before opening the model test. Motors were not accessed.')
if __name__=='__main__':main()
