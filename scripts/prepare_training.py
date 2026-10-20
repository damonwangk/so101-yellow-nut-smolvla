"""Create a NEW training run; source dataset is read-only."""
import json,subprocess,sys
from datetime import datetime
from so101_paths import ROOT
if __name__=='__main__':
    spec=json.loads((ROOT/'configs/training-reference.json').read_text(encoding='utf-8'))
    name='yellow_nut_smolvla_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    logs=ROOT/'logs'/name;logs.mkdir(parents=True)
    spec.update(run_name=name,log_dir=str(logs),output_dir=str(ROOT/'outputs/train'/name),dataset_root=str(ROOT/'datasets/yellow_nut_to_brown_tray/formal/lerobot'),steps=12000)
    path=logs/'run_spec.json';path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    raise SystemExit(subprocess.call([sys.executable,'-u',str(ROOT/'scripts/train-yellow-smolvla.py'),'--run-spec',str(path)],cwd=ROOT))
