"""Release paths and explicit local-hardware opt-in; imports open no devices."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(os.environ.get('SO101_ROOT',Path(__file__).resolve().parents[1])).resolve()
CHECKPOINT=Path(os.environ.get('SO101_CHECKPOINT',ROOT/'models/yellow_nut_smolvla_005000'))
TOKENIZER=CHECKPOINT/'tokenizer'

def require_local_hardware():
    marker=ROOT/'configs/local-hardware-confirmed.json'
    if not marker.is_file():raise RuntimeError('Import your reviewed local robot configuration with ConfigureHardware.cmd before live control. Published calibration is reference-only.')
    profile=json.loads(marker.read_text(encoding='utf-8'))
    for rel,expected in profile['sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Local hardware configuration changed; review and re-import it: '+rel)
