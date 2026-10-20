"""Download released model/data and verify SHA-256. Never opens robot hardware."""
import argparse,hashlib,json,os,urllib.request,zipfile
from pathlib import Path
from so101_paths import ROOT,CHECKPOINT


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def github_download(entry):
    cache=ROOT/'cache/releases';cache.mkdir(parents=True,exist_ok=True)
    target=cache/entry['name'];partial=target.with_suffix(target.suffix+'.partial')
    if not target.exists() or sha256(target)!=entry['sha256']:
        request=urllib.request.Request(entry['url'],headers={'User-Agent':'so101-reproduce/1.0'})
        with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as out:
            while block:=response.read(4*1024*1024):out.write(block)
        if sha256(partial)!=entry['sha256']:raise RuntimeError('Archive checksum mismatch: '+entry['name'])
        os.replace(partial,target)
    with zipfile.ZipFile(target) as archive:
        for info in archive.infolist():
            dest=(ROOT/info.filename).resolve()
            if not dest.is_relative_to(ROOT.resolve()):raise RuntimeError('Unsafe archive path')
            if (info.external_attr>>16)&0o170000==0o120000:raise RuntimeError('Archive contains a symlink')
        archive.extractall(ROOT)


def verify_files(kind,dest,manifest):
    entries=manifest[kind]
    for relative,expected in entries.items():
        path=dest/relative
        if not path.is_file() or sha256(path)!=expected['sha256']:
            raise RuntimeError('Missing or altered released file: '+kind+'/'+relative)
    print(f'CHECKED {kind}: {len(entries)} files; SHA-256 matches.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['hf','github'],default='hf');parser.add_argument('--training-base',action='store_true');args=parser.parse_args()
    manifest=json.loads((ROOT/'release.json').read_text(encoding='utf-8'))
    files=json.loads((ROOT/'provenance/artifact-manifest.json').read_text(encoding='utf-8'))
    for kind,dest in [('model',CHECKPOINT),('dataset',ROOT/'datasets/yellow_nut_to_brown_tray/formal/lerobot')]:
        entry=manifest[kind]
        if args.source=='github':
            if CHECKPOINT!=ROOT/'models/yellow_nut_smolvla_005000':raise RuntimeError('GitHub archives use the default checkpoint path; unset SO101_CHECKPOINT.')
            github_download(entry['github_asset'])
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(entry['repo_id'],repo_type='model' if kind=='model' else 'dataset',revision=entry.get('revision') or 'main',local_dir=str(dest),allow_patterns=entry['allow_patterns'])
        verify_files(kind,dest,files)
    if args.training_base:
        from huggingface_hub import snapshot_download
        snapshot_download('lerobot/smolvla_base',local_dir=str(ROOT/'models/smolvla_base'))
    print('Download and checksum verification complete. No robot or training process was started.')
if __name__=='__main__':main()
