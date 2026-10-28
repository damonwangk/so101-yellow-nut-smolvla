param([ValidateSet('install','download','configure','offline','test','train','collect','trial')][string]$Mode)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
try {
 if($Mode -eq 'install'){
  if(-not(Test-Path '.venv\Scripts\python.exe')){ & py -3.12 -m venv .venv; if($LASTEXITCODE){throw 'Python 3.12 with the py launcher is required.'} }
  & '.\.venv\Scripts\python.exe' -m pip install 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
  if($LASTEXITCODE){throw 'CUDA PyTorch installation failed'}
  & '.\.venv\Scripts\python.exe' -m pip install -r requirements-windows.lock.txt
  if($LASTEXITCODE){throw 'Dependency installation failed'}
  Write-Host 'Installed. Add FFmpeg shared DLLs to tools\ffmpeg\bin or PATH; see README.'
  exit
 }
 $python=Join-Path $root '.venv\Scripts\python.exe'
 if(-not(Test-Path $python)){throw 'Run Install.cmd first (Python 3.12 is required).'}
 switch($Mode){
  'download' { & $python -u scripts\download_assets.py }
  'configure' { & $python -u scripts\configure_hardware.py }
  'offline' { & $python -u scripts\release_offline_check.py }
  'test' { & $python scripts\check_local_hardware.py; if($LASTEXITCODE){throw 'Local hardware profile is required'}; & (Join-Path $PSScriptRoot 'start-yellow-model-test.ps1') }
  'train' { & $python -u scripts\prepare_training.py }
  'collect' { & $python scripts\check_local_hardware.py; if($LASTEXITCODE){throw 'Local hardware profile is required'}; & (Join-Path $PSScriptRoot 'record-yellow-batch.ps1') }
  'trial' { & $python scripts\check_local_hardware.py; if($LASTEXITCODE){throw 'Local hardware profile is required'}; & $python -u scripts\record-so101.py --task 'Pick up the yellow nut and place it into the brown tray.' --dataset-id yellow_nut_trial --dataset-root (Join-Path $root 'captures\yellow_nut_to_brown_tray\trial\lerobot') --episode-root (Join-Path $root 'captures\yellow_nut_to_brown_tray\trial\episodes') --log-root (Join-Path $root 'captures\yellow_nut_to_brown_tray\trial\sessions') }
 }
 if($LASTEXITCODE){throw ('Program returned '+$LASTEXITCODE)}
} catch {Write-Host $_.Exception.Message -ForegroundColor Yellow;exit 1}
