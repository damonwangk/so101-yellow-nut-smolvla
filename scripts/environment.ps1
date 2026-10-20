$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:SO101_ROOT=$root
$env:PYTHONUTF8='1'
$env:HF_HOME=Join-Path $root 'cache\huggingface'
$env:HF_LEROBOT_HOME=Join-Path $root 'cache\lerobot'
$env:HF_LEROBOT_CALIBRATION=Join-Path $root 'calibration'
$env:TORCH_HOME=Join-Path $root 'cache\torch'
$env:TEMP=Join-Path $root 'tmp'
$env:TMP=$env:TEMP
foreach($name in @('logs','tmp','outputs','captures')){New-Item -ItemType Directory -Force -Path (Join-Path $root $name) | Out-Null}
$env:PATH=(Join-Path $root '.venv\Scripts')+';'+(Join-Path $root 'tools\ffmpeg\bin')+';'+$env:PATH
