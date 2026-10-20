$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$Host.UI.RawUI.WindowTitle='SO101 Yellow Nut - Model 5000 TEST'
Set-Location (Split-Path -Parent $PSScriptRoot)
. '.\scripts\environment.ps1'
$busy=Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match '(follow-practical|record-so101|recover-so101-torque|test-yellow-smolvla-robot)\.py' -and $_.CommandLine -notmatch '--offline-check|--read-only'}
if($busy){Write-Host 'Another robot program is active. Close its window before opening this test.';Read-Host 'Press ENTER to close';exit 1}
try {
 Write-Host 'Checkpoint 5000 guarded test. Model loading takes about 1-2 minutes.' -ForegroundColor Cyan
 Write-Host 'Please wait for model loading, then follow the next on-screen prompt.'
 Write-Host 'Q pauses and HOLDS torque. R uses leader to reset.'
 Write-Host 'S marks success; F marks failure. Support follower before X or Ctrl+C releases torque.'
 & '.\.venv\Scripts\python.exe' -u '.\scripts\test-yellow-smolvla-robot.py'
 Write-Host ('Test process ended. Exit code: '+$LASTEXITCODE)
} catch {Write-Host $_.Exception.Message -ForegroundColor Yellow}
Read-Host 'Program ended. Press ENTER to close this window'
