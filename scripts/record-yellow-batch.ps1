param([ValidateRange(1,500)][int]$TargetTotal = 50)
$ErrorActionPreference='Stop'
$utf8=[Text.UTF8Encoding]::new($false)
[Console]::InputEncoding=$utf8
[Console]::OutputEncoding=$utf8
$OutputEncoding=$utf8
$Host.UI.RawUI.WindowTitle='SO-101 单黄色螺母正式采集 50条'
. (Join-Path $PSScriptRoot 'environment.ps1')
$taskRoot=Join-Path $root 'captures\yellow_nut_to_brown_tray'
$phaseRoot=Join-Path $taskRoot 'formal'
New-Item -ItemType Directory -Force -Path $phaseRoot,(Join-Path $phaseRoot 'sessions') | Out-Null
$datasetRoot=Join-Path $phaseRoot 'lerobot'
$episodeRoot=Join-Path $phaseRoot 'episodes'
$logRoot=Join-Path $phaseRoot 'sessions'
$config=Get-Content -LiteralPath (Join-Path $root 'configs\task.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$python=Join-Path $root '.venv\Scripts\python.exe'
$recorder=Join-Path $PSScriptRoot 'record-so101.py'
$progressPath=Join-Path $phaseRoot 'batch-progress.json'
function Get-SavedCount {
    $infoPath=Join-Path $datasetRoot 'meta\info.json'
    if (-not (Test-Path -LiteralPath $infoPath)) { return 0 }
    $info=Get-Content -LiteralPath $infoPath -Raw -Encoding UTF8 | ConvertFrom-Json
    return [int]$info.total_episodes
}
function Set-BatchProgress([string]$Status,[int]$Count) {
    $json=[ordered]@{task_id='yellow_nut_to_brown_tray';phase='formal';target_saved_episodes=$TargetTotal;saved_episodes=$Count;status=$Status;updated=(Get-Date -Format o);note='保存数为技术提交数量，任务成功需依据实际示范确认。'} | ConvertTo-Json
    $tempPath=$progressPath+'.tmp'
    [IO.File]::WriteAllText($tempPath,$json,$utf8)
    Move-Item -LiteralPath $tempPath -Destination $progressPath -Force
}
function Wait-BatchAction([bool]$AllowRetry=$false,[bool]$ReleaseTorque=$false) {
    while ($true) {
        $prompt=if ($ReleaseTorque) { '确认托稳从臂、放稳主臂后输入R并回车，释放残留力矩后重试；输入X退出' } elseif ($AllowRetry) { '调整就位后输入R再回车重试；输入X再回车退出（单独回车会保留窗口）' } else { '输入X再回车关闭窗口（单独回车会保留窗口）' }
        $answer=([string](Read-Host $prompt)).Trim()
        if ($answer -ieq 'X') { return 'X' }
        if ($AllowRetry -and $answer -ieq 'R') { return 'R' }
    }
}
$exitRequested=$false
$mutex=[Threading.Mutex]::new($false,'Local\SO101YellowFormalBatch')
$ownsMutex=$false
try {
    $ownsMutex=$mutex.WaitOne(0)
    if (-not $ownsMutex) { throw '已有黄色螺母正式批次窗口，请回到原窗口继续。' }
    Write-Host ('单黄色螺母正式采集：目标 '+$TargetTotal+' 条。') -ForegroundColor Cyan
    Write-Host '每条Enter开始；放好后保持画面1-2秒，按Q结束本条并等待SAVED。Q后主从继续跟随。'
    Write-Host '用主臂将从臂带回起点，重摆螺母；READY后Enter开始下一条，复位过程不录制。托稳从臂后按X退出控制。'
    Write-Host '再次打开此入口会从已保存数量继续；本窗口只采集，不启动训练。'
    while ($true) {
        $count=Get-SavedCount
        if ($count -ge $TargetTotal) { Set-BatchProgress 'target_saved_count_reached' $count; Write-Host ('已保存 '+$count+' 条，批次结束，等待质量核对。') -ForegroundColor Green; break }
        $running=@(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match '(follow-practical|record-so101)\.py' -and $_.CommandLine -notmatch '--offline-check|--read-only' })
        if ($running.Count -gt 0) { throw '已有跟随或采集程序，停止本批次以免重复启动。' }
        if (Test-Path -LiteralPath (Join-Path $datasetRoot '.recording-pending.json')) { throw '上一条仍有未完成提交标记，请先处理保存状态。' }
        Set-BatchProgress 'episode_in_progress' $count
        Write-Host ('--- 准备第 '+($count+1)+'/'+$TargetTotal+' 条；已保存 '+$count+' 条 ---') -ForegroundColor Cyan
        $previousSummary=Get-ChildItem -LiteralPath $logRoot -Filter '*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        & $python -u $recorder --task $config.task --dataset-id 'yellow_nut_to_brown_tray_formal' --dataset-root $datasetRoot --episode-root $episodeRoot --log-root $logRoot --target-total $TargetTotal
        $code=$LASTEXITCODE
        $after=Get-SavedCount
        if ($code -eq 0) {
            $status=if($after -ge $TargetTotal){'target_saved_count_reached'}else{'stopped'}
            Set-BatchProgress $status $after
            Write-Host ('控制已退出，已保存 '+$after+'/'+$TargetTotal+' 条；再次打开可续接。') -ForegroundColor Green
            break
        }
        if ($code -ne 0) {
            Set-BatchProgress 'stopped_without_new_commit' $after
            Write-Host '本条未正常新增一条已提交数据。' -ForegroundColor Yellow
            $latest=Get-ChildItem -LiteralPath $logRoot -Filter '*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $summary=$null
            if ($latest -and (-not $previousSummary -or $latest.Name -ne $previousSummary.Name)) {
                try { $summary=Get-Content -LiteralPath $latest.FullName -Raw -Encoding UTF8 | ConvertFrom-Json } catch { }
            }
            $canRetry=$false
            $staleTorque=$false
            if ($summary) {
                Write-Host ('原因：'+$summary.error) -ForegroundColor Yellow
                Write-Host ('本次摘要：'+$latest.FullName)
                $staleTorque=$summary.error -match '(leader|follower) torque already enabled' -and $summary.motor_enable_attempted -eq $false
                $poseError=$summary.error -match 'starting position outside usable range|Starting pose exceeds smooth recovery margin'
                $canRetry=$poseError -and $summary.motor_enable_attempted -eq $false -and $summary.frames_written -eq 0 -and $summary.cleanup.follower -eq 'closed without motion setup' -and $summary.cleanup.leader -eq 'closed without motion setup'
                if ($canRetry) {
                    Set-BatchProgress 'waiting_for_pose_adjustment' $after
                    Write-Host '尚未启用电机。请托稳从臂，轻微调整提示的关节；不要强推。就位后可重试，本条编号不变。' -ForegroundColor Cyan
                }
                if ($staleTorque) {
                    $canRetry=$true
                    Set-BatchProgress 'waiting_for_operator_torque_recovery' $after
                    Write-Host '检测到上次启动留下的电机力矩。请先托稳从臂、放稳主臂；输入R并回车表示确认释放残留力矩后重试。释放时从臂需要手扶。' -ForegroundColor Cyan
                }
            }
            $choice=Wait-BatchAction $canRetry $staleTorque
            if ($choice -eq 'R') {
                if ($staleTorque) {
                    $other=@(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match '(follow-practical|record-so101)\.py' })
                    if($other.Count -gt 0){throw '检测到其他控制程序，未执行力矩恢复；请先退出其他控制窗口。'}
                    $recovery=Join-Path $PSScriptRoot 'recover-so101-torque.py'
                    & $python -u $recovery --operator-confirmed-release
                    if($LASTEXITCODE -ne 0){throw '残留力矩未能确认全部释放。保持支撑并检查窗口提示，暂不启动采集。'}
                }
                continue
            }
            $exitRequested=$true
            break
        }
    }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Yellow
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
if (-not $exitRequested) { [void](Wait-BatchAction $false) }
