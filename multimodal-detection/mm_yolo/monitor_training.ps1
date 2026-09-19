# Monitor the unattended A0 -> B0 training until B0 finishes or the run dies.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File code\mm_yolo\monitor_training.ps1
#
# Liveness is judged by the training log's mtime plus the presence of python processes,
# not by counting powershell processes (the user may have other windows open).
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\35482\Desktop\人工智能精英\2'
Set-Location $root
$log  = Join-Path $root 'code\runs\monitor.log'
$aLog = Join-Path $root 'code\runs\a0_rgb\train.log'
$bLog = Join-Path $root 'code\runs\b0_mm\train.log'
$wLog = Join-Path $root 'code\runs\watchdog.log'

function L([string]$m) {
    $stamp = Get-Date -Format 'MM-dd HH:mm:ss'
    Add-Content -LiteralPath $log -Value ('[' + $stamp + '] ' + $m) -Encoding UTF8
}

function EpCount([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return 0 }
    $m = @(Select-String -LiteralPath $p -Pattern 'ep \d+/60' -ErrorAction SilentlyContinue)
    return $m.Count
}

function IsDone([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return $false }
    $c = Get-Content -LiteralPath $p -Raw -Encoding UTF8
    return [bool]($c -match '\[train\] ')
}

function LastLine([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return 'n/a' }
    $m = @(Select-String -LiteralPath $p -Pattern 'ep \d+/60' -ErrorAction SilentlyContinue)
    if ($m.Count -eq 0) { return 'no epoch yet' }
    $t = $m[$m.Count - 1].Line
    return ($t -replace '^\[train\] ', '')
}

function Finished([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return $false }
    $c = Get-Content -LiteralPath $p -Raw -Encoding UTF8
    return [bool]($c -match '完成')
}

function PyCount {
    $p = @(Get-Process python -ErrorAction SilentlyContinue)
    return $p.Count
}

L '==== monitor start ===='
$lastA = -1
$lastB = -1
$prevStamp = ''

for ($i = 0; $i -lt 1400; $i++) {
    $aN = EpCount $aLog
    $bN = EpCount $bLog
    $stamp = '' + $aN + '/' + $bN
    $np = PyCount

    $changed = $stamp -ne $prevStamp
    $tick = ($i % 10) -eq 0
    if ($tick -or $changed) {
        $msg = 'A0 ep=' + $aN + '/60 :: ' + (LastLine $aLog)
        $msg = $msg + '   ||   B0 ep=' + $bN + '/60 :: ' + (LastLine $bLog)
        $msg = $msg + '   [py=' + $np + ']'
        L $msg
        $prevStamp = $stamp
    }

    if (Finished $bLog) {
        L 'ALL-DONE: B0 finished'
        break
    }
    if ((Finished $aLog) -and (-not (Test-Path -LiteralPath $bLog)) -and $np -eq 0) {
        L 'A0-DONE-NO-B0: A0 finished but B0 never started and no python process'
        break
    }

    if ($np -eq 0) {
        $newest = $null
        foreach ($f in @($aLog, $bLog)) {
            if (Test-Path -LiteralPath $f) {
                $t = (Get-Item -LiteralPath $f).LastWriteTime
                if ($null -eq $newest) { $newest = $t }
                elseif ($t -gt $newest) { $newest = $t }
            }
        }
        if ($null -ne $newest) {
            $idle = [int]((Get-Date) - $newest).TotalMinutes
            if ($idle -gt 20) {
                L ('DEAD: no python process and logs idle ' + $idle + ' min')
                L ('      A0 ep=' + $aN + ' ; B0 ep=' + $bN + ' ; see ' + $wLog)
                break
            }
        }
    }
    Start-Sleep -Seconds 30
}
L '==== monitor exit ===='
