# 无人值守看门狗：A0（RGB 锚点）→ B0（三模态），崩溃自动从 last.pt 续训。
#
# 【启动方式】必须在**普通 PowerShell 窗口**里跑（不要从智能体会话里启动，
#   否则会话结束时会连带杀掉进程树）：
#     powershell -ExecutionPolicy Bypass -File code\mm_yolo\watchdog.ps1
#   只启动正式三模态 B1：
#     powershell -ExecutionPolicy Bypass -File code\mm_yolo\watchdog.ps1 -Only B1
#   启动 B2（默认从 B1 best.pt 仅初始化，不覆盖 B1）：
#     powershell -ExecutionPolicy Bypass -File code\mm_yolo\watchdog.ps1 -Only B2
#
# 【本文件的编码要求】保存为 **UTF-8 with BOM**。
#   powershell.exe（Windows PowerShell 5.1）对**无 BOM** 的 .ps1 按当前 ANSI 代码页解码，
#   本脚本里的中文数据路径会变成乱码 → 训练直接报"缺少模态目录"。
#   本文件由写入工具直接产出 UTF-8（含 BOM 由 Add-Content 决定）；若手工编辑过，
#   请务必另存为 "UTF-8 with BOM"。
#
# 【修复记录】
#   * 旧版第 19 行的日期格式串里混入了**全角冒号**（'MM-dd HH:mm:ss'），
#     PowerShell 直接报"意外的标记 ')'" → 看门狗从来没有真正跑起来过；
#   * `--resume` 现在恢复 optimizer / GradScaler / EMA updates / RNG / DataLoader generator；
#   * `--resume-split`（默认开）复用 split.json → 崩溃重启后验证集完全一致，成绩可比；
#   * A1/B1 都用 `--val-limit 0` 全量验证，保证两条基线可直接比较；
#   * 每轮验证日志都会打印"(有效类 n/12)"，缺类不再隐形。
param(
    [ValidateSet('all', 'A1', 'B1', 'B2')]
    [string]$Only = 'all'
)

$ErrorActionPreference = 'Continue'
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONIOENCODING = 'utf-8'
$root = 'C:\Users\35482\Desktop\人工智能精英\2'
Set-Location $root
$py   = 'D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe'
$data = '初赛数据集-面向城市场景的多模态目标检测\train_extracted'
$lab  = '初赛数据集-面向城市场景的多模态目标检测\训练集\new_labels_2000'
$wlog = Join-Path $root 'code\runs\watchdog.log'

function W([string]$m) {
    $line = '[' + (Get-Date -Format 'MM-dd HH:mm:ss') + '] ' + $m
    Write-Host $line
    Add-Content -LiteralPath $wlog -Value $line -Encoding UTF8
}

# ---- 前置校验：路径/编码/解释器任一不对就**立刻报错退出**，不要盲重试 15 次 ----
$fatal = $null
if (-not (Test-Path -LiteralPath $py))   { $fatal = "找不到 conda 解释器: $py" }
elseif (-not (Test-Path -LiteralPath (Join-Path $root $data))) { $fatal = "找不到数据目录: $data" }
elseif (-not (Test-Path -LiteralPath (Join-Path $root $lab)))  { $fatal = "找不到标签目录: $lab" }
elseif (-not (Test-Path -LiteralPath (Join-Path $root 'code\mm_yolo\train.py'))) { $fatal = "找不到 code\mm_yolo\train.py" }
if ($fatal) {
    W "FATAL: $fatal"
    W "若路径看起来正常但仍然报错，多半是本 .ps1 被存成了 **无 BOM 的 UTF-8**："
    W "  powershell.exe 会按 ANSI 代码页解码，中文路径会变乱码。请另存为 UTF-8 with BOM，或用"
    W "  python code\mm_yolo\train.py --root <...> --workers 6 直接启动。"
    exit 2
}

# A1：修复后重新训练的 RGB-only 锚点（不要覆盖历史 a0_rgb）
$AName = 'a1_rgb_clean'
$A = @('code\mm_yolo\train.py','--root',$data,'--labels',$lab,'--name',$AName,'--modalities','rgb',
       '--depth-channels','2','--misalign-px','15','--degrade-p','0.3','--ir-noise-p','0',
       '--depth-hole-p','0','--target-crop-p','0','--rare-sample-max','1',
       '--imgsz','544x960','--batch','8','--accum','2','--workers','6','--epochs','60',
       '--freeze-epochs','5','--val-every','5','--val-limit','0','--val-conf','0.01',
       '--save-every','1')
# B1：三模态 L2 + Depth P4/P5 对齐 + P3→P5 持久 Register
# 本机 RTX 4050 6GB 实测 batch=4 峰值约 3.1GB；accum=4，等效批大小 16。
# 若其他机器报 CUDA out of memory，可改为 batch=2 / accum=8，保持等效批大小 16。
$BName = 'b1_register_s'
$B = @('code\mm_yolo\train.py','--root',$data,'--labels',$lab,'--name',$BName,'--modalities','all',
       '--weights','code\yolo11s.pt','--device','cuda:0','--register-bus','--depth-scales','p4p5',
       '--depth-channels','2','--misalign-px','15','--degrade-p','0.3','--ir-noise-p','0',
       '--depth-hole-p','0','--target-crop-p','0','--rare-sample-max','1',
       '--imgsz','544x960','--batch','4','--accum','4',
       '--workers','6','--epochs','100','--freeze-epochs','5','--rgb-dropout','0.05',
       '--aux-dropout','0.05','--dropout-start-epoch','5','--val-every','5','--val-limit','0',
       '--val-conf','0.01','--save-every','1')

# B2：保留融合注意力 + P3→P5 register，重点修正数据语义与小目标采样。
# 6GB 本机配方：608×1088, batch=3, accum=5（等效 batch 15）。
$B2Name = 'b2_depth4_s_safe'
$B2Init = 'code\runs\b1_register_s\weights\best.pt'
$B2 = @('code\mm_yolo\train.py','--root',$data,'--labels',$lab,'--name',$B2Name,'--modalities','all',
        '--weights','code\yolo11s.pt','--device','cuda:0',
        '--register-bus','--depth-scales','p4p5','--depth-channels','4',
        '--imgsz','608x1088','--batch','3','--accum','5','--workers','6','--epochs','80',
        '--lr','0.0001','--freeze-epochs','3','--rgb-dropout','0.02','--aux-dropout','0.02',
        '--dropout-start-epoch','10','--misalign-px','5','--degrade-p','0.15',
        '--ir-noise-p','0.10','--depth-hole-p','0.10','--target-crop-p','0.08',
        '--rare-sample-max','1.5','--eval-initial','--val-every','5','--val-limit','0','--val-conf','0.01',
        '--save-every','1')

$jobs = @()
if ($Only -in @('all', 'A1')) { $jobs += ,@('A1', $A, $AName) }
if ($Only -in @('all', 'B1')) { $jobs += ,@('B1', $B, $BName) }
if ($Only -in @('all', 'B2')) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $B2Init))) {
        W "FATAL: B2 初始权重不存在: $B2Init"
        exit 2
    }
    $jobs += ,@('B2', $B2, $B2Name, $B2Init)
}

W "watchdog 启动（PID $PID，Only=$Only）"
foreach ($item in $jobs) {
    $tag = $item[0]
    $arr = $item[1]
    $runName = $item[2]
    $init = if ($item.Count -gt 3) { $item[3] } else { '' }
    $runDir = Join-Path $root ("code\runs\" + $runName)
    # 新实验首次启动时 train.py 还没来得及建目录；Out-File 不会自动创建
    # 父目录，会让 Python 在真正运行前就被管道终止。
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $console = Join-Path $runDir 'console.log'
    for ($i = 1; $i -le 15; $i++) {
        W "$tag 第 $i 次启动"
        $runArgs = @($arr)
        $last = Join-Path $root ("code\runs\" + $runName + "\weights\last.pt")
        if (Test-Path -LiteralPath $last) { $runArgs += '--resume' }
        elseif ($init) { $runArgs += @('--init-checkpoint', $init) }
        # 保留完整 stdout/stderr；否则 Python traceback 被 Out-Null 吞掉，只剩 code=1。
        & $py @runArgs 2>&1 | Out-File -LiteralPath $console -Append -Encoding UTF8
        $code = $LASTEXITCODE
        W "$tag 退出 code=$code"
        if ($code -eq 0) { break }
        Start-Sleep -Seconds 30
    }
}
W "watchdog 全部结束"
