# Checkpoint scan around 4000 steps (2026-09-25): checkpoints 2000/3000/5000 for both
# baseline training seeds, scored exactly like the overnight sweep (guidance 2, 30 steps,
# 100 images/class, generation seed 0, cleaned validation set). Log: runs\ckpt_scan.log
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:HF_HUB_DISABLE_PROGRESS_BARS = '1'; $env:PYTHONUNBUFFERED = '1'
$py = '.\.venv\Scripts\python.exe'
function Log($m) { Add-Content -Encoding utf8 runs\ckpt_scan.log ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $m) }
Log "=== checkpoint scan start"
foreach ($run in 'lora_r8', 'lora_r8_seed43') {
    Log "$run`: scoring checkpoints 2000, 3000, 5000"
    & $py diffusion\eval_fid.py "runs\$run\checkpoint-2000" "runs\$run\checkpoint-3000" "runs\$run\checkpoint-5000" `
        --guidance-scale 2 --steps 30 --num-per-class 100 --seed 0 --output-dir "runs\$run\fid" *>> "runs\$run\fid\ckpt_scan.log"
    Log "$run`: finished, exit code $LASTEXITCODE"
}
Log "=== checkpoint scan done"
