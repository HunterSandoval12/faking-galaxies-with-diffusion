# Main synthetic set (2026-09-26): 12,330 images (working-pool per-class
# counts) with the final settings (lora_r8/checkpoint-10000, guidance 3, 30 steps), then
# integrity / realism / memorization checks. Resumable (re-run skips complete shards).
# Log: runs\main_generation.log
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:HF_HUB_DISABLE_PROGRESS_BARS = '1'; $env:PYTHONUNBUFFERED = '1'
$py = '.\.venv\Scripts\python.exe'
$dir = 'data\synthetic\main_cfg3_steps30'
New-Item -ItemType Directory -Force $dir | Out-Null
function Log($m) { Add-Content -Encoding utf8 runs\main_generation.log ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $m) }
Log "=== main generation start -> $dir"
& $py diffusion\generate_synthetic.py --output-dir $dir *>> "$dir\generation.log"
Log "generation exit code $LASTEXITCODE"
& $py diffusion\check_synthetic.py $dir *>> "$dir\check.log"
Log "checks exit code $LASTEXITCODE"
Log "=== main generation done"
