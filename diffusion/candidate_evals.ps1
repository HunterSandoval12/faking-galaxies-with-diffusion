# Candidate evidence queue (2026-09-26). Scores the generator candidates exactly like
# the sweep (100 images/class, cleaned validation set, images+features saved):
#   1. guidance re-check: 1.5 and 3 on checkpoint 4000 of both baseline seeds
#      (guidance was tuned on checkpoint 10000; guidance 2 is already scored)
#   2. extras (evidence only): generation-seed replicate (seed 1) and 100 steps at
#      guidance 2 for both checkpoint-4000 candidates (42@10000 already has both).
# Log: runs\candidate_evals.log; per-run output runs\<run>\fid\candidate_evals.log
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:HF_HUB_DISABLE_PROGRESS_BARS = '1'; $env:PYTHONUNBUFFERED = '1'
$py = '.\.venv\Scripts\python.exe'
function Log($m) { Add-Content -Encoding utf8 runs\candidate_evals.log ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $m) }
function Score($run, $guidance, $steps, $seed) {
    Log "$run checkpoint-4000: guidance $guidance, steps $steps, seed $seed"
    & $py diffusion\eval_fid.py "runs\$run\checkpoint-4000" --guidance-scale @($guidance -split ' ') `
        --steps $steps --num-per-class 100 --seed $seed --output-dir "runs\$run\fid" *>> "runs\$run\fid\candidate_evals.log"
    Log "$run checkpoint-4000: guidance $guidance, steps $steps, seed $seed -> exit code $LASTEXITCODE"
}
Log "=== candidate evals start"
foreach ($run in 'lora_r8', 'lora_r8_seed43') { Score $run '1.5 3' 30 0 }   # 1. guidance re-check
foreach ($run in 'lora_r8', 'lora_r8_seed43') { Score $run '2' 30 1 }       # 2a. generation-seed replicate
foreach ($run in 'lora_r8', 'lora_r8_seed43') { Score $run '2' 100 0 }      # 2b. 100 steps
Log "=== candidate evals done"
