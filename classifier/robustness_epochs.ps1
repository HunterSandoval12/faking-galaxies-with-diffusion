# Robustness check (2026-09-27; VALIDATION ONLY - the test set is never read).
# Question: is the real-only vs synthetic-only gap sensitive to the classifier's training
# schedule? Fixed in advance: retrain real-only and synthetic-only (seeds 0-2) with 50 epochs
# instead of 30 (same 1-epoch warmup + cosine shape, LR 1e-3, everything else identical) and
# compare the validation gap with the original recipe's (same seeds). Reported as a
# robustness statement if it holds, or as an exploratory limitation if it changes a lot.
# Log: runs_cls\robustness_epochs.log
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = '1'
$py = '.\.venv\Scripts\python.exe'
$out = 'runs_cls\robustness\epochs50'
function Log($m) { Add-Content -Encoding utf8 runs_cls\robustness_epochs.log ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $m) }
Log "=== robustness check start (50 epochs, validation only)"
foreach ($frac in '0', '1') {
    foreach ($seed in 0, 1, 2) {
        $d = "$out\A_replace${frac}_seed${seed}"
        if (Test-Path "$d\metrics.json") { Log "replace ${frac} seed ${seed}: already done"; continue }
        New-Item -ItemType Directory -Force $d | Out-Null
        & $py classifier\train_classifier.py --output-dir $d --learning-rate 1e-3 --epochs 50 --seed $seed `
            --synthetic-dirs data\synthetic\main_cfg3_steps30 --replace-fraction $frac *>> "$d\console.log"
        Log "replace ${frac} seed ${seed}: exit code $LASTEXITCODE"
    }
}
Log "=== robustness check done"
