# Guidance pilot (2026-09-26). Which guidance scale makes the most USEFUL
# synthetic training data? Train ResNet-18 classifiers on ONLY the saved generated images
# of the chosen generator (lora_r8 checkpoint-10000, 30 steps, generation seed 0,
# 100 images/class) at guidance 1.5 / 2 / 3 / 4 / 6, and measure accuracy on REAL
# validation images. Control: 100 real working-pool images/class (same size).
# Fixed in advance: LR 1e-3 (best real-only LR), 30 epochs, 2 classifier seeds each;
# deciding metric = real-validation accuracy (mean of the 2 seeds), macro-F1 secondary.
# Log: runs_cls\guidance_pilot.log. No test data is read.
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = '1'
$py = '.\.venv\Scripts\python.exe'
$out = 'runs_cls\pilot_guidance'
New-Item -ItemType Directory -Force $out | Out-Null
function Log($m) { Add-Content -Encoding utf8 runs_cls\guidance_pilot.log ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $m) }
function Train($name, $extra) {
    foreach ($seed in 0, 1) {
        $d = "$out\${name}_seed$seed"
        if (Test-Path "$d\metrics.json") { Log "$name seed ${seed}: already done"; continue }
        New-Item -ItemType Directory -Force $d | Out-Null
        & $py classifier\train_classifier.py --output-dir $d --learning-rate 1e-3 --seed $seed @extra *>> "$d\console.log"
        Log "$name seed ${seed}: exit code $LASTEXITCODE"
    }
}
Log "=== guidance pilot start"
foreach ($g in '1.5', '2', '3', '4', '6') {
    Train "synthetic_cfg$g" @('--train-npz', "runs\lora_r8\fid\generated\checkpoint-10000_cfg${g}_steps30_seed0_n100.npz")
}
Train 'real_100perclass' @('--per-class', '100')
Log "=== guidance pilot done"
