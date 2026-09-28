# Steps pilot (2026-09-26). How many sampling steps make the most USEFUL
# synthetic training data at the chosen settings (lora_r8 checkpoint-10000, guidance 3)?
#   A. generate + score (KID/FID, images saved) 100 images/class at 50 / 75 / 100 steps
#      (30 steps already exists from the guidance pilot), generation seed 0
#   B. class fidelity of the new sets with the two real-only classifiers
#   C. ResNet-18 trained on ONLY each generated set (LR 1e-3, 30 epochs, 2 seeds),
#      accuracy on REAL validation images
# Decision rule (fixed in advance): the FEWEST steps whose mean real-validation accuracy
# is within 2 points of the best. Log: runs_cls\steps_pilot.log. No test data is read.
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:HF_HUB_DISABLE_PROGRESS_BARS = '1'; $env:PYTHONUNBUFFERED = '1'
$py = '.\.venv\Scripts\python.exe'
$gen = 'runs\lora_r8\fid\generated'
function Log($m) { Add-Content -Encoding utf8 runs_cls\steps_pilot.log ("[" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "] " + $m) }
Log "=== steps pilot start"

# A. generate + score the missing step counts
$todo = @('50', '75', '100') | Where-Object { -not (Test-Path "$gen\checkpoint-10000_cfg3_steps${_}_seed0_n100.npz") }
if ($todo) {
    Log ("A. generating guidance 3 at steps " + ($todo -join ', '))
    & $py diffusion\eval_fid.py runs\lora_r8\checkpoint-10000 --guidance-scale 3 --steps @todo `
        --num-per-class 100 --seed 0 --output-dir runs\lora_r8\fid *>> runs\lora_r8\fid\steps_pilot.log
    Log "A. generation exit code $LASTEXITCODE"
} else { Log "A. all generated sets already exist" }

# B. class fidelity of the guidance-3 sets (30 steps included for comparison)
Log "B. class fidelity"
& $py classifier\class_fidelity.py --classifiers runs_cls\real_lr1e-3_seed0\best.pt runs_cls\real_lr1e-3_seed1\best.pt `
    --generated "$gen\checkpoint-10000_cfg3_steps*_seed0_n100.npz" --output runs_cls\class_fidelity_steps_pilot.jsonl *> runs_cls\steps_pilot_fidelity.txt
Log "B. class fidelity exit code $LASTEXITCODE"

# C. synthetic-only classifiers (30 steps = runs_cls\pilot_guidance\synthetic_cfg3_seed*)
foreach ($steps in '50', '75', '100') {
    foreach ($seed in 0, 1) {
        $d = "runs_cls\pilot_steps\synthetic_cfg3_steps${steps}_seed${seed}"
        if (Test-Path "$d\metrics.json") { Log "C. steps ${steps} seed ${seed}: already done"; continue }
        New-Item -ItemType Directory -Force $d | Out-Null
        & $py classifier\train_classifier.py --output-dir $d --learning-rate 1e-3 --seed $seed `
            --train-npz "$gen\checkpoint-10000_cfg3_steps${steps}_seed0_n100.npz" *>> "$d\console.log"
        Log "C. steps ${steps} seed ${seed}: exit code $LASTEXITCODE"
    }
}
Log "=== steps pilot done"
