### Focus-class analysis: class F1 on the test set when adding synthetic images of that class

| Class (real count) | Synthetic added | Class F1 [95% CI] | Delta F1 vs no synthetic [95% CI] | Overall acc. % | Seeds |
|---|---|---|---|---|---|
| Round Smooth (1,844 real) | 0x | 0.927 [0.910, 0.942] | - | 86.6 | 8 |
| Round Smooth (1,844 real) | 0.5x | 0.929 [0.912, 0.945] | +0.003 [-0.003, +0.009] | 86.5 | 3 |
| Round Smooth (1,844 real) | 1x | 0.933 [0.916, 0.948] | +0.006 [+0.001, +0.013] | 86.6 | 3 |
| Round Smooth (1,844 real) | 2x | 0.931 [0.914, 0.946] | +0.004 [-0.002, +0.011] | 86.5 | 3 |
| Round Smooth (1,844 real) | 4x | 0.928 [0.910, 0.943] | +0.001 [-0.005, +0.007] | 86.7 | 3 |
| Cigar-Shaped Smooth (233 real) | 0x | 0.769 [0.677, 0.842] | - | 86.6 | 8 |
| Cigar-Shaped Smooth (233 real) | 0.5x | 0.770 [0.684, 0.843] | +0.001 [-0.027, +0.028] | 86.8 | 8 |
| Cigar-Shaped Smooth (233 real) | 1x | 0.791 [0.707, 0.859] | +0.021 [-0.006, +0.050] | 86.3 | 8 |
| Cigar-Shaped Smooth (233 real) | 2x | 0.779 [0.695, 0.852] | +0.010 [-0.024, +0.044] | 86.3 | 8 |
| Cigar-Shaped Smooth (233 real) | 4x | 0.789 [0.705, 0.858] | +0.020 [-0.009, +0.048] | 86.6 | 8 |
| Round Smooth cut to 233 (control) | 0x | 0.882 [0.862, 0.900] | - | 85.0 | 8 |
| Round Smooth cut to 233 (control) | 0.5x | 0.874 [0.853, 0.893] | -0.008 [-0.016, -0.001] | 84.9 | 8 |
| Round Smooth cut to 233 (control) | 1x | 0.882 [0.861, 0.900] | -0.001 [-0.008, +0.006] | 84.6 | 8 |
| Round Smooth cut to 233 (control) | 2x | 0.876 [0.856, 0.895] | -0.006 [-0.013, +0.001] | 84.9 | 8 |
| Round Smooth cut to 233 (control) | 4x | 0.895 [0.875, 0.912] | +0.012 [+0.005, +0.019] | 85.6 | 8 |

Pre-registered rule: ~20 differences are tested, so isolated CIs excluding 0 are expected by chance; no consistent pattern in any class.
