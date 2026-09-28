### Dataset composition after cleaning (Galaxy10 DECaLS)

| Class | Original | Exact duplicates removed | Overlapping cutouts removed | Final | Working pool (train) | Validation | Test |
|---|---|---|---|---|---|---|---|
| 0 Disturbed | 1081 | 30 | 0 | 1051 | 736 | 158 | 157 |
| 1 Merging | 1853 | 14 | 48 | 1791 | 1287 | 253 | 251 |
| 2 Round Smooth | 2645 | 10 | 4 | 2631 | 1844 | 391 | 396 |
| 3 In-between Round | 2027 | 6 | 4 | 2017 | 1415 | 301 | 301 |
| 4 Cigar-Shaped Smooth | 334 | 1 | 0 | 333 | 233 | 50 | 50 |
| 5 Barred Spiral | 2043 | 5 | 2 | 2036 | 1427 | 304 | 305 |
| 6 Unbarred Tight Spiral | 1829 | 10 | 1 | 1818 | 1273 | 272 | 273 |
| 7 Unbarred Loose Spiral | 2628 | 33 | 6 | 2589 | 1816 | 387 | 386 |
| 8 Edge-on w/o Bulge | 1423 | 4 | 2 | 1417 | 993 | 213 | 211 |
| 9 Edge-on w/ Bulge | 1873 | 8 | 2 | 1863 | 1306 | 278 | 279 |
| Total | 17736 | 121 | 69 | 17546 | 12330 | 2607 | 2609 |

Exact duplicates: pixel-identical images with conflicting labels, removed before the stratified 70/15/15 split. Overlapping cutouts: validation/test images covering >= 25% of the same sky as an image in another split, removed after splitting (working pool unchanged).
