Questions: 93 ({'single_hop': 31, 'multi_constraint': 43, 'version_conditional': 8, 'error_code': 11})
Verified: 93/93 (by claude-opus-5-5); auto-screen passed: 66/93

| metric | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| correctness | 0.31 [0.26, 0.37] | 0.65 [0.58, 0.71] | 0.62 [0.55, 0.68] | 0.58 [0.52, 0.64] |
| key_fact_recall | 0.29 [0.22, 0.36] | 0.78 [0.71, 0.85] | 0.77 [0.69, 0.84] | 0.71 [0.63, 0.78] |
| recall@8 | 0.00 [0.00, 0.00] | 0.91 [0.87, 0.95] | 0.84 [0.77, 0.90] | 0.80 [0.73, 0.87] |
| support_complete@8 | 0.00 [0.00, 0.00] | 0.83 [0.75, 0.90] | 0.74 [0.66, 0.83] | 0.74 [0.66, 0.83] |
| citation_precision | — | 0.57 [0.51, 0.64] | 0.57 [0.50, 0.65] | 0.59 [0.52, 0.67] |
| citation_recall | — | 0.77 [0.71, 0.83] | 0.71 [0.64, 0.78] | 0.70 [0.62, 0.77] |
| unsupported_rate | — | 0.10 [0.07, 0.13] | 0.11 [0.09, 0.15] | 0.10 [0.07, 0.14] |
| abstention_precision | — | — | — | — |
| abstention_recall | — | — | — | — |
| latency_p50_s | 3.62 | 6.09 | 6.18 | 5.25 |
| latency_p95_s | 4.51 | 7.90 | 8.47 | 7.11 |

Paired permutation tests vs S1 (Holm-corrected):

| metric | system | n | diff | p | p (Holm) | d_z |
|---|---|---|---|---|---|---|
| correctness | S0 | 93 | -0.333 | 0.000 | 0.000 | -1.00 |
| correctness | S2 | 93 | -0.027 | 0.427 | 0.427 | -0.10 |
| correctness | S3 | 93 | -0.065 | 0.081 | 0.163 | -0.20 |
| recall@8 | S0 | 93 | -0.915 | 0.000 | 0.000 | -4.62 |
| recall@8 | S2 | 93 | -0.079 | 0.012 | 0.013 | -0.26 |
| recall@8 | S3 | 93 | -0.116 | 0.006 | 0.013 | -0.29 |
| support_complete@8 | S0 | 93 | -0.828 | 0.000 | 0.000 | -2.18 |
| support_complete@8 | S2 | 93 | -0.086 | 0.073 | 0.146 | -0.21 |
| support_complete@8 | S3 | 93 | -0.086 | 0.167 | 0.167 | -0.16 |
| unsupported_rate | S2 | 93 | +0.015 | 0.362 | 0.724 | +0.10 |
| unsupported_rate | S3 | 93 | +0.004 | 0.828 | 0.828 | +0.02 |

By evidence span (correctness / recall@8, n):

| evidence | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| single-article | 0.28 / — (n=44) | 0.69 / 1.00 (n=44) | 0.62 / 0.98 (n=44) | 0.57 / 0.89 (n=44) |
| multi-article | 0.34 / — (n=49) | 0.60 / 0.84 (n=49) | 0.61 / 0.71 (n=49) | 0.59 / 0.71 (n=49) |

Multi-article questions only, vs S1 (Holm-corrected):

| metric | system | n | diff | p | p (Holm) | d_z |
|---|---|---|---|---|---|---|
| correctness | S0 | 49 | -0.265 | 0.000 | 0.000 | -0.78 |
| correctness | S2 | 49 | +0.010 | 1.000 | 1.000 | +0.04 |
| correctness | S3 | 49 | -0.010 | 1.000 | 1.000 | -0.03 |
| recall@8 | S0 | 49 | -0.838 | 0.000 | 0.000 | -3.35 |
| recall@8 | S2 | 49 | -0.129 | 0.026 | 0.053 | -0.33 |
| recall@8 | S3 | 49 | -0.124 | 0.076 | 0.076 | -0.26 |

Correctness by question type:

| type | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| error_code | 0.09 | 0.55 | 0.41 | 0.50 |
| multi_constraint | 0.34 | 0.58 | 0.57 | 0.56 |
| single_hop | 0.32 | 0.74 | 0.71 | 0.58 |
| version_conditional | 0.44 | 0.75 | 0.81 | 0.81 |

Recall@8 by question type:

| type | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| error_code | — | 1.00 | 1.00 | 1.00 |
| multi_constraint | — | 0.85 | 0.72 | 0.71 |
| single_hop | — | 0.98 | 0.95 | 0.82 |
| version_conditional | — | 0.88 | 0.81 | 0.94 |
