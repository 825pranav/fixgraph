Questions: 191 ({'single_hop': 40, 'multi_constraint': 80, 'version_conditional': 30, 'cross_device': 30, 'error_code': 11})
Verified: 93/191 (by claude-opus-5-5); auto-screen passed: 112/191

| metric | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| correctness | 0.33 [0.29, 0.37] | 0.57 [0.53, 0.61] | 0.57 [0.53, 0.61] | 0.53 [0.49, 0.57] |
| key_fact_recall | 0.35 [0.30, 0.40] | 0.71 [0.66, 0.75] | 0.70 [0.64, 0.75] | 0.66 [0.60, 0.71] |
| recall@8 | 0.00 [0.00, 0.00] | 0.81 [0.77, 0.84] | 0.77 [0.73, 0.81] | 0.71 [0.65, 0.76] |
| support_complete@8 | 0.00 [0.00, 0.00] | 0.62 [0.55, 0.69] | 0.59 [0.52, 0.65] | 0.57 [0.49, 0.63] |
| citation_precision | — | 0.48 [0.43, 0.52] | 0.50 [0.45, 0.54] | 0.55 [0.50, 0.60] |
| citation_recall | — | 0.60 [0.55, 0.65] | 0.61 [0.56, 0.66] | 0.59 [0.54, 0.65] |
| unsupported_rate | — | 0.13 [0.11, 0.15] | 0.14 [0.11, 0.16] | 0.15 [0.12, 0.17] |
| abstention_precision | — | — | 0.00 | 0.00 |
| abstention_recall | — | — | — | — |
| latency_p50_s | 3.72 | 6.14 | 6.31 | 5.28 |
| latency_p95_s | 4.59 | 7.85 | 8.42 | 6.97 |

Paired permutation tests vs S1 (Holm-corrected):

| metric | system | n | diff | p | p (Holm) | d_z |
|---|---|---|---|---|---|---|
| correctness | S0 | 191 | -0.243 | 0.000 | 0.000 | -0.69 |
| correctness | S2 | 191 | -0.008 | 0.792 | 0.792 | -0.03 |
| correctness | S3 | 191 | -0.039 | 0.086 | 0.173 | -0.13 |
| recall@8 | S0 | 191 | -0.805 | 0.000 | 0.000 | -3.06 |
| recall@8 | S2 | 191 | -0.038 | 0.115 | 0.115 | -0.12 |
| recall@8 | S3 | 191 | -0.100 | 0.001 | 0.002 | -0.25 |
| support_complete@8 | S0 | 191 | -0.623 | 0.000 | 0.000 | -1.28 |
| support_complete@8 | S2 | 191 | -0.037 | 0.377 | 0.380 | -0.08 |
| support_complete@8 | S3 | 191 | -0.058 | 0.190 | 0.380 | -0.11 |
| unsupported_rate | S2 | 190 | +0.009 | 0.543 | 0.543 | +0.05 |
| unsupported_rate | S3 | 190 | +0.018 | 0.271 | 0.542 | +0.08 |

By evidence span (correctness / recall@8, n):

| evidence | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| single-article | 0.32 / — (n=54) | 0.67 / 1.00 (n=54) | 0.60 / 0.98 (n=54) | 0.55 / 0.88 (n=54) |
| multi-article | 0.33 / — (n=137) | 0.54 / 0.73 (n=137) | 0.55 / 0.68 (n=137) | 0.53 / 0.64 (n=137) |

Multi-article questions only, vs S1 (Holm-corrected):

| metric | system | n | diff | p | p (Holm) | d_z |
|---|---|---|---|---|---|---|
| correctness | S0 | 137 | -0.204 | 0.000 | 0.000 | -0.58 |
| correctness | S2 | 137 | +0.015 | 0.651 | 1.000 | +0.05 |
| correctness | S3 | 137 | -0.007 | 0.882 | 1.000 | -0.03 |
| recall@8 | S0 | 137 | -0.729 | 0.000 | 0.000 | -2.64 |
| recall@8 | S2 | 137 | -0.045 | 0.168 | 0.168 | -0.12 |
| recall@8 | S3 | 137 | -0.091 | 0.013 | 0.025 | -0.22 |

Correctness by question type:

| type | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| cross_device | 0.23 | 0.47 | 0.52 | 0.42 |
| error_code | 0.09 | 0.55 | 0.41 | 0.50 |
| multi_constraint | 0.39 | 0.57 | 0.56 | 0.56 |
| single_hop | 0.35 | 0.70 | 0.66 | 0.55 |
| version_conditional | 0.33 | 0.53 | 0.55 | 0.57 |

Recall@8 by question type:

| type | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| cross_device | — | 0.56 | 0.73 | 0.45 |
| error_code | — | 1.00 | 1.00 | 1.00 |
| multi_constraint | — | 0.85 | 0.71 | 0.72 |
| single_hop | — | 0.99 | 0.96 | 0.81 |
| version_conditional | — | 0.62 | 0.63 | 0.67 |
