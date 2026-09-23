Questions: 30 ({'single_hop': 8, 'cross_device': 6, 'version_conditional': 5, 'error_code': 5, 'multi_constraint': 3, 'unanswerable': 3})

| metric | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| correctness | 0.40 [0.30, 0.50] | 0.73 [0.62, 0.83] | 0.68 [0.57, 0.80] | 0.63 [0.52, 0.75] |
| key_fact_recall | 0.46 [0.31, 0.61] | 0.99 [0.96, 1.00] | 0.94 [0.85, 1.00] | 0.82 [0.69, 0.93] |
| recall@8 | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.96 [0.89, 1.00] | 0.83 [0.70, 0.94] |
| support_complete@8 | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.96 [0.89, 1.00] | 0.78 [0.63, 0.93] |
| citation_precision | — | 0.87 [0.78, 0.94] | 0.80 [0.70, 0.89] | 0.78 [0.64, 0.91] |
| citation_recall | — | 0.90 [0.81, 0.98] | 0.93 [0.83, 1.00] | 0.75 [0.62, 0.87] |
| unsupported_rate | — | 0.08 [0.04, 0.13] | 0.08 [0.04, 0.13] | 0.08 [0.03, 0.12] |
| abstention_precision | — | 1.00 | 1.00 | 1.00 |
| abstention_recall | 0.00 | 0.33 | 0.33 | 0.33 |
| latency_p50_s | 3.70 | 6.05 | 6.05 | 5.25 |
| latency_p95_s | 4.53 | 8.51 | 7.97 | 7.43 |

Paired permutation tests vs S1 (Holm-corrected):

| metric | system | n | diff | p | p (Holm) | d_z |
|---|---|---|---|---|---|---|
| correctness | S0 | 30 | -0.333 | 0.000 | 0.001 | -0.88 |
| correctness | S2 | 30 | -0.050 | 0.376 | 0.376 | -0.25 |
| correctness | S3 | 30 | -0.100 | 0.125 | 0.249 | -0.36 |
| recall@8 | S0 | 27 | -1.000 | 0.000 | 0.000 | +0.00 |
| recall@8 | S2 | 27 | -0.037 | 1.000 | 1.000 | -0.19 |
| recall@8 | S3 | 27 | -0.173 | 0.035 | 0.069 | -0.50 |
| support_complete@8 | S0 | 27 | -1.000 | 0.000 | 0.000 | +0.00 |
| support_complete@8 | S2 | 27 | -0.037 | 1.000 | 1.000 | -0.19 |
| support_complete@8 | S3 | 27 | -0.222 | 0.035 | 0.069 | -0.52 |
| unsupported_rate | S2 | 27 | +0.007 | 0.765 | 1.000 | +0.07 |
| unsupported_rate | S3 | 27 | -0.001 | 0.993 | 1.000 | -0.01 |

Correctness by question type:

| type | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| cross_device | 0.42 | 0.75 | 0.58 | 0.67 |
| error_code | 0.40 | 0.60 | 0.70 | 0.60 |
| multi_constraint | 0.50 | 0.83 | 0.67 | 0.67 |
| single_hop | 0.56 | 0.88 | 0.88 | 0.75 |
| unanswerable | 0.00 | 0.33 | 0.33 | 0.33 |
| version_conditional | 0.30 | 0.80 | 0.70 | 0.60 |
