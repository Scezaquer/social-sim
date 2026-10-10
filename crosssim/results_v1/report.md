# Cross-simulator replication results

Completed runs: 180 (Concordia: 58, OASIS: 64, SiliSocS: 58)

## Health (failed turns / survey errors)

| simulator   | family   | finetuned   |   runs |   failed_turn_rate |   turns |   survey_errors |
|:------------|:---------|:------------|-------:|-------------------:|--------:|----------------:|
| concordia   | minitaur | False       |     16 |          0.184277  |   10240 |               0 |
| concordia   | minitaur | True        |     16 |          0.0178711 |   10240 |               0 |
| concordia   | qwen     | False       |     12 |          0.204167  |    7680 |               0 |
| concordia   | qwen     | True        |     14 |          0.259263  |    8960 |               0 |
| oasis       | minitaur | False       |     16 |          0.209766  |   10240 |               0 |
| oasis       | minitaur | True        |     16 |          0.591797  |   10240 |               0 |
| oasis       | qwen     | False       |     16 |          0.385742  |   10240 |              33 |
| oasis       | qwen     | True        |     16 |          0.550391  |   10240 |               0 |
| silisocs    | minitaur | False       |     16 |          0.0290039 |   10240 |               0 |
| silisocs    | minitaur | True        |     16 |          0.0618164 |   10240 |               0 |
| silisocs    | qwen     | False       |     12 |          0.254688  |    7680 |               0 |
| silisocs    | qwen     | True        |     14 |          0.245793  |    8320 |              12 |

## Claim 1: fine-tuning on social-media data is a dominant, model-gated driver

| Simulator | Metric | ft η²p | p | ft×model η²p | model η²p | graph η²p | Δ Minitaur [CI] | Δ Qwen [CI] |
|---|---|---|---|---|---|---|---|---|
| OASIS | OSR | 0.530 | $<10^{-3}$ | 0.030 | 0.091 | 0.012 | 0.146 [0.102, 0.189] | 0.104 [0.055, 0.155] |
| OASIS | MFR | 0.780 | $<10^{-3}$ | 0.096 | 0.200 | 0.028 | 0.533 [0.474, 0.580] | 0.376 [0.245, 0.504] |
| OASIS | NASR | 0.570 | $<10^{-3}$ | 0.032 | 0.004 | 0.038 | 0.063 [0.042, 0.084] | 0.046 [0.028, 0.065] |
| Concordia | OSR | 0.641 | $<10^{-3}$ | 0.038 | 0.056 | 0.049 | 0.171 [0.133, 0.209] | 0.119 [0.064, 0.173] |
| Concordia | MFR | 0.767 | $<10^{-3}$ | 0.063 | 0.049 | 0.041 | 0.483 [0.399, 0.564] | 0.359 [0.230, 0.474] |
| Concordia | NASR | 0.596 | $<10^{-3}$ | 0.009 | 0.049 | 0.176 | 0.066 [0.046, 0.085] | 0.050 [0.026, 0.075] |
| SiliSocS | OSR | 0.560 | $<10^{-3}$ | 0.128 | 0.012 | 0.050 | 0.142 [0.108, 0.180] | 0.073 [0.028, 0.118] |
| SiliSocS | MFR | 0.719 | $<10^{-3}$ | 0.482 | 0.597 | 0.018 | 0.454 [0.378, 0.533] | 0.109 [-0.002, 0.224] |
| SiliSocS | NASR | 0.476 | $<10^{-3}$ | 0.090 | 0.055 | 0.149 | 0.054 [0.035, 0.071] | 0.029 [0.008, 0.049] |

Pooled across simulators (Type-III partial η², sum contrasts):

| Metric | term | η²p | p |
|---|---|---|---|
| OSR | C(ft, Sum) | 0.566 | $<10^{-3}$ |
| OSR | C(ft, Sum):C(family, Sum) | 0.055 | 0.01 |
| OSR | C(family, Sum):C(simulator, Sum) | 0.050 | 0.051 |
| OSR | C(simulator, Sum) | 0.021 | 0.29 |
| OSR | C(ft, Sum):C(simulator, Sum) | 0.019 | 0.32 |
| OSR | C(graph, Sum) | 0.009 | 0.6 |
| MFR | C(ft, Sum) | 0.742 | $<10^{-3}$ |
| MFR | C(family, Sum) | 0.252 | $<10^{-3}$ |
| MFR | C(ft, Sum):C(family, Sum) | 0.177 | $<10^{-3}$ |
| MFR | C(ft, Sum):C(simulator, Sum) | 0.096 | 0.0026 |
| MFR | C(family, Sum):C(simulator, Sum) | 0.070 | 0.014 |
| MFR | C(question, Sum) | 0.069 | 0.004 |
| NASR | C(ft, Sum) | 0.537 | $<10^{-3}$ |
| NASR | C(graph, Sum) | 0.099 | 0.0022 |
| NASR | C(ft, Sum):C(family, Sum) | 0.037 | 0.037 |
| NASR | C(family, Sum):C(simulator, Sum) | 0.028 | 0.19 |
| NASR | C(ft, Sum):C(simulator, Sum) | 0.025 | 0.23 |
| NASR | C(question, Sum) | 0.013 | 0.22 |

## Claim 2: much of the raw shift rate is context-perturbation noise (scrambled floor)

| simulator   | family   | finetuned   |   OSR_normal |   OSR_scrambled |   excess |   ci_lo |   ci_hi |   floor_share |
|:------------|:---------|:------------|-------------:|----------------:|---------:|--------:|--------:|--------------:|
| oasis       | minitaur | True        |        0.140 |           0.070 |    0.070 |  -0.020 |   0.158 |         0.497 |
| oasis       | minitaur | False       |        0.000 |           0.000 |    0.000 |   0.000 |   0.000 |       nan     |
| oasis       | qwen     | True        |        0.175 |           0.131 |    0.044 |  -0.055 |   0.142 |         0.750 |
| oasis       | qwen     | False       |        0.063 |           0.118 |   -0.055 |  -0.175 |   0.053 |         1.864 |
| concordia   | minitaur | True        |        0.182 |           0.100 |    0.082 |  -0.023 |   0.191 |         0.549 |
| concordia   | minitaur | False       |        0.000 |           0.000 |    0.000 |   0.000 |   0.000 |       nan     |
| concordia   | qwen     | True        |        0.144 |           0.095 |    0.048 |  -0.051 |   0.155 |         0.663 |
| concordia   | qwen     | False       |        0.002 |           0.065 |   -0.062 |  -0.128 |   0.001 |        27.667 |
| silisocs    | minitaur | True        |        0.170 |           0.105 |    0.065 |  -0.061 |   0.190 |         0.618 |
| silisocs    | minitaur | False       |        0.000 |           0.000 |    0.000 |   0.000 |   0.000 |       nan     |
| silisocs    | qwen     | True        |        0.103 |           0.098 |    0.005 |  -0.067 |   0.078 |         0.951 |
| silisocs    | qwen     | False       |        0.037 |           0.130 |   -0.092 |  -0.201 |   0.011 |         3.458 |

## Claim 3: answers are prompt-sensitive (dual-order consistency)

| simulator   | family   | finetuned   |   mean |   std |   size |
|:------------|:---------|:------------|-------:|------:|-------:|
| concordia   | minitaur | False       |  0.389 | 0.239 |     16 |
| concordia   | minitaur | True        |  0.536 | 0.126 |     16 |
| concordia   | qwen     | False       |  0.966 | 0.069 |     12 |
| concordia   | qwen     | True        |  0.792 | 0.230 |     14 |
| oasis       | minitaur | False       |  0.348 | 0.111 |     16 |
| oasis       | minitaur | True        |  0.500 | 0.060 |     16 |
| oasis       | qwen     | False       |  0.885 | 0.115 |     16 |
| oasis       | qwen     | True        |  0.745 | 0.256 |     16 |
| silisocs    | minitaur | False       |  0.269 | 0.251 |     16 |
| silisocs    | minitaur | True        |  0.519 | 0.122 |     16 |
| silisocs    | qwen     | False       |  0.909 | 0.142 |     12 |
| silisocs    | qwen     | True        |  0.788 | 0.231 |     14 |

Order-consistency partial η² per simulator:

- OASIS: C(ft, Sum)=0.001, C(family, Sum)=0.695, C(question, Sum)=0.228, C(stimulus, Sum)=0.028, C(ft, Sum):C(family, Sum)=0.242
- Concordia: C(ft, Sum)=0.002, C(family, Sum)=0.581, C(question, Sum)=0.034, C(stimulus, Sum)=0.001, C(ft, Sum):C(family, Sum)=0.176
- SiliSocS: C(ft, Sum)=0.027, C(family, Sum)=0.594, C(question, Sum)=0.009, C(stimulus, Sum)=0.015, C(ft, Sum):C(family, Sum)=0.196

## Claim 4: flip probability falls with answer confidence (margin)

- OASIS: quintile flip rates 0.233, 0.090, 0.053, 0.056, 0.025 (Spearman ρ = -0.235)
- Concordia: quintile flip rates 0.282, 0.072, 0.028, 0.013, 0.001 (Spearman ρ = -0.341)
- SiliSocS: quintile flip rates 0.253, 0.068, 0.027, 0.023, 0.015 (Spearman ρ = -0.288)

## Claim 5: topology effect on NASR is mechanical (response-shuffle null)

| simulator   | graph           |   n |   NASR_real |   NASR_null |   mean_z |   pct_sig |   graph_eta2_real |   graph_eta2_real_p |   graph_eta2_null |   graph_eta2_null_p |
|:------------|:----------------|----:|------------:|------------:|---------:|----------:|------------------:|--------------------:|------------------:|--------------------:|
| oasis       | cycle           |  16 |       0.027 |       0.029 |   -0.300 |     0.062 |             0.016 |               0.704 |             0.013 |               0.745 |
| oasis       | random          |  16 |       0.038 |       0.039 |   -0.305 |     0.000 |             0.016 |               0.704 |             0.013 |               0.745 |
| oasis       | barabasi_albert |  16 |       0.034 |       0.034 |    0.036 |     0.062 |             0.016 |               0.704 |             0.013 |               0.745 |
| concordia   | cycle           |  10 |       0.017 |       0.017 |   -0.095 |     0.000 |             0.042 |               0.437 |             0.043 |               0.423 |
| concordia   | random          |  16 |       0.035 |       0.037 |   -0.224 |     0.000 |             0.042 |               0.437 |             0.043 |               0.423 |
| concordia   | barabasi_albert |  16 |       0.038 |       0.038 |    0.188 |     0.000 |             0.042 |               0.437 |             0.043 |               0.423 |
| silisocs    | cycle           |  12 |       0.018 |       0.018 |   -0.640 |     0.083 |             0.092 |               0.146 |             0.093 |               0.145 |
| silisocs    | random          |  16 |       0.036 |       0.035 |    0.290 |     0.062 |             0.092 |               0.146 |             0.093 |               0.145 |
| silisocs    | barabasi_albert |  14 |       0.042 |       0.043 |    0.113 |     0.000 |             0.092 |               0.146 |             0.093 |               0.145 |
