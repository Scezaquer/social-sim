# V2 Type-III ANOVA — pass `pooled` (eta^2, partial eta^2, BH-FDR, bootstrap CIs)

Runs: 720; factors: model_family, lora_finetuned, question_number, num_agents, graph_type, homophily, add_survey_to_context, num_news_agents, activity_exponent

## net_consensus_change (n=720)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(model_family, Sum):C(question_number, Sum) | 0.195 | 0.247 | [0.200, 0.311] | 33.27 | 9.5e-35 | 4.3e-33 | yes |
| C(model_family, Sum):C(lora_finetuned, Sum) | 0.038 | 0.060 | [0.026, 0.106] | 13.00 | 3.1e-08 | 7e-07 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.017 | 0.028 | [0.015, 0.073] | 2.97 | 0.0073 | 0.046 | yes |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.016 | 0.026 | [0.010, 0.065] | 4.00 | 0.0033 | 0.025 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.014 | 0.024 | [0.006, 0.058] | 7.37 | 0.00069 | 0.01 | yes |
| C(lora_finetuned, Sum):C(num_agents, Sum) | 0.013 | 0.021 | [0.005, 0.051] | 6.52 | 0.0016 | 0.014 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.011 | 0.019 | [0.006, 0.047] | 3.83 | 0.0097 | 0.049 | yes |
| C(lora_finetuned, Sum) | 0.010 | 0.017 | [0.003, 0.039] | 10.29 | 0.0014 | 0.014 | yes |
| C(num_agents, Sum) | 0.009 | 0.016 | [0.004, 0.044] | 4.84 | 0.0082 | 0.046 | yes |
| C(lora_finetuned, Sum):C(question_number, Sum) | 0.008 | 0.014 | [0.003, 0.039] | 4.24 | 0.015 | 0.067 |  |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.008 | 0.013 | [0.007, 0.048] | 1.33 | 0.24 | 0.6 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.007 | 0.011 | [0.005, 0.046] | 1.16 | 0.33 | 0.64 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.006 | 0.010 | [0.001, 0.037] | 3.00 | 0.05 | 0.21 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.006 | 0.009 | [0.003, 0.041] | 1.44 | 0.22 | 0.6 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.005 | 0.008 | [0.001, 0.033] | 2.42 | 0.089 | 0.31 |  |
| C(question_number, Sum):C(num_agents, Sum) | 0.004 | 0.007 | [0.002, 0.038] | 1.13 | 0.34 | 0.64 |  |
| C(lora_finetuned, Sum):C(graph_type, Sum) | 0.004 | 0.006 | [0.000, 0.022] | 3.66 | 0.056 | 0.21 |  |
| C(model_family, Sum) | 0.004 | 0.006 | [0.001, 0.028] | 1.21 | 0.31 | 0.64 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.003 | 0.006 | [0.002, 0.033] | 0.87 | 0.48 | 0.77 |  |
| C(question_number, Sum) | 0.003 | 0.005 | [0.000, 0.023] | 1.53 | 0.22 | 0.6 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.002 | 0.004 | [0.000, 0.024] | 1.17 | 0.31 | 0.64 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.002 | 0.004 | [0.000, 0.024] | 1.15 | 0.32 | 0.64 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.003 | [0.000, 0.021] | 0.91 | 0.4 | 0.72 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.002 | 0.003 | [0.001, 0.025] | 0.43 | 0.79 | 0.89 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.003 | [0.000, 0.016] | 1.59 | 0.21 | 0.6 |  |
| C(lora_finetuned, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.018] | 0.73 | 0.48 | 0.77 |  |
| C(lora_finetuned, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.002 | [0.000, 0.015] | 1.44 | 0.23 | 0.6 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.020] | 0.65 | 0.52 | 0.78 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.001, 0.027] | 0.31 | 0.87 | 0.93 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.000, 0.020] | 0.61 | 0.54 | 0.78 |  |
| C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.015] | 0.59 | 0.56 | 0.78 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.001, 0.020] | 0.37 | 0.77 | 0.89 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.002 | [0.000, 0.014] | 0.96 | 0.33 | 0.64 |  |
| C(num_news_agents, Sum) | 0.001 | 0.002 | [0.000, 0.015] | 0.47 | 0.63 | 0.83 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.001 | 0.001 | [0.000, 0.017] | 0.42 | 0.66 | 0.83 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.001 | [0.000, 0.017] | 0.37 | 0.69 | 0.83 |  |
| C(lora_finetuned, Sum):C(homophily, Sum) | 0.001 | 0.001 | [0.000, 0.010] | 0.52 | 0.47 | 0.77 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.000 | 0.001 | [0.000, 0.019] | 0.16 | 0.92 | 0.97 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.012] | 0.46 | 0.5 | 0.77 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.14 | 0.87 | 0.93 |  |
| C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.010] | 0.20 | 0.66 | 0.83 |  |
| C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.008] | 0.19 | 0.66 | 0.83 |  |
| C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.15 | 0.7 | 0.83 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.000 | [0.000, 0.013] | 0.06 | 0.95 | 0.97 |  |
| C(lora_finetuned, Sum):C(num_news_agents, Sum) | 0.000 | 0.000 | [0.000, 0.010] | 0.02 | 0.98 | 0.98 |  |

## mean_opinion_shift_rate (n=720)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(lora_finetuned, Sum) | 0.505 | 0.754 | [0.718, 0.780] | 1861.83 | 3.2e-187 | 1.4e-185 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.086 | 0.342 | [0.271, 0.418] | 52.68 | 3e-52 | 4.5e-51 | yes |
| C(num_agents, Sum) | 0.085 | 0.340 | [0.278, 0.397] | 156.33 | 1.7e-55 | 3.7e-54 | yes |
| C(model_family, Sum):C(lora_finetuned, Sum) | 0.032 | 0.164 | [0.107, 0.234] | 39.86 | 1.6e-23 | 1.8e-22 | yes |
| C(model_family, Sum) | 0.026 | 0.137 | [0.090, 0.196] | 32.25 | 2.4e-19 | 2.1e-18 | yes |
| C(lora_finetuned, Sum):C(question_number, Sum) | 0.024 | 0.126 | [0.078, 0.180] | 43.93 | 1.5e-18 | 1.1e-17 | yes |
| C(question_number, Sum):C(num_agents, Sum) | 0.012 | 0.070 | [0.037, 0.120] | 11.40 | 6.3e-09 | 4.1e-08 | yes |
| C(lora_finetuned, Sum):C(add_survey_to_context, Sum) | 0.008 | 0.045 | [0.017, 0.086] | 28.35 | 1.4e-07 | 8e-07 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.007 | 0.042 | [0.018, 0.080] | 13.43 | 2e-06 | 9e-06 | yes |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.007 | 0.042 | [0.019, 0.078] | 13.41 | 2e-06 | 9e-06 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.007 | 0.041 | [0.020, 0.089] | 4.30 | 0.00029 | 0.0012 | yes |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.006 | 0.036 | [0.017, 0.082] | 3.79 | 0.001 | 0.0039 | yes |
| C(num_news_agents, Sum) | 0.003 | 0.017 | [0.004, 0.045] | 5.15 | 0.0061 | 0.021 | yes |
| C(model_family, Sum):C(graph_type, Sum) | 0.002 | 0.013 | [0.003, 0.044] | 2.68 | 0.046 | 0.11 |  |
| C(question_number, Sum) | 0.002 | 0.012 | [0.002, 0.039] | 3.74 | 0.024 | 0.078 |  |
| C(lora_finetuned, Sum):C(num_agents, Sum) | 0.002 | 0.012 | [0.002, 0.036] | 3.57 | 0.029 | 0.087 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.002 | 0.010 | [0.001, 0.036] | 3.19 | 0.042 | 0.11 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.001 | 0.008 | [0.001, 0.034] | 2.45 | 0.087 | 0.19 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.001 | 0.008 | [0.002, 0.032] | 1.60 | 0.19 | 0.34 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.001 | 0.008 | [0.002, 0.037] | 1.20 | 0.31 | 0.5 |  |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.008 | [0.001, 0.032] | 1.55 | 0.2 | 0.35 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.008 | [0.002, 0.035] | 1.15 | 0.33 | 0.51 |  |
| C(lora_finetuned, Sum):C(homophily, Sum) | 0.001 | 0.007 | [0.000, 0.027] | 4.36 | 0.037 | 0.1 |  |
| C(lora_finetuned, Sum):C(graph_type, Sum) | 0.001 | 0.007 | [0.000, 0.028] | 4.01 | 0.046 | 0.11 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.001 | 0.006 | [0.004, 0.036] | 0.65 | 0.69 | 0.81 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.001 | 0.006 | [0.002, 0.033] | 0.93 | 0.45 | 0.65 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.001 | 0.006 | [0.002, 0.031] | 0.84 | 0.5 | 0.66 |  |
| C(graph_type, Sum) | 0.001 | 0.005 | [0.000, 0.026] | 3.27 | 0.071 | 0.16 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.001 | 0.004 | [0.000, 0.024] | 1.33 | 0.27 | 0.44 |  |
| C(homophily, Sum) | 0.001 | 0.004 | [0.000, 0.021] | 2.47 | 0.12 | 0.24 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.004 | [0.000, 0.020] | 2.23 | 0.14 | 0.27 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.001 | 0.003 | [0.000, 0.021] | 1.00 | 0.37 | 0.55 |  |
| C(add_survey_to_context, Sum) | 0.000 | 0.003 | [0.000, 0.016] | 1.78 | 0.18 | 0.34 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.000 | 0.002 | [0.000, 0.021] | 0.74 | 0.48 | 0.66 |  |
| C(lora_finetuned, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.000, 0.022] | 0.73 | 0.48 | 0.66 |  |
| C(activity_exponent, Sum) | 0.000 | 0.002 | [0.000, 0.019] | 0.61 | 0.54 | 0.7 |  |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.001, 0.023] | 0.27 | 0.9 | 0.96 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.002 | [0.000, 0.016] | 0.51 | 0.6 | 0.74 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.002 | [0.000, 0.018] | 0.50 | 0.61 | 0.74 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.35 | 0.7 | 0.81 |  |
| C(lora_finetuned, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.32 | 0.73 | 0.82 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.014] | 0.08 | 0.92 | 0.96 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.10 | 0.75 | 0.82 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.000 | 0.000 | [0.000, 0.013] | 0.01 | 0.99 | 0.99 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.008] | 0.00 | 0.96 | 0.98 |  |

## mean_current_majority_follow_rate (n=720)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(lora_finetuned, Sum) | 0.607 | 0.893 | [0.875, 0.912] | 5081.15 | 2e-297 | 9.1e-296 | yes |
| C(num_agents, Sum) | 0.104 | 0.589 | [0.532, 0.650] | 435.92 | 3.6e-118 | 8.2e-117 | yes |
| C(lora_finetuned, Sum):C(num_agents, Sum) | 0.100 | 0.579 | [0.519, 0.646] | 418.71 | 4.7e-115 | 7e-114 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.028 | 0.279 | [0.221, 0.344] | 39.26 | 2.2e-40 | 2.4e-39 | yes |
| C(add_survey_to_context, Sum) | 0.024 | 0.247 | [0.169, 0.344] | 199.07 | 2.6e-39 | 2.4e-38 | yes |
| C(lora_finetuned, Sum):C(add_survey_to_context, Sum) | 0.019 | 0.209 | [0.134, 0.301] | 160.83 | 7.3e-33 | 5.5e-32 | yes |
| C(model_family, Sum) | 0.014 | 0.163 | [0.083, 0.269] | 39.34 | 3e-23 | 2e-22 | yes |
| C(model_family, Sum):C(lora_finetuned, Sum) | 0.008 | 0.101 | [0.043, 0.196] | 22.79 | 5.4e-14 | 3e-13 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.005 | 0.062 | [0.032, 0.102] | 13.51 | 1.5e-08 | 7.6e-08 | yes |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.002 | 0.028 | [0.012, 0.078] | 2.91 | 0.0084 | 0.029 | yes |
| C(num_news_agents, Sum) | 0.002 | 0.027 | [0.005, 0.087] | 8.51 | 0.00023 | 0.001 | yes |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.002 | 0.023 | [0.010, 0.065] | 2.34 | 0.031 | 0.092 |  |
| C(lora_finetuned, Sum):C(num_news_agents, Sum) | 0.001 | 0.019 | [0.003, 0.070] | 5.83 | 0.0031 | 0.013 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.017 | [0.004, 0.045] | 5.34 | 0.005 | 0.019 | yes |
| C(model_family, Sum):C(homophily, Sum) | 0.001 | 0.014 | [0.003, 0.043] | 2.86 | 0.037 | 0.1 |  |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.013 | [0.005, 0.034] | 1.95 | 0.1 | 0.25 |  |
| C(activity_exponent, Sum) | 0.001 | 0.012 | [0.001, 0.055] | 3.71 | 0.025 | 0.08 |  |
| C(model_family, Sum):C(num_agents, Sum) | 0.001 | 0.010 | [0.005, 0.032] | 0.99 | 0.43 | 0.7 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.001 | 0.009 | [0.003, 0.030] | 1.39 | 0.23 | 0.5 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.008 | [0.002, 0.035] | 1.24 | 0.29 | 0.59 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.000 | 0.007 | [0.001, 0.029] | 2.07 | 0.13 | 0.3 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.000 | 0.006 | [0.002, 0.033] | 0.91 | 0.46 | 0.71 |  |
| C(homophily, Sum) | 0.000 | 0.006 | [0.000, 0.036] | 3.46 | 0.063 | 0.17 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.000 | 0.005 | [0.001, 0.031] | 1.09 | 0.35 | 0.69 |  |
| C(lora_finetuned, Sum):C(activity_exponent, Sum) | 0.000 | 0.005 | [0.000, 0.039] | 1.62 | 0.2 | 0.45 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.000 | 0.005 | [0.002, 0.030] | 0.76 | 0.55 | 0.73 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.000 | 0.003 | [0.000, 0.022] | 0.89 | 0.41 | 0.7 |  |
| C(question_number, Sum):C(num_agents, Sum) | 0.000 | 0.003 | [0.001, 0.018] | 0.43 | 0.79 | 0.87 |  |
| C(question_number, Sum) | 0.000 | 0.003 | [0.000, 0.034] | 0.85 | 0.43 | 0.7 |  |
| C(lora_finetuned, Sum):C(question_number, Sum) | 0.000 | 0.003 | [0.000, 0.035] | 0.84 | 0.43 | 0.7 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.000 | 0.002 | [0.000, 0.013] | 0.73 | 0.48 | 0.72 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.002 | [0.000, 0.014] | 0.70 | 0.5 | 0.72 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.000 | 0.002 | [0.000, 0.013] | 0.66 | 0.52 | 0.72 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.000, 0.019] | 0.63 | 0.53 | 0.72 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.001 | [0.000, 0.013] | 0.73 | 0.39 | 0.7 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.017] | 0.32 | 0.73 | 0.86 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.27 | 0.76 | 0.87 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.015] | 0.24 | 0.79 | 0.87 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.000 | [0.000, 0.014] | 0.14 | 0.87 | 0.9 |  |
| C(lora_finetuned, Sum):C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.019] | 0.25 | 0.62 | 0.79 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.000 | 0.000 | [0.000, 0.013] | 0.13 | 0.88 | 0.9 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.010] | 0.20 | 0.66 | 0.82 |  |
| C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.15 | 0.7 | 0.85 |  |
| C(lora_finetuned, Sum):C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.03 | 0.86 | 0.9 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.008] | 0.00 | 0.99 | 0.99 |  |

## mean_neighbor_alignment_shift_rate (n=720)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(lora_finetuned, Sum) | 0.548 | 0.783 | [0.749, 0.809] | 2193.19 | 7.5e-204 | 3.4e-202 | yes |
| C(num_agents, Sum) | 0.078 | 0.339 | [0.274, 0.397] | 155.61 | 2.7e-55 | 6e-54 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.074 | 0.328 | [0.264, 0.400] | 49.48 | 1.6e-49 | 2.5e-48 | yes |
| C(model_family, Sum):C(lora_finetuned, Sum) | 0.033 | 0.180 | [0.119, 0.248] | 44.52 | 5.1e-26 | 5.7e-25 | yes |
| C(lora_finetuned, Sum):C(question_number, Sum) | 0.027 | 0.149 | [0.100, 0.209] | 53.09 | 5.6e-22 | 5e-21 | yes |
| C(model_family, Sum) | 0.021 | 0.124 | [0.081, 0.178] | 28.61 | 2.6e-17 | 1.9e-16 | yes |
| C(question_number, Sum):C(num_agents, Sum) | 0.008 | 0.050 | [0.027, 0.094] | 8.03 | 2.6e-06 | 1.5e-05 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.008 | 0.050 | [0.026, 0.102] | 5.32 | 2.3e-05 | 0.0001 | yes |
| C(lora_finetuned, Sum):C(add_survey_to_context, Sum) | 0.007 | 0.042 | [0.017, 0.078] | 26.43 | 3.7e-07 | 2.4e-06 | yes |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.005 | 0.032 | [0.012, 0.083] | 3.33 | 0.0031 | 0.011 | yes |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.005 | 0.031 | [0.012, 0.068] | 9.85 | 6.2e-05 | 0.00025 | yes |
| C(lora_finetuned, Sum):C(graph_type, Sum) | 0.005 | 0.031 | [0.008, 0.063] | 19.44 | 1.2e-05 | 6.1e-05 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.023 | [0.008, 0.053] | 7.00 | 0.00099 | 0.0037 | yes |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.003 | 0.018 | [0.004, 0.050] | 5.46 | 0.0045 | 0.014 | yes |
| C(num_agents, Sum):C(graph_type, Sum) | 0.002 | 0.014 | [0.002, 0.041] | 4.36 | 0.013 | 0.037 | yes |
| C(num_news_agents, Sum) | 0.002 | 0.013 | [0.002, 0.043] | 4.10 | 0.017 | 0.045 | yes |
| C(lora_finetuned, Sum):C(num_agents, Sum) | 0.002 | 0.013 | [0.002, 0.036] | 3.92 | 0.02 | 0.051 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.002 | 0.012 | [0.002, 0.043] | 2.47 | 0.061 | 0.14 |  |
| C(lora_finetuned, Sum):C(homophily, Sum) | 0.002 | 0.011 | [0.001, 0.033] | 6.77 | 0.0095 | 0.029 | yes |
| C(model_family, Sum):C(homophily, Sum) | 0.001 | 0.010 | [0.002, 0.036] | 1.96 | 0.12 | 0.25 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.009 | [0.003, 0.037] | 1.34 | 0.26 | 0.46 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.001 | 0.008 | [0.002, 0.036] | 1.29 | 0.27 | 0.47 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.001 | 0.007 | [0.003, 0.034] | 1.10 | 0.35 | 0.57 |  |
| C(add_survey_to_context, Sum) | 0.001 | 0.007 | [0.000, 0.023] | 4.00 | 0.046 | 0.11 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.001 | 0.006 | [0.002, 0.031] | 0.87 | 0.48 | 0.67 |  |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.005 | [0.001, 0.026] | 0.93 | 0.43 | 0.62 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.001 | 0.005 | [0.000, 0.022] | 1.39 | 0.25 | 0.46 |  |
| C(question_number, Sum) | 0.001 | 0.004 | [0.000, 0.021] | 1.08 | 0.34 | 0.57 |  |
| C(graph_type, Sum) | 0.001 | 0.003 | [0.000, 0.019] | 2.12 | 0.15 | 0.3 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.000 | 0.003 | [0.000, 0.021] | 0.92 | 0.4 | 0.62 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.000 | 0.003 | [0.000, 0.021] | 0.88 | 0.42 | 0.62 |  |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.001, 0.028] | 0.37 | 0.83 | 0.91 |  |
| C(homophily, Sum) | 0.000 | 0.002 | [0.000, 0.016] | 1.50 | 0.22 | 0.43 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.002 | [0.000, 0.019] | 0.71 | 0.49 | 0.67 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.003, 0.030] | 0.24 | 0.96 | 0.96 |  |
| C(lora_finetuned, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.000, 0.019] | 0.68 | 0.51 | 0.67 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.000, 0.019] | 0.62 | 0.54 | 0.69 |  |
| C(activity_exponent, Sum) | 0.000 | 0.002 | [0.000, 0.019] | 0.59 | 0.55 | 0.69 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.43 | 0.65 | 0.79 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.34 | 0.71 | 0.82 |  |
| C(lora_finetuned, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.24 | 0.79 | 0.89 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.010] | 0.17 | 0.68 | 0.8 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.000 | 0.000 | [0.000, 0.012] | 0.05 | 0.96 | 0.96 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.04 | 0.85 | 0.91 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.010] | 0.02 | 0.88 | 0.92 |  |

## delta_assortativity (n=720)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(lora_finetuned, Sum) | 0.088 | 0.130 | [0.085, 0.175] | 90.55 | 4.2e-20 | 1.9e-18 | yes |
| C(model_family, Sum):C(homophily, Sum) | 0.073 | 0.110 | [0.073, 0.167] | 25.05 | 2.7e-15 | 6.1e-14 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.072 | 0.109 | [0.071, 0.167] | 12.35 | 3.8e-13 | 5.8e-12 | yes |
| C(question_number, Sum):C(homophily, Sum) | 0.029 | 0.047 | [0.018, 0.095] | 14.91 | 4.8e-07 | 4.3e-06 | yes |
| C(lora_finetuned, Sum):C(homophily, Sum) | 0.029 | 0.046 | [0.021, 0.078] | 29.37 | 8.6e-08 | 9.7e-07 | yes |
| C(homophily, Sum) | 0.022 | 0.037 | [0.014, 0.066] | 23.04 | 2e-06 | 1.5e-05 | yes |
| C(model_family, Sum):C(lora_finetuned, Sum) | 0.017 | 0.028 | [0.012, 0.056] | 5.87 | 0.00059 | 0.0038 | yes |
| C(model_family, Sum) | 0.011 | 0.019 | [0.007, 0.049] | 3.94 | 0.0084 | 0.042 | yes |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.007 | 0.012 | [0.001, 0.039] | 7.50 | 0.0064 | 0.036 | yes |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.007 | 0.011 | [0.007, 0.046] | 1.13 | 0.34 | 0.89 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.006 | 0.010 | [0.001, 0.034] | 2.97 | 0.052 | 0.23 |  |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.005 | 0.008 | [0.005, 0.043] | 0.85 | 0.53 | 0.93 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.003 | 0.005 | [0.001, 0.028] | 1.08 | 0.36 | 0.89 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.003 | 0.005 | [0.000, 0.025] | 1.59 | 0.21 | 0.73 |  |
| C(question_number, Sum):C(num_agents, Sum) | 0.003 | 0.005 | [0.002, 0.034] | 0.74 | 0.56 | 0.93 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.003 | 0.005 | [0.000, 0.026] | 1.46 | 0.23 | 0.73 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.003 | 0.005 | [0.002, 0.030] | 0.72 | 0.58 | 0.93 |  |
| C(num_news_agents, Sum) | 0.003 | 0.004 | [0.000, 0.021] | 1.37 | 0.25 | 0.73 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.003 | 0.004 | [0.002, 0.032] | 0.69 | 0.6 | 0.93 |  |
| C(lora_finetuned, Sum):C(num_agents, Sum) | 0.003 | 0.004 | [0.000, 0.025] | 1.36 | 0.26 | 0.73 |  |
| C(question_number, Sum) | 0.003 | 0.004 | [0.000, 0.022] | 1.36 | 0.26 | 0.73 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.002 | 0.003 | [0.000, 0.020] | 0.78 | 0.46 | 0.93 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.002 | [0.000, 0.017] | 1.50 | 0.22 | 0.73 |  |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.002 | [0.000, 0.021] | 0.70 | 0.5 | 0.93 |  |
| C(model_family, Sum):C(num_agents, Sum) | 0.001 | 0.002 | [0.003, 0.035] | 0.23 | 0.97 | 0.98 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.019] | 0.66 | 0.52 | 0.93 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.001 | 0.002 | [0.001, 0.027] | 0.32 | 0.86 | 0.98 |  |
| C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.015] | 0.60 | 0.55 | 0.93 |  |
| C(lora_finetuned, Sum):C(num_news_agents, Sum) | 0.001 | 0.002 | [0.000, 0.017] | 0.56 | 0.57 | 0.93 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.002 | [0.000, 0.021] | 0.55 | 0.57 | 0.93 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.001 | 0.002 | [0.000, 0.019] | 0.55 | 0.58 | 0.93 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.001, 0.022] | 0.25 | 0.91 | 0.98 |  |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.001 | [0.001, 0.027] | 0.21 | 0.93 | 0.98 |  |
| C(lora_finetuned, Sum):C(activity_exponent, Sum) | 0.001 | 0.001 | [0.000, 0.016] | 0.41 | 0.66 | 0.95 |  |
| C(num_agents, Sum) | 0.001 | 0.001 | [0.000, 0.020] | 0.41 | 0.66 | 0.95 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.001 | 0.001 | [0.000, 0.017] | 0.36 | 0.7 | 0.95 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.001 | 0.001 | [0.000, 0.015] | 0.26 | 0.77 | 0.98 |  |
| C(lora_finetuned, Sum):C(question_number, Sum) | 0.000 | 0.001 | [0.000, 0.015] | 0.19 | 0.83 | 0.98 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.017] | 0.11 | 0.9 | 0.98 |  |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.017] | 0.06 | 0.98 | 0.98 |  |
| C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.007] | 0.15 | 0.7 | 0.95 |  |
| C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.007] | 0.13 | 0.72 | 0.95 |  |
| C(lora_finetuned, Sum):C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.006] | 0.05 | 0.83 | 0.98 |  |
| C(lora_finetuned, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.007] | 0.02 | 0.87 | 0.98 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.00 | 0.98 | 0.98 |  |

## mean_local_agreement (n=720)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(num_agents, Sum) | 0.390 | 0.890 | [0.871, 0.904] | 2451.27 | 9.6e-292 | 4.3e-290 | yes |
| C(lora_finetuned, Sum):C(num_agents, Sum) | 0.377 | 0.886 | [0.868, 0.902] | 2369.62 | 9e-288 | 2e-286 | yes |
| C(lora_finetuned, Sum) | 0.118 | 0.709 | [0.667, 0.742] | 1479.13 | 5.6e-165 | 8.4e-164 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.038 | 0.438 | [0.369, 0.504] | 78.92 | 8.3e-73 | 9.3e-72 | yes |
| C(model_family, Sum):C(lora_finetuned, Sum) | 0.005 | 0.093 | [0.058, 0.141] | 20.74 | 8.4e-13 | 7.5e-12 | yes |
| C(lora_finetuned, Sum):C(question_number, Sum) | 0.003 | 0.061 | [0.029, 0.099] | 19.77 | 4.8e-09 | 3.6e-08 | yes |
| C(model_family, Sum) | 0.003 | 0.052 | [0.023, 0.095] | 11.03 | 4.6e-07 | 3e-06 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.045 | [0.022, 0.082] | 9.58 | 3.4e-06 | 1.9e-05 | yes |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.002 | 0.040 | [0.017, 0.088] | 6.37 | 5e-05 | 0.00025 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.031 | [0.009, 0.067] | 9.66 | 7.4e-05 | 0.00033 | yes |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.002 | 0.030 | [0.017, 0.072] | 3.17 | 0.0045 | 0.018 | yes |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.001 | 0.023 | [0.011, 0.063] | 2.37 | 0.029 | 0.081 |  |
| C(model_family, Sum):C(num_agents, Sum) | 0.001 | 0.022 | [0.010, 0.068] | 2.29 | 0.034 | 0.086 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.001 | 0.017 | [0.003, 0.048] | 5.13 | 0.0062 | 0.022 | yes |
| C(question_number, Sum) | 0.001 | 0.014 | [0.003, 0.042] | 4.35 | 0.013 | 0.043 | yes |
| C(question_number, Sum):C(num_agents, Sum) | 0.001 | 0.013 | [0.003, 0.047] | 1.96 | 0.099 | 0.19 |  |
| C(lora_finetuned, Sum):C(graph_type, Sum) | 0.001 | 0.012 | [0.002, 0.033] | 7.46 | 0.0065 | 0.022 | yes |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.001 | 0.011 | [0.001, 0.038] | 3.47 | 0.032 | 0.084 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.000 | 0.010 | [0.002, 0.037] | 2.07 | 0.1 | 0.19 |  |
| C(num_news_agents, Sum) | 0.000 | 0.010 | [0.002, 0.034] | 3.04 | 0.049 | 0.11 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.000 | 0.010 | [0.001, 0.032] | 3.00 | 0.051 | 0.11 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.000 | 0.009 | [0.003, 0.037] | 1.35 | 0.25 | 0.42 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.009 | [0.000, 0.030] | 5.37 | 0.021 | 0.063 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.000 | 0.008 | [0.001, 0.030] | 2.43 | 0.088 | 0.18 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.006 | [0.000, 0.029] | 1.98 | 0.14 | 0.25 |  |
| C(add_survey_to_context, Sum) | 0.000 | 0.005 | [0.000, 0.023] | 3.04 | 0.082 | 0.17 |  |
| C(lora_finetuned, Sum):C(activity_exponent, Sum) | 0.000 | 0.004 | [0.000, 0.021] | 1.22 | 0.3 | 0.46 |  |
| C(lora_finetuned, Sum):C(homophily, Sum) | 0.000 | 0.003 | [0.000, 0.018] | 2.06 | 0.15 | 0.26 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.000 | 0.003 | [0.000, 0.021] | 0.98 | 0.37 | 0.54 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.000 | 0.003 | [0.001, 0.025] | 0.45 | 0.77 | 0.94 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.003 | [0.000, 0.021] | 0.78 | 0.46 | 0.62 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.000 | 0.002 | [0.000, 0.018] | 0.67 | 0.51 | 0.68 |  |
| C(graph_type, Sum) | 0.000 | 0.002 | [0.000, 0.014] | 1.17 | 0.28 | 0.45 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.001, 0.027] | 0.29 | 0.88 | 0.95 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.001, 0.026] | 0.29 | 0.89 | 0.95 |  |
| C(lora_finetuned, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.002 | [0.000, 0.013] | 1.02 | 0.31 | 0.47 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.000 | 0.001 | [0.000, 0.019] | 0.25 | 0.86 | 0.95 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.014] | 0.75 | 0.39 | 0.54 |  |
| C(activity_exponent, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.28 | 0.75 | 0.94 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.28 | 0.76 | 0.94 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.017] | 0.20 | 0.82 | 0.95 |  |
| C(lora_finetuned, Sum):C(num_news_agents, Sum) | 0.000 | 0.000 | [0.000, 0.011] | 0.08 | 0.92 | 0.96 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.06 | 0.81 | 0.95 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.000 | 0.000 | [0.000, 0.014] | 0.01 | 0.99 | 0.99 |  |
| C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.00 | 0.96 | 0.98 |  |

## cross_cutting_edge_fraction (n=720)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(lora_finetuned, Sum) | 0.570 | 0.770 | [0.733, 0.801] | 2030.99 | 5.7e-196 | 2.6e-194 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.137 | 0.446 | [0.378, 0.510] | 81.47 | 1.2e-74 | 2.7e-73 | yes |
| C(model_family, Sum):C(lora_finetuned, Sum) | 0.016 | 0.087 | [0.052, 0.139] | 19.40 | 5e-12 | 7.5e-11 | yes |
| C(lora_finetuned, Sum):C(num_agents, Sum) | 0.012 | 0.065 | [0.032, 0.108] | 21.06 | 1.4e-09 | 1.6e-08 | yes |
| C(lora_finetuned, Sum):C(question_number, Sum) | 0.011 | 0.059 | [0.030, 0.100] | 19.04 | 9.5e-09 | 8.6e-08 | yes |
| C(model_family, Sum) | 0.009 | 0.052 | [0.024, 0.099] | 11.08 | 4.3e-07 | 3.2e-06 | yes |
| C(num_agents, Sum) | 0.008 | 0.044 | [0.019, 0.078] | 13.85 | 1.3e-06 | 8.4e-06 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.008 | 0.043 | [0.022, 0.081] | 9.15 | 6.3e-06 | 3.5e-05 | yes |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.007 | 0.042 | [0.022, 0.090] | 6.59 | 3.4e-05 | 0.00015 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.006 | 0.035 | [0.012, 0.076] | 10.94 | 2.1e-05 | 0.00011 | yes |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.006 | 0.033 | [0.016, 0.076] | 3.48 | 0.0022 | 0.0089 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.005 | 0.026 | [0.012, 0.076] | 2.68 | 0.014 | 0.043 | yes |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.004 | 0.025 | [0.013, 0.065] | 2.57 | 0.018 | 0.051 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.003 | 0.017 | [0.004, 0.047] | 5.27 | 0.0054 | 0.019 | yes |
| C(question_number, Sum) | 0.003 | 0.017 | [0.004, 0.046] | 5.19 | 0.0058 | 0.019 | yes |
| C(question_number, Sum):C(num_agents, Sum) | 0.003 | 0.016 | [0.005, 0.053] | 2.52 | 0.04 | 0.1 |  |
| C(lora_finetuned, Sum):C(graph_type, Sum) | 0.002 | 0.014 | [0.002, 0.036] | 8.59 | 0.0035 | 0.013 | yes |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.002 | 0.013 | [0.003, 0.046] | 1.95 | 0.1 | 0.2 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.002 | 0.010 | [0.001, 0.034] | 2.97 | 0.052 | 0.12 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.002 | 0.009 | [0.002, 0.037] | 1.92 | 0.13 | 0.23 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.009 | [0.001, 0.035] | 2.80 | 0.062 | 0.14 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.001 | 0.009 | [0.001, 0.030] | 2.62 | 0.074 | 0.16 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.001 | 0.008 | [0.001, 0.030] | 2.57 | 0.078 | 0.16 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.008 | [0.000, 0.029] | 4.93 | 0.027 | 0.071 |  |
| C(num_news_agents, Sum) | 0.001 | 0.006 | [0.001, 0.026] | 1.92 | 0.15 | 0.25 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.005 | [0.000, 0.026] | 1.44 | 0.24 | 0.37 |  |
| C(add_survey_to_context, Sum) | 0.001 | 0.004 | [0.000, 0.020] | 2.35 | 0.13 | 0.23 |  |
| C(graph_type, Sum) | 0.001 | 0.003 | [0.000, 0.017] | 2.05 | 0.15 | 0.25 |  |
| C(lora_finetuned, Sum):C(homophily, Sum) | 0.001 | 0.003 | [0.000, 0.017] | 1.90 | 0.17 | 0.27 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.003 | [0.000, 0.022] | 0.86 | 0.42 | 0.59 |  |
| C(lora_finetuned, Sum):C(activity_exponent, Sum) | 0.000 | 0.003 | [0.000, 0.019] | 0.83 | 0.44 | 0.59 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.000 | 0.003 | [0.001, 0.026] | 0.39 | 0.81 | 0.9 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.000 | 0.002 | [0.000, 0.018] | 0.72 | 0.49 | 0.64 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.001, 0.025] | 0.35 | 0.84 | 0.9 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.001, 0.025] | 0.35 | 0.84 | 0.9 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.000 | 0.002 | [0.001, 0.022] | 0.39 | 0.76 | 0.88 |  |
| C(lora_finetuned, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.001 | [0.000, 0.012] | 0.90 | 0.34 | 0.52 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.37 | 0.69 | 0.88 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.013] | 0.69 | 0.4 | 0.59 |  |
| C(activity_exponent, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.34 | 0.71 | 0.88 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.015] | 0.27 | 0.76 | 0.88 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.000 | 0.000 | [0.000, 0.015] | 0.11 | 0.9 | 0.92 |  |
| C(lora_finetuned, Sum):C(num_news_agents, Sum) | 0.000 | 0.000 | [0.000, 0.011] | 0.08 | 0.92 | 0.92 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.09 | 0.76 | 0.88 |  |
| C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.03 | 0.87 | 0.91 |  |

## order_consistency_rate (n=720)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(lora_finetuned, Sum) | 0.337 | 0.837 | [0.813, 0.859] | 3129.92 | 6e-242 | 2.7e-240 | yes |
| C(lora_finetuned, Sum):C(num_agents, Sum) | 0.210 | 0.762 | [0.729, 0.792] | 971.83 | 4.3e-190 | 9.6e-189 | yes |
| C(num_agents, Sum) | 0.157 | 0.705 | [0.662, 0.742] | 725.91 | 8e-162 | 1.2e-160 | yes |
| C(question_number, Sum) | 0.091 | 0.581 | [0.526, 0.632] | 420.75 | 2e-115 | 2.2e-114 | yes |
| C(model_family, Sum) | 0.075 | 0.535 | [0.463, 0.599] | 233.40 | 1e-100 | 9e-100 | yes |
| C(model_family, Sum):C(lora_finetuned, Sum) | 0.010 | 0.129 | [0.078, 0.194] | 30.10 | 3.7e-18 | 2.8e-17 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.009 | 0.116 | [0.075, 0.172] | 39.85 | 5.5e-17 | 3.5e-16 | yes |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.008 | 0.108 | [0.072, 0.153] | 36.84 | 7.9e-16 | 4.5e-15 | yes |
| C(add_survey_to_context, Sum) | 0.007 | 0.094 | [0.042, 0.162] | 62.89 | 1.1e-14 | 5.3e-14 | yes |
| C(lora_finetuned, Sum):C(question_number, Sum) | 0.005 | 0.066 | [0.032, 0.117] | 21.55 | 9.1e-10 | 4.1e-09 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.004 | 0.062 | [0.038, 0.109] | 6.74 | 6.3e-07 | 2.4e-06 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.004 | 0.057 | [0.029, 0.109] | 12.21 | 9.1e-08 | 3.7e-07 | yes |
| C(question_number, Sum):C(num_agents, Sum) | 0.003 | 0.051 | [0.026, 0.092] | 8.09 | 2.3e-06 | 7.5e-06 | yes |
| C(lora_finetuned, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.036 | [0.008, 0.083] | 22.93 | 2.1e-06 | 7.3e-06 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.002 | 0.035 | [0.020, 0.084] | 3.66 | 0.0014 | 0.004 | yes |
| C(lora_finetuned, Sum):C(num_news_agents, Sum) | 0.002 | 0.024 | [0.005, 0.061] | 7.44 | 0.00064 | 0.0019 | yes |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.002 | 0.023 | [0.008, 0.062] | 3.65 | 0.006 | 0.016 | yes |
| C(num_news_agents, Sum) | 0.001 | 0.016 | [0.002, 0.050] | 4.84 | 0.0083 | 0.021 | yes |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.001 | 0.014 | [0.007, 0.054] | 1.47 | 0.19 | 0.38 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.001 | 0.009 | [0.005, 0.044] | 0.95 | 0.46 | 0.65 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.000 | 0.007 | [0.002, 0.034] | 0.99 | 0.41 | 0.61 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.000 | 0.006 | [0.001, 0.026] | 1.74 | 0.18 | 0.38 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.000 | 0.006 | [0.001, 0.030] | 1.14 | 0.33 | 0.53 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.000 | 0.006 | [0.002, 0.031] | 0.85 | 0.49 | 0.67 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.005 | [0.000, 0.028] | 1.51 | 0.22 | 0.42 |  |
| C(lora_finetuned, Sum):C(activity_exponent, Sum) | 0.000 | 0.005 | [0.000, 0.032] | 1.50 | 0.22 | 0.42 |  |
| C(activity_exponent, Sum) | 0.000 | 0.005 | [0.000, 0.030] | 1.44 | 0.24 | 0.43 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.000 | 0.004 | [0.000, 0.021] | 1.17 | 0.31 | 0.52 |  |
| C(homophily, Sum) | 0.000 | 0.004 | [0.000, 0.027] | 2.17 | 0.14 | 0.33 |  |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.000 | 0.004 | [0.001, 0.022] | 0.54 | 0.71 | 0.81 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.000 | 0.003 | [0.000, 0.023] | 1.02 | 0.36 | 0.56 |  |
| C(graph_type, Sum) | 0.000 | 0.003 | [0.000, 0.024] | 1.96 | 0.16 | 0.36 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.000 | 0.003 | [0.001, 0.022] | 0.45 | 0.77 | 0.84 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.000 | 0.002 | [0.000, 0.016] | 0.65 | 0.52 | 0.67 |  |
| C(lora_finetuned, Sum):C(graph_type, Sum) | 0.000 | 0.002 | [0.000, 0.020] | 1.30 | 0.25 | 0.44 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.000 | 0.002 | [0.000, 0.022] | 0.38 | 0.77 | 0.84 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.000 | 0.002 | [0.000, 0.017] | 0.50 | 0.61 | 0.76 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.000, 0.018] | 0.46 | 0.63 | 0.77 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.012] | 0.62 | 0.43 | 0.63 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.000 | 0.001 | [0.000, 0.015] | 0.24 | 0.79 | 0.84 |  |
| C(lora_finetuned, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.017] | 0.44 | 0.51 | 0.67 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.009] | 0.20 | 0.65 | 0.77 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.000 | 0.000 | [0.000, 0.013] | 0.06 | 0.94 | 0.96 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.000 | [0.000, 0.014] | 0.04 | 0.96 | 0.96 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.010] | 0.01 | 0.94 | 0.96 |  |

