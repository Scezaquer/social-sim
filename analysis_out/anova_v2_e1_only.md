# V2 Type-III ANOVA — pass `e1_only` (eta^2, partial eta^2, BH-FDR, bootstrap CIs)

Runs: 576; factors: model_family, proportions_option, question_number, num_agents, graph_type, homophily, add_survey_to_context, num_news_agents, activity_exponent

## net_consensus_change (n=576)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(model_family, Sum):C(question_number, Sum) | 0.212 | 0.369 | [0.300, 0.456] | 42.16 | 1.9e-40 | 8.6e-39 | yes |
| C(model_family, Sum) | 0.093 | 0.205 | [0.134, 0.288] | 37.19 | 2.1e-21 | 4.8e-20 | yes |
| C(model_family, Sum):C(proportions_option, Sum) | 0.061 | 0.144 | [0.099, 0.237] | 8.06 | 4.6e-11 | 5.1e-10 | yes |
| C(proportions_option, Sum):C(question_number, Sum) | 0.059 | 0.140 | [0.093, 0.228] | 11.72 | 3.5e-12 | 5.2e-11 | yes |
| C(proportions_option, Sum) | 0.025 | 0.064 | [0.028, 0.130] | 9.83 | 2.8e-06 | 2.5e-05 | yes |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.024 | 0.062 | [0.032, 0.131] | 7.18 | 1.3e-05 | 9.9e-05 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.021 | 0.056 | [0.026, 0.119] | 8.56 | 1.6e-05 | 0.0001 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.017 | 0.046 | [0.013, 0.111] | 10.42 | 3.8e-05 | 0.00021 | yes |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.011 | 0.028 | [0.010, 0.081] | 3.14 | 0.014 | 0.059 |  |
| C(model_family, Sum):C(num_agents, Sum) | 0.010 | 0.027 | [0.015, 0.085] | 1.99 | 0.065 | 0.21 |  |
| C(num_agents, Sum) | 0.009 | 0.025 | [0.005, 0.068] | 5.54 | 0.0042 | 0.021 | yes |
| C(proportions_option, Sum):C(add_survey_to_context, Sum) | 0.009 | 0.025 | [0.005, 0.080] | 3.62 | 0.013 | 0.059 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.009 | 0.023 | [0.011, 0.081] | 1.73 | 0.11 | 0.28 |  |
| C(activity_exponent, Sum) | 0.007 | 0.019 | [0.003, 0.061] | 4.14 | 0.017 | 0.062 |  |
| C(proportions_option, Sum):C(num_agents, Sum) | 0.006 | 0.017 | [0.008, 0.069] | 1.21 | 0.3 | 0.5 |  |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.005 | 0.014 | [0.009, 0.069] | 1.05 | 0.39 | 0.61 |  |
| C(proportions_option, Sum):C(num_news_agents, Sum) | 0.005 | 0.014 | [0.008, 0.068] | 1.01 | 0.42 | 0.61 |  |
| C(proportions_option, Sum):C(activity_exponent, Sum) | 0.005 | 0.013 | [0.007, 0.058] | 0.95 | 0.46 | 0.62 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.005 | 0.013 | [0.001, 0.049] | 2.77 | 0.064 | 0.21 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.004 | 0.011 | [0.001, 0.045] | 2.35 | 0.097 | 0.26 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.003 | 0.009 | [0.003, 0.051] | 0.97 | 0.42 | 0.61 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.003 | 0.009 | [0.001, 0.041] | 1.92 | 0.15 | 0.32 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.003 | 0.009 | [0.003, 0.049] | 0.95 | 0.44 | 0.61 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.003 | 0.008 | [0.001, 0.036] | 1.76 | 0.17 | 0.35 |  |
| C(question_number, Sum) | 0.003 | 0.008 | [0.001, 0.045] | 1.71 | 0.18 | 0.35 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.003 | 0.007 | [0.000, 0.034] | 3.12 | 0.078 | 0.23 |  |
| C(graph_type, Sum) | 0.003 | 0.007 | [0.000, 0.032] | 3.02 | 0.083 | 0.23 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.002 | 0.007 | [0.000, 0.037] | 1.45 | 0.23 | 0.44 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.002 | 0.007 | [0.000, 0.040] | 1.42 | 0.24 | 0.44 |  |
| C(proportions_option, Sum):C(graph_type, Sum) | 0.002 | 0.006 | [0.002, 0.038] | 0.94 | 0.42 | 0.61 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.006 | [0.001, 0.043] | 1.20 | 0.3 | 0.5 |  |
| C(add_survey_to_context, Sum) | 0.002 | 0.005 | [0.000, 0.030] | 2.33 | 0.13 | 0.3 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.005 | [0.000, 0.032] | 2.27 | 0.13 | 0.3 |  |
| C(question_number, Sum):C(num_agents, Sum) | 0.001 | 0.004 | [0.002, 0.040] | 0.43 | 0.79 | 0.84 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.001 | 0.004 | [0.001, 0.035] | 0.57 | 0.63 | 0.8 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.001 | 0.003 | [0.002, 0.039] | 0.34 | 0.85 | 0.86 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.001 | 0.003 | [0.000, 0.030] | 0.56 | 0.57 | 0.76 |  |
| C(proportions_option, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.001, 0.033] | 0.35 | 0.79 | 0.84 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.001, 0.031] | 0.34 | 0.8 | 0.84 |  |
| C(num_news_agents, Sum) | 0.001 | 0.002 | [0.000, 0.025] | 0.44 | 0.65 | 0.8 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.025] | 0.41 | 0.66 | 0.8 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.000 | 0.001 | [0.000, 0.025] | 0.26 | 0.77 | 0.84 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.025] | 0.25 | 0.78 | 0.84 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.07 | 0.79 | 0.84 |  |
| C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.03 | 0.86 | 0.86 |  |

## mean_opinion_shift_rate (n=576)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(question_number, Sum) | 0.153 | 0.415 | [0.335, 0.486] | 153.50 | 4.3e-51 | 1.9e-49 | yes |
| C(model_family, Sum):C(proportions_option, Sum) | 0.100 | 0.317 | [0.260, 0.412] | 22.23 | 4.8e-31 | 5.4e-30 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.096 | 0.309 | [0.227, 0.403] | 32.15 | 5.4e-32 | 8.1e-31 | yes |
| C(model_family, Sum) | 0.092 | 0.299 | [0.231, 0.382] | 61.55 | 3.8e-33 | 8.5e-32 | yes |
| C(num_agents, Sum) | 0.078 | 0.266 | [0.185, 0.348] | 78.42 | 8.8e-30 | 7.9e-29 | yes |
| C(add_survey_to_context, Sum) | 0.048 | 0.182 | [0.111, 0.253] | 96.37 | 1.1e-20 | 8.6e-20 | yes |
| C(question_number, Sum):C(num_agents, Sum) | 0.034 | 0.137 | [0.080, 0.219] | 17.10 | 5e-13 | 3.2e-12 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.026 | 0.109 | [0.065, 0.189] | 8.77 | 5e-09 | 2.8e-08 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.018 | 0.076 | [0.031, 0.139] | 17.77 | 3.8e-08 | 1.9e-07 | yes |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.014 | 0.061 | [0.025, 0.124] | 14.13 | 1.1e-06 | 5.1e-06 | yes |
| C(num_news_agents, Sum) | 0.013 | 0.056 | [0.017, 0.108] | 12.93 | 3.5e-06 | 1.4e-05 | yes |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.011 | 0.049 | [0.019, 0.109] | 5.51 | 0.00025 | 0.00093 | yes |
| C(proportions_option, Sum):C(question_number, Sum) | 0.010 | 0.043 | [0.018, 0.111] | 3.25 | 0.0039 | 0.011 | yes |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.009 | 0.039 | [0.016, 0.094] | 4.43 | 0.0016 | 0.0055 | yes |
| C(proportions_option, Sum) | 0.008 | 0.034 | [0.011, 0.088] | 5.12 | 0.0017 | 0.0055 | yes |
| C(proportions_option, Sum):C(homophily, Sum) | 0.007 | 0.031 | [0.009, 0.080] | 4.66 | 0.0032 | 0.0097 | yes |
| C(proportions_option, Sum):C(num_agents, Sum) | 0.006 | 0.028 | [0.013, 0.089] | 2.07 | 0.056 | 0.11 |  |
| C(proportions_option, Sum):C(activity_exponent, Sum) | 0.005 | 0.023 | [0.012, 0.077] | 1.73 | 0.11 | 0.19 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.005 | 0.022 | [0.007, 0.070] | 2.46 | 0.045 | 0.096 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.004 | 0.020 | [0.003, 0.060] | 4.32 | 0.014 | 0.037 | yes |
| C(proportions_option, Sum):C(add_survey_to_context, Sum) | 0.004 | 0.019 | [0.004, 0.060] | 2.85 | 0.037 | 0.083 |  |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.004 | 0.019 | [0.010, 0.074] | 1.41 | 0.21 | 0.31 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.004 | 0.018 | [0.006, 0.068] | 1.98 | 0.097 | 0.17 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.004 | 0.018 | [0.009, 0.069] | 1.31 | 0.25 | 0.35 |  |
| C(proportions_option, Sum):C(num_news_agents, Sum) | 0.004 | 0.017 | [0.008, 0.065] | 1.22 | 0.3 | 0.4 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.003 | 0.015 | [0.002, 0.055] | 3.39 | 0.035 | 0.082 |  |
| C(proportions_option, Sum):C(graph_type, Sum) | 0.003 | 0.014 | [0.002, 0.058] | 2.02 | 0.11 | 0.19 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.014 | [0.000, 0.047] | 5.99 | 0.015 | 0.037 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.013 | [0.002, 0.057] | 1.85 | 0.14 | 0.21 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.003 | 0.012 | [0.001, 0.044] | 2.54 | 0.08 | 0.15 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.002 | 0.011 | [0.003, 0.054] | 1.16 | 0.33 | 0.43 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.002 | 0.008 | [0.000, 0.034] | 3.31 | 0.069 | 0.14 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.002 | 0.007 | [0.001, 0.042] | 1.04 | 0.37 | 0.48 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.001 | 0.007 | [0.000, 0.037] | 1.44 | 0.24 | 0.35 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.006 | [0.000, 0.034] | 2.46 | 0.12 | 0.19 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.001 | 0.003 | [0.000, 0.029] | 0.76 | 0.47 | 0.59 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.001 | 0.003 | [0.001, 0.038] | 0.45 | 0.72 | 0.77 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.001 | 0.003 | [0.000, 0.031] | 0.58 | 0.56 | 0.68 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.029] | 0.54 | 0.58 | 0.68 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.002 | [0.000, 0.025] | 0.45 | 0.64 | 0.72 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.024] | 0.28 | 0.75 | 0.79 |  |
| C(graph_type, Sum) | 0.000 | 0.001 | [0.000, 0.017] | 0.29 | 0.59 | 0.68 |  |
| C(homophily, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.13 | 0.72 | 0.77 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.021] | 0.05 | 0.95 | 0.98 |  |
| C(activity_exponent, Sum) | 0.000 | 0.000 | [0.000, 0.021] | 0.02 | 0.98 | 0.98 |  |

## mean_current_majority_follow_rate (n=576)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(model_family, Sum):C(question_number, Sum) | 0.309 | 0.451 | [0.383, 0.532] | 59.13 | 2.8e-53 | 1.3e-51 | yes |
| C(model_family, Sum) | 0.070 | 0.157 | [0.096, 0.244] | 26.75 | 6.9e-16 | 1.6e-14 | yes |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.028 | 0.068 | [0.036, 0.144] | 5.26 | 3e-05 | 0.00045 | yes |
| C(model_family, Sum):C(proportions_option, Sum) | 0.026 | 0.065 | [0.041, 0.144] | 3.34 | 0.00058 | 0.0052 | yes |
| C(question_number, Sum) | 0.017 | 0.043 | [0.013, 0.098] | 9.77 | 7.1e-05 | 0.0008 | yes |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.016 | 0.041 | [0.016, 0.100] | 4.64 | 0.0011 | 0.0076 | yes |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.012 | 0.031 | [0.006, 0.080] | 6.84 | 0.0012 | 0.0076 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.011 | 0.028 | [0.008, 0.075] | 4.22 | 0.0059 | 0.029 | yes |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.011 | 0.028 | [0.010, 0.075] | 6.22 | 0.0022 | 0.012 | yes |
| C(proportions_option, Sum):C(question_number, Sum) | 0.011 | 0.027 | [0.011, 0.088] | 2.03 | 0.061 | 0.18 |  |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.010 | 0.026 | [0.012, 0.088] | 1.92 | 0.077 | 0.2 |  |
| C(model_family, Sum):C(num_agents, Sum) | 0.008 | 0.022 | [0.010, 0.078] | 1.62 | 0.14 | 0.3 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.008 | 0.022 | [0.006, 0.078] | 2.42 | 0.048 | 0.15 |  |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.008 | 0.021 | [0.004, 0.061] | 4.59 | 0.011 | 0.048 | yes |
| C(num_news_agents, Sum) | 0.007 | 0.018 | [0.002, 0.056] | 3.95 | 0.02 | 0.075 |  |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.006 | 0.017 | [0.005, 0.068] | 1.86 | 0.12 | 0.26 |  |
| C(activity_exponent, Sum) | 0.006 | 0.017 | [0.002, 0.057] | 3.65 | 0.027 | 0.093 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.006 | 0.015 | [0.003, 0.057] | 2.14 | 0.094 | 0.24 |  |
| C(add_survey_to_context, Sum) | 0.005 | 0.014 | [0.001, 0.049] | 6.27 | 0.013 | 0.052 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.005 | 0.012 | [0.001, 0.050] | 2.68 | 0.069 | 0.2 |  |
| C(proportions_option, Sum):C(num_agents, Sum) | 0.004 | 0.012 | [0.006, 0.063] | 0.84 | 0.54 | 0.73 |  |
| C(question_number, Sum):C(num_agents, Sum) | 0.004 | 0.011 | [0.003, 0.053] | 1.15 | 0.33 | 0.62 |  |
| C(proportions_option, Sum):C(activity_exponent, Sum) | 0.004 | 0.010 | [0.006, 0.059] | 0.73 | 0.62 | 0.74 |  |
| C(proportions_option, Sum):C(homophily, Sum) | 0.003 | 0.008 | [0.001, 0.047] | 1.19 | 0.31 | 0.62 |  |
| C(proportions_option, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.007 | [0.001, 0.039] | 1.02 | 0.38 | 0.62 |  |
| C(homophily, Sum) | 0.002 | 0.006 | [0.000, 0.032] | 2.54 | 0.11 | 0.26 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.002 | 0.006 | [0.002, 0.047] | 0.60 | 0.66 | 0.77 |  |
| C(proportions_option, Sum):C(num_news_agents, Sum) | 0.002 | 0.005 | [0.004, 0.052] | 0.39 | 0.89 | 0.91 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.002 | 0.005 | [0.000, 0.032] | 1.12 | 0.33 | 0.62 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.002 | 0.005 | [0.000, 0.038] | 1.06 | 0.35 | 0.62 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.002 | 0.005 | [0.001, 0.038] | 0.68 | 0.57 | 0.73 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.002 | 0.004 | [0.000, 0.034] | 0.97 | 0.38 | 0.62 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.002 | 0.004 | [0.000, 0.033] | 0.89 | 0.41 | 0.62 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.002 | 0.004 | [0.002, 0.043] | 0.43 | 0.78 | 0.84 |  |
| C(proportions_option, Sum) | 0.001 | 0.003 | [0.001, 0.035] | 0.42 | 0.74 | 0.83 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.003 | [0.000, 0.030] | 0.59 | 0.55 | 0.73 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.001 | 0.002 | [0.000, 0.029] | 0.50 | 0.61 | 0.74 |  |
| C(proportions_option, Sum):C(graph_type, Sum) | 0.001 | 0.002 | [0.001, 0.033] | 0.32 | 0.81 | 0.85 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.000, 0.029] | 0.47 | 0.63 | 0.74 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.000, 0.023] | 0.83 | 0.36 | 0.62 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.002 | [0.000, 0.021] | 0.69 | 0.41 | 0.62 |  |
| C(num_agents, Sum) | 0.000 | 0.001 | [0.000, 0.023] | 0.25 | 0.78 | 0.84 |  |
| C(graph_type, Sum) | 0.000 | 0.001 | [0.000, 0.017] | 0.46 | 0.5 | 0.72 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.001 | [0.000, 0.017] | 0.35 | 0.55 | 0.73 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.000 | [0.000, 0.021] | 0.06 | 0.95 | 0.95 |  |

## mean_neighbor_alignment_shift_rate (n=576)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(question_number, Sum) | 0.139 | 0.386 | [0.305, 0.464] | 135.86 | 1.7e-46 | 7.6e-45 | yes |
| C(model_family, Sum) | 0.098 | 0.308 | [0.231, 0.389] | 64.02 | 2.9e-34 | 6.5e-33 | yes |
| C(model_family, Sum):C(proportions_option, Sum) | 0.092 | 0.294 | [0.233, 0.393] | 19.98 | 4.3e-28 | 3.9e-27 | yes |
| C(num_agents, Sum) | 0.090 | 0.290 | [0.208, 0.370] | 88.14 | 7.9e-33 | 1.2e-31 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.088 | 0.285 | [0.209, 0.374] | 28.70 | 6.6e-29 | 7.4e-28 | yes |
| C(add_survey_to_context, Sum) | 0.054 | 0.197 | [0.133, 0.272] | 105.86 | 2.4e-22 | 1.8e-21 | yes |
| C(graph_type, Sum) | 0.040 | 0.154 | [0.094, 0.228] | 78.93 | 1.7e-17 | 1.1e-16 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.028 | 0.113 | [0.069, 0.197] | 9.18 | 1.8e-09 | 9.1e-09 | yes |
| C(question_number, Sum):C(num_agents, Sum) | 0.026 | 0.106 | [0.057, 0.187] | 12.84 | 6.9e-10 | 3.9e-09 | yes |
| C(proportions_option, Sum):C(num_agents, Sum) | 0.011 | 0.046 | [0.022, 0.117] | 3.48 | 0.0023 | 0.0078 | yes |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.008 | 0.036 | [0.014, 0.095] | 4.08 | 0.003 | 0.0089 | yes |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.008 | 0.035 | [0.013, 0.091] | 3.94 | 0.0037 | 0.011 | yes |
| C(proportions_option, Sum):C(question_number, Sum) | 0.007 | 0.032 | [0.016, 0.095] | 2.39 | 0.028 | 0.066 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.007 | 0.032 | [0.009, 0.082] | 7.11 | 0.00092 | 0.0041 | yes |
| C(num_news_agents, Sum) | 0.007 | 0.031 | [0.007, 0.076] | 6.87 | 0.0012 | 0.0047 | yes |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.007 | 0.030 | [0.009, 0.074] | 6.63 | 0.0015 | 0.0055 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.006 | 0.027 | [0.005, 0.075] | 6.10 | 0.0024 | 0.0078 | yes |
| C(proportions_option, Sum):C(homophily, Sum) | 0.006 | 0.027 | [0.008, 0.074] | 3.94 | 0.0086 | 0.023 | yes |
| C(num_agents, Sum):C(graph_type, Sum) | 0.005 | 0.022 | [0.005, 0.067] | 4.76 | 0.009 | 0.023 | yes |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.005 | 0.020 | [0.007, 0.065] | 2.26 | 0.062 | 0.13 |  |
| C(proportions_option, Sum):C(num_news_agents, Sum) | 0.004 | 0.020 | [0.009, 0.073] | 1.44 | 0.2 | 0.29 |  |
| C(proportions_option, Sum):C(add_survey_to_context, Sum) | 0.004 | 0.019 | [0.003, 0.060] | 2.74 | 0.043 | 0.097 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.004 | 0.018 | [0.005, 0.063] | 1.96 | 0.1 | 0.19 |  |
| C(proportions_option, Sum):C(activity_exponent, Sum) | 0.003 | 0.015 | [0.008, 0.067] | 1.09 | 0.37 | 0.5 |  |
| C(proportions_option, Sum):C(graph_type, Sum) | 0.003 | 0.015 | [0.003, 0.061] | 2.18 | 0.09 | 0.18 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.003 | 0.013 | [0.002, 0.055] | 1.84 | 0.14 | 0.25 |  |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.011 | [0.002, 0.051] | 1.64 | 0.18 | 0.28 |  |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.002 | 0.010 | [0.005, 0.060] | 0.76 | 0.6 | 0.69 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.002 | 0.009 | [0.001, 0.045] | 2.04 | 0.13 | 0.25 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.002 | 0.009 | [0.001, 0.041] | 1.89 | 0.15 | 0.25 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.002 | 0.008 | [0.005, 0.060] | 0.56 | 0.76 | 0.83 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.001 | 0.006 | [0.000, 0.034] | 1.33 | 0.26 | 0.38 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.006 | [0.002, 0.044] | 0.64 | 0.64 | 0.71 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.001 | 0.005 | [0.000, 0.040] | 1.13 | 0.32 | 0.45 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.005 | [0.000, 0.029] | 2.16 | 0.14 | 0.25 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.001 | 0.005 | [0.000, 0.028] | 2.02 | 0.16 | 0.25 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.001 | 0.004 | [0.000, 0.032] | 0.86 | 0.42 | 0.53 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.004 | [0.000, 0.036] | 0.77 | 0.46 | 0.56 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.001 | 0.003 | [0.000, 0.027] | 0.64 | 0.53 | 0.62 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.001, 0.034] | 0.36 | 0.78 | 0.83 |  |
| C(proportions_option, Sum) | 0.000 | 0.002 | [0.001, 0.029] | 0.32 | 0.81 | 0.83 |  |
| C(homophily, Sum) | 0.000 | 0.002 | [0.000, 0.023] | 0.75 | 0.39 | 0.51 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.002 | [0.000, 0.019] | 0.72 | 0.4 | 0.51 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.022] | 0.21 | 0.81 | 0.83 |  |
| C(activity_exponent, Sum) | 0.000 | 0.001 | [0.000, 0.021] | 0.17 | 0.85 | 0.85 |  |

## delta_assortativity (n=576)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(homophily, Sum) | 0.150 | 0.277 | [0.203, 0.349] | 165.44 | 2.8e-32 | 1.3e-30 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.082 | 0.174 | [0.110, 0.266] | 15.13 | 9.5e-16 | 1.1e-14 | yes |
| C(model_family, Sum) | 0.081 | 0.171 | [0.112, 0.250] | 29.65 | 1.9e-17 | 4.2e-16 | yes |
| C(model_family, Sum):C(homophily, Sum) | 0.073 | 0.156 | [0.100, 0.242] | 26.63 | 8e-16 | 1.1e-14 | yes |
| C(model_family, Sum):C(proportions_option, Sum) | 0.054 | 0.121 | [0.084, 0.213] | 6.58 | 8e-09 | 7.2e-08 | yes |
| C(proportions_option, Sum) | 0.035 | 0.083 | [0.041, 0.152] | 12.99 | 3.9e-08 | 2.9e-07 | yes |
| C(question_number, Sum):C(homophily, Sum) | 0.031 | 0.072 | [0.027, 0.147] | 16.81 | 9.3e-08 | 6e-07 | yes |
| C(proportions_option, Sum):C(homophily, Sum) | 0.013 | 0.032 | [0.010, 0.079] | 4.70 | 0.0031 | 0.017 | yes |
| C(proportions_option, Sum):C(num_agents, Sum) | 0.010 | 0.024 | [0.012, 0.084] | 1.81 | 0.097 | 0.36 |  |
| C(proportions_option, Sum):C(num_news_agents, Sum) | 0.007 | 0.018 | [0.008, 0.077] | 1.29 | 0.26 | 0.65 |  |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.007 | 0.017 | [0.009, 0.068] | 1.23 | 0.29 | 0.68 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.006 | 0.015 | [0.001, 0.056] | 3.26 | 0.039 | 0.18 |  |
| C(question_number, Sum) | 0.006 | 0.015 | [0.001, 0.059] | 3.22 | 0.041 | 0.18 |  |
| C(model_family, Sum):C(num_agents, Sum) | 0.005 | 0.013 | [0.007, 0.064] | 0.98 | 0.44 | 0.86 |  |
| C(num_news_agents, Sum) | 0.005 | 0.012 | [0.001, 0.048] | 2.65 | 0.072 | 0.29 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.004 | 0.009 | [0.006, 0.060] | 0.69 | 0.66 | 0.98 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.003 | 0.008 | [0.003, 0.049] | 0.86 | 0.49 | 0.91 |  |
| C(proportions_option, Sum):C(question_number, Sum) | 0.003 | 0.008 | [0.006, 0.060] | 0.57 | 0.76 | 0.99 |  |
| C(proportions_option, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.008 | [0.002, 0.046] | 1.12 | 0.34 | 0.77 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.007 | [0.001, 0.038] | 1.60 | 0.2 | 0.58 |  |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.007 | [0.000, 0.037] | 1.56 | 0.21 | 0.58 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.003 | 0.007 | [0.000, 0.039] | 1.52 | 0.22 | 0.58 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.002 | 0.005 | [0.003, 0.044] | 0.58 | 0.68 | 0.98 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.005 | [0.000, 0.029] | 2.25 | 0.13 | 0.44 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.002 | 0.005 | [0.001, 0.041] | 0.75 | 0.52 | 0.91 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.005 | [0.000, 0.030] | 2.24 | 0.14 | 0.44 |  |
| C(proportions_option, Sum):C(activity_exponent, Sum) | 0.002 | 0.005 | [0.005, 0.052] | 0.37 | 0.9 | 0.99 |  |
| C(activity_exponent, Sum) | 0.002 | 0.005 | [0.000, 0.035] | 1.03 | 0.36 | 0.77 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.002 | 0.004 | [0.000, 0.029] | 0.90 | 0.41 | 0.83 |  |
| C(question_number, Sum):C(num_agents, Sum) | 0.001 | 0.004 | [0.002, 0.047] | 0.40 | 0.81 | 0.99 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.001 | 0.003 | [0.002, 0.042] | 0.35 | 0.84 | 0.99 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.003 | [0.000, 0.029] | 0.65 | 0.52 | 0.91 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.027] | 0.47 | 0.63 | 0.98 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.001 | 0.002 | [0.000, 0.027] | 0.44 | 0.65 | 0.98 |  |
| C(num_agents, Sum) | 0.001 | 0.002 | [0.000, 0.027] | 0.36 | 0.7 | 0.98 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.001 | [0.002, 0.036] | 0.15 | 0.96 | 0.99 |  |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.001 | [0.002, 0.038] | 0.14 | 0.97 | 0.99 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.001 | 0.001 | [0.000, 0.024] | 0.28 | 0.76 | 0.99 |  |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.001 | [0.001, 0.028] | 0.12 | 0.95 | 0.99 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.000 | 0.001 | [0.000, 0.023] | 0.16 | 0.85 | 0.99 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.001 | [0.000, 0.024] | 0.15 | 0.86 | 0.99 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.000, 0.016] | 0.23 | 0.63 | 0.98 |  |
| C(proportions_option, Sum):C(graph_type, Sum) | 0.000 | 0.000 | [0.001, 0.026] | 0.04 | 0.99 | 0.99 |  |
| C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.02 | 0.88 | 0.99 |  |
| C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.014] | 0.00 | 0.99 | 0.99 |  |

## mean_local_agreement (n=576)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(model_family, Sum):C(question_number, Sum) | 0.259 | 0.483 | [0.387, 0.573] | 67.29 | 7e-59 | 3.1e-57 | yes |
| C(model_family, Sum) | 0.127 | 0.314 | [0.245, 0.390] | 65.90 | 4.2e-35 | 9.4e-34 | yes |
| C(question_number, Sum) | 0.090 | 0.244 | [0.173, 0.325] | 69.84 | 5.2e-27 | 7.9e-26 | yes |
| C(model_family, Sum):C(proportions_option, Sum) | 0.036 | 0.114 | [0.070, 0.207] | 6.15 | 3.7e-08 | 4.2e-07 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.025 | 0.082 | [0.040, 0.153] | 12.84 | 4.8e-08 | 4.3e-07 | yes |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.023 | 0.075 | [0.037, 0.150] | 8.77 | 8.1e-07 | 6.1e-06 | yes |
| C(proportions_option, Sum):C(question_number, Sum) | 0.021 | 0.069 | [0.039, 0.142] | 5.36 | 2.3e-05 | 0.00015 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.012 | 0.042 | [0.010, 0.097] | 9.38 | 0.0001 | 0.00058 | yes |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.011 | 0.040 | [0.019, 0.101] | 2.97 | 0.0075 | 0.028 | yes |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.011 | 0.039 | [0.018, 0.106] | 2.90 | 0.0089 | 0.031 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.009 | 0.033 | [0.015, 0.100] | 2.43 | 0.025 | 0.082 |  |
| C(proportions_option, Sum):C(num_agents, Sum) | 0.008 | 0.028 | [0.012, 0.087] | 2.07 | 0.055 | 0.14 |  |
| C(num_agents, Sum) | 0.007 | 0.026 | [0.005, 0.079] | 5.77 | 0.0034 | 0.014 | yes |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.007 | 0.024 | [0.006, 0.074] | 2.60 | 0.035 | 0.11 |  |
| C(add_survey_to_context, Sum) | 0.007 | 0.023 | [0.002, 0.062] | 10.19 | 0.0015 | 0.0076 | yes |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.006 | 0.020 | [0.002, 0.058] | 8.90 | 0.003 | 0.014 | yes |
| C(proportions_option, Sum):C(activity_exponent, Sum) | 0.005 | 0.018 | [0.009, 0.073] | 1.32 | 0.25 | 0.39 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.004 | 0.015 | [0.003, 0.054] | 2.22 | 0.085 | 0.17 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.004 | 0.015 | [0.002, 0.051] | 3.28 | 0.039 | 0.11 |  |
| C(proportions_option, Sum):C(num_news_agents, Sum) | 0.004 | 0.015 | [0.007, 0.064] | 1.08 | 0.37 | 0.49 |  |
| C(proportions_option, Sum) | 0.004 | 0.014 | [0.003, 0.060] | 2.11 | 0.098 | 0.19 |  |
| C(activity_exponent, Sum) | 0.004 | 0.014 | [0.002, 0.051] | 3.09 | 0.046 | 0.12 |  |
| C(num_news_agents, Sum) | 0.003 | 0.012 | [0.001, 0.048] | 2.65 | 0.072 | 0.16 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.003 | 0.012 | [0.001, 0.047] | 2.63 | 0.073 | 0.16 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.003 | 0.009 | [0.002, 0.049] | 1.03 | 0.39 | 0.49 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.002 | 0.009 | [0.001, 0.044] | 1.88 | 0.15 | 0.27 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.002 | 0.009 | [0.001, 0.045] | 1.25 | 0.29 | 0.44 |  |
| C(graph_type, Sum) | 0.002 | 0.008 | [0.000, 0.035] | 3.59 | 0.059 | 0.14 |  |
| C(question_number, Sum):C(num_agents, Sum) | 0.002 | 0.008 | [0.002, 0.052] | 0.87 | 0.48 | 0.54 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.008 | [0.000, 0.042] | 1.67 | 0.19 | 0.32 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.002 | 0.008 | [0.003, 0.047] | 0.83 | 0.51 | 0.55 |  |
| C(proportions_option, Sum):C(graph_type, Sum) | 0.002 | 0.007 | [0.001, 0.045] | 1.06 | 0.36 | 0.49 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.002 | 0.006 | [0.000, 0.036] | 1.38 | 0.25 | 0.39 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.002 | 0.006 | [0.000, 0.033] | 2.70 | 0.1 | 0.19 |  |
| C(proportions_option, Sum):C(add_survey_to_context, Sum) | 0.002 | 0.006 | [0.001, 0.039] | 0.85 | 0.47 | 0.54 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.002 | 0.005 | [0.000, 0.036] | 1.18 | 0.31 | 0.45 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.001 | 0.005 | [0.000, 0.033] | 1.05 | 0.35 | 0.49 |  |
| C(homophily, Sum) | 0.001 | 0.005 | [0.000, 0.026] | 2.05 | 0.15 | 0.27 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.004 | [0.000, 0.030] | 0.91 | 0.4 | 0.49 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.001 | 0.004 | [0.000, 0.032] | 0.87 | 0.42 | 0.5 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.001 | 0.003 | [0.000, 0.028] | 0.58 | 0.56 | 0.6 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.001 | 0.003 | [0.001, 0.036] | 0.29 | 0.89 | 0.91 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.000, 0.025] | 0.43 | 0.65 | 0.68 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.002 | [0.000, 0.020] | 0.73 | 0.39 | 0.49 |  |
| C(proportions_option, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.001, 0.027] | 0.13 | 0.94 | 0.94 |  |

## cross_cutting_edge_fraction (n=576)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(model_family, Sum):C(question_number, Sum) | 0.267 | 0.492 | [0.410, 0.588] | 69.68 | 1.8e-60 | 8.2e-59 | yes |
| C(model_family, Sum) | 0.119 | 0.301 | [0.230, 0.381] | 61.96 | 2.5e-33 | 5.5e-32 | yes |
| C(question_number, Sum) | 0.092 | 0.251 | [0.171, 0.327] | 72.23 | 8.7e-28 | 1.3e-26 | yes |
| C(model_family, Sum):C(proportions_option, Sum) | 0.033 | 0.107 | [0.066, 0.199] | 5.74 | 1.5e-07 | 1.4e-06 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.024 | 0.080 | [0.040, 0.150] | 12.51 | 7.4e-08 | 8.3e-07 | yes |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.024 | 0.079 | [0.042, 0.148] | 9.28 | 3.3e-07 | 2.5e-06 | yes |
| C(proportions_option, Sum):C(question_number, Sum) | 0.019 | 0.066 | [0.038, 0.137] | 5.07 | 4.9e-05 | 0.00027 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.013 | 0.046 | [0.013, 0.098] | 10.36 | 4e-05 | 0.00026 | yes |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.012 | 0.041 | [0.018, 0.102] | 3.04 | 0.0063 | 0.024 | yes |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.011 | 0.040 | [0.021, 0.107] | 2.98 | 0.0074 | 0.026 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.011 | 0.037 | [0.017, 0.103] | 2.75 | 0.012 | 0.037 | yes |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.009 | 0.030 | [0.009, 0.086] | 3.37 | 0.0098 | 0.032 | yes |
| C(num_agents, Sum) | 0.007 | 0.026 | [0.006, 0.076] | 5.82 | 0.0032 | 0.016 | yes |
| C(proportions_option, Sum):C(num_agents, Sum) | 0.007 | 0.026 | [0.012, 0.084] | 1.91 | 0.078 | 0.17 |  |
| C(proportions_option, Sum) | 0.006 | 0.023 | [0.006, 0.071] | 3.39 | 0.018 | 0.051 |  |
| C(add_survey_to_context, Sum) | 0.005 | 0.019 | [0.002, 0.057] | 8.51 | 0.0037 | 0.017 | yes |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.005 | 0.019 | [0.001, 0.057] | 8.18 | 0.0044 | 0.018 | yes |
| C(proportions_option, Sum):C(activity_exponent, Sum) | 0.005 | 0.018 | [0.010, 0.069] | 1.29 | 0.26 | 0.4 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.004 | 0.015 | [0.002, 0.049] | 3.33 | 0.037 | 0.097 |  |
| C(proportions_option, Sum):C(num_news_agents, Sum) | 0.004 | 0.014 | [0.006, 0.066] | 1.02 | 0.41 | 0.51 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.004 | 0.013 | [0.002, 0.049] | 1.95 | 0.12 | 0.22 |  |
| C(activity_exponent, Sum) | 0.003 | 0.012 | [0.001, 0.049] | 2.70 | 0.068 | 0.16 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.003 | 0.012 | [0.001, 0.050] | 2.70 | 0.069 | 0.16 |  |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.003 | 0.012 | [0.004, 0.062] | 1.31 | 0.27 | 0.4 |  |
| C(question_number, Sum):C(num_agents, Sum) | 0.003 | 0.011 | [0.003, 0.058] | 1.21 | 0.31 | 0.43 |  |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.003 | 0.011 | [0.001, 0.047] | 2.40 | 0.092 | 0.19 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.003 | 0.011 | [0.002, 0.049] | 1.56 | 0.2 | 0.32 |  |
| C(num_news_agents, Sum) | 0.003 | 0.011 | [0.001, 0.042] | 2.30 | 0.1 | 0.19 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.003 | 0.009 | [0.001, 0.045] | 2.04 | 0.13 | 0.23 |  |
| C(graph_type, Sum) | 0.002 | 0.008 | [0.000, 0.032] | 3.27 | 0.071 | 0.16 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.002 | 0.007 | [0.003, 0.049] | 0.76 | 0.55 | 0.64 |  |
| C(homophily, Sum) | 0.002 | 0.006 | [0.000, 0.031] | 2.70 | 0.1 | 0.19 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.002 | 0.005 | [0.000, 0.032] | 1.19 | 0.3 | 0.43 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.001 | 0.005 | [0.000, 0.033] | 1.14 | 0.32 | 0.44 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.001 | 0.005 | [0.000, 0.033] | 1.10 | 0.33 | 0.44 |  |
| C(proportions_option, Sum):C(add_survey_to_context, Sum) | 0.001 | 0.005 | [0.001, 0.041] | 0.72 | 0.54 | 0.64 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.001 | 0.004 | [0.000, 0.031] | 1.94 | 0.16 | 0.27 |  |
| C(proportions_option, Sum):C(graph_type, Sum) | 0.001 | 0.004 | [0.001, 0.038] | 0.63 | 0.6 | 0.65 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.001 | 0.003 | [0.000, 0.026] | 0.63 | 0.53 | 0.64 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.001 | 0.003 | [0.000, 0.028] | 0.55 | 0.58 | 0.65 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.001 | 0.002 | [0.002, 0.037] | 0.25 | 0.91 | 0.93 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.001 | 0.002 | [0.000, 0.028] | 0.48 | 0.62 | 0.65 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.001 | 0.002 | [0.000, 0.029] | 0.48 | 0.62 | 0.65 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.002 | [0.000, 0.019] | 0.78 | 0.38 | 0.48 |  |
| C(proportions_option, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.001, 0.030] | 0.08 | 0.97 | 0.97 |  |

## order_consistency_rate (n=576)

| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |
|---|---|---|---|---|---|---|---|
| C(model_family, Sum) | 0.391 | 0.849 | [0.821, 0.876] | 811.49 | 4.6e-177 | 2.1e-175 | yes |
| C(question_number, Sum) | 0.277 | 0.800 | [0.757, 0.836] | 862.07 | 1.6e-151 | 3.5e-150 | yes |
| C(add_survey_to_context, Sum) | 0.086 | 0.553 | [0.484, 0.614] | 533.68 | 1.8e-77 | 2.7e-76 | yes |
| C(num_agents, Sum) | 0.028 | 0.284 | [0.202, 0.372] | 85.79 | 4.2e-32 | 4.7e-31 | yes |
| C(num_agents, Sum):C(add_survey_to_context, Sum) | 0.027 | 0.282 | [0.205, 0.372] | 84.90 | 8e-32 | 7.2e-31 | yes |
| C(model_family, Sum):C(proportions_option, Sum) | 0.027 | 0.280 | [0.221, 0.377] | 18.71 | 2.1e-26 | 1.4e-25 | yes |
| C(model_family, Sum):C(add_survey_to_context, Sum) | 0.026 | 0.272 | [0.200, 0.364] | 53.89 | 1.3e-29 | 9.8e-29 | yes |
| C(model_family, Sum):C(num_agents, Sum) | 0.013 | 0.157 | [0.094, 0.261] | 13.37 | 6.4e-14 | 3.2e-13 | yes |
| C(question_number, Sum):C(add_survey_to_context, Sum) | 0.013 | 0.156 | [0.090, 0.239] | 39.93 | 1.2e-16 | 6.9e-16 | yes |
| C(model_family, Sum):C(question_number, Sum) | 0.011 | 0.140 | [0.087, 0.227] | 11.72 | 3.5e-12 | 1.6e-11 | yes |
| C(question_number, Sum):C(num_agents, Sum) | 0.008 | 0.109 | [0.063, 0.195] | 13.16 | 4e-10 | 1.6e-09 | yes |
| C(proportions_option, Sum):C(question_number, Sum) | 0.005 | 0.066 | [0.035, 0.140] | 5.08 | 4.7e-05 | 0.00018 | yes |
| C(question_number, Sum):C(activity_exponent, Sum) | 0.003 | 0.045 | [0.016, 0.112] | 5.10 | 0.0005 | 0.0017 | yes |
| C(proportions_option, Sum):C(num_agents, Sum) | 0.002 | 0.032 | [0.017, 0.099] | 2.39 | 0.028 | 0.09 |  |
| C(proportions_option, Sum):C(num_news_agents, Sum) | 0.001 | 0.016 | [0.008, 0.068] | 1.15 | 0.33 | 0.62 |  |
| C(num_news_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.013 | [0.003, 0.062] | 1.47 | 0.21 | 0.5 |  |
| C(proportions_option, Sum) | 0.001 | 0.012 | [0.003, 0.054] | 1.79 | 0.15 | 0.37 |  |
| C(proportions_option, Sum):C(activity_exponent, Sum) | 0.001 | 0.012 | [0.006, 0.066] | 0.87 | 0.52 | 0.72 |  |
| C(question_number, Sum):C(num_news_agents, Sum) | 0.001 | 0.012 | [0.004, 0.062] | 1.28 | 0.28 | 0.54 |  |
| C(add_survey_to_context, Sum):C(activity_exponent, Sum) | 0.001 | 0.011 | [0.001, 0.051] | 2.51 | 0.083 | 0.23 |  |
| C(model_family, Sum):C(activity_exponent, Sum) | 0.001 | 0.011 | [0.007, 0.063] | 0.82 | 0.55 | 0.76 |  |
| C(homophily, Sum) | 0.001 | 0.010 | [0.000, 0.042] | 4.51 | 0.034 | 0.1 |  |
| C(num_agents, Sum):C(activity_exponent, Sum) | 0.001 | 0.009 | [0.004, 0.050] | 1.02 | 0.4 | 0.66 |  |
| C(question_number, Sum):C(homophily, Sum) | 0.001 | 0.009 | [0.001, 0.045] | 2.03 | 0.13 | 0.35 |  |
| C(num_agents, Sum):C(num_news_agents, Sum) | 0.001 | 0.008 | [0.003, 0.054] | 0.90 | 0.46 | 0.72 |  |
| C(homophily, Sum):C(num_news_agents, Sum) | 0.000 | 0.007 | [0.000, 0.041] | 1.50 | 0.22 | 0.5 |  |
| C(num_agents, Sum):C(homophily, Sum) | 0.000 | 0.007 | [0.000, 0.041] | 1.46 | 0.23 | 0.5 |  |
| C(model_family, Sum):C(num_news_agents, Sum) | 0.000 | 0.006 | [0.006, 0.054] | 0.47 | 0.83 | 0.87 |  |
| C(model_family, Sum):C(graph_type, Sum) | 0.000 | 0.005 | [0.001, 0.040] | 0.78 | 0.51 | 0.72 |  |
| C(add_survey_to_context, Sum):C(num_news_agents, Sum) | 0.000 | 0.004 | [0.000, 0.034] | 0.95 | 0.39 | 0.66 |  |
| C(proportions_option, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.004 | [0.001, 0.038] | 0.52 | 0.67 | 0.81 |  |
| C(activity_exponent, Sum) | 0.000 | 0.003 | [0.000, 0.028] | 0.74 | 0.48 | 0.72 |  |
| C(graph_type, Sum):C(num_news_agents, Sum) | 0.000 | 0.003 | [0.000, 0.029] | 0.70 | 0.5 | 0.72 |  |
| C(graph_type, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.003 | [0.000, 0.025] | 1.33 | 0.25 | 0.51 |  |
| C(proportions_option, Sum):C(graph_type, Sum) | 0.000 | 0.003 | [0.001, 0.037] | 0.37 | 0.78 | 0.85 |  |
| C(proportions_option, Sum):C(homophily, Sum) | 0.000 | 0.002 | [0.001, 0.034] | 0.34 | 0.8 | 0.85 |  |
| C(num_news_agents, Sum) | 0.000 | 0.002 | [0.000, 0.027] | 0.45 | 0.64 | 0.81 |  |
| C(graph_type, Sum):C(homophily, Sum) | 0.000 | 0.002 | [0.000, 0.023] | 0.88 | 0.35 | 0.63 |  |
| C(graph_type, Sum):C(activity_exponent, Sum) | 0.000 | 0.002 | [0.000, 0.028] | 0.42 | 0.66 | 0.81 |  |
| C(question_number, Sum):C(graph_type, Sum) | 0.000 | 0.002 | [0.000, 0.026] | 0.39 | 0.68 | 0.81 |  |
| C(homophily, Sum):C(activity_exponent, Sum) | 0.000 | 0.001 | [0.000, 0.026] | 0.32 | 0.73 | 0.82 |  |
| C(model_family, Sum):C(homophily, Sum) | 0.000 | 0.001 | [0.001, 0.029] | 0.10 | 0.96 | 0.97 |  |
| C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.21 | 0.65 | 0.81 |  |
| C(homophily, Sum):C(add_survey_to_context, Sum) | 0.000 | 0.000 | [0.000, 0.016] | 0.13 | 0.72 | 0.82 |  |
| C(num_agents, Sum):C(graph_type, Sum) | 0.000 | 0.000 | [0.000, 0.020] | 0.03 | 0.97 | 0.97 |  |

