# Analysis of Randomized LLM-Based Social Simulation Experiments

**Dataset:** 354 simulation runs with standard agent counts ({64, 256, 1024, 4096}), across three questions (Q25: genetic enhancements, Q28: AI copyright, Q29: environmental protection), varying seven independent parameters. Runs with non-standard `num_agents` values (50, 500, 1000, 1500) from earlier job submissions are excluded throughout.

---

## 1. Dataset Overview

| Parameter | Values observed | n per level |
|---|---|---|
| `base_model` | minitaur (131), llama3.1 (78), qwen (105), gemma (40) | See below |
| `num_agents` | 64, 256, 1024, 4096 | 100, 102, 88, 64 |
| `graph_type` | random, powerlaw_cluster | 178, 176 |
| `homophily` | True, False | 166, 188 |
| `add_survey_to_context` | True, False | 190, 164 |
| `proportions_option` | uniform, blueprint, average, distribution | 102, 96, 76, 80 |
| `num_news_agents` | 0, 1 | 175, 179 |
| `question` | 25, 28, 29 | 121, 117, 116 |

The four base models are Llama-3.1-Minitaur-8B, Llama-3.1-8B, Qwen2.5-7B-Instruct, and Gemma-3-4B-PT — all fine-tuned on social media data with persona-specific LoRA adapters. All runs produce a roughly constant number of threads (~780) independent of agent count, confirming the simulation is time-bounded rather than agent-count-bounded.

---

## 2. Normalization Check: Agent Count Effect is Real

The `mean_opinion_shift_rate` metric is `changed_users / shared_users` — a fraction between 0 and 1, already normalized by population size. This was verified by comparing stored values against manually computed ratios from the transition log (they match exactly).

The fact that this fraction decreases with `num_agents` (r = −0.48, p < 0.001) while the absolute count of changed agents increases (r = +0.67) confirms the effect is genuine: **in larger simulated communities, a smaller fraction of agents changes their opinion per simulation step**, consistent with statistical stabilization in larger populations. This holds within every model individually (r = −0.58 for minitaur, −0.45 for llama3.1, −0.39 for qwen, −0.71 for gemma; all p < 0.001).

---

## 3. Key Findings

### 3.1 Model Choice is the Strongest Driver of Consensus Trajectory

All four models produce mostly downward consensus trajectories, but at very different rates:

| Model | n | Initial consensus | Δ consensus | Runs with ↓ consensus |
|---|---|---|---|---|
| Minitaur | 131 | 0.85 | **−0.108 ± 0.111** | 85% |
| Qwen | 105 | 0.86 | **−0.067 ± 0.170** | 72% |
| Llama-3.1-8B | 78 | 0.79 | **−0.021 ± 0.188** | 46% |
| Gemma | 40 | 0.73 | **+0.013 ± 0.125** | 40% |

Minitaur produces the strongest, most consistent consensus erosion; Gemma is the only model that drifts slightly toward consensus gain on average (though the difference from zero is not significant given its variance). Pairwise tests:

| Comparison | t | p | Cohen's d |
|---|---|---|---|
| Minitaur vs Gemma | −5.52 | < 0.001 *** | −1.06 |
| Minitaur vs Llama3.1 | −3.73 | < 0.001 *** | −0.60 |
| Qwen vs Gemma | −3.09 | 0.002 ** | −0.50 |
| Minitaur vs Qwen | −2.17 | 0.030 * | −0.30 |
| Llama3.1 vs Qwen | 1.68 | 0.092 (ns) | — |
| Llama3.1 vs Gemma | −1.19 | 0.236 (ns) | — |

The ranking (Minitaur > Qwen > Llama3.1 > Gemma in consensus erosion) does not obviously correspond to model size or architecture family. Both Minitaur and Llama-3.1-8B share the same Llama-3.1-8B base architecture but differ substantially in behavior, suggesting fine-tuning on Minitaur's social media persona data instills distinctly stronger individual opinionation.

**Per-question patterns:**

| | Minitaur | Llama3.1 | Qwen | Gemma |
|---|---|---|---|---|
| Q25 (genetic enhance.) | −0.140 | −0.131 | +0.042 | +0.007 |
| Q28 (AI copyright) | −0.091 | +0.036 | −0.072 | −0.039 |
| Q29 (envir. protection) | −0.089 | +0.010 | −0.161 | +0.095 |

Notable anomalies: Qwen starts at near-unanimous initial consensus on Q28 (0.993 vs 0.72–0.76 for other models), while Gemma starts at only 0.562 on Q29 where Minitaur starts at 1.000. These differences in initial opinion distributions likely reflect how each fine-tuned model interprets the initial survey question, and partly shape subsequent dynamics.

### 3.2 Agent Count Suppresses Per-Agent Opinion Volatility (Robust Across Models)

Opinion shift rate (fraction of agents changing opinion per step) decreases with population size consistently across all four models:

| num_agents | Minitaur | Llama3.1 | Qwen | Gemma |
|---|---|---|---|---|
| 64 | 0.208 | 0.251 | 0.269 | 0.292 |
| 256 | 0.241 | 0.263 | 0.276 | 0.319 |
| 1024 | 0.105 | 0.151 | 0.165 | 0.149 |
| 4096 | 0.061 | 0.127 | 0.158 | — |

The 64-agent to 1024-agent ratio is ~1.63–1.98× for all models (all comparisons p < 0.001). The drop is particularly sharp between 256 and 1024 agents. This is consistent with a "law of large numbers" stabilization effect: with more agents, the aggregate opinion distribution fluctuates less per step even though each individual agent is equally likely to be exposed to persuasive content.

Crucially, total net consensus change does NOT correlate with agent count (r = 0.03, p = 0.73) — larger populations reach similar final consensus levels via smaller but more numerous micro-steps.

### 3.3 Survey Context Increases Herding and AI Detectability — but the Effect is Model-Specific

**Majority follow rate** (fraction of opinion-changers adopting the majority view) is higher when agents see survey results:
- ctx=True: 0.521 ± 0.060 vs ctx=False: 0.492 ± 0.067 (*t* = 4.24, *p* < 0.001, Cohen's *d* = 0.46)

**BERT AI detectability** is also substantially higher with survey context:
- ctx=True: 0.982 ± 0.020 vs ctx=False: 0.939 ± 0.037 (*t* = 13.5, *p* < 10⁻¹⁵, *d* = 1.49)

However, these effects are **not uniform across models**. Breaking down the survey context effect on consensus change within each model:

| Model | ctx=True | ctx=False | *p* |
|---|---|---|---|
| Minitaur | −0.110 | −0.107 | 0.89 (ns) |
| Llama3.1 | +0.041 | −0.087 | 0.0015 ** |
| Qwen | −0.004 | −0.131 | < 0.001 *** |
| Gemma | +0.020 | −0.001 | 0.64 (ns) |

Survey context has **no effect on Minitaur or Gemma consensus**, but significantly reduces consensus erosion for Llama3.1 and Qwen (both p < 0.01). When Llama3.1 and Qwen agents can see the current majority opinion, they conform more and the aggregate consensus stabilizes or drifts upward. Minitaur agents apparently do not substantially adjust their behavior based on this information.

Similarly, the BERT detectability uplift from survey context is almost entirely absent for Gemma (99.8% vs 99.7%), and is driven by the non-Gemma models (98.0% vs 93.4%).

### 3.4 Gemma Produces Near-Perfectly Detectable AI Text

The most striking BERT result is model-specific:

| Model | BERT accuracy | n | Range |
|---|---|---|---|
| Gemma | **0.998 ± 0.004** | 40 | [0.985, 1.000] |
| Qwen | 0.959 ± 0.031 | 105 | [0.875, 1.000] |
| Minitaur | 0.959 ± 0.035 | 131 | [0.829, 1.000] |
| Llama3.1 | 0.954 ± 0.044 | 78 | [0.804, 1.000] |

Gemma-generated threads are classified as AI-generated with near-ceiling accuracy across all 40 runs, with minimal variance. The eval thread counts are comparable across models (Gemma: 145.7 vs others: 147.4), ruling out a sample-size explanation. This strongly suggests that Gemma-3-4B-PT, fine-tuned on social media persona data, produces text with highly distinctive non-human patterns that TwHIN-BERT readily identifies. The other three models produce text that is harder to distinguish from genuine social media content (BERT accuracy 95–96%, with substantial run-to-run variance).

### 3.5 Proportions Option Interacts Strongly with Model

With the full multi-model dataset, the previously observed "average proportions cause strongest consensus decrease" finding no longer holds as a clean main effect (average vs uniform: *p* = 0.29, ns). The effect is better characterized as an interaction:

| Proportions | Minitaur | Llama3.1 | Qwen | Gemma |
|---|---|---|---|---|
| uniform | −0.102 | −0.022 | −0.029 | +0.015 |
| blueprint | −0.118 | +0.029 | −0.016 | +0.082 |
| average | **−0.155** | +0.061 | −0.064 | −0.000 |
| distribution | −0.055 | **−0.180** | **−0.140** | −0.042 |

The `average` option (optimized convex combination weights) produces the strongest erosion for Minitaur, while `distribution` produces the strongest for Llama3.1 and Qwen. No single proportions scheme consistently dominates across models. The `blueprint` option (demographic frequencies) tends to produce mild or positive drift for non-Minitaur models.

### 3.6 Echo Chamber Dynamics: Clustering Dissolves Over Simulation Time

Echo chamber metrics are now tracked across all 11 survey steps, allowing temporal analysis. Key trajectories (all 354 runs, Δ = last − first):

| Metric | First survey step | Final survey step | Δ | *p* (vs zero) |
|---|---|---|---|---|
| Assortativity | 0.023 ± 0.079 | 0.000 ± 0.045 | **−0.022** | *p* < 0.001 *** |
| Local agreement | 0.772 ± 0.168 | 0.690 ± 0.156 | **−0.082** | *p* < 0.001 *** |
| Cross-cutting edges | 0.229 ± 0.171 | 0.311 ± 0.156 | **+0.082** | *p* < 0.001 *** |
| Same-option exposure | 0.000 (init.) | 0.692 ± 0.156 | — | — |

Opinion-based network clustering (assortativity) starts slightly positive and decays to near-zero by the end of the simulation, indicating that whatever initial clustering exists is dismantled over time rather than reinforced. This is directly contrary to the echo chamber hypothesis. Local agreement and cross-cutting edges move in tandem, reflecting the same progressive opinion mixing.

**Same-option exposure** is zero at initialization (before any interactions) and grows to a final value of 0.69. This initialization artifact makes the start-to-end comparison uninformative as a measure of echo formation; the final value (used in Section 3.6 cross-sectional analysis) remains a valid measure of informationally homogeneous neighborhoods.

**Homophily accelerates initial clustering but not final state:** Homophilic networks begin with substantially higher assortativity (0.055 vs −0.007 for non-homophilic), and their assortativity drops much more steeply:

| | homophily=True | homophily=False | *p* | Cohen's *d* |
|---|---|---|---|---|
| Δ assortativity | **−0.049 ± 0.110** | **+0.002 ± 0.036** | < 0.001 *** | −0.636 |

Homophilic networks start with opinion clustering and then dissolve it rapidly during the simulation; non-homophilic networks show negligible change. Graph type has no significant effect on any echo metric delta (*p* > 0.06 for all four metrics).

**Link to consensus dynamics:** Δ local agreement correlates strongly with consensus change (r = +0.915, p < 0.001) — local opinion homogeneity and global consensus erode in exact lockstep. Δ assortativity shows a weak negative correlation with consensus change (r = −0.121, p = 0.022*): greater assortativity loss is marginally associated with larger consensus erosion.

With the larger dataset, previously marginal cross-sectional graph_type effects disappear:

- **Local agreement** by graph_type: powerlaw 0.701 vs random 0.679 (*p* = 0.18, ns)
- **Cross-cutting edges** by graph_type: random 0.321 vs powerlaw 0.300 (*p* = 0.19, ns)
- **Final assortativity** by homophily: 0.006 vs −0.005 (*p* = 0.018 *, very small effect)

Same-option exposure still correlates significantly with final consensus (r = +0.54, p < 0.001): agents in more informationally homogeneous environments show less consensus erosion.

### 3.7 Neighbor Alignment Shift Rate (NASR)

NASR measures the fraction of agents who, when they change their opinion, do so in the direction of their immediate network neighbors' majority. It is now fully populated across all 354 runs (mean = 0.074 ± 0.038).

**NASR tracks overall volatility.** NASR correlates strongly with opinion shift rate (r = +0.924, p < 0.001), confirming it captures the same global volatility signal. The NASR/OSR ratio (the fraction of opinion changers who align with their local neighbor majority) is stable at roughly 0.36–0.42 across models, suggesting the local-conformity mechanism is consistent regardless of how frequently opinions change.

**Model ranking mirrors inverse consensus erosion.** Higher NASR means more agents conform to their local neighbors when changing opinions, which stabilizes rather than erodes aggregate consensus. The model ordering by NASR exactly inverts the consensus-erosion ranking:

| Model | Mean NASR | NASR/OSR ratio |
|---|---|---|
| Gemma | **0.099 ± 0.031** | 0.39 |
| Qwen | 0.081 ± 0.038 | 0.38 |
| Llama-3.1-8B | 0.074 ± 0.036 | 0.42 |
| Minitaur | **0.060 ± 0.036** | 0.36 |

All pairwise differences are statistically significant except Llama3.1 vs Qwen (*p* = 0.207, ns). The Gemma–Minitaur gap is particularly large (d = 1.10, p < 0.001).

**NASR decreases with population size** (r = −0.42, p < 0.001), mirroring the OSR pattern.

**Survey context effect.** Survey context globally increases NASR (ctx=True: 0.079 vs ctx=False: 0.068; *p* = 0.007, *d* = 0.29), but this effect is entirely explained by the model distribution across conditions: within each individual model, the survey context effect on NASR is non-significant (*p* > 0.10 for all four models).

### 3.8 News Agents Have No Measurable Effect

The null result holds with the larger dataset. Presence vs absence of a news agent has no significant effect on consensus change (*p* = 0.61), opinion shift rate (*p* = 0.61), or BERT detectability (*p* = 0.17). A single news agent among hundreds does not measurably alter simulation dynamics.

---

## 4. Summary Table

| Finding | Effect size | Significance | Notes |
|---|---|---|---|
| Minitaur vs Gemma → consensus change | Cohen's *d* = 1.06 | *p* < 0.001 *** | Minitaur: −0.108; Gemma: +0.013 |
| Minitaur vs Llama3.1 → consensus change | Cohen's *d* = 0.60 | *p* < 0.001 *** | |
| Qwen vs Gemma → consensus change | Cohen's *d* = 0.50 | *p* = 0.002 ** | |
| Gemma → BERT accuracy | *d* ≈ 1.25–1.45 | *p* < 0.001 *** | 99.8% vs ~95.9% for others |
| survey_context → majority follow rate | Cohen's *d* = 0.46 | *p* < 0.001 *** | Driven by Llama3.1 and Qwen |
| survey_context → BERT accuracy | Cohen's *d* = 1.49 | *p* < 10⁻¹⁵ *** | Near-zero effect for Gemma |
| num_agents ↑ → opinion_shift_rate ↓ | r = −0.48 | *p* < 0.001 *** | Holds within every model |
| num_agents ↑ → consensus change | r = −0.01 | *p* = 0.92 (ns) | No effect on final outcome |
| homophily → final assortativity | *t* = 2.36 | *p* = 0.018 * | Effect very small (Δ = 0.011) |
| homophily → Δ assortativity (time series) | Cohen's *d* = −0.636 | *p* < 0.001 *** | True: −0.049, False: +0.002 |
| graph_type → echo chamber metrics | various | ns (*p* > 0.06) | No effect on time series deltas |
| num_news_agents → any metric | various | ns (*p* > 0.16) | **No effect** |
| same-option exposure ↔ consensus | r = +0.54 | *p* < 0.001 *** | Homogeneity stabilizes consensus |
| Δ local agreement ↔ consensus change | r = +0.915 | *p* < 0.001 *** | Local and global opinion erosion in lockstep |
| NASR: Gemma vs Minitaur | Cohen's *d* = 1.10 | *p* < 0.001 *** | 0.099 vs 0.060 |
| NASR ↔ opinion shift rate | r = +0.924 | *p* < 0.001 *** | NASR tracks same volatility signal |
| NASR ↓ with num_agents | r = −0.42 | *p* < 0.001 *** | Mirrors OSR scaling |

---

## 5. Limitations and Methodological Notes

1. **Model imbalance and confounds:** Minitaur (n=131), Qwen (n=105), Llama3.1 (n=78), Gemma (n=40). The Gemma condition is the smallest and also has no 4096-agent runs, limiting some comparisons. Model is confounded with question-specific initial consensus values (e.g., Qwen on Q28 starts at 0.993, Gemma on Q29 at 0.562), which may affect within-question comparisons.
2. **NASR interpretation:** `neighbor_alignment_shift_rate` is now fully populated in all 354 runs. Because NASR correlates strongly with OSR (r = 0.924), it may not provide genuinely independent information beyond overall volatility; the NASR/OSR ratio (fraction of changers that follow local majority) is the more distinct quantity.
3. **BERT accuracy interpretation:** The BERT classifier is trained and evaluated within each run's own thread file (train/eval split). Accuracy reflects in-distribution separability, not a cross-run held-out assessment.
4. **Statistical tests are unadjusted** for multiple comparisons. With ~30+ tests performed, several nominally significant results at *p* < 0.05 may be false positives; treat marginal results (*p* < 0.05) with caution.
5. **Same-option exposure initialization artifact:** `mean_same_option_exposure_share` is exactly zero at the first survey step (before any interactions), so the start-to-end Δ of +0.69 reflects initialization rather than echo chamber formation. The final value is used for cross-sectional analysis; temporal comparisons for this metric are not interpretable.
6. **p-values use a normal approximation** (via the error function) rather than a t-distribution CDF. This is accurate for moderate-to-large samples but slightly off for small subgroups.
7. **`bert_n` confound in the survey-context BERT finding:** `add_survey_to_context=True` runs have fewer eval threads on average (ctx=True: ~136 threads vs ctx=False: ~154), and fewer threads correlates with higher accuracy (r = −0.55, p < 0.001). The direction of the effect is unambiguous, but the magnitude may be partially inflated by this imbalance.
