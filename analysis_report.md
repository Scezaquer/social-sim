# Analysis of Randomized LLM-Based Social Simulation Experiments

**Dataset:** 595 simulation runs with standard agent counts ({64, 256, 1024, 4096}), across three questions (Q25: genetic enhancements, Q28: AI copyright, Q29: environmental protection), varying seven independent parameters. Runs with non-standard `num_agents` values (50, 500, 1000, 1500) from earlier job submissions are excluded throughout. The dataset now includes a fifth model condition, `qwen_base`, which uses Qwen2.5-7B-Instruct with no persona LoRA fine-tuning, enabling a direct ablation of the social media fine-tuning.

---

## 1. Dataset Overview

| Parameter | Values observed | n per level |
|---|---|---|
| `base_model` | minitaur (169), llama3.1 (111), qwen (151), qwen_base (72), gemma (92) | See below |
| `num_agents` | 64, 256, 1024, 4096 | — |
| `graph_type` | random, powerlaw_cluster | — |
| `homophily` | True, False | — |
| `add_survey_to_context` | True, False | — |
| `proportions_option` | uniform, blueprint, average, distribution, None (qwen_base) | — |
| `num_news_agents` | 0, 1 | — |
| `question` | 25, 28, 29 | — |

Five model conditions are present. Four — Llama-3.1-Minitaur-8B, Llama-3.1-8B, Qwen2.5-7B-Instruct, and Gemma-3-4B-PT — are fine-tuned on social media data with persona-specific LoRA adapters (`proportions_option` ∈ {uniform, blueprint, average, distribution}). The fifth, **`qwen_base`**, uses the same Qwen2.5-7B-Instruct architecture but with no persona LoRAs applied (`proportions_option = None`), serving as a direct ablation of the social media fine-tuning within the same base model family. All runs produce a roughly constant number of threads (~780) independent of agent count, confirming the simulation is time-bounded rather than agent-count-bounded.

---

## 2. Normalization Check: Agent Count Effect is Real

The `mean_opinion_shift_rate` metric is `changed_users / shared_users` — a fraction between 0 and 1, already normalized by population size. This was verified by comparing stored values against manually computed ratios from the transition log (they match exactly).

The fact that this fraction decreases with `num_agents` (r = −0.42, p < 0.001) while the absolute count of changed agents increases (r = +0.64) confirms the effect is genuine: **in larger simulated communities, a smaller fraction of agents changes their opinion per simulation step**, consistent with statistical stabilization in larger populations. This holds within every LoRA-equipped model individually (r = −0.57 for minitaur, −0.45 for llama3.1, −0.42 for qwen, −0.73 for gemma; all p < 0.001). Notably, it does **not** hold for qwen_base (r = −0.18, p = 0.12, ns), which is explored in Section 3.10.

---

## 3. Key Findings

### 3.1 Model Choice is the Strongest Driver of Consensus Trajectory

All four models produce mostly downward consensus trajectories, but at very different rates:

| Model | n | Initial consensus | Δ consensus | Runs with ↓ consensus |
|---|---|---|---|---|
| Minitaur | 169 | 0.86 | **−0.095 ± 0.112** | 83% |
| Qwen | 151 | 0.85 | **−0.055 ± 0.161** | 70% |
| Llama-3.1-8B | 111 | 0.79 | **−0.027 ± 0.179** | 50% |
| Gemma | 92 | 0.73 | **+0.025 ± 0.122** | 36% |

Minitaur produces the strongest, most consistent consensus erosion; Gemma is the only model that drifts slightly toward consensus gain on average (though the difference from zero is not significant given its variance). Pairwise tests:

| Comparison | t | p | Cohen's d |
|---|---|---|---|
| Minitaur vs Gemma | −7.78 | < 0.001 *** | −1.03 |
| Minitaur vs Llama3.1 | −3.57 | < 0.001 *** | −0.48 |
| Qwen vs Gemma | −4.40 | < 0.001 *** | −0.55 |
| Minitaur vs Qwen | −2.52 | 0.012 * | −0.29 |
| Llama3.1 vs Qwen | 1.33 | 0.183 (ns) | — |
| Llama3.1 vs Gemma | −2.44 | 0.015 * | −0.33 |

The ranking (Minitaur > Qwen > Llama3.1 > Gemma in consensus erosion) does not obviously correspond to model size or architecture family. Both Minitaur and Llama-3.1-8B share the same Llama-3.1-8B base architecture but differ substantially in behavior, suggesting fine-tuning on Minitaur's social media persona data instills distinctly stronger individual opinionation. With the expanded dataset, Llama3.1 vs Gemma also reaches significance (*p* = 0.015, *d* = −0.33).

**Per-question patterns:**

| | Minitaur | Llama3.1 | Qwen | Gemma |
|---|---|---|---|---|
| Q25 (genetic enhance.) | −0.119 | −0.141 | +0.020 | +0.007 |
| Q28 (AI copyright) | −0.072 | +0.023 | −0.075 | −0.017 |
| Q29 (envir. protection) | −0.089 | +0.009 | −0.129 | +0.091 |

Notable anomalies: Qwen starts at near-unanimous initial consensus on Q28 (0.993 vs 0.72–0.76 for other models), while Gemma starts at only 0.562 on Q29 where Minitaur starts at 1.000. These differences in initial opinion distributions likely reflect how each fine-tuned model interprets the initial survey question, and partly shape subsequent dynamics.

### 3.2 Agent Count Suppresses Per-Agent Opinion Volatility (Robust Across Models)

Opinion shift rate (fraction of agents changing opinion per step) decreases with population size consistently across all four models:

| num_agents | Minitaur | Llama3.1 | Qwen | Gemma |
|---|---|---|---|---|
| 64 | 0.199 | 0.255 | 0.269 | 0.282 |
| 256 | 0.230 | 0.267 | 0.274 | 0.315 |
| 1024 | 0.110 | 0.151 | 0.160 | 0.178 |
| 4096 | 0.060 | 0.129 | 0.149 | 0.117 |

The 64-agent to 1024-agent ratio is ~1.59–1.82× for all models (all comparisons p < 0.001). The drop is particularly sharp between 256 and 1024 agents. This is consistent with a "law of large numbers" stabilization effect: with more agents, the aggregate opinion distribution fluctuates less per step even though each individual agent is equally likely to be exposed to persuasive content.

Crucially, total net consensus change does NOT correlate with agent count (r = 0.05, p = 0.22) — larger populations reach similar final consensus levels via smaller but more numerous micro-steps.

### 3.3 Survey Context Increases Herding and AI Detectability — but the Effect is Model-Specific

**Majority follow rate** (fraction of opinion-changers adopting the majority view) is higher when agents see survey results:
- ctx=True: 0.495 ± 0.111 vs ctx=False: 0.461 ± 0.138 (*t* = 3.30, *p* = 0.001 ***, Cohen's *d* = 0.27)

**BERT AI detectability** is also substantially higher with survey context:
- ctx=True: 0.984 ± 0.023 vs ctx=False: 0.943 ± 0.043 (*t* = 14.2, *p* < 10⁻¹⁵, *d* = 1.20)

However, these effects are **not uniform across models**. Breaking down the survey context effect on consensus change within each model:

| Model | ctx=True | ctx=False | *p* |
|---|---|---|---|
| Minitaur | −0.095 | −0.095 | 0.99 (ns) |
| Llama3.1 | +0.015 | −0.067 | 0.014 * |
| Qwen | −0.004 | −0.107 | < 0.001 *** |
| Gemma | +0.042 | +0.002 | 0.11 (ns) |

Survey context has **no effect on Minitaur or Gemma consensus**, but significantly reduces consensus erosion for Llama3.1 and Qwen (Qwen p < 0.001, Llama3.1 p = 0.014). When Llama3.1 and Qwen agents can see the current majority opinion, they conform more and the aggregate consensus stabilizes or drifts upward. Minitaur agents apparently do not substantially adjust their behavior based on this information.

Similarly, the BERT detectability uplift from survey context is almost entirely absent for Gemma (99.8% vs 99.1%), and is driven by the non-Gemma models (98.1% vs 93.6%).

### 3.4 Gemma Produces Near-Perfectly Detectable AI Text

The most striking BERT result is model-specific:

| Model | BERT accuracy | n | Range |
|---|---|---|---|
| qwen_base | **1.000 ± 0.001** | 72 | [0.994, 1.000] |
| Gemma | **0.995 ± 0.008** | 92 | [0.954, 1.000] |
| Minitaur | 0.954 ± 0.038 | 169 | [0.839, 1.000] |
| Qwen | 0.953 ± 0.035 | 151 | [0.849, 1.000] |
| Llama3.1 | 0.947 ± 0.048 | 111 | [0.779, 1.000] |

qwen_base now tops the ranking at near-perfect accuracy (1.000) — without social media persona fine-tuning, the base model writes in a trivially AI-identifiable style (see Section 3.10). Among LoRA-equipped models, Gemma-generated threads are classified with near-ceiling accuracy across all 92 runs, with minimal variance. The eval thread counts are comparable across models (Gemma: 145.0 vs others: ~146–150), ruling out a sample-size explanation. The other three LoRA models produce text that is harder to distinguish (BERT accuracy ~95%, with substantial run-to-run variance). With the expanded dataset, Gemma vs each other LoRA model: d = 1.30–1.49, all *p* < 0.001. Overall BERT accuracy: mean = 96.4%, range [77.9%, 100%].

### 3.5 Proportions Option Interacts Strongly with Model

With the full multi-model dataset, the previously observed "average proportions cause strongest consensus decrease" finding no longer holds as a clean main effect (average vs uniform: *p* = 0.29, ns). The effect is better characterized as an interaction:

| Proportions | Minitaur | Llama3.1 | Qwen | Gemma |
|---|---|---|---|---|
| uniform | −0.097 | −0.050 | −0.036 | +0.016 |
| blueprint | −0.101 | +0.030 | +0.007 | +0.067 |
| average | **−0.126** | +0.045 | −0.063 | −0.010 |
| distribution | −0.045 | **−0.180** | **−0.129** | −0.042 |

The `average` option (optimized convex combination weights) produces the strongest erosion for Minitaur, while `distribution` produces the strongest for Llama3.1 and Qwen. No single proportions scheme consistently dominates across models. The `blueprint` option (demographic frequencies) tends to produce mild or positive drift for non-Minitaur models.

### 3.6 Echo Chamber Dynamics: Clustering Dissolves Over Simulation Time

Echo chamber metrics are now tracked across all 11 survey steps, allowing temporal analysis. Key trajectories (all 354 runs, Δ = last − first):

| Metric | First survey step | Final survey step | Δ | *p* (vs zero) |
|---|---|---|---|---|
| Assortativity | 0.022 ± 0.082 | 0.001 ± 0.042 | **−0.021** | *p* < 0.001 *** |
| Local agreement | 0.782 ± 0.174 | 0.723 ± 0.170 | **−0.059** | *p* < 0.001 *** |
| Cross-cutting edges | 0.218 ± 0.176 | 0.277 ± 0.170 | **+0.059** | *p* < 0.001 *** |
| Same-option exposure | 0.000 (init.) | 0.724 ± 0.170 | — | — |

Opinion-based network clustering (assortativity) starts slightly positive and decays to near-zero by the end of the simulation, indicating that whatever initial clustering exists is dismantled over time rather than reinforced. This is directly contrary to the echo chamber hypothesis. Local agreement and cross-cutting edges move in tandem, reflecting the same progressive opinion mixing.

**Same-option exposure** is zero at initialization (before any interactions) and grows to a final value of 0.69. This initialization artifact makes the start-to-end comparison uninformative as a measure of echo formation; the final value (used in Section 3.6 cross-sectional analysis) remains a valid measure of informationally homogeneous neighborhoods.

**Homophily accelerates initial clustering but not final state:** Homophilic networks begin with substantially higher assortativity (0.055 vs −0.007 for non-homophilic), and their assortativity drops much more steeply:

| | homophily=True | homophily=False | *p* | Cohen's *d* |
|---|---|---|---|---|
| Δ assortativity | **−0.046 ± 0.107** | **+0.002 ± 0.037** | < 0.001 *** | −0.622 |

Homophilic networks start with opinion clustering (initial assortativity 0.052 vs −0.006) and then dissolve it rapidly during the simulation; non-homophilic networks show negligible change. Graph type has no significant effect on local agreement, cross-cutting, or same-option-exposure deltas (*p* > 0.22), but does reach significance for Δ assortativity (random: −0.030 vs powerlaw: −0.013; *p* = 0.016*) — powerlaw topology partially dampens the assortativity collapse.

**Link to consensus dynamics:** Δ local agreement correlates strongly with consensus change (r = +0.904, p < 0.001) — local opinion homogeneity and global consensus erode in exact lockstep. Δ assortativity shows a weak negative correlation with consensus change (r = −0.132, p = 0.001**): greater assortativity loss is associated with larger consensus erosion.

Cross-sectional distributions are shifted upward in local agreement and same-option exposure relative to earlier analyses, because qwen_base runs (which barely change opinions) dominate the high end of both metrics. Restricting to LoRA-equipped models recovers more typical values. Cross-sectional graph-type effects remain non-significant:

- **Local agreement** by graph_type: powerlaw 0.726 vs random 0.720 (*p* = 0.66, ns)
- **Cross-cutting edges** by graph_type: random 0.280 vs powerlaw 0.275 (*p* = 0.70, ns)
- **Final assortativity** by homophily: 0.006 vs −0.004 (*p* = 0.004 **, small effect)

Same-option exposure correlates significantly with final consensus (r = +0.46, p < 0.001), somewhat weaker than before because qwen_base compresses the range: agents in more informationally homogeneous environments show less consensus erosion.

### 3.7 Neighbor Alignment Shift Rate (NASR)

NASR measures the fraction of agents who, when they change their opinion, do so in the direction of their immediate network neighbors' majority. It is fully populated across all 595 runs (mean = 0.066 ± 0.041).

**NASR tracks overall volatility.** NASR correlates strongly with opinion shift rate (r = +0.943, p < 0.001), confirming it captures the same global volatility signal. The NASR/OSR ratio (the fraction of opinion changers who align with their local neighbor majority) is stable at roughly 0.36–0.41 across LoRA-equipped models, suggesting the local-conformity mechanism is consistent regardless of how frequently opinions change.

**Model ranking mirrors inverse consensus erosion.** Higher NASR means more agents conform to their local neighbors when changing opinions, which stabilizes rather than erodes aggregate consensus. The model ordering by NASR exactly inverts the consensus-erosion ranking:

| Model | Mean NASR | NASR/OSR ratio |
|---|---|---|
| Gemma | **0.088 ± 0.036** | 0.39 |
| Qwen | 0.078 ± 0.039 | 0.38 |
| Llama-3.1-8B | 0.074 ± 0.035 | 0.41 |
| Minitaur | **0.057 ± 0.034** | 0.36 |

All pairwise differences are statistically significant except Llama3.1 vs Qwen (*p* = 0.41, ns). Effect sizes are more moderate with the expanded dataset: Gemma–Minitaur d = 0.89, Gemma–Llama3.1 d = 0.40, Gemma–Qwen d = 0.27 (*p* = 0.035*).

**NASR decreases with population size** (r = −0.36, p < 0.001), mirroring the OSR pattern.

**Survey context effect.** Survey context globally increases NASR (ctx=True: 0.072 vs ctx=False: 0.059; *p* < 0.001, *d* = 0.32). With the larger dataset, this effect is now also detectable within individual models: minitaur (*p* = 0.034*), llama3.1 (*p* = 0.031*), and gemma (*p* = 0.042*) all show a significant within-model ctx uplift; qwen does not reach significance (*p* = 0.26). The global effect is therefore not purely a model-distribution confound as previously reported.

### 3.8 News Agents Have No Measurable Effect

The null result holds with the larger dataset. Presence vs absence of a news agent has no significant effect on consensus change (*p* = 0.62), opinion shift rate (*p* = 0.23), or BERT detectability (*p* = 0.34). A single news agent among hundreds does not measurably alter simulation dynamics.

### 3.9 Parameter Interactions: Mostly Additive, with One Critical Non-Linearity

To assess whether parameters combine additively (knowing A increases X and B increases X implies A+B increases X more than either alone), we computed 2×2 interaction contrasts for all key binary parameter pairs and examined whether the effect of each parameter is consistent across levels of the others.

**Variance explained by each parameter individually** (η², single-factor):

| Metric | Dominant factor | η² | Secondary |
|---|---|---|---|
| opinion_shift_rate | model | **0.200** | num_agents 0.230 (near-tied) |
| net_consensus_change | model | **0.090** | prop_opt 0.025, ctx 0.028 |
| BERT accuracy | model ≈ ctx | **0.266 / 0.264** | question 0.024 |
| Δ assortativity | homophily | **0.088** | num_agents 0.037 |

Note: with qwen_base added, the η² distribution shifts substantially. qwen_base's near-zero OSR and near-100% BERT accuracy means `model` now explains as much variance as `num_agents` in OSR, and nearly as much as `ctx` in BERT accuracy.

The sum of all individual η² values falls well below 1.0 for each metric, reflecting both residual noise and potential interactions.

#### Where the space is approximately additive

**num_agents × model → OSR.** The negative scaling of OSR with population size holds consistently across all four models (r = −0.38 to −0.71, all *p* < 0.001). The direction and approximate magnitude of each model's OSR offset are preserved at every population size; no meaningful interaction.

**homophily → Δ assortativity.** The homophily effect is significant within every model and at every population size tested (all *p* < 0.02, consistent direction). The magnitude varies somewhat, but there is no reversal or suppression.

**ctx + news → BERT accuracy.** The interaction contrast is essentially zero (IC = −0.006): survey context and news agents contribute independently to detection accuracy.

#### Where it is not additive

**1. Model × survey_context → consensus change (strongest non-linearity)**

The survey context benefit on consensus change is entirely concentrated in two models:

| Model | ctx=True | ctx=False | Δ | *p* |
|---|---|---|---|---|
| Qwen | −0.004 | −0.107 | **+0.103** | *p* < 0.001 *** |
| Llama-3.1-8B | +0.015 | −0.067 | **+0.082** | *p* = 0.014 * |
| Minitaur | −0.095 | −0.095 | +0.000 | *p* = 0.99 (ns) |
| Gemma | +0.042 | +0.002 | +0.040 | *p* = 0.11 (ns) |

For Minitaur and Gemma, survey context has no significant effect on consensus. For Qwen and Llama3.1 it is a meaningful driver (Qwen: +0.10, Llama3.1: +0.08), though with the larger dataset Llama3.1's effect is weaker than previously estimated.

**2. Survey context × num_agents → majority_follow_rate**

The ctx effect on herding is absent at small populations and grows with scale:

| num_agents | ctx effect on MFR | *p* |
|---|---|---|
| 64 | −0.020 | *p* = 0.36 (ns) |
| 256 | +0.024 | *p* = 0.25 (ns) |
| 1024 | +0.075 | *p* < 0.001 *** |
| 4096 | +0.064 | *p* = 0.001 *** |

At 64 and 256 agents, providing the survey in context has no significant effect on whether agents follow the majority. At 1024+ agents, the effect is substantial and robust. The two parameters are synergistic: the survey-context herding effect only emerges at larger population sizes.

**3. Homophily × graph_type → Δ assortativity**

Powerlaw topology affects assortativity change only when homophily is also active:

| | homophily=False | homophily=True |
|---|---|---|
| random graph | +0.002 | −0.067 |
| powerlaw graph | +0.002 | −0.029 |

The interaction contrast is +0.038: powerlaw graphs partially dampen the homophily-driven assortativity collapse (−0.029 vs −0.067), but this effect exists only in the homophilic condition. In non-homophilic networks, graph topology has near-zero effect on assortativity dynamics. The marginal significance of the graph type main effect on Δ assortativity (p = 0.016*) is entirely attributable to this interaction.

#### Summary of additivity

The parameter space is approximately linear for the dominant effects (OSR scaling with agents, homophily → assortativity, model-level differences). The main exception is survey context, whose effect on consensus change is gated by model identity and whose effect on herding is gated by population size. Predicting the combined effect of ctx + model (or ctx + num_agents) from their marginal effects in isolation would give systematically wrong answers for these metrics.

### 3.10 Ablation: Persona LoRA Fine-tuning is the Core Driver of Social Dynamics

The `qwen_base` condition — Qwen2.5-7B-Instruct with no persona LoRA adapters, run against the same parameter grid as the LoRA-equipped models — provides a direct ablation of the social media fine-tuning. The contrast with `qwen` (same base architecture, with LoRAs) isolates what the fine-tuning contributes.

#### Opinion dynamics collapse without LoRAs

| Metric | qwen (LoRA) | qwen_base (no LoRA) | Cohen's *d* | *p* |
|---|---|---|---|---|
| opinion_shift_rate | 0.210 ± 0.108 | **0.057 ± 0.097** | 1.46 | < 0.001 *** |
| majority_follow_rate | 0.505 ± 0.064 | **0.275 ± 0.236** | 1.60 | < 0.001 *** |
| NASR | 0.078 ± 0.039 | **0.021 ± 0.036** | 1.52 | < 0.001 *** |
| net_consensus_change | −0.055 ± 0.161 | **+0.004 ± 0.048** | 0.44 | < 0.001 *** |

Without persona LoRAs, agents change opinions at one quarter the rate (OSR 0.057 vs 0.210), follow the majority half as often (MFR 0.275 vs 0.505), and produce essentially zero net change in group consensus. The simulation runs, but the social dynamics are nearly absent.

#### Initial consensus: LoRAs create opinion diversity

qwen_base starts every simulation at near-perfect consensus (init = 1.000 for Q25 and Q28, 0.907 for Q29). The base model, without persona conditioning, produces the same "default" opinion on each topic from every agent. The persona LoRAs are what introduce the heterogeneity of starting positions that makes disagreement, persuasion, and consensus change possible. Correspondingly, qwen_base's near-zero NCC (+0.004) reflects a floor effect: you cannot erode consensus that was never present.

#### BERT detectability: LoRAs make AI text look human

qwen_base achieves near-perfect BERT detection accuracy (0.9999 ± 0.0008) — even higher than Gemma (0.995). Without the social media fine-tuning, the model writes in a formulaic, characteristically "AI" style that is trivially distinguishable from real social media text. The LoRA-equipped qwen achieves 0.953, meaning the social media fine-tuning closes most (but not all) of the style gap.

#### Echo chamber structure: static without LoRAs

Without opinion change, network structure is preserved rather than mixed. qwen_base has dramatically higher local agreement (0.960 vs 0.722, d = 1.74) and lower cross-cutting edge fraction (0.040 vs 0.282, d = 1.74). This is not evidence of echo chamber *formation* — it reflects the absence of the opinion dynamics that would otherwise disrupt initial clustering. Correspondingly, Δ assortativity for qwen_base is essentially zero (−0.0004) and the homophily effect on Δ assortativity disappears entirely (p = 0.83, ns vs p = 0.006 for qwen with LoRAs).

#### Parameter effects vanish without LoRAs

Every simulation parameter effect that was significant for LoRA-equipped models is absent or much weaker for qwen_base:

- **Survey context → consensus change**: qwen shows +0.103 uplift (p < 0.001), qwen_base shows +0.000 (p = 0.97, ns)
- **Survey context → majority_follow_rate**: qwen p = 0.047*, qwen_base p = 0.22 (ns)
- **num_agents → OSR**: qwen r = −0.42 (p < 0.001), qwen_base r = −0.18 (p = 0.12, ns)
- **Homophily → Δ assortativity**: qwen p = 0.006**, qwen_base p = 0.83 (ns)

The persona LoRA fine-tuning is not merely a stylistic adjustment — it is the mechanism through which simulation parameters have any effect at all. Without it, the agents are effectively identical, inert, and unresponsive to context.

---

## 4. Summary Table

| Finding | Effect size | Significance | Notes |
|---|---|---|---|
| Minitaur vs Gemma → consensus change | Cohen's *d* = 1.03 | *p* < 0.001 *** | Minitaur: −0.095; Gemma: +0.025 |
| Minitaur vs Llama3.1 → consensus change | Cohen's *d* = 0.48 | *p* < 0.001 *** | |
| Qwen vs Gemma → consensus change | Cohen's *d* = 0.55 | *p* < 0.001 *** | |
| Llama3.1 vs Gemma → consensus change | Cohen's *d* = 0.33 | *p* = 0.015 * | Newly significant with n=111/92 |
| Gemma → BERT accuracy (vs LoRA models) | *d* = 1.30–1.49 | *p* < 0.001 *** | 99.5% vs ~95% for LoRA models |
| survey_context → majority follow rate | Cohen's *d* = 0.27 | *p* = 0.001 *** | Effect at 1024+ agents only |
| survey_context → BERT accuracy | Cohen's *d* = 1.20 | *p* < 10⁻¹⁵ *** | Near-zero effect for Gemma |
| num_agents ↑ → opinion_shift_rate ↓ | r = −0.42 | *p* < 0.001 *** | Holds within every LoRA model |
| num_agents ↑ → consensus change | r = +0.05 | *p* = 0.22 (ns) | No effect on final outcome |
| homophily → final assortativity | *t* = 2.85 | *p* = 0.004 ** | Effect very small (Δ = 0.010) |
| homophily → Δ assortativity (time series) | Cohen's *d* = −0.622 | *p* < 0.001 *** | True: −0.046, False: +0.002 |
| graph_type → Δ assortativity | *t* = 2.41 | *p* = 0.016 * | Only in homophilic runs; other echo metrics ns |
| num_news_agents → any metric | various | ns (*p* > 0.23) | **No effect** |
| same-option exposure ↔ consensus | r = +0.46 | *p* < 0.001 *** | Weakened by qwen_base range compression |
| Δ local agreement ↔ consensus change | r = +0.904 | *p* < 0.001 *** | Local and global opinion erosion in lockstep |
| NASR: Gemma vs Minitaur | Cohen's *d* = 0.89 | *p* < 0.001 *** | 0.088 vs 0.057 |
| NASR ↔ opinion shift rate | r = +0.943 | *p* < 0.001 *** | NASR tracks same volatility signal |
| NASR ↓ with num_agents | r = −0.36 | *p* < 0.001 *** | Mirrors OSR scaling |
| **LoRA ablation (qwen_base vs qwen)** | | | |
| qwen_base vs qwen → OSR | Cohen's *d* = 1.46 | *p* < 0.001 *** | 0.057 vs 0.210; dynamics collapse without LoRAs |
| qwen_base vs qwen → MFR | Cohen's *d* = 1.60 | *p* < 0.001 *** | 0.275 vs 0.505; herding halved |
| qwen_base vs qwen → BERT accuracy | Cohen's *d* = 1.62 | *p* < 0.001 *** | 0.9999 vs 0.953; trivially detectable without LoRAs |
| qwen_base initial consensus | init ≈ 1.000 on 2/3 questions | — | All agents agree by default without persona diversity |
| **Interactions** | | | |
| model × ctx → consensus change | IC = ±0.128 | *p* < 0.001 *** (Llama/Qwen only) | Non-additive: ctx only helps 2 of 4 models |
| ctx × num_agents → majority_follow_rate | IC grows with scale | ns at 64, *p* < 0.001 at 1024+ | Synergistic: herding effect emerges at scale |
| homophily × graph_type → Δ assortativity | IC = +0.038 | Powerlaw dampens homophily effect | Only matters when homophily is active |
| num_agents × model → OSR | consistent r = −0.38 to −0.71 | All *p* < 0.001 | **Additive**: effect holds across all models |

---

## 5. Limitations and Methodological Notes

1. **Model imbalance and confounds:** Minitaur (n=131), Qwen (n=105), Llama3.1 (n=78), Gemma (n=40). The Gemma condition is the smallest and also has no 4096-agent runs, limiting some comparisons. Model is confounded with question-specific initial consensus values (e.g., Qwen on Q28 starts at 0.993, Gemma on Q29 at 0.562), which may affect within-question comparisons.
2. **NASR interpretation:** `neighbor_alignment_shift_rate` is now fully populated in all 354 runs. Because NASR correlates strongly with OSR (r = 0.924), it may not provide genuinely independent information beyond overall volatility; the NASR/OSR ratio (fraction of changers that follow local majority) is the more distinct quantity.
3. **BERT accuracy interpretation:** The BERT classifier is trained and evaluated within each run's own thread file (train/eval split). Accuracy reflects in-distribution separability, not a cross-run held-out assessment.
4. **Statistical tests are unadjusted** for multiple comparisons. With ~30+ tests performed, several nominally significant results at *p* < 0.05 may be false positives; treat marginal results (*p* < 0.05) with caution.
5. **Same-option exposure initialization artifact:** `mean_same_option_exposure_share` is exactly zero at the first survey step (before any interactions), so the start-to-end Δ of +0.69 reflects initialization rather than echo chamber formation. The final value is used for cross-sectional analysis; temporal comparisons for this metric are not interpretable.
6. **p-values use a normal approximation** (via the error function) rather than a t-distribution CDF. This is accurate for moderate-to-large samples but slightly off for small subgroups.
7. **`bert_n` confound in the survey-context BERT finding:** `add_survey_to_context=True` runs have fewer eval threads on average (ctx=True: ~136 threads vs ctx=False: ~154), and fewer threads correlates with higher accuracy (r = −0.55, p < 0.001). The direction of the effect is unambiguous, but the magnitude may be partially inflated by this imbalance.
