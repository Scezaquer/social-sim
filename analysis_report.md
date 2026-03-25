# Analysis of Randomized LLM-Based Social Simulation Experiments

To update:
Ignore all runs with num_agents not in {64, 256, 1024, 4096}. There are now more base_model, include the new ones in the analysis. See if you can find other interesting things with them. Update this report. You see that num_agents dominates, double check that this is an actual real result, and not just metrics being bigger when there are more agents because they aren't normalized.

**Dataset:** 159 completed simulation runs across three questions (Q25: genetic enhancements, Q28: AI copyright, Q29: environmental protection), varying seven independent parameters.

---

## 1. Dataset Overview

| Parameter | Values observed | Notes |
|---|---|---|
| `base_model` | Minitaur (n=133), Llama-3.1-8B (n=26) | Highly imbalanced; Llama-3.1-8B is the base (no persona fine-tuning) |
| `num_agents` | 50, 64, 256, 500, 1000, 1024, 1500, 4096 | Values 50/500/1000/1500 not in original parameter list; all are blueprint proportions |
| `graph_type` | random (n=86), powerlaw_cluster (n=73) | |
| `homophily` | True (n=83), False (n=76) | |
| `add_survey_to_context` | True (n=86), False (n=73) | |
| `proportions_option` | blueprint (n=80), uniform (n=41), distribution (n=20), average (n=18) | |
| `num_news_agents` | 1 (n=80), 0 (n=79) | |

**Note:** ~37 runs used non-standard `num_agents` values (50, 500, 1000, 1500) not present in the job script's `NUM_AGENTS_CHOICES=(64 256 1024 4096)`. All of these are Minitaur runs with `blueprint` proportions, suggesting an earlier version of the job script was used.

All runs produce a roughly constant number of threads (~780, sd≈13) independent of agent count, confirming the simulation is time-bounded rather than agent-count-bounded.

---

## 2. ANOVA Summary (stratified by timestep)

The table below shows eta² (fraction of within-timestep variance explained by each parameter) for the most informative metrics, averaged across questions. Full output in `anova_randomized_q{25,28,29}.csv`.

| Metric | Top predictor (η²) | 2nd predictor (η²) | 3rd predictor (η²) |
|---|---|---|---|
| `vote_pct::Yes/No` | num_agents (0.39–0.43) | base_model (0.28–0.35) | add_survey_to_context (0.002–0.24) |
| `opinion_shift_rate` | num_agents (0.17–0.72) | base_model/proportions (0.06–0.32) | add_survey_to_context (0.01–0.25) |
| `majority_follow` | num_agents (0.16–0.28) | proportions/base_model (0.05–0.17) | add_survey_to_context (0.03–0.07) |
| `consensus` | proportions (0.04–0.10) | num_agents (0.04–0.06) | graph_type/homophily (0.03) |
| `neighbor_alignment_shift_rate` | proportions/num_agents (0.52–0.86) | add_survey_to_context (0.26–0.50) | graph_type/homophily (0.14–0.38) |

`homophily` and `graph_type` are generally weak predictors for opinion metrics (η² < 0.08 consistently), while `num_agents`, `base_model`, and `add_survey_to_context` dominate.

---

## 3. Key Findings

### 3.1 Consensus Universally Decreases — But Much More Under Fine-Tuned Personas

Across all three questions, the simulation consistently erodes opinion consensus (measured as fraction of agents holding the majority opinion):

| Question | Topic | Initial consensus (Minitaur) | Δ consensus (Minitaur) | Δ consensus (Llama-3.1-8B) |
|---|---|---|---|---|
| Q25 | Genetic enhancements | 0.861 | −0.156 ± 0.115 | −0.029 ± 0.134 |
| Q28 | AI copyright | 0.749 | −0.100 ± 0.125 | +0.039 ± 0.180 |
| Q29 | Environmental protection | 1.000 | −0.092 ± 0.071 | −0.032 ± 0.205 |

The **base model distinction is the single strongest driver of consensus trajectory** (Welch *t* = −3.47, *p* = 0.0005):

- **Minitaur** (persona fine-tuned, n=133): mean Δ = **−0.119 ± 0.111**; 87% of runs show decreased consensus.
- **Llama-3.1-8B** (base model, no persona adaptation, n=26): mean Δ = **+0.004 ± 0.175**; 57% of runs show *increased* consensus.

This is a notable divergence. The fine-tuned Minitaur personas introduce genuine opinion diversity and are willing to dissent from majority opinion, causing the simulated discourse to erode initial consensus. The untuned base Llama-3.1-8B model shows near-random drift — it lacks persona grounding and drifts toward the majority or is simply noisier.

**Caveat:** The two model conditions are heavily confounded with run count (133 vs 26), and Llama-3.1-8B is also confounded with the `proportions_option` distribution (all four options present in Minitaur; Llama-3.1-8B runs cover all options but with different counts). This should be interpreted cautiously.

### 3.2 Agent Count Strongly Suppresses Per-Agent Opinion Volatility

`num_agents` is the dominant predictor of `opinion_shift_rate` (r = −0.52, *p* < 0.001 with raw agent count; η² up to 0.72 in ANOVA for Q25). The effect is large:

| num_agents | Mean opinion shift rate (Minitaur) | n |
|---|---|---|
| 64 | 0.211 ± 0.097 | 29 |
| 256 | 0.244 ± 0.093 | 41 |
| 1024 | 0.118 ± 0.060 | 24 |
| 4096 | 0.060 ± 0.072 | 19 |

The ratio between 64-agent and 4096-agent runs is ~3.5× (t = 7.57, *p* < 0.001 for Minitaur). Larger populations produce lower per-agent opinion volatility per simulation step, consistent with the law of large numbers stabilizing opinion distributions in bigger communities. Interestingly, **total consensus does not correlate significantly with num_agents** (r = 0.03, *p* = 0.73) — larger populations churn less per step, but end up at similar final consensus levels, suggesting the overall opinion trajectory is governed by other factors.

### 3.3 Survey Context in Agent Prompts Increases Conformity AND Detectability

Adding survey results to each agent's context (`add_survey_to_context=True`) has two distinct effects:

**A. Increased majority conformity (herding):**
Majority follow rate — the fraction of opinion-changers who adopted the majority view — increases significantly with survey context:
- ctx=True: 0.517 ± 0.079 (44/86 runs show majority-follow rate > 50%)
- ctx=False: 0.472 ± 0.060 (18/73 runs show rate > 50%)
- Welch *t* = 4.11, *p* < 0.001

When agents are explicitly told what the majority thinks, they conform to it more. This is a herding effect induced by showing agents the aggregate survey results.

**B. Increased AI detectability by BERT classifier:**
This is perhaps the most surprising finding:

| Condition | BERT accuracy (LLM thread detection) | n |
|---|---|---|
| add_survey_to_context = True | **0.986 ± 0.017** | 86 |
| add_survey_to_context = False | **0.952 ± 0.026** | 73 |

Difference: +3.3 percentage points. Welch *t* = 9.36, *p* < 10⁻¹⁵.

When agents have the survey results in their context, the resulting discussion threads are significantly more detectable as AI-generated. The most likely explanation is that survey-aware agents explicitly reference consensus statistics, producing formulaic and AI-distinctive text patterns (e.g., "According to the survey, X% of agents think..."). This finding is consistent across all proportions options (the ~3 pp gap is visible in every sub-group).

Overall BERT accuracy is very high across all conditions: mean = **97.0% ± 2.7%**, range [88.7%, 100%]. LLM-generated social simulation threads are highly detectable regardless of parameter setting.

### 3.4 Persona Proportions Shape Consensus Trajectory

The weighting scheme used to mix LoRA personas significantly affects consensus dynamics:

| Proportions option | Initial consensus | Δ consensus | n |
|---|---|---|---|
| distribution | 0.845 | −0.056 ± 0.087 | 20 |
| uniform | 0.823 | −0.074 ± 0.148 | 41 |
| blueprint | 0.852 | −0.102 ± 0.128 | 80 |
| average | **0.922** | **−0.193 ± 0.103** | 18 |

`average` proportions (from optimized convex combination weights) produce the **strongest consensus decrease** (*p* = 0.0004 vs uniform, t = −3.56). The `average` runs also start at higher initial consensus (0.922 vs ~0.845 for others), providing more room to erode. The `distribution` option (demographic-distribution-matching weights) produces the most stable consensus trajectories.

These results suggest that the weighting scheme used to populate the agent population from trained personas is a meaningful design choice with measurable downstream effects on opinion dynamics.

### 3.5 Echo Chamber Formation: Weak Effects of Homophily and Graph Topology

Despite these being the canonical drivers of echo chambers in network science, both homophily and graph topology show only **weak, inconsistent effects** on echo chamber metrics.

**Network assortativity** (opinion-based sorting) is very low overall (mean = 0.003 ± 0.043, range [−0.17, 0.27]):
- homophily=True: 0.011 ± 0.048 vs homophily=False: −0.005 ± 0.035 (*p* = 0.014, barely significant)
- random vs powerlaw_cluster: 0.008 vs −0.002 (*p* = 0.27, ns)

**Local agreement** (fraction of neighbors sharing your opinion) differs only marginally:
- powerlaw graph: 0.700 ± 0.150 vs random graph: 0.651 ± 0.151 (*p* = 0.042, modest)
- homophily=True: 0.689 vs homophily=False: 0.657 (*p* = 0.19, ns)

**Cross-cutting edges** (opinion-bridging connections) are slightly higher in random graphs:
- random: 0.349 ± 0.152 vs powerlaw: 0.298 ± 0.153 (*p* = 0.034)

**Notable absence:** Homophily does not produce meaningfully stronger echo chambers in these simulations. In classical network models, homophilic attachment strongly concentrates opinion-similar nodes, but here the effect is negligible. This may be because:
1. The simulation uses a fixed graph structure that only weakly encodes homophily preferences.
2. LLM agents do not sufficiently restrict their discourse to within-group opinions.
3. The single-survey-question context limits the salience of opinion-based social sorting.

Echo chamber metrics do, however, strongly correlate with final consensus outcomes. Same-option exposure (fraction of content an agent sees from opinion-same neighbors) is strongly correlated with consensus: r = **0.65**, *p* < 0.001. Agents in more informationally homogeneous environments are more likely to converge, not less. This is counterintuitive — higher same-option exposure correlates with *higher* (less eroded) final consensus — suggesting that the echo chamber effect here is one of stabilizing existing majorities rather than radicalizing fringe positions.

### 3.6 News Agents Have No Measurable Effect

The presence or absence of a news agent (`num_news_agents` = 0 vs 1) has **no statistically significant effect** on any measured outcome:

| Metric | news=0 | news=1 | *p* |
|---|---|---|---|
| Δ consensus | −0.106 ± 0.116 | −0.092 ± 0.145 | 0.51 (ns) |
| opinion shift rate | 0.172 ± 0.103 | 0.179 ± 0.102 | ~0.68 (ns) |
| BERT accuracy | 0.969 ± 0.024 | 0.972 ± 0.030 | 0.51 (ns) |

This is a notable null result. In real social media systems, news agents typically drive agenda-setting and opinion shifts. The absence of any measurable effect here could indicate that:
- A single news agent among hundreds produces insufficient coverage to measurably alter agent behavior.
- LLM agents do not adequately respond to news-style information injections in the current prompt design.

### 3.7 Question Content Matters: Q25 Erodes Most, Q28 Most Variable

| Question | Topic | Median Δ consensus | Runs with increased consensus |
|---|---|---|---|
| Q25 | Genetic enhancements | −0.148 | 12% |
| Q28 | AI copyright | −0.059 | 34% |
| Q29 | Environmental protection | −0.084 | 8% |

Q25 (genetic enhancements) shows the most consistent consensus erosion. Q28 (AI copyright) is the most variable, with 34% of runs showing consensus *increase*. Q29 starts at near-unanimous initial consensus (Minitaur: 1.00) — essentially floor effects on opinion diversity at the start, yet still erodes. Variation across questions is significant and suggests topic-specific persona responses matter.

---

## 4. Summary Table

| Finding | Effect size | Significance | Direction |
|---|---|---|---|
| Minitaur vs Llama3.1-8B → consensus change | Cohen's d ≈ 1.00 | *p* = 0.0005 *** | Minitaur: −0.12; Llama3.1: ≈0 |
| num_agents ↑ → opinion_shift_rate ↓ | r = −0.52 | *p* < 0.001 *** | 3.5× difference (64 vs 4096) |
| survey_context → majority follow rate | Cohen's d ≈ 0.64 | *p* < 0.001 *** | +0.045 pp (ctx=True) |
| survey_context → BERT accuracy | Cohen's d ≈ 1.54 | *p* < 10⁻¹⁵ *** | +3.3 pp (ctx=True) |
| `average` vs `uniform` proportions → consensus | t = −3.56 | *p* = 0.0004 *** | −0.119 stronger erosion |
| powerlaw vs random graph → local agreement | t = 2.04 | *p* = 0.042 * | +0.049 pp |
| homophily → assortativity | t = 2.47 | *p* = 0.014 * | +0.016 (very small) |
| homophily → echo chamber strength | various | ns (p > 0.19) | **No meaningful effect** |
| num_news_agents → any metric | various | ns (p > 0.5) | **No effect** |
| same-option exposure ↔ final consensus | r = +0.65 | *p* < 0.001 *** | Higher exposure → higher consensus |

---

## 5. Limitations and Methodological Notes

1. **Model imbalance:** 133/159 runs used Minitaur; only 26 used Llama-3.1-8B. The base model comparison is confounded with other variables and should be treated carefully.
2. **Non-standard num_agents values** (50, 500, 1000, 1500) appear in 37 runs from earlier job submissions. These runs are included in analyses but flag that the sampling is not perfectly balanced across the intended 4-level design.
3. **NASR metric sparsity:** `neighbor_alignment_shift_rate` is only populated in 26/159 runs (a newer metric addition). The large ANOVA η² values for this metric should be interpreted with caution given the limited sample.
4. **BERT accuracy interpretation:** The BERT classifier was trained partly on these same files (train/eval split per file). Accuracy reflects within-run consistency between training and evaluation splits, not a truly held-out assessment.
5. **Statistical tests are unadjusted** for multiple comparisons. With ~30+ tests performed, several nominally significant results at p<0.05 may be false positives.
6. **Echo chamber metrics are computed at a single survey timepoint**, limiting temporal analysis of echo chamber dynamics.
7. **p-values use a normal approximation** (via the error function) rather than a t-distribution CDF. This is accurate for moderate-to-large samples but slightly off for very small groups (e.g., the Q25 Llama-3.1-8B comparison with n=6).
8. **`bert_n` confound in the survey-context BERT finding:** `add_survey_to_context=True` runs have fewer eval threads on average (136 vs 154), and fewer eval threads correlates with higher accuracy (r=−0.55, p<0.001). The direction of the effect is unambiguous, but the magnitude (+3.3 pp) may be partially inflated by this imbalance.
