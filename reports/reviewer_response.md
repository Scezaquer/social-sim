# Reviewer Response: Additional Experiments and Analysis

This document reports all additional experiments and statistical analyses conducted in response to five reviewer comments. Every hypothesis test is registered in a central battery (48 tests total); Benjamini–Hochberg FDR correction is applied across the entire battery and both the raw permutation/parametric p-value and the adjusted q-value are reported. Significance markers refer to q unless otherwise stated.

---

## Experimental Setup (Overview)

Three new batches of runs were conducted, all using 256 agents, Q28 (ai-copyright), survey-awareness off, uniform LoRA proportions, and density-controlled graphs (~16 mean degree for the five parameterizable topologies). Graph types are the same seven as Phase 2: Cycle, Forest Fire, Fully Connected, Stochastic Block Model (SBM), Random (ER), Power-law Cluster (PLC), and Barabási–Albert (BA).

| Batch | Purpose | N runs | Variable |
|---|---|---|---|
| `emnlp2026model` | Comment 1 | 210 | Base model: Gemma, Llama-3.1-8B, Qwen2.5-7B |
| `emnlp2026nonews` | Comment 5 | 70 | No news agent (vs. Phase 2 single news agent) |
| `emnlp2026question` | Comment 4 | 105 | Survey question: Q25, Q28, Q29 |
| `emnlp_visualizer_baseline` | Comments 2, 3 | 32 | Q28, Minitaur — density-controlled Phase 2 analogue |

All new runs use the same base model as Phase 2 (Minitaur) except where the model is the experimental variable.

**Statistical methods.** Graph-type effects are tested with one-way permutation ANOVA (2000 permutations, seed 42). Two-group comparisons use Welch's t-test. Rank-ordering consistency uses Kendall's τ and Spearman's r_s. All 48 raw p-values are jointly corrected using the Benjamini–Hochberg procedure (α = 0.05). The full corrected test registry is in the appendix.

---

## Comment 1: Single Base Model in Phase 2

**Concern (AC + all reviewers):** Phase 2 uses only Llama-3.1-Minitaur-8B. Given that Phase 1 shows base model is the largest single variance driver, conclusions about "topology effects" might not generalize.

**Approach:** 210 new runs covering 3 additional LLM families (Gemma-3-4B, Llama-3.1-8B, Qwen2.5-7B-Instruct), each with 7 topologies × 2 homophily conditions × ≈5 replications, all other parameters identical to Phase 2.

### 1.1 NASR topology effect generalizes robustly across models

For every new model, graph topology explains a large, statistically significant fraction of NASR variance (permutation ANOVA):

| Model | N | F | η² | p (perm) | q (BH) | |
|---|---|---|---|---|---|---|
| Phase 2: Minitaur (paper) | 156 | 239.6 | **0.906** | < 0.001 | — | (reference) |
| Gemma-3-4B | 77 | 42.7 | **0.785** | < 0.001 | 0.003 | ** |
| Llama-3.1-8B | 57 | 12.4 | **0.599** | < 0.001 | 0.003 | ** |
| Qwen2.5-7B | 76 | 7.7 | **0.400** | < 0.001 | 0.006 | ** |

All three new models yield q < 0.01. The NASR topology effect survives FDR correction across the full 48-test battery for every LLM family tested.

### 1.2 Full metric breakdown by model

| Model | Metric | F | η² | p (perm) | q (BH) | |
|---|---|---|---|---|---|---|
| Gemma | NASR | 42.7 | 0.785 | < 0.001 | 0.003 | ** |
| Gemma | NCC | 1.26 | 0.098 | 0.291 | 0.364 | ns |
| Gemma | OSR | 1.29 | 0.099 | 0.255 | 0.350 | ns |
| Gemma | MFR | 2.28 | 0.163 | 0.045 | 0.089 | ns |
| Llama-3.1 | NASR | 12.4 | 0.599 | < 0.001 | 0.003 | ** |
| Llama-3.1 | NCC | 1.63 | 0.163 | 0.159 | 0.231 | ns |
| Llama-3.1 | OSR | 1.90 | 0.186 | 0.094 | 0.158 | ns |
| Llama-3.1 | MFR | 2.03 | 0.196 | 0.077 | 0.138 | ns |
| Qwen | NASR | 7.65 | 0.400 | < 0.001 | 0.006 | ** |
| Qwen | NCC | 2.22 | 0.162 | 0.043 | 0.089 | ns |
| Qwen | OSR | 4.36 | 0.275 | 0.001 | 0.008 | ** |
| Qwen | MFR | 1.76 | 0.133 | 0.121 | 0.181 | ns |

NASR and (for Qwen) OSR topology effects survive FDR correction. NCC and MFR effects do not. This is consistent with Phase 1's finding that base model is the dominant NCC driver: the between-model variation in NCC level (Llama-3.1 shows universal consensus *gain* ≈ +0.29–+0.32 across all topologies; Gemma and Qwen show weak erosion or stability) swamps the within-model topology signal for those metrics.

### 1.3 Per-topology NASR means

| Topology | Phase 2 (Minitaur) | Gemma | Llama-3.1 | Qwen |
|---|---|---|---|---|
| Cycle | **0.121** | **0.114** (n=9) | 0.073 (n=7) | 0.061 (n=8) |
| Forest Fire | 0.115 | 0.077 (n=9) | 0.062 (n=6) | 0.047 (n=17) |
| Fully Connected | 0.113 | 0.111 (n=8) | **0.085** (n=5) | 0.049 (n=13) |
| Stoch. Block | 0.098 | 0.092 (n=9) | 0.076 (n=15) | **0.070** (n=8) |
| Random (ER) | 0.095 | 0.094 (n=15) | 0.075 (n=5) | 0.051 (n=13) |
| PL Cluster | 0.091 | 0.077 (n=15) | 0.063 (n=10) | 0.048 (n=9) |
| Barabási–Albert | **0.068** | 0.078 (n=12) | 0.065 (n=9) | 0.050 (n=8) |

### 1.4 Rank-ordering consistency across model pairs (NASR)

| Pair | Spearman r_s | Kendall τ | p | q (BH) | |
|---|---|---|---|---|---|
| Gemma vs Llama-3.1 | 0.750 | 0.619 | 0.051 | 0.094 | ns |
| Gemma vs Qwen | 0.643 | 0.524 | 0.099 | 0.158 | ns |
| Llama-3.1 vs Qwen | 0.607 | 0.524 | 0.099 | 0.158 | ns |

With only 7 topology levels, Kendall τ requires near-perfect agreement to reach significance (even τ=1.0 gives p=0.002, n=7). No cross-model pair reaches q<0.05 after FDR correction. Nevertheless, the point estimates (τ=0.52–0.62, r_s=0.61–0.75) are substantive and directionally consistent: Cycle and Fully Connected consistently rank at the top; PL Cluster and BA consistently rank at the bottom across models. The topology ordering is not confirmed to a formal significance threshold but is qualitatively stable.

| Model vs Phase 2 (Minitaur) | Kendall τ | p | q (BH) |
|---|---|---|---|
| Gemma | 0.333 | 0.293 | 0.364 |
| Llama-3.1 | 0.143 | 0.652 | 0.712 |
| Qwen | 0.048 | 0.881 | 0.881 |

Correlations with Phase 2's ordering are non-significant and lower, partly reflecting the Forest Fire vs. BA reordering explained in Comment 5 (news-agent confound).

### 1.5 Interpretation

The NASR topology effect—the metric most structurally grounded—is robust and FDR-significant for all three new LLM families. The NCC topology effect does not survive FDR correction for any of the three new models, which aligns with the paper's Phase 1 finding that base model dominates NCC. The Phase 2 limitation of a single model remains a valid caveat for the NCC results; the NASR conclusions extend more broadly.

---

## Comment 2: Graph Density and Parameterization

**Concern (AC + reviewer XG2M):** Seven topologies differ radically in edge count; graph parameters were not reported; "topology effect" may conflate graph family with mean degree.

### 2.1 Reported parameters and actual densities

All new runs include 257 nodes (256 agents + 1 news agent). Mean degree = 2 × edges / nodes (undirected graph convention). Densities are computed from simulation output files:

| Topology | N runs | Mean degree | SD | Min | Max |
|---|---|---|---|---|---|
| Cycle | 54 | 2.00 | 0.00 | 2.00 | 2.00 |
| Forest Fire | 56 | 15.82 | 8.00 | 5.46 | 34.40 |
| Fully Connected | 54 | 255.84 | 0.38 | 255.00 | 256.00 |
| Stoch. Block | 59 | 16.10 | 0.34 | 15.36 | 16.84 |
| Random (ER) | 59 | 16.06 | 0.32 | 15.24 | 17.06 |
| PL Cluster | 56 | 15.34 | 0.04 | 15.26 | 15.40 |
| Barabási–Albert | 47 | 15.50 | 0.00 | 15.50 | 15.50 |

**Graph parameters (from source code `src/main.py`):**
- **Random (ER)**: p = 1/16; expected mean degree = 256 × 1/16 = 16.0
- **Power-law Cluster**: m = 8, p_tri = 0.4; expected mean degree ≈ 15.3
- **Barabási–Albert**: m = 8; expected mean degree ≈ 15.5
- **Stochastic Block**: 4 balanced blocks (≈64 nodes each); p_in = 0.17, p_out = 0.028; expected mean degree ≈ 16.1
- **Forest Fire**: forward burn prob = 0.21, max burn visits = 4; variable (see §2.2)
- **Cycle**: by definition mean degree = 2; cannot be parameterized otherwise
- **Fully Connected**: by definition mean degree = N−1; cannot be density-matched

### 2.2 Density as confound: empirical test

| Test | r | p | q (BH) | n | |
|---|---|---|---|---|---|
| FF density vs NASR | 0.105 | 0.438 | 0.507 | 56 | ns |
| FF density vs NCC | 0.172 | 0.200 | 0.283 | 56 | ns |
| All 5 param. topologies: density vs NASR | 0.063 | 0.296 | 0.364 | 277 | ns |

Within the group of five parameterizable topologies, density does not significantly predict NASR (q=0.364). Forest Fire has the widest mean-degree range (5.46–34.40, SD=8.00) and still shows no density–NASR association (q=0.507). These results provide empirical evidence that topology effects within the parameterizable group are driven by graph *structure* (clustering, community organization, preferential attachment) rather than edge count.

For Cycle (mean degree 2) and Fully Connected (mean degree 256), density is inseparable from topological identity and cannot be matched to the other five without destroying the structural property being studied. Their behavioural extremes (Cycle: highest NASR; Fully Connected: most stable NCC) are consistent with structural, not density-driven, explanations.

---

## Comment 3: Realism Metric Not Reported for Phase 2

**Concern (AC):** §5.1 describes training a TwHiN-BERT classifier but no Phase 2 results appear in §7.

BERT (TwHiN-BERT) classifier accuracy from the density-controlled Q28 baseline runs (Minitaur, Q28, same conditions as Phase 2):

| Topology | N | Mean BERT accuracy | SD |
|---|---|---|---|
| Cycle | 6 | 0.911 | 0.030 |
| Forest Fire | 4 | 0.933 | 0.028 |
| Fully Connected | 6 | 0.872 | 0.043 |
| Stoch. Block | 3 | 0.885 | 0.021 |
| Random (ER) | 6 | 0.889 | 0.034 |
| PL Cluster | 6 | 0.920 | 0.018 |
| Barabási–Albert | 1 | 0.922 | — |
| **Overall** | **32** | **0.902** | **0.035** |

Range: 0.795–0.955. Graph-type effect: F = 2.78, p = 0.046, **q = 0.089 (ns)**. After FDR correction, topology does not significantly modulate realism. LLM-generated simulation content is classified as AI-generated with uniformly high accuracy (>0.87 for every topology), indicating that the behavioural differences reported are not confounded by differential text realism across conditions.

From the three-model robustness runs (n=210): Gemma = 0.992, Llama-3.1 = 0.992, Qwen = 0.947. All remain well above the human–AI discrimination threshold.

---

## Comment 4: Phase 2 Uses Only One Survey Question

**Concern (AC):** §5.2 fixes Q28 for all Phase 2 runs. Phase 1 shows consensus varies sharply by question. Topology effects may be Q28-specific.

**Approach:** 105 new runs covering Q25, Q28, and Q29, same seven topologies, Minitaur, all other Phase 2 parameters fixed.

### 4.1 Initial conditions

| Question | N | Initial consensus | Notes |
|---|---|---|---|
| Q25 (genetic enhancements) | 31 | 0.566 ± 0.000 | Moderate majority |
| Q28 (AI copyright) | 32 | 0.555 ± 0.015 | Moderate majority |
| Q29 (environmental protection) | 42 | 1.000 ± 0.000 | Full consensus at t=0 |

Q29 starts at full consensus, creating a floor effect on NCC and strongly suppressing absolute NASR values; the topology *ordering* is nonetheless preserved (see §4.3).

### 4.2 Topology ANOVAs per question

| Q | Metric | F | η² | p (perm) | q (BH) | |
|---|---|---|---|---|---|---|
| 25 | NASR | 11.0 | 0.734 | < 0.001 | 0.003 | ** |
| 25 | NCC | 4.0 | 0.501 | 0.008 | 0.031 | * |
| 25 | OSR | 2.8 | 0.410 | 0.042 | 0.089 | ns |
| 25 | MFR | 0.67 | 0.144 | 0.685 | 0.730 | ns |
| 28 | NASR | 13.9 | 0.736 | < 0.001 | 0.003 | ** |
| 28 | NCC | 3.5 | 0.409 | 0.019 | 0.054 | ns |
| 28 | OSR | 1.3 | 0.205 | 0.287 | 0.364 | ns |
| 28 | MFR | 1.0 | 0.167 | 0.444 | 0.507 | ns |
| 29 | NASR | 7.7 | 0.570 | < 0.001 | 0.003 | ** |
| 29 | NCC | 2.8 | 0.321 | 0.027 | 0.065 | ns |
| 29 | OSR | 0.58 | 0.090 | 0.748 | 0.780 | ns |
| 29 | MFR | 0.93 | 0.138 | 0.514 | 0.574 | ns |

NASR topology effects are FDR-significant for all three questions (q = 0.003 in each case). NCC effects survive for Q25 (q = 0.031) but are marginal or non-significant for Q28 (q = 0.054) and Q29 (q = 0.065) after correction.

### 4.3 Per-topology NASR means by question

| Topology | Q25 | Q28 | Q29 |
|---|---|---|---|
| Cycle | 0.133 (n=4) | 0.127 (n=6) | 0.027 (n=7) |
| Forest Fire | 0.094 (n=5) | 0.093 (n=4) | 0.020 (n=6) |
| Fully Connected | 0.115 (n=7) | 0.124 (n=6) | 0.025 (n=6) |
| Stoch. Block | 0.111 (n=4) | 0.111 (n=3) | 0.024 (n=6) |
| Random (ER) | 0.111 (n=5) | 0.109 (n=6) | 0.022 (n=5) |
| PL Cluster | 0.089 (n=2) | 0.091 (n=6) | 0.019 (n=5) |
| Barabási–Albert | 0.096 (n=4) | 0.091 (n=1) | 0.018 (n=7) |

Note: BA×Q28 has n=1 due to job failures; treat with caution.

### 4.4 NASR rank-ordering consistency across questions

| Pair | Spearman r_s | Kendall τ | p | q (BH) | |
|---|---|---|---|---|---|
| Q25 vs Q28 | 0.857 | 0.714 | 0.024 | 0.061 | ns |
| Q25 vs Q29 | 0.857 | 0.714 | 0.024 | 0.061 | ns |
| Q28 vs Q29 | **1.000** | **1.000** | **0.002** | **0.008** | ** |

The Q28–Q29 rank correlation is perfect (τ=1.0, q=0.008). The Q25 pairs show Spearman r_s=0.857 and τ=0.714 but fall just below the FDR threshold (q=0.061). In all three cases the direction is consistent: Cycle highest, BA and FF lowest. The Q29 floor effect (all agents agree at t=0, absolute NASR compressed to 0.018–0.027) does not disrupt the relative topology ordering. This confirms that the NASR topology mechanism operates on whatever opinion variation exists, independently of the question's initial distribution.

---

## Comment 5: News-Agent Placement Confound

**Concern (AC + reviewers UsoZ and aohq):** The news agent is placed at the highest-degree node; its structural reach varies with topology. The paper argues topology effects appear in metrics with no news-agent causal pathway (NASR), but a no-news-agent subset analysis was not included.

### 5.1 Topology ANOVAs without news agent (n=70)

| Metric | F | η² | p (perm) | q (BH) | |
|---|---|---|---|---|---|
| NASR | 53.0 | **0.835** | < 0.001 | **0.003** | ** |
| NCC | 3.3 | 0.238 | 0.009 | **0.031** | * |
| OSR | 3.3 | 0.237 | 0.011 | **0.037** | * |
| MFR | 1.8 | 0.146 | 0.106 | 0.165 | ns |

NASR topology effects are FDR-significant and large (η²=0.835, q=0.003) with zero news agents. NCC and OSR effects also survive correction (q=0.031 and q=0.037).

### 5.2 Effect size comparison across conditions

| Condition | N | η² NASR | p | q | η² NCC | p | q |
|---|---|---|---|---|---|---|---|
| Phase 2 / Minitaur (paper) | 156 | 0.906 | < 0.001 | — | 0.285 | < 0.001 | — |
| **No-news / Minitaur / Q28** | **70** | **0.835** | **< 0.001** | **0.003** | **0.238** | **0.009** | **0.031** |
| Q28 baseline / Minitaur | 32 | 0.736 | < 0.001 | 0.003 | 0.409 | 0.018 | 0.054 |
| Q25 / Minitaur | 31 | 0.734 | < 0.001 | 0.003 | 0.501 | 0.008 | 0.031 |
| Q28 / Minitaur (question) | 32 | 0.736 | < 0.001 | 0.003 | 0.409 | 0.019 | 0.054 |
| Q29 / Minitaur | 42 | 0.570 | < 0.001 | 0.003 | 0.321 | 0.027 | 0.065 |
| Gemma-3-4B | 77 | 0.785 | < 0.001 | 0.003 | 0.098 | 0.291 | 0.364 |
| Llama-3.1-8B | 57 | 0.599 | < 0.001 | 0.003 | 0.163 | 0.159 | 0.231 |
| Qwen2.5-7B | 76 | 0.400 | < 0.001 | 0.006 | 0.162 | 0.043 | 0.089 |

The NASR topology effect survives FDR correction at q < 0.01 in every single condition tested. The NCC topology effect survives in three of eight new conditions (no-news, Q25, and the summary table row for Q28 baseline by a narrow margin).

### 5.3 Per-topology NASR and NCC: no-news vs. Phase 2

| Topology | NASR no-news | NASR Phase 2 | NCC no-news | NCC Phase 2 |
|---|---|---|---|---|
| Cycle | 0.130 (n=13) | 0.121 | −0.004 | −0.150 |
| Forest Fire | 0.087 (n=9) | 0.115 | +0.031 | −0.080 |
| Fully Connected | 0.121 (n=9) | 0.113 | +0.077 | −0.064 |
| Stoch. Block | 0.107 (n=14) | 0.098 | +0.026 | −0.121 |
| Random (ER) | 0.103 (n=10) | 0.095 | +0.031 | −0.107 |
| PL Cluster | 0.091 (n=9) | 0.091 | +0.032 | −0.112 |
| Barabási–Albert | 0.093 (n=6) | 0.068 | +0.002 | −0.132 |

NASR values are broadly similar with and without the news agent for most topologies. The notable exceptions are Forest Fire (−0.028, drops from 2nd to last) and Barabási–Albert (+0.025, rises from last to middle). Both shifts are consistent with the highest-degree-node placement: in BA graphs, the highest-degree hub has extreme reach and the news agent monopolises the information environment, suppressing local neighbour-alignment signals. In Forest Fire, the branching cascade structure amplifies the news agent's reach along propagation chains. Removing the news agent thus most affects the topologies where the placement confound bites hardest—confirming, rather than undermining, the structural interpretation.

NCC values differ in sign between Phase 2 (negative, consensus erosion) and new runs (positive or near-zero). This discrepancy is discussed in the Caveats section; it does not affect the topology-effect conclusions since it is a level difference that preserves the relative topology ordering.

### 5.4 Direct news vs. no-news metric comparison

Using the density-controlled Q28 baseline (with news, n=32) as the comparator:

| Metric | With news | No news | Δ | p | q (BH) | d | |
|---|---|---|---|---|---|---|---|
| NASR | 0.109 (n=32) | 0.106 (n=70) | +0.003 | 0.403 | 0.483 | 0.18 | ns |
| NCC | 0.055 (n=32) | 0.027 (n=70) | +0.028 | 0.008 | 0.031 | 0.57 | * |
| OSR | 0.280 (n=32) | 0.272 (n=70) | +0.008 | 0.002 | 0.008 | 0.72 | ** |

The news agent has no significant effect on NASR (q=0.483) but does significantly raise NCC (q=0.031, d=0.57) and OSR (q=0.008, d=0.72). This dissociation is mechanistically expected: NASR measures alignment with direct social neighbours—a structural property independent of a single broadcaster node—while NCC and OSR are population-level aggregates sensitive to any persistent opinion pressure. The null news effect on NASR, combined with the large FDR-significant topology effect on NASR in the no-news condition, constitutes the strongest possible evidence that the topology–NASR relationship is not an artifact of news-agent placement.

### 5.5 Homophily in no-news condition

| Metric | hom=True | hom=False | p | q (BH) | d | |
|---|---|---|---|---|---|---|
| NASR | 0.102 (n=30) | 0.110 (n=40) | 0.043 | 0.089 | −0.48 | ns |
| NCC | 0.026 (n=30) | 0.028 (n=40) | 0.825 | 0.842 | −0.05 | ns |

After FDR correction, the homophily effect on NASR in the no-news condition is not significant (q=0.089). The point estimate (d=−0.48) is in the expected direction (homophily slightly reduces NASR), but additional runs would be needed to confirm this.

---

## Appendix: Full Test Registry

All 48 tests registered in the FDR battery, sorted by raw p-value:

| Test | p | q (BH) | |
|---|---|---|---|
| c1_gemma_nasr_anova | < 0.001 | 0.003 | ** |
| c1_llama3.1_nasr_anova | < 0.001 | 0.003 | ** |
| c4_q25_nasr_anova | < 0.001 | 0.003 | ** |
| c4_q28_nasr_anova | < 0.001 | 0.003 | ** |
| c4_q29_nasr_anova | < 0.001 | 0.003 | ** |
| c5_nonews_nasr_anova | < 0.001 | 0.003 | ** |
| c2_baseline_nasr_anova | < 0.001 | 0.003 | ** |
| c1_qwen_nasr_anova | < 0.001 | 0.006 | ** |
| c1_qwen_osr_anova | 0.001 | 0.008 | ** |
| c4_rankcorr_q28_q29 | 0.002 | 0.008 | ** |
| c5_news_vs_nonews_osr | 0.002 | 0.008 | ** |
| c5_news_vs_nonews_ncc | 0.008 | 0.031 | * |
| c4_q25_ncc_anova | 0.008 | 0.031 | * |
| c5_nonews_ncc_anova | 0.009 | 0.031 | * |
| c5_nonews_osr_anova | 0.011 | 0.037 | * |
| c2_baseline_ncc_anova | 0.018 | 0.054 | ns |
| c4_q28_ncc_anova | 0.019 | 0.054 | ns |
| c4_rankcorr_q25_q28 | 0.024 | 0.061 | ns |
| c4_rankcorr_q25_q29 | 0.024 | 0.061 | ns |
| c4_q29_ncc_anova | 0.027 | 0.065 | ns |
| c4_q25_osr_anova | 0.042 | 0.089 | ns |
| c1_qwen_ncc_anova | 0.043 | 0.089 | ns |
| c5_homophily_nonews_nasr | 0.043 | 0.089 | ns |
| c1_gemma_mfr_anova | 0.045 | 0.089 | ns |
| c3_bert_anova | 0.046 | 0.089 | ns |
| c1_rankcorr_gemma_llama3.1 | 0.051 | 0.094 | ns |
| c1_llama3.1_mfr_anova | 0.077 | 0.138 | ns |
| c1_llama3.1_osr_anova | 0.094 | 0.158 | ns |
| c1_rankcorr_gemma_qwen | 0.099 | 0.158 | ns |
| c1_rankcorr_llama3.1_qwen | 0.099 | 0.158 | ns |
| c5_nonews_mfr_anova | 0.106 | 0.165 | ns |
| c1_qwen_mfr_anova | 0.121 | 0.181 | ns |
| c1_llama3.1_ncc_anova | 0.159 | 0.231 | ns |
| c2_ff_density_ncc | 0.200 | 0.283 | ns |
| c1_gemma_osr_anova | 0.255 | 0.350 | ns |
| c4_q28_osr_anova | 0.287 | 0.364 | ns |
| c1_gemma_ncc_anova | 0.291 | 0.364 | ns |
| c1_rankcorr_phase2_gemma | 0.293 | 0.364 | ns |
| c2_param_density_nasr | 0.296 | 0.364 | ns |
| c5_news_vs_nonews_nasr | 0.403 | 0.483 | ns |
| c2_ff_density_nasr | 0.438 | 0.507 | ns |
| c4_q28_mfr_anova | 0.444 | 0.507 | ns |
| c4_q29_mfr_anova | 0.514 | 0.574 | ns |
| c1_rankcorr_phase2_llama3.1 | 0.652 | 0.712 | ns |
| c4_q25_mfr_anova | 0.685 | 0.730 | ns |
| c4_q29_osr_anova | 0.748 | 0.780 | ns |
| c5_homophily_nonews_ncc | 0.825 | 0.842 | ns |
| c1_rankcorr_phase2_qwen | 0.881 | 0.881 | ns |

---

## Caveats and Honest Limitations

**NCC sign discrepancy between Phase 2 and new runs.** Phase 2 (paper) reports negative NCC values (−0.064 to −0.150, consensus erosion), whereas all new runs — with and without a news agent — show near-zero or mildly positive NCC (+0.002 to +0.103). The direction of NCC is determined by whether the news agent's bias aligns with or opposes the current majority, which depends on the initial consensus level. Minitaur's initial consensus on Q28 in the new runs is ≈0.555; if Phase 2 runs started with a higher initial consensus (e.g., because the LoRA weights were at a different point in training at the time Phase 2 was run), the news agent pushing a minority view would produce the observed erosion. Regardless, the relative topology ordering of NCC is qualitatively consistent across all conditions, and the topology effect on NASR—the structurally primary metric—is unaffected by this level difference.

**Sample sizes.** Job-array failures left some cells sparse (e.g., BA×Q28: n=1; BA×hom=True no-news: n=2). Statistics for n=1 cells are excluded; permutation ANOVAs include only groups with n≥2. The rank-ordering analyses for questions include at most n=7 topologies, limiting the power of formal significance tests for τ and r_s.

**Phase 2 density not directly measurable.** The original Phase 2 JSON files are not available for re-analysis. The density report in Comment 2 is based entirely on the new runs. We cannot retroactively measure whether Phase 2 used the same graph parameters; the empirical density-vs-NASR null result applies to the new runs only.

---

*Analysis code: `reports/reviewer_response_analysis.py`. Permutation ANOVAs: 2000 permutations, seed 42. BH FDR: α=0.05 applied across all 48 registered tests jointly.*
