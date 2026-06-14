# Silicon Society Cookbook — V2 Plan

Synthesis of the four COLM reviews (all scored 4–5, "marginally below"), ordered from most to
least important by (a) how many reviewers raised the issue and (b) how much weight they put on
it (final comments after rebuttal weigh heaviest, since those are the reasons scores did not
move). Since V1 was not accepted and the V1 data is lost, **all experiments below produce new
data and the V2 paper must be self-contained** — no design decision may be justified by
pointing at V1 results.

---

## Priority 1 — Realism claims: scope down AND strengthen the one realism metric
**Raised by: all 4 reviewers** (R1-W1, R2-W2, R3-W2 + final comment, R4-W1 + Q1/Q2).

The paper is *motivated* by validation/realism but the evidence is internal sensitivity
analysis. The only realism-grounded metric is a BERT classifier that is (i) trained and tested
in-domain on BluePrint, (ii) stylistic only, and (iii) still at ~95% accuracy, i.e. the text is
"basically always distinguishable" (R2).

**What changes in V2:**
1. Reframe: realism language confined to *stylistic* realism; everything survey-based is
   explicitly "internal sensitivity of simulator dynamics". The title/intro framing becomes
   "design-space characterization", not validation.
2. Strengthen the stylistic metric so it survives scrutiny:
   - Leakage-safe protocol: split human users AND simulation runs between train/test
     (`analysis/train_bert_detector.py`).
   - Out-of-domain evaluation: held-out human text from users never seen in training, and
     AI text from a simulator that is not ours (`analysis/eval_detector.py --external-ai-dir`).
   - Cross-check against an off-the-shelf open AI-text detector and report agreement
     (`analysis/eval_detector.py --crosscheck-model`).
3. Report detector AUC / accuracy with run-level bootstrap CIs instead of bare means.

## Priority 2 — Statistical methodology overhaul + balanced experimental design
**Raised by: R3-W3 (confidence-5 reviewer; explicitly kept score because "the 'model matters
most' claim still needs the proposed corrected analysis"), R4-W3 (rated 4 over it), R1 implicitly.**

V1 used single-factor η² on an unbalanced, ad-hoc randomized dataset ($RANDOM in bash, no
seeds, unequal cell counts), unadjusted pairwise t-tests, and a normal approximation.

**What changes in V2 (this is fixed at data-collection time, not just analysis time):**
1. **Pre-registered balanced design.** Runs are enumerated by a deterministic, seeded design
   generator (`bash_job_scripts/v2/generate_design.py`) that emits committed CSVs; every job
   array index maps to one fully specified, fully seeded configuration. Marginals are exactly
   balanced; pairwise cells near-balanced. No more on-node $RANDOM.
2. **Type-III ANOVA with partial η²** over main effects + all two-way interactions
   (`analysis/anova_v2.py`), so "model matters most" is claimed from variance properly
   partitioned among correlated factors.
3. **Mixed-effects models** with run batch (SLURM job) as a random effect.
4. **Benjamini–Hochberg FDR** q-values for every multi-comparison family, declared families,
   and bootstrap CIs on every reported effect size.
5. The LoRA-vs-no-LoRA contrast is **crossed with model** (see Priority 5) instead of being
   nested under "Qwen base", which is what let "model" absorb the LoRA effect in V1.

## Priority 3 — Survey-response measurement validity (noise floor, order effects, terminology)
**Raised by: R4-W2/W4/Q4 (heavy focus), R3 final comment ("belief/opinion measurements remain
difficult to interpret given prompt sensitivity and lack of persistent agent state").**

V1 itself showed option-order flips answer distributions dramatically, yet "opinion change"
was interpreted as dynamics. Agents have no persistent belief state, so "opinion dynamics" is
an overclaim.

**What changes in V2:**
1. **Dual-order surveying** (`--survey_order_mode both`, the V2 default): every survey asks
   each agent the question in canonical AND order-flipped phrasing; we record both answers,
   the per-answer log-prob margins, and a per-survey *order-consistency rate*. The flipped
   phrasings are hand-written in `flipped_questions.json` (option order reversed throughout
   the question text, not just in the answer-format suffix — necessary for Q29, where the
   order appears in the question body), and `main.py` validates them against the canonical
   question at startup. Opinion metrics are computed on canonical order; order-inconsistent
   answers quantify the prompt-sensitivity floor inside every single run.
2. **No-stimulus baseline** (`--stimulus_mode none`): agents are surveyed on the same cadence
   with no interaction at all. With greedy log-prob choice this should be exactly zero drift
   when ctx=False (a sanity check we report) and quantifies pure self-feedback drift when
   ctx=True.
3. **Scrambled-stimulus baseline** (`--stimulus_mode scrambled --scrambled_corpus ...`):
   agents observe socially meaningless threads (random messages from an unrelated corpus, no
   coupling between agents). The shift rate under scrambled stimulus is the *context-
   perturbation noise floor*; only the excess of the real simulation over this floor is
   interpreted as socially driven.
4. **Log-prob margins recorded per survey answer**, so flips can be stratified by decision
   confidence (low-margin flips ≈ noise; high-margin flips ≈ genuine context-driven change).
5. Terminology: "opinion dynamics" → **"survey-response dynamics"** throughout the paper.

## Priority 4 — Make it an actual design-space study: balance coverage, topology first-class
**Raised by: R2-W1 (their main rejection reason), R1 final comment ("the paper would benefit
from reorganizing the content, putting in more design space analyses").**

V1 promised a design-space study but spent most pages on the model and had no real topology
analysis (2 topologies, buried in an appendix).

**What changes in V2:**
1. **Controlled topology study** as a first-class experiment: 7 topologies (ER, Barabási–
   Albert, stochastic block, cycle, power-law cluster, fully connected, forest fire), all
   parameterizable ones **density-matched to mean degree ≈ 16** at every population size
   (V1's ER had mean degree = N/16, confounding topology with scale — fixed in
   `src/main.py::_build_graph`; empirical mean degree is now logged in `run_parameters` for
   every run so density can be used as a covariate). Replicated across all 4 model families
   and all 3 questions; permutation-test η² + BH-FDR.
2. **Two new design axes** that are "non-trivial simulation components" (R2-W3) at ~zero
   compute cost:
   - `--activity_exponent` ∈ {0.0, 0.5, 1.0}: the Zipf exponent of the activity distribution,
     i.e. an *opinion-leader / power-user* axis (0 = egalitarian, 1 = strongly leader-driven).
     Was hardcoded to 0.5 in V1.
   - `num_news_agents` ∈ {0, 1, 4}: V1's null result for a single news agent is upgraded to a
     dose-response test of the paper's own "signal too weak" hypothesis.
3. The results section is reorganized per-axis (one subsection per design axis), so the paper
   reads as the design-space study it claims to be.

## Priority 5 — Disentangle the base-model confound
**Raised by: R1-W2, R4-W3 (overlapping), R3 final comment.**

"Base model" bundles scale, instruction tuning, alignment, and — worst — in V1 the no-LoRA
condition existed only for Qwen, so "model" and "fine-tuning" were partially the same variable.

**What changes in V2:**
1. **Base-only (no-LoRA) arm for all four base models**, not just Qwen, giving a crossed
   4 (model) × 2 (LoRA on/off) design. This is the single most important new data for the
   headline claim.
2. The corrected ANOVA (Priority 2) then reports how much variance "model identity" retains
   once LoRA-status, agent count, etc. are partitioned out.
3. The Minitaur vs Llama-3.1-8B pair (same architecture, different post-training) is analyzed
   explicitly as the one available controlled contrast within "model identity"; remaining
   confounds (scale, tokenizer, instruction tuning) are enumerated in the limitations.

## Priority 6 — Actually answer the Average-vs-Distribution question
**Raised by: R1-W3.**

V1 motivated the contrast (match the full distribution of human opinions vs only the modal
answer) with a page of math, then never returned to it.

**What changes in V2:**
1. Proportions (uniform / blueprint / average / distribution) is exactly balanced against
   model and question in the core sweep, so the proportions × model interaction (which the V1
   rebuttal analysis suggested is the real story) is estimable with equal power in every cell.
2. A dedicated results subsection reports: main effect of proportions, proportions × model
   interaction, and the Distribution-vs-Average contrast specifically, with FDR-corrected
   q-values.
3. The SimBench-weights analysis (why Distribution and Average reach similar scores with very
   different weights) is reported alongside.

## Priority 7 — Scope and generalizability framing
**Raised by: R1-W4, R3-W1.**

Claims must be scoped to controlled simulators of this architecture; "Silicon Societies"
stays as motivation only. Each finding gets a flag: expected-to-generalize (model dominance,
non-additivity) vs implementation-tied (survey-context effect, news-agent placement). The
cross-simulator detector evaluation (Priority 1) and qualitative alignment with OASIS-style
published results partially externalize the comparison. This is mostly writing, but the
detector OOD evaluation and the topology robustness-across-models data are the supporting
experiments.

## Priority 8 — Known internal artifacts get fixed or measured
(Self-identified in V1 + touched by R1/R4.)
1. **Survey-cadence artifact**: OSR vs num_agents was an artifact of fixed survey interval. V2
   adds `--survey_interval` / `--max_steps` and a dedicated scale study with per-agent-matched
   cadence (interval ∝ N), so the scale axis is interpretable.
2. `sample_choice` crash-bug (`self._measurements` undefined) fixed.
3. Full seeding (`--seed`: python / numpy / torch) so every run is reproducible from its
   design row.

---

# Experiment plan (all new data)

| # | Experiment | Script | Runs | Agents | Addresses |
|---|-----------|--------|------|--------|-----------|
| E1 | Core balanced sweep: model(4) × proportions(4) × question(3) × N(64/256/1024) × graph(ER/PLC) × homophily(2) × ctx(2) × news(0/1/4) × activity(0/.5/1) | `mila_v2_core_sweep.sh` | 576 | 64–1024 | P2, P4, P6 |
| E2 | Base-vs-LoRA crossed arm: 4 base models, no LoRAs, balanced over the other axes at N=256 | `mila_v2_base_vs_lora.sh` | 144 | 256 | P5, P2 |
| E3 | Controlled topology study: 7 topologies × 4 models × 3 questions, density-matched, news=0 | `mila_v2_topology.sh` | 336 | 256 | P4 |
| E4 | Measurement-validity: stimulus_mode ∈ {none, scrambled} × ctx(2) × 4 models × 3 questions | `mila_v2_noise_floor.sh` | 96 | 256 | P3 |
| E5 | Scale study with per-agent-matched survey cadence: N(64/256/1024) × 2 models × 3 questions | `mila_v2_scale_cadence.sh` | 72 | 64–1024 | P8 |
| — | Detector training + OOD/cross-check eval (CPU/1 GPU, post-hoc on E1–E3 threads) | `analysis/train_bert_detector.py`, `analysis/eval_detector.py` | — | — | P1 |

Total ≈ 1,224 GPU runs (V1 was 595). If compute-constrained, cut in this order: E5 → E3
robustness models (keep Minitaur + one other) → E1 from 576 to 384 (generator flag
`--num-runs`); do **not** cut E2 or E4 — they are the direct answers to the score-blocking
reviews.

Every run uses `--survey_order_mode both` (dual-order surveying with log-prob margins) at no
meaningful extra cost; this turns *every* run into measurement-validity evidence.

**Power note (self-contained, no V1 data):** with 576 balanced runs, a two-level factor has
288 runs/level → minimum detectable effect d ≈ 0.38 at α = 0.001 with power 0.9; four-level
factors have 144/level → d ≈ 0.46 for pairwise contrasts. Per-cell counts for all two-way
interactions are ≥ 36. Replicate counts in E3/E4 give ≥ 8 runs per primary cell, which the
permutation-based η² analysis requires.

# Analysis pipeline (in `analysis/`)

1. `build_dataset.py` — parse `visualizer_*.json` / `survey_*.json` into one tidy CSV (one row
   per run: all design factors, seeds, all behavioral metrics, order-consistency, margins,
   mean degree).
2. `anova_v2.py` — Type-III ANOVA (partial η², main + 2-way interactions), mixed-effects with
   batch random intercept, BH-FDR per declared family, bootstrap CIs.
3. `noise_floor_analysis.py` — E4: shift rate under none/scrambled vs matched real runs,
   margin-stratified flip analysis, order-consistency reporting.
4. `topology_analysis.py` — E3: permutation η² per metric, density-as-covariate check,
   per-model robustness with FDR.
5. `train_bert_detector.py` / `eval_detector.py` — leakage-safe detector training, OOD and
   off-the-shelf cross-checks (P1).
