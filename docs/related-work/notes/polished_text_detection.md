# Detection of AI-polished / machine-revised human text, and detectors aligned to machine style (state as of 2026-11-14)

Scope note: covers (1) benchmarks and findings on AI-polished/revised text, (2) detectors trained or aligned to machine style, (3) mixed human-AI text and boundary detection, (4) how zero-shot detectors (Binoculars, Fast-DetectGPT, likelihood) do on polished text, and (5) a novelty verdict for a training-free "temporal pair" detector, i.e. log p(post-ChatGPT-crawl model) minus log p(pre-ChatGPT-crawl model). Items marked **[unverified]** come only from search-engine summaries or prior knowledge and were not checked against the primary text in this session.

---

## Q1. APT-Eval: "Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing" (Findings ACL 2025)

### Takeaway
APT-Eval (Saha & Feizi) is the main benchmark for human text that has been lightly-to-heavily polished by an LLM. Its main result is about false positives: with thresholds calibrated on pure human text (FPR 0-8%), many detectors flag even "extreme-minor" or 1%-of-words polishing as AI. They cannot rank degrees of polishing, and they flag text from older/smaller polishers more often. Scientific-paper abstracts were the domain with the *lowest* flag rates.

### Cited Findings
- **Bibliographic:** Shoumik Saha, Soheil Feizi. arXiv:2502.15666 (v1 21 Feb 2025; v2 5 May 2025). Published in Findings of ACL 2025 (Anthology ID 2025.findings-acl.1303). — [arXiv abs](https://arxiv.org/abs/2502.15666); [ACL Anthology preview](https://preview.aclanthology.org/credits/2025.findings-acl.1303)
- **Sample count by version:** v1 abstract says 11.7K samples and 11 detectors. v2 says 14.7K samples (exactly 14,713) and 12 detectors. — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2); [search summary of v1](https://arxiv.org/html/2502.15666v1)
- **Source data:** 300 pure human texts taken from MixSet (Zhang et al. 2024), 50 per domain across 6 domains: blogs (2006), Enron emails (2015), BBC news (2006), Steam game reviews (2021), **ArXiv-10 paper abstracts (2022)**, TED talks (2021). — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **Polishers:** GPT-4o, Llama3.1-70B, Llama3-8B, Llama2-7B, DeepSeek-V3. — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **Polishing settings:** Degree-based: extreme-minor, minor, slight-major, major. Percentage-based: polish 1/5/10/20/35/50/75% of words. Each human text gets 11 polished variants. Samples with BERT semantic similarity < 0.85 to the source were filtered out.
  - Degree-based counts: extreme-minor 1,383; minor 1,358; slight-major 1,322; major 1,184.
  - Percentage-based total: 9,166.
  - — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **How much text changes (GPT-4o):**

  | Setting | Semantic similarity | Normalized Levenshtein | Jaccard distance |
  |---|---|---|---|
  | Extreme-minor | 0.98 | 0.15 | 0.20 |
  | Major | 0.93 | 0.65 | 0.62 |
  | 1% edits | 1.00 | 0.02 | 0.03 |
  | 75% edits | 0.96 | 0.38 | 0.42 |

  - Llama3.1-70B: extreme-minor Levenshtein 0.06; major 0.77.
  - — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **The 12 detectors tested:**
  - Model-based: RADAR, RoBERTa-base (ChatGPT), RoBERTa-base (GPT-2), RoBERTa-large (GPT-2).
  - Metric/zero-shot: GLTR, DetectGPT, **Fast-DetectGPT**, LLMDet, **Binoculars**.
  - Commercial: ZeroGPT, GPTZero, Pangram.
  - Loglikelihood/LogRank as standalone baselines, and ImBD/DetectAnyLLM, were not tested.
  - — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **Calibration on pure human vs AI text (threshold, accuracy, FPR on human text):**

  | Detector | Threshold | Accuracy | FPR |
  |---|---|---|---|
  | Binoculars | 0.1075 | 0.88 | 0.018 |
  | Fast-DetectGPT | 0.778 | 0.8317 | 0.06 |
  | GLTR | 0.7038 | 0.845 | 0.0683 |
  | DetectGPT | — | 0.725 | 0.085 |
  | RADAR | — | 0.8017 | 0.082 |
  | LLMDet | — | 0.605 | 0.2383 |
  | ZeroGPT | — | 0.8067 | 0.0367 |
  | GPTZero | — | 0.862 | 0.075 |
  | Pangram | — | 0.875 | 0.00 |

  — [arXiv v2 HTML (Table 6)](https://arxiv.org/html/2502.15666v2)
- **False positives on lightly polished text:**
  - GLTR: 6.83% FPR on pure human text, but flags 40.87% of GPT-4o extreme-minor texts, 42.81% of GPT-4o minor texts, and 26.85% of texts with 1% of words polished.
  - Commercial detectors on Llama2-7B extreme-minor polishing: ZeroGPT 32.31%, Pangram 42.56%, GPTZero 64.71%.
  - Across detectors, detection on minimally GPT-4o-polished text ranges from about 10% to 75%.
  - — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **Fast-DetectGPT on percentage-based polishing is flat:** 10.07% flagged at 1% edits vs 9.59% at 75% edits. The number of words polished does not move this zero-shot detector at a fixed threshold. — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **RoBERTa-large (GPT-2) cannot tell degrees apart:** it flags 47.69% of "minor" and 51.98% of "major" polished texts. — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **Older/smaller polishers are flagged more:**
  - About 45% of Llama2-7B extreme-minor texts are flagged, vs 25-32% for DeepSeek-V3, GPT-4o and Llama-3.
  - Average flag rate: 46% for Llama2-7B-polished vs 23% for DeepSeek-V3-polished.
  - GLTR extreme-minor by polisher: DeepSeek-V3 34.11%, Llama3.1-70B 39.19%, Llama3-8B 44.86%, Llama2-7B 52.31%.
  - — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **Domain effect:**
  - Speech had the highest flag rates at extreme-minor (33-56%). **Paper abstracts had the lowest (16-31%).**
  - For the larger polishers (GPT-4o, DeepSeek-V3, Llama3.1-70B), the flag rate on abstracts *fell* as polishing increased.
  - — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)
- **Authors' recommendations:**
  - Replace binary labels with tiered/probabilistic ones.
  - Train detectors on AI-polished samples.
  - Use domain-specific fine-tuning and interpretable segment-highlighting.
  - Keep a human in the loop for high-stakes decisions.
  - — [arXiv v2 HTML](https://arxiv.org/html/2502.15666v2)

### Inferences
- APT-Eval frames the problem as **false positives**: lightly polished text should arguably count as "human". The temporal-pair project frames it as **detection**: heavily polished abstracts should be caught. The project's finding that light proofreading is invisible (response roughly proportional to edit-ratio²) is close to the opposite failure mode from APT-Eval's. For a policy that tolerates light polishing, this could be a feature: low FPR on lightly edited text.
- APT-Eval reports thresholded flag rates, not AUROC at each polish level. The project's AUROC numbers are therefore not directly comparable. Running the temporal score on APT-Eval's ArXiv-10 abstract subset (degree and percentage splits) would be the natural external check, and it would give a flag-rate-vs-edit-percentage curve to compare against Fast-DetectGPT's flat 10%.
- Abstracts having the lowest false-flag rate in APT-Eval suggests abstracts are already "low-perplexity, formal" text. That compresses the margin likelihood-based detectors can exploit, which is consistent with plain likelihood doing only ~0.79 in the project.

### Gaps
- Per-detector, per-degree numbers for Binoculars and Fast-DetectGPT on the *degree-based* splits exist only in figures. They were not extractable as text, and no exact Binoculars flag rate on polished text was obtained.
- I did not inspect the APT-Eval GitHub release for raw score dumps.

---

## Q2. ImBD ("Imitate Before Detect", AAAI 2025) and other detectors trained or aligned to machine style

### Takeaway
ImBD fine-tunes a scoring model (GPT-Neo-2.7B, LoRA) with a DPO-style "style preference optimization" (SPO) on about 500-1,000 pairs of (human text, machine-revised version of the same text). It then scores with a Fast-DetectGPT-style conditional curvature. On GPT-4o-polished text it reaches AUROC ~0.89, vs ~0.69 for Fast-DetectGPT and ~0.57 for likelihood. Follow-ups are trained-scoring-model descendants (DetectAnyLLM/DDL, ~0.93 AUROC on MIRAGE-polish), plus related "preference/reward" detectors (ReMoDetect) and learned rewriters (L2R). In 2026 there are also **training-free aligned-vs-base log-ratio detectors** (IRM, Alignment Imprint/LAPD). These are the closest prior art to the temporal-pair idea.

### Cited Findings
**ImBD**
- **Bibliographic:** Jiaqi Chen et al. (14 authors), "Imitate Before Detect: Aligning Machine Stylistic Preference for Machine-Revised Text Detection". AAAI 2025, DOI 10.1609/aaai.v39i22.34525; arXiv:2412.10432. Project site: https://machine-text-detection.github.io/ImBD — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/34525); [arXiv v2 HTML](https://arxiv.org/html/2412.10432v2)
- **SPO method:**
  - Preference pairs are (x_m = machine-revised text, x_h = the same human text). The model is trained to prefer x_m.
  - Loss is Bradley-Terry with DPO reparameterization **r(x) = β·log(p_θ(x)/p_ref(x))**, where p_ref is the frozen initial scoring model; β = 0.05.
  - Base scoring model GPT-Neo-2.7B with LoRA (r = 8, α = 32); learning rate 1e-4.
  - — [arXiv v2 HTML](https://arxiv.org/html/2412.10432v2)
- **Style-CPC score:** d(x) = (log p̂(x|x) − μ̃)/σ̃, computed with the style-aligned model p̂ and the same conditional-sampling trick as Fast-DetectGPT. — [arXiv v2 HTML](https://arxiv.org/html/2412.10432v2)
- **Data:**
  - Test domains: XSum, SQuAD, WritingPrompts, PubMedQA.
  - Tasks: rewrite, expand, polish, plus "generate" for pure MGT.
  - Revisers: Qwen2-7B, Llama-3-8B, Mixtral-7B, Deepseek-7B, GPT-3.5-turbo-0125, GPT-4o-2024-05-13.
  - Training set size is inconsistent within the paper: 500 pairs in Sec. 3.1/A.2 vs "1,000 samples" in the abstract. Training takes about 220 s on one L20 GPU (abstract: "five minutes").
  - — [arXiv v2 HTML](https://arxiv.org/html/2412.10432v2); [search summary](https://arxiv.org/abs/2412.10432v1)
- **Polish-task AUROC (Table 1, averages over domains):**

  | Method | GPT-3.5 polished | GPT-4o polished |
  |---|---|---|
  | ImBD | **0.9449** | **0.8899** |
  | Fast-DetectGPT | 0.7933 | 0.6931 |
  | DetectGPT | 0.6915 | 0.6867 |
  | Likelihood | 0.6433 | 0.5690 |
  | RoBERTa-large | 0.6158 | 0.4193 |
  | RoBERTa-base | 0.5800 | 0.4064 |

  - GPTZero on GPT-3.5-polished text: 0.9351.
  - **Binoculars is not evaluated in ImBD.**
  - — [arXiv v2 HTML](https://arxiv.org/html/2412.10432v2)
- **Other ImBD results:**
  - Open-source-polished text: ImBD 0.9550 vs Fast-DetectGPT 0.8261.
  - Ablation on the polish task (GPT-3.5/GPT-4o): SPO 0.95/0.89; ORPO 0.86/0.71; RLHF 0.80/0.66; SFT 0.70/0.67, even though SFT used 3× data.
  - Removing the "imitation" step costs about 16-20 points.
  - **Polish degree and edit ratio were not studied.**
  - — [arXiv v2 HTML](https://arxiv.org/html/2412.10432v2)
- **Headline claims:** +13% AUC for open-source-revised text, +5% for GPT-3.5 and +19% for GPT-4o. Beats GPTZero with about 1,000 samples and 5 minutes of SPO. — [arXiv abstract via search](https://arxiv.org/abs/2412.10432v1)

**DetectAnyLLM / Direct Discrepancy Learning (DDL), 2025 — the direct follow-up of ImBD**
- **Bibliographic:** Jiachen Fu, Chun-Le Guo, Chongyi Li. arXiv:2509.14268 (Sep 2025). Project page: fjc2005.github.io/detectanyllm. — [arXiv HTML](https://arxiv.org/html/2509.14268v1)
- **Method:** Optimizes the scoring model directly on the detection discrepancy. Loss = ‖dc(x_h)‖₁ + ‖γ − dc(x_m)‖₁ with γ = 100. The DPO KL/reference term is removed. Base model GPT-Neo-2.7B; trained on the same 500 GPT-3.5-polished pairs as ImBD. — [arXiv HTML](https://arxiv.org/html/2509.14268v1)
- **MIRAGE benchmark:**
  - 10 corpora / 5 domains (academic, e-mail, website, news, comment), about 93K human/machine pairs, 17 LLMs (13 proprietary).
  - Tasks Generate / Polish / Rewrite; settings DIG (disjoint inputs) and SIG (shared inputs).
  - — [arXiv HTML](https://arxiv.org/html/2509.14268v1)
- **MIRAGE-DIG Polish task:**

  | Method | AUROC | TPR@5%FPR |
  |---|---|---|
  | DetectAnyLLM | **0.9297** | 0.7756 |
  | ImBD | 0.7888 | 0.2730 |
  | Fast-DetectGPT | 0.5720 | 0.1189 |
  | DetectGPT | 0.5469 | — |
  | Entropy | 0.5543 | — |
  | RoBERTa-large | 0.5171 | — |
  | **Likelihood** | **0.4653** | — |
  | LogRank | 0.4512 | — |
  | LRR | 0.4081 | — |
  | DNA-GPT | 0.4771 | — |

  - Binoculars is not reported.
  - Task difficulty (Generate → Polish → Rewrite): Fast-DetectGPT 0.7768 → 0.5720 → 0.5455; ImBD 0.8597 → 0.7888 → 0.7825.
  - — [arXiv HTML](https://arxiv.org/html/2509.14268v1)

**ReMoDetect (NeurIPS 2024)**
- **Bibliographic:** Hyunseok Lee, Jihoon Tack, Jinwoo Shin (KAIST), "ReMoDetect: Reward Models Recognize Aligned LLM's Generations", arXiv:2405.17382. — [arXiv](https://arxiv.org/abs/2405.17382v2)
- **Method:**
  - Uses a reward model as the detector, since aligned LLMs' outputs get higher predicted preference than human text.
  - Continual preference fine-tuning pushes the reward model to prefer LLM text even more, with replay.
  - A "Human/LLM mixed text" corpus, made by partially rephrasing human text with aligned LLMs, serves as *median-preference* data.
  - — [search summary of arXiv](https://arxiv.org/html/2405.17382v1)
- **Reported numbers [unverified, secondary summaries]:** about 97.9-98.8% AUROC on GPT-4 text vs Fast-DetectGPT about 96.1%. On Claude 3 Opus: 92.6% (Fast-DetectGPT) → 98.6%. — [Moonlight review](https://www.themoonlight.io/review/remodetect-reward-models-recognize-aligned-llms-generations)

**Learning to Rewrite (L2R), ACL 2025**
- **Bibliographic:** Wei Hao, Ran Li, Weiliang Zhao, Junfeng Yang, Chengzhi Mao. ACL 2025 long, pp. 6421-6434; arXiv:2408.04237. — [arXiv](https://arxiv.org/html/2408.04237v2); [ACL Anthology](https://preview.aclanthology.org/setup/2025.acl-long.322)
- **Method:** Fine-tunes a rewriter LLM so that it edits LLM text little and human text a lot. Detection uses edit distance. It extends RAIDAR, which does the same with an untuned rewriter. Evaluated on 21 domains with 4 generators. — [arXiv](https://arxiv.org/html/2408.04237v2)
- **Exact AUROC numbers:** not extracted in this session.

**AdaDetectGPT (2025)**
- **Bibliographic:** arXiv:2510.01268; code at Mamba413/AdaDetectGPT. — [arXiv HTML v4](https://arxiv.org/html/2510.01268v4)
- **Method:** Learns an adaptive "witness function" on top of logit-based statistics (closed-form, B-spline features per a secondary summary), with finite-sample bounds on TNR/FNR. — [arXiv HTML v4](https://arxiv.org/html/2510.01268v4)
- **Headline claim:** up to 37% improvement. No polished-text evaluation was found. — [arXiv HTML v4](https://arxiv.org/html/2510.01268v4)

**Training-free aligned-vs-base contrast detectors (2026) — closest conceptual neighbours of the temporal pair**
- **"Alignment Imprint: Zero-Shot AI-Generated Text Detection via Provable Preference Discrepancy"**
  - Junxi Wu, Kailin Huang, Dongjian Hu, Bin Chen, Hao Wu, Shu-Tao Xia, Changliang Zou. arXiv:2604.16923 (v1 18 Apr 2026, v2 2 Nov 2026); comments list NeurIPS 2026.
  - Treats the **log-likelihood ratio between an aligned model and its base model** as an "alignment imprint" that decomposes into instructional biases plus preference rewards.
  - LAPD standardizes and information-weights it. Claims +45.82% relative over the strongest baselines and "dominates Fast-DetectGPT".
  - Polished text: not stated in the abstract.
  - — [arXiv abs](https://arxiv.org/abs/2604.16923)
- **"Zero-Shot Detection of LLM-Generated Text via Implicit Reward Model" (IRM)**
  - Runheng Liu, Heyan Huang, Xingchen Xiao, Zhijing Wu. arXiv:2604.21223 (23 Apr 2026). Venue comment "NeurIPS 2025" is inconsistent with the date.
  - Derives an implicit reward from public instruct + base model pairs with no training; evaluated on DetectRL.
  - — [arXiv abs](https://arxiv.org/abs/2604.21223)
  - From a search snippet: Log-Likelihood improves from 66.44% to 77.46% when scored with Llama-3.2-1B-Instruct, and IRM averages 91.77% on the Llama-3.2-1B family. — [search snippet, arXiv HTML](https://arxiv.org/html/2604.21223v1) **[unverified]**
- **"Zero-shot detection of LLM-generated text via dual-network preference divergence"** (Expert Systems with Applications, 2026, per ScienceDirect PII S0957417426001260). Contrasts a "human-preference" model with a "machine-preference" model to capture SFT/RLHF-induced shift. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426001260) **[unverified beyond snippet]**
- **Binoculars (Hans et al., ICML 2024)**
  - Contrasts two closely related models (an observer/performer pair; in practice Falcon-7B and Falcon-7B-instruct, i.e. base vs instruct — this pair is from prior knowledge).
  - Detects >90% of ChatGPT samples at 0.01% FPR with no ChatGPT training data.
  - — [arXiv 2401.12070](https://arxiv.org/abs/2401.12070); [OpenReview](https://openreview.net/pdf?id=iARAKITHTH)

### Inferences
- ImBD's SPO objective *literally* maximizes the margin of β·log(p_θ/p_ref) between machine-revised and human versions of the same text. So "log p_post − log p_pre" is the **same functional form as the DPO implicit reward**. The "alignment" was done by the web (post-ChatGPT crawl contamination) instead of by SPO on 500 pairs. The analogy is precise, not just rhetorical. One nuance: ImBD's final score is a curvature on the aligned model alone, not the ratio. IRM and Alignment Imprint use the ratio itself as the score.
- Reference numbers for scale:
  - Project's temporal score on locally polished abstracts: AUROC ~0.88-0.90, vs likelihood ~0.79.
  - ImBD on GPT-4o-polished text: 0.89, a *trained* detector.
  - MIRAGE-polish: ImBD 0.79 and DetectAnyLLM 0.93.
  - These numbers come from different data, polishers and polish degrees, so they are only order-of-magnitude comparisons. A head-to-head on ImBD's released polish data (XSum/SQuAD/Writing/PubMed, GPT-3.5/GPT-4o) and on MIRAGE-polish would make the comparison direct.
- In MIRAGE-polish, likelihood is *below chance* (0.47). The project's likelihood at 0.79 suggests its "heavy local polishing" protocol is easier than MIRAGE's "preserve details" polish. Reviewers will likely ask for evaluation on a public polish benchmark.

### Gaps
- Exact AUROC tables for L2R, AdaDetectGPT, ReMoDetect (primary), IRM and LAPD were not extracted. I could not verify whether IRM or LAPD evaluate polished/revised text.
- I found no published follow-up that evaluates **Binoculars on ImBD's polish data**.
- DALD (distribution-aligned surrogate fine-tuning) and other surrogate-alignment detectors were not checked in this session.

---

## Q3. Other benchmarks and findings for revised, edited and mixed human-AI text (2024-2026)

### Takeaway
The field has moved from binary MGT detection toward three things:
- edit-aware benchmarks (Beemo, DetectRL, MIRAGE, APT-Eval, OpAI-Bench, ARB);
- continuous "how much AI editing" estimation (EditLens, ICLR 2026);
- multi-class detectors with an explicit "human-written, AI-polished" class (LLM-DetectAIve, FAID, DETree/HART, RACE, GPTZero's 2026 hierarchical model).

OpAI-Bench (June 2026) reports that detection is **non-monotonic in AI-edit fraction**: intermediate mixed versions can be harder than both endpoints.

### Cited Findings
- **Beemo** (Ekaterina Artemova et al.; NAACL 2025; arXiv:2411.04032)
  - Data: 6.5K human-written texts plus 13.1K machine-generated and LLM-edited texts. Outputs from 10 instruction-tuned LLMs were edited by experts.
  - Evaluation: 33 detector configurations on 7 binary tasks.
  - Findings: expert edits evade detection, while LLM-edited MGTs are unlikely to be recognized as human. Zero-shot detectors generalize better than pretrained ones to both expert- and LLM-edited MGT.
  - Note the direction: machine text edited by humans/LLMs, the opposite of human text polished by an LLM.
  - — [arXiv abs](https://arxiv.org/abs/2411.04032); [search summary](https://arxiv.org/pdf/2411.04032)
  - Expert target edit ratio of 20-40%, with Llama3.1-70B-Instruct and GPT-4o as LLM editors — [search summary](https://huggingface.co/papers/2411.04032) **[unverified]**
- **DetectRL** (Junchao Wu, Derek F. Wong et al.; NeurIPS 2024 Datasets & Benchmarks; arXiv:2410.23746)
  - Real-world high-risk domains, with attacks simulating prompt usage, human revisions (word substitution), writing errors, paraphrasing and data mixing. SOTA detectors underperform.
  - Code: NLP2CT/DetectRL.
  - — [arXiv](https://arxiv.org/abs/2410.23746); [NeurIPS page](https://neurips.cc/virtual/2024/poster/97633)
  - It has no dedicated "LLM-polished human text" class (not confirmed).
- **MIRAGE** (DetectAnyLLM paper, 2025): the polish task at scale (17 LLMs, 5 domains). Numbers are in Q2. — [arXiv HTML](https://arxiv.org/html/2509.14268v1)
- **EditLens** (Katherine Thai, Bradley Emi, Elyas Masrour, Mohit Iyyer; ICLR 2026; arXiv:2510.03154; Pangram Labs / UMass / UMD)
  - Treats AI editing as a continuous quantity.
  - Similarity metrics between source and edited text (e.g., embedding cosine distance), validated with human annotators, serve as intermediate supervision for a regression model.
  - Binary F1 = 94.7%; ternary (human/AI/mixed) F1 = 90.4%. Includes a case study of Grammarly edits.
  - Code: github.com/pangramlabs/EditLens (CC BY-NC-SA 4.0).
  - — [arXiv](https://arxiv.org/abs/2510.03154); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008153)
- **OpAI-Bench** (Sondos Mahmoud Bsharat, ..., Zhiqiang Shen; arXiv:2606.06481, 4 Jun 2026)
  - Each human document is progressively revised into 9 sequential versions at predefined AI-coverage levels, using 5 edit operations across 4 domains. Labels at document, sentence, token and span level.
  - Evaluates 8 document-level, 7 sentence-level and 2 token/span-level detectors.
  - Detectability depends on edit proportion, operation, domain and revision history. **Mixed-authorship intermediate versions are often harder to detect than both fully human and heavily AI-edited endpoints.**
  - — [arXiv abs](https://arxiv.org/abs/2606.06481)
- **ARB: A Matched Authorship-Rewriting Benchmark Dataset for AI-Text Detector Evaluation** (arXiv:2607.29539, 2026). Exists, but details were not fetched. — [arXiv PDF](https://arxiv.org/pdf/2607.29539) **[unverified content]**
- **Fine-grained / multi-class detectors with a "polished" class:**
  - **LLM-DetectAIve** (MBZUAI; arXiv:2408.04284): four classes — human, machine, machine-humanized, and **human-written then machine-polished**. The authors note that polishing is typically acceptable in academic writing but not in education. — [arXiv](https://arxiv.org/pdf/2408.04284); [GitHub](https://github.com/mbzuai-nlp/LLM-DetectAIve)
  - **FAID** (arXiv:2505.14271): 83,350 examples; human-LLM collaborative classes include human-written/LLM-polished. — [arXiv](https://arxiv.org/pdf/2505.14271)
  - **DETree** (NeurIPS 2025; arXiv:2510.17489): a Hierarchical Affinity Tree over generation processes with a tree-structured contrastive loss. RealBench has about 16.4M samples / 1,204 categories, including categories like "human_polish_Gemini1.5". — [arXiv](https://arxiv.org/pdf/2510.17489); [NeurIPS](https://neurips.cc/virtual/2025/poster/115391)
  - A snippet shows "HART (Binoculars) 0.838 vs DETree 0.988" on one HART task, but the metric and column were unclear. — [search snippet](https://arxiv.org/html/2510.17489v1) **[unverified]**
  - **RACE** (arXiv:2604.04932): four classes (human, LLM, LLM-polished human, humanized LLM), evaluated only on HART. — [arXiv](https://arxiv.org/html/2604.04932)
  - **GPTZero technical report** (arXiv:2602.13042): hierarchical classifier with a "Polished (human written and then AI polished)" class, evaluated on 4,631 polished texts. — [arXiv](https://arxiv.org/pdf/2602.13042) **[details unverified]**
  - **Pangram 4 Technical Report** (arXiv:2607.27183): vendor-reported FPR on AI-polished text of 0.01% vs 0.18% for Pangram 3.3.2. This implies Pangram counts lightly polished text as human. — [search summary](https://arxiv.org/html/2607.27183) **[unverified, vendor claim]**
- **Corpus-level estimation for scientific abstracts** (Weixin Liang et al., "Mapping the Increasing Use of LLMs in Scientific Papers", arXiv:2404.01268, COLM 2024 per prior knowledge)
  - A distributional (population-level) estimator of the fraction of "LLM-modified" text, defined as substantially rewritten beyond spelling/grammar fixes.
  - 950,965 papers (Jan 2020-Feb 2024) from arXiv, bioRxiv and Nature.
  - The estimated LLM-modified fraction in **CS abstracts reached 17.5%** (introductions 15.3%) by Feb 2024; mathematics 4.9%, Nature portfolio 6.3%.
  - — [arXiv](https://arxiv.org/abs/2404.01268); [HTML](https://arxiv.org/html/2404.01268v1)
- **RAID** (Dugan et al., ACL 2024; arXiv:2405.07940): its "edited" conditions are 11 adversarial attacks (paraphrase, synonym swap, misspelling, homoglyph, etc.) applied to *machine* text. It has no split for human text polished by an LLM. — [arXiv 2405.07940](https://arxiv.org/abs/2405.07940) **[from prior knowledge, not re-fetched]**

### Inferences
- Paper-abstract data for the project's own domain:
  - Liang et al.'s abstract-level estimates (about 17.5% of CS abstracts LLM-modified by early 2024) mean a post-ChatGPT crawl contains *many* LLM-polished abstracts. That supports the "implicitly style-aligned by the web" story for this domain.
  - The same fact implies that **"human" abstracts dated 2023+ are contaminated**. Human-label test sets should use pre-2023 abstracts.
- EditLens's continuous target (edit magnitude from source/edited similarity) is the natural evaluation axis for the project's edit-ratio² finding. A Spearman correlation between the temporal score and EditLens-style edit magnitude, or Levenshtein ratio as in APT-Eval, would sit well alongside the AUROC.
- OpAI-Bench's non-monotonicity finding is a useful contrast: the project reports a smooth, monotone (quadratic) response to edit ratio. Whether that holds on OpAI-Bench's progressive versions would be a strong test.

### Gaps
- Exact numbers for zero-shot detectors on OpAI-Bench, ARB, HART and FAID were not obtained (only the abstracts were read).
- MixSet ("LLM-as-a-Coauthor", Zhang et al. 2024), the source of APT-Eval's human data, was not fetched. Its detector numbers on polished/mixed text are therefore missing.

---

## Q4. Mixed human-AI text and boundary detection

### Takeaway
Boundary and segment detection is a separate, mostly supervised line of work:
- SemEval-2024 Task 8 / M4GT-Bench subtask on human→machine boundaries;
- sentence-level CRF taggers;
- DAMASHA/MAS with adversarial segmentation;
- English-BMAS (46,830 mixed samples).

It targets "human then AI continuation" or interleaved spans, not uniform polishing spread across a whole text. Token-level zero-shot signals (e.g., per-token log-ratio of two models) are what SeqXGPT-style methods use as features.

### Cited Findings
- **Background — SemEval 2024 / M4GT-Bench:** TM-TREK (Qu & Meng 2024) won the human-machine mixed-text boundary subtask. In RoFT, humans identified the boundary correctly in only 23.4% of cases. — [search summary of arXiv 2509.17830](https://arxiv.org/html/2509.17830v2)
- **Sentence-level segmentation** (arXiv:2509.17830; Findings IJCNLP 2025): DeBERTa + BiGRU + CRF on TriBERT and M4GT-Bench Task 3; best configuration about 0.806 F1-style score in one column. — [arXiv](https://arxiv.org/html/2509.17830v2); [ACL Anthology PDF](https://aclanthology.org/2025.findings-ijcnlp.48.pdf) **[number from snippet]**
- **DAMASHA / MAS** (arXiv:2512.04838, Dec 2025): segmentation benchmark with adversarial attacks and new partial-overlap metrics (Segment-wise Boundary Detection Accuracy, Segment Precision); Info-Mask method. — [arXiv](https://arxiv.org/abs/2512.04838)
- **English-BMAS** ("Mixture of Detectors", arXiv:2509.22147): 46,830 mixed-authorship samples with human-end and machine-end boundaries. — [arXiv](https://www.arxiv.org/pdf/2509.22147)
- **D2C-Routing** (arXiv:2608.27380, 2026): evidence routing for mixed-origin detection. — [arXiv](https://arxiv.org/pdf/2608.27380) **[content unverified]**
- **SeqXGPT** (Wang et al., EMNLP 2023; arXiv:2310.08903): sentence-level detection using token log-probability sequences from several white-box LLMs as features. **M4GT-Bench** (Wang et al., ACL 2024; arXiv:2402.11175) added a mixed-text boundary task. — [SeqXGPT arXiv](https://arxiv.org/abs/2310.08903); [M4GT-Bench arXiv](https://arxiv.org/abs/2402.11175) **[from prior knowledge, not re-fetched]**

### Inferences
- A per-token temporal log-ratio, log p_post(t) − log p_pre(t), is a natural SeqXGPT-style feature. It could localize polished sentences without training. This is a possible extension, but uniform light proofreading has no boundary to find, so it would not fix the edit-ratio² blind spot.

### Gaps
- I found no boundary-detection paper that uses a training-free two-model contrast as its per-token feature. This was not searched exhaustively.

---

## Q5. How zero-shot detectors (Binoculars, Fast-DetectGPT, likelihood) perform on polished text

### Takeaway
Zero-shot curvature and likelihood detectors degrade sharply from "generate" to "polish". In MIRAGE they are near or below chance on polish, and Fast-DetectGPT reaches only ~0.69-0.79 on ImBD's GPT-4o/GPT-3.5 polish. Fast-DetectGPT's flag rate on APT-Eval does not rise with the fraction of words polished. Binoculars has the lowest human FPR in APT-Eval's calibration (1.8%), but **no published exact AUROC for Binoculars on human-text-polished-by-LLM was found** in ImBD, MIRAGE or APT-Eval text.

### Cited Findings
- **Fast-DetectGPT AUROC:**
  - ImBD: 0.7933 (GPT-3.5 polish), 0.6931 (GPT-4o polish).
  - MIRAGE-DIG: 0.7768 (generate) → **0.5720 (polish)** → 0.5455 (rewrite).
  - — [ImBD arXiv HTML](https://arxiv.org/html/2412.10432v2); [DetectAnyLLM arXiv HTML](https://arxiv.org/html/2509.14268v1)
- **Likelihood AUROC:** 0.6433 (GPT-3.5 polish) and 0.5690 (GPT-4o polish) in ImBD; **0.4653** on MIRAGE-DIG polish (LogRank 0.4512; LRR 0.4081). — [ImBD](https://arxiv.org/html/2412.10432v2); [DetectAnyLLM](https://arxiv.org/html/2509.14268v1)
- **Fast-DetectGPT flag rate on APT-Eval percentage-based polishing:** 10.07% at 1% edits vs 9.59% at 75% edits (calibrated human FPR 6%). — [APT-Eval v2](https://arxiv.org/html/2502.15666v2)
- **Binoculars in APT-Eval calibration:** threshold 0.1075, accuracy 0.88, FPR 1.8%, the best non-commercial detector. Its polished-text flag rates appear only in figures. — [APT-Eval v2](https://arxiv.org/html/2502.15666v2)
- **Beemo:** zero-shot detectors generalize better than pretrained classifiers to both expert- and LLM-edited MGT. — [arXiv](https://arxiv.org/abs/2411.04032)
- **Instruct models as scorers help zero-shot detection:** IRM reports that using instruction-tuned scorers improves 5 of 6 baselines, e.g. Log-Likelihood 66.44 → 77.46 with Llama-3.2-1B-Instruct. — [arXiv HTML](https://arxiv.org/html/2604.21223v1) **[snippet, unverified]**

### Inferences
- The project's numbers on its own "heavily locally polished abstracts":
  - Plain likelihood ~0.79 is *higher* than likelihood in ImBD (0.57-0.64) and MIRAGE (0.47).
  - The temporal score ~0.88-0.90 is in the range of *trained* ImBD (0.89 on GPT-4o polish) but below DetectAnyLLM on MIRAGE (0.93).
  - Because the evaluation sets differ, the honest claim is that the score "closes much of the gap between zero-shot and style-aligned trained detectors in-domain". It is not "beats ImBD".
- Binoculars is the obvious missing baseline. It is a two-model contrast, and the IRM results suggest that instruct-vs-base contrasts already carry machine-style signal. The key ablation is temporal pair vs base/instruct pair (Binoculars-style or IRM-style) on the same polished abstracts.

### Gaps
- No exact AUROC for Binoculars on LLM-polished human text was found in any primary source read here. MIRAGE and ImBD omit it, and APT-Eval gives only figures.

---

## Q6. Policy angle: why polished-text detection matters

### Takeaway
Detectors that penalize low-perplexity text falsely accuse non-native writers: 61% average FPR on TOEFL essays in Liang et al. 2023. LLM polishing is often legitimate, especially for non-native authors. So a detector that is *insensitive to light proofreading* but sensitive to heavy rewriting arguably matches policy needs better than one that flags any AI touch.

### Cited Findings
- **Liang et al. 2023 ("GPT detectors are biased against non-native English writers"):** Weixin Liang, Mert Yuksekgonul, Yining Mao, Eric Wu, James Zou; Patterns 2023; arXiv:2304.02819.
  - Seven detectors on 91 TOEFL essays and 88 US 8th-grade essays.
  - Average FPR on TOEFL essays: **61.22%**.
  - 18/91 (19.78%) TOEFL essays were flagged unanimously; 89/91 (97.80%) were flagged by at least one detector.
  - The hypothesized cause is low perplexity from limited linguistic variability.
  - — [arXiv](https://arxiv.org/pdf/2304.02819); [Times Higher Education](https://timeshighereducation.com/node/727071)
- **APT-Eval's motivation:** false plagiarism accusations against minimally polished human writing; the authors recommend tiered labels and human-in-the-loop review. — [arXiv v2](https://arxiv.org/html/2502.15666v2)
- **LLM-DetectAIve's rationale:** machine polishing is typically acceptable in academic writing but not in education, hence a separate class. — [arXiv](https://arxiv.org/pdf/2408.04284)
- **EditLens** is explicitly motivated by the fact that a binary label mischaracterizes AI-edited text. Its demo warns against consequential use. — [arXiv](https://arxiv.org/abs/2510.03154); [HF demo](https://multimodalart-editlens.hf.space/)
- **Liang et al. 2024:** "LLM-modified" is defined as beyond spelling/grammar fixes, i.e. light proofreading is deliberately excluded from what is measured. — [arXiv HTML](https://arxiv.org/html/2404.01268v1)

### Inferences
- The project's edit-ratio² response can be framed as a *desirable* property under these policies: low sensitivity to proofreading, which is common for non-native authors, and high sensitivity to substantive rewriting. That framing should be backed by an explicit FPR evaluation on lightly proofread text and on non-native human writing (e.g., the TOEFL set released by Liang et al.).

### Gaps
- I did not check whether the post- vs pre-ChatGPT log-ratio itself is biased against non-native writing. That is an open empirical question for the project.

---

## Q7. Verdict: how novel is a training-free "temporal" detector for polished text?

### Takeaway
**Partially novel.** The *mechanism* is well precedented: a log-ratio between two models, one of which is tilted toward machine style.
- Binoculars (base/instruct contrast, 2024).
- The DPO implicit reward inside ImBD's SPO (2025).
- Training-free aligned-vs-base detectors IRM and Alignment Imprint/LAPD (2026, the latter listed for NeurIPS 2026).

The *specific* idea is not found in any source located in this search: the pair differs only in **training-data date** (pre- vs post-ChatGPT crawl), the "alignment" comes **implicitly from web contamination**, and it is applied to **LLM-polished human text** with an edit-ratio scaling analysis.

### Cited Findings
- A targeted search for detectors contrasting checkpoints with different training cutoffs or pre/post-ChatGPT crawls returned no such paper. Binoculars was the closest. — [search: Binoculars](https://arxiv.org/abs/2401.12070)
- A separate search found only temporal *measurement* studies (e.g., Fast-DetectGPT scores on Reddit before and after ChatGPT), not temporal-pair detectors. — [arXiv 2609.12447](https://arxiv.org/pdf/2609.12447)
- **Training-free aligned-vs-base log-ratio is already published:**
  - Alignment Imprint/LAPD: "log-likelihood ratio between aligned and base models", arXiv:2604.16923, Apr/Nov 2026. — [arXiv](https://arxiv.org/abs/2604.16923)
  - IRM: implicit reward from public instruct and base models, no training, arXiv:2604.21223. — [arXiv](https://arxiv.org/abs/2604.21223)
  - Dual-network preference divergence: ESWA 2026. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426001260)
- **ImBD's SPO** reward is exactly r(x) = β·log(p_θ(x)/p_ref(x)), trained to prefer machine-revised over human versions of the same text. — [arXiv HTML](https://arxiv.org/html/2412.10432v2)
- **Trained style-aligned detectors on polish tasks:** ImBD 0.89 AUROC on GPT-4o polish; DetectAnyLLM 0.93 on MIRAGE-polish. — [ImBD](https://arxiv.org/html/2412.10432v2); [DetectAnyLLM](https://arxiv.org/html/2509.14268v1)

### Inferences
- **Novelty claims that seem defensible:**
  - (a) The source of the stylistic tilt is *temporal data contamination*, not explicit SFT/RLHF/DPO. This yields a "found" alignment that needs no machine text at all, not even the 500 pairs ImBD uses.
  - (b) An analysis of how polished-text detection scales with edit ratio (≈ quadratic), which none of ImBD, DetectAnyLLM, IRM or LAPD study.
  - (c) The policy-relevant property of tolerating light proofreading.
- **Claims that would be over-reach:**
  - "First two-model contrast detector" (Binoculars).
  - "First training-free analogue of style-aligned detection": IRM and LAPD already present base-vs-aligned log-ratios as training-free implicit-reward detectors. The project must cite them and position the temporal pair as a *different source of alignment*.
- **Likely reviewer demands:**
  1. An ablation against the instruct-vs-base pair of the same family (IRM/LAPD/Binoculars-style) on the same polished abstracts.
  2. Controls for **temporal topic drift**: post-2022 human abstracts discuss new topics and vocabulary the pre-ChatGPT model hasn't seen, so the log-ratio could partly measure recency rather than machine style. Use human abstracts from before both cutoffs, or topic-matched sets.
  3. Contamination of "human" test abstracts written after 2022. Liang et al. estimate about 17.5% of CS abstracts were LLM-modified by Feb 2024.
  4. Evaluation on public polish benchmarks: ImBD's polish split, MIRAGE-polish, APT-Eval's ArXiv-10 subset.
  5. Comparison with Binoculars, which is missing from both ImBD and MIRAGE tables, so the project could fill that gap.
- **Overall:** a workshop or Findings-level contribution, provided the paper is framed as "temporal contamination as implicit style alignment" and includes the base-vs-instruct ablation. It is not novel as a general "two-model ratio detector".

### Gaps
- No exhaustive check of whether IRM or LAPD evaluate polished/revised text. If they do, the novelty margin for "polished text" shrinks further. Their full papers should be read.
- I did not verify which public model families offer clean pre/post-ChatGPT checkpoint pairs that differ only in crawl date. That was outside this note's scope.
