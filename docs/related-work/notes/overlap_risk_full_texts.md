# Overlap ("already done / scooped") risk: full-text check of the closest prior works (as of 2026-11-14)

Idea under test (four claims): (C1) a date-split pair of otherwise identical LMs (DatedGPT 1.3B, yearly 2013-2024, FineWeb-Edu) used as a *measurement instrument*; (C2) a *year-curve* of chat-style preference (flat for 2013-2022 vintages, jump at the 2023 crawl) and log p_new - log p_old as a measure of AI-style diffusion into human writing; (C3) *separating the membership-inference / knowledge confound* (the newer model simply having seen the text or topic); (C4) a *training-free temporal-pair detector* s_lambda = log p_new - lambda * log p_old.

Method note: full texts were read from arXiv HTML for every version listed below, downloaded with curl and keyword-grepped locally, plus WebFetch reading. Quotes are verbatim from the HTML text unless marked "(tool-extracted)", which means a WebFetch summarizer read the text and I did not re-grep the exact string.

---

## Q1. Kelly, Malamud, Schwab, Xu, "Scaling Point-in-Time Language Models" (arXiv 2607.11889; NBER w35247): post-2022 checkpoints, AI-generated/ChatGPT-style text, style drift, deferred work, released checkpoints?

### Takeaway
None of the three arXiv versions (v1 24 Apr 2026, v2 17 Jul 2026, v3 12 Oct 2026) analyzes AI-generated text, ChatGPT-style text, contamination of the web by LLM output, style drift, or perplexity across vintages. The only post-2022 analysis is a coarse IFEval comparison of 2013-2020 vs 2021-2024 checkpoints, added in v3. v1 and v2 promised that "updated results for 2022–2025 will be reported in a subsequent version"; v3 dropped that sentence and added only that IFEval appendix. **Scoop risk is at the resource level, not the claim level.** 2023 and 2024 checkpoints are publicly released (36 models: PIT-1B, PIT-4B and PIT-4B-FT for 2013-2024), so anyone could run the same before/after-2023 likelihood analysis on a second, larger family.

### Cited Findings
- Version history: v1 Fri 24 Apr 2026, v2 Fri 17 Jul 2026, v3 Mon 12 Oct 2026 (current). Title in v1 and v3: "Scaling Point-in-Time Language Models: Enabling Economic Evaluation of Embeddings". v2 HTML title: "Scaling Point-in-Time Language Models". — [arXiv abs](https://arxiv.org/abs/2607.11889)
- v1 and v2 abstracts: "we construct a sequence of monthly model checkpoints spanning 2013–2024"; method: "We checkpoint the model weights on a monthly basis." — [v1 HTML](https://arxiv.org/html/2607.11889v1); [v2 HTML](https://arxiv.org/html/2607.11889v2)
- v3 abstract changes to "a sequence of yearly model checkpoints spanning 2013–2024", and the contributions list says "we release the complete project resources—covering dataset construction, model training, evaluation code, and yearly model checkpoints". — [v3 HTML](https://arxiv.org/html/2607.11889v3)
- Training design (all versions): "The model is trained on a temporally ordered stream of tokens, directly aligning our setup with the incremental and continual learning literature". The checkpoints therefore look like snapshots of one cumulative continual run, not independent from-scratch models. Pretraining data is "HuggingFaceFW/fineweb 170B / 1T" (FineWeb, not FineWeb-Edu). — [v1 HTML](https://arxiv.org/html/2607.11889v1); [v3 HTML](https://arxiv.org/html/2607.11889v3)
- Deferral in v1 and v2 (verbatim, identical): "Finally, our current evaluation is incomplete for the most recent period, and updated results for 2022–2025 will be reported in a subsequent version." — [v1 HTML](https://arxiv.org/html/2607.11889v1); [v2 HTML](https://arxiv.org/html/2607.11889v2)
- In v3, "subsequent version" and "2022–2025" have zero hits. The other limitations remain ("broader evidence across tasks and domains remains an important direction for future work"; v1 and v2 also say "stronger methods for temporally robust post-training and preference alignment warrant further study"). — [v3 HTML](https://arxiv.org/html/2607.11889v3); [v1 HTML](https://arxiv.org/html/2607.11889v1)
- The only post-2022 analysis (v3 Appendix C) is "Table 4: Mean IFEval performance: early (2013–2020) vs. later (2021–2024) checkpoints". Base Prompt-S is 17.14% vs 15.40% (−1.74 pp) and Base Inst-L is 31.25% vs 29.27% (−1.98). FT Prompt-S is 20.05% vs 23.05% (+3.00) and FT Inst-S is 33.42% vs 37.85% (+4.43). The paper offers no interpretation tied to AI-generated web content. — [v3 HTML](https://arxiv.org/html/2607.11889v3)
- Grep of the v3 full text: "AI-generated" has 0 hits and "subsequent version" has 0 hits. "ChatGPT" appears only in related work ("prompting ChatGPT for headline sentiment yields robust trading signals") and the bibliography. "synthetic" appears once, about LLMs as "synthetic agents" for in-silico experiments. "perplexity across vintages" is not reported. — [v3 HTML](https://arxiv.org/html/2607.11889v3)
- An LLM classifier is used only for temporal leakage in the SFT data ("we employ gpt-5-nano to identify and remove temporally sensitive examples") (tool-extracted). — [v3 HTML](https://arxiv.org/html/2607.11889v3)
- Release: v2 says "Code is available on Github and models are available on HuggingFace". The v3 checklist says "The model weights and training data are publicly available on Hugging Face". The Hugging Face API lists 36 models under Diamegs: PIT-1B-{201312…202412}, PIT-4B-{201312…202412} and PIT-4B-FT-{201312…202412}, with 2015 dated 201511, plus the dataset Diamegs/pre-training-tokens. — [v2 HTML](https://arxiv.org/html/2607.11889v2); [HF org](https://huggingface.co/Diamegs); [HF API listing](https://huggingface.co/api/models?author=Diamegs)
- Code URL differs by version: v1 footnote github.com/DjoFE2021/Point-In-Time-LLM, v2 github.com/anonymouspitllm-collab/PIT-LLM (tool-extracted). — [v1 HTML](https://arxiv.org/html/2607.11889v1); [v2 HTML](https://arxiv.org/html/2607.11889v2)
- A downstream paper describes the same models differently: "PIT-4B (Kelly et al., 2026), a suite of 4B-parameter base models trained on yearly partitions of FineWeb". This conflicts with the PIT paper's own "temporally ordered stream" (continual) description. — [arXiv 2610.37631 HTML](https://arxiv.org/html/2610.37631v1)
- Related point-in-time work cited by PIT v3: ChronoGPT (He et al. 2025a), ChronoGPT-Instruct (2025b), DatedGPT (Yan et al. 2026) (tool-extracted). — [v3 HTML](https://arxiv.org/html/2607.11889v3)

### Inferences
- Overlap with the four claims: C1 none, C2 none, C3 none, C4 none. The Appendix C IFEval drop for base vintages 2021-2024 is the only quantity that touches post-2022 vintages, and it concerns instruction following, not the likelihood of chat text.
- Residual risk: v1 and v2 explicitly announced "updated results for 2022–2025". The authors (finance/economics) could still add post-2022 analyses in a v4 or the NBER version, but nothing in the text points toward AI-text measurement.
- The released PIT-4B and PIT-1B 2022 and 2023 checkpoints are a ready-made robustness or replication family (FineWeb, 4B, continual training). They are also an easy route for a competitor. If PIT really is one continual run, a PIT 2023-vs-2022 pair differs by roughly "one more year of continued training", which is closer to a fine-tune contrast than DatedGPT's independent from-scratch pairs. The idea's paper should state which design it relies on.

### Gaps
- NBER w35247: I did not fetch the NBER page or PDF, so whether the NBER version differs from arXiv v3 (for example, contains the promised 2022-2025 results) is **unverified**.
- About 27.6k characters at the end of the v3 HTML (checklist tail) were read only via WebFetch, which found no relevant terms. Local grep covered the full text.

---

## Q2. Forward citations of DatedGPT (2603.11838), ChronoGPT (2502.21206), and the point-in-time paper: does any citing paper use these models for AI-text detection, AI-contamination measurement, or pre/post-ChatGPT vintage comparison?

### Takeaway
No. Across all Semantic Scholar citing records (DatedGPT: 15; ChronoGPT: 29 entries, 28 unique; PIT: 10), every citing paper is about finance lookahead bias, temporal leakage, historical or knowledge-cutoff LMs, or pretraining dynamics. None uses the models for AI-text detection, measuring LLM-text contamination, or comparing pre/post-ChatGPT vintages. The closest is "What Improves During Pretraining?" (2610.37631), which uses cross-vintage NLL variation of PIT-4B (2017-2024) as a measurement instrument for knowledge prevalence. That is partial overlap with C1 in form only. The source papers themselves (DatedGPT, ChronoGPT, ChronoGPT-Instruct) also do not touch AI-generated text.

### Cited Findings
- DatedGPT citing papers (15): Point-in-Time Encoders for Earnings Calls (2611.14325); FinCutBench (2611.11588); LLMs Don't Know What They Know (2611.06808); What Improves During Pretraining? (2610.37631); Lüshi-1840 Classical Chinese knowledge cutoff (2610.16201); SciTBERT (2610.12207); Temporal Leakage in LLM Backtesting (2608.02985); HindsightBench (2607.18867); Hindcast (2607.14051); A Language Model from 1913 (2606.02991); Scaling PIT LMs (2607.11889); Debiasing LLMs by Fine-tuning (2604.02921); The Self Driving Portfolio (2604.02279); All Leaks Count (2602.17234); Look-Ahead-Bench (2601.13770). Keyword hits for "contamina" and "detect" in their abstracts refer to *temporal leakage* contamination, not AI-generated text. — [Semantic Scholar API, DatedGPT citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2603.11838/citations?fields=title,externalIds,year,abstract&limit=100)
- ChronoGPT citing papers (29 entries; MemGuard-Alpha is listed twice). All are finance, forecasting or temporal-knowledge papers (for example SciTBERT, Wieszcz-XIX pre-1918 Polish LMs, NoLBERT Topics, Temporal Leakage in LLM Backtesting, HindsightBench, TEMPO, MemGuard-Alpha membership inference for *memorization-contaminated financial signals*, "Do LLMs Understand Chronology?"). None concerns AI-text detection or web contamination by LLM output. — [Semantic Scholar API, ChronoGPT citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2502.21206/citations?fields=title,externalIds,year,abstract&limit=200)
- PIT citing papers (10): Point-in-Time Encoders for Earnings Calls; FinCutBench; What Improves During Pretraining? (2610.37631); Uncertainty Quantification for Financial Foundation Models survey (2610.33109); When Does "Post-Call" Begin? (2610.21659); AI in Equity and Crypto Markets review (2609.04917); PALM: Point-in-Time Adaptation (2609.30316); A Language Model from 1913 (2606.02991); DatedGPT (2603.11838; its v2 cites "Kelly et al. (2026)"); plus one record without an arXiv ID. — [Semantic Scholar API, PIT citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.11889/citations?fields=title,externalIds,year,abstract&limit=100)
- 2610.37631 (Su, Wu, Yilmaz, Lu; AACL 2026; v1 27 Oct 2026) defines TVS as "the coefficient of variation of the negative log-likelihood assigned to an item's question and correct answer across eight yearly PIT-4B models (Kelly et al., 2026)". It also says "TVS measures likelihood variation, not prevalence". Its appendices include "Contamination Checks: Release-date test. Verbatim overlap" for benchmark contamination. Grep: "AI-generated" 0, "LLM-generated" 0, "ChatGPT" 0. "Synthetic" refers only to a toy "synthetic era" intervention. — [arXiv 2610.37631 HTML](https://arxiv.org/html/2610.37631v1); [abs](https://arxiv.org/abs/2610.37631)
- DatedGPT itself: "a family of twelve 1.3B-parameter language models trained from scratch on approximately 100 billion tokens each with strict annual data cutoffs spanning 2013 to 2024". Data: "We adopt FineWeb-Edu (Penedo et al., 2024) as our base dataset … Each FineWeb-Edu document carries the timestamp at which it was crawled from Common Crawl". Grep: "AI-generated", "LLM-generated", "machine-generated" and "synthetic" all 0. "ChatGPT" appears only in reference titles. Checkpoints: "Model checkpoints are available at hf.co/datedGPT". — [DatedGPT HTML](https://arxiv.org/html/2603.11838)
- DatedGPT's perplexity probing is directly relevant to C3: "we adopt the perplexity-based probing method of Cheng et al. (2024) and evaluate the DatedGPT-base series on quarterly news headlines … A clear reversal appears around the model's 2020 data cutoff, after which perplexity rises steadily … The same pattern holds across all cutoff years." — [DatedGPT HTML](https://arxiv.org/html/2603.11838)
- ChronoGPT: grep for "AI-generated", "LLM-generated" and "machine-generated" all 0. ChatGPT appears only as a post-cutoff *event* in knowledge probes ("In 2022, a major milestone for generative AI was marked by the public release of the AI assistant known as ChatGPT"). — [ChronoGPT HTML](https://arxiv.org/html/2502.21206)
- ChronoGPT-Instruct (2510.11677): "more recent training data consistently enhances instruction-following capabilities, with the 2024 vintage showing the strongest performance". There is no discussion of AI-generated text in post-2022 pretraining data; grep for "AI-generated", "LLM-generated" and "machine-generated" is 0. — [ChronoGPT-Instruct HTML](https://arxiv.org/html/2510.11677)

### Inferences
- Overlap of citing works with the four claims: none for C2, C3 and C4. Only 2610.37631 overlaps partially with C1, as a cross-vintage likelihood-as-instrument design for *knowledge*, not AI style. Cite it as the nearest "vintage-likelihood-as-instrument" precedent.
- DatedGPT's own cutoff "perplexity reversal" is the strongest published evidence *for* the confound in C3. A newer vintage assigns lower perplexity to post-cutoff content because of topical knowledge or membership. The idea's C3 separation, for example same-title human/AI pairs where topic knowledge is matched, should be framed against this figure.
- Two unremarked cross-vintage regularities could be partly driven by AI text entering post-2022 crawls, but neither paper says so: ChronoGPT-Instruct's monotone instruction-following gain to 2024, and PIT's FT IFEval gain for 2021-2024. These are opportunities to cite, not scoops.

### Gaps
- Google Scholar "cited by" was not checked (no automated access attempted). Semantic Scholar coverage of November 2026 arXiv postings may lag by days to weeks.
- Full texts of the citing papers other than 2610.37631 were not read. Exclusion is based on titles and S2 abstracts.

---

## Q3. LAPD / "Alignment Imprint" (arXiv 2604.16923) and IRM (arXiv 2604.21223): formula, model pairs, datasets, LLM-polished human text, training-data date / ChatGPT boundary, numbers vs Binoculars and Fast-DetectGPT

### Takeaway
Both are two-model log-likelihood-ratio detectors of exactly the s_lambda form with lambda = 1, but the pair is **aligned/instruct vs base of the same release**, not two base models split by crawl date. Neither paper mentions training-data dates, knowledge cutoffs, pre-ChatGPT data, or the ChatGPT boundary (0 grep hits for "cutoff" and "knowledge cut"). In LAPD, "temporal" appears once, about Lastde. Neither has a lambda coefficient. LAPD standardizes and information-weights instead; IRM uses the raw sum. IRM evaluates DetectRL's LLM *polishing of human-written text* and reports it as its weakest attack. LAPD does not evaluate polished human text. **Overlap with C4 is substantial in functional form and partial in substance.** The idea's novelty for C4 rests on (a) using a date-split base/base pair, so the "imprint" is diffusion of chat style into the web rather than RLHF, and (b) lambda.

### Cited Findings
- LAPD versions: v1 18 Apr 2026, v2 2 Nov 2026; comment "NeurIPS 2026". Authors: Wu, Huang, Hu, Chen, Wu, Xia, Zou. — [arXiv abs](https://arxiv.org/abs/2604.16923)
- LAPD formula (tool-extracted from v2): Raw Alignment Imprint Δ(x) = log P_φ(x) − log P_θ(x), with P_φ the aligned model and P_θ the base. The weighted score is S(x) = Σ_t I_φ(x_t)·Δ(x_t), with I_φ = −log P_φ(x_t). S_LAPD = (S(x) − μ̃_x)/σ̃_x, with mean and SD estimated by Fast-DetectGPT-style conditional sampling. There is no lambda. — [LAPD v2 HTML](https://arxiv.org/html/2604.16923v2)
- LAPD model pairs: default Llama2-7B (base and perturbation model) with Llama2-7B-Instruct (aligned). Ablations use Falcon-7B, GPT-J-6B and Llama3.1-8B pairs. Cross-family pairs (Falcon-7B/Llama2-7B-Instruct, etc.) "lead to an average AUROC drop ranging from 8.73% to 29.80%" (tool-extracted). — [LAPD v2 HTML](https://arxiv.org/html/2604.16923v2)
- LAPD datasets: M4, DetectRL (Multi-LLM and Multi-Domain), RAID and RealDet (2,000 samples each), plus XSum, WritingPrompts and Arxiv with GPT-4-Turbo, Gemini-2.0-Flash and Claude-3.7-Sonnet sources (tool-extracted). Attacks: token insertion, deletion and replacement, and RAID's "paraphrase, synonym substitution, homoglyph, and zero-width space insertion", applied to AI text. "polish" has 0 hits. "Imitate before detect … machine-revised text detection" appears only as a citation (§2.1, Appendix H). — [LAPD v2 HTML](https://arxiv.org/html/2604.16923v2)
- LAPD headline claim (verbatim): "achieving an average relative improvement of 56.99% over Fast-DetectGPT and 45.82% over the strongest baselines". Code: "https://github.com/creator-xi/LAPD". "Relative" is defined against headroom, (new − old)/(1 − old) (tool-extracted). — [LAPD v2 HTML](https://arxiv.org/html/2604.16923v2)
- LAPD AUROC averages (tool-extracted). Table 1 (M4/DetectRL×2/RAID/RealDet): Fast-DetectGPT 82.26, Binoculars 84.03, DNA-DetectLLM 86.12, raw imprint 85.60, LAPD 92.37. Table 2 (XSum/WP/Arxiv): Fast-DetectGPT 95.36, Binoculars 97.46, DNA-DetectLLM 98.36, raw imprint 95.69, LAPD 99.49. Appendix H also reports "LAPD achieves the best average AUROC of 92.37%" vs supervised ReMoDetect 80.82 and ImBD 83.27. — [LAPD v2 HTML](https://arxiv.org/html/2604.16923v2)
- LAPD temporal discussion: "cutoff" 0 hits, "knowledge cut" 0, "limitation" 0. The single "temporal" hit is "Lastde [38] incorporates temporal dynamics". — [LAPD v2 HTML](https://arxiv.org/html/2604.16923v2)
- IRM: single version v1, 23 Apr 2026; comment "NeurIPS 2025". Authors: Runheng Liu, Heyan Huang, Xingchen Xiao, Zhijing Wu. — [arXiv abs](https://arxiv.org/abs/2604.21223)
- IRM formula (tool-extracted, Eq. 5): r(y) = log(π_θ(y)/π_ref(y)) = Σ_i log(π_θ(y_i|y_<i)/π_ref(y_i|y_<i)), with instruct over base, β log Z dropped as "a constant", and no length normalization. Pairs: Gemma-2-2B-it/Gemma-2-2B and Llama-3.2-1B-Instruct/Llama-3.2-1B, plus Qwen2/2.5 and Llama-3.2-3B families in ablations. Datasets: DetectRL (main) with sources GPT-3.5-turbo, PaLM-2-Bison, Claude-Instant and LLaMA-2-70B; RAID and DivScore in the appendix. — [IRM v1 HTML](https://arxiv.org/html/2604.21223v1)
- IRM on polished human text (verbatim): "IRM is particularly affected by the polishing attack, where LLMs are prompted to improve human-written texts … since the polished texts are generated by LLMs and typically exhibit improved fluency or coherence, they tend to receive higher reward scores than their original, unpolished counterparts. As a result, the distinction between human-written and LLM-generated texts become less clear, leading to a moderate drop in detection performance". — [IRM v1 HTML](https://arxiv.org/html/2604.21223v1)
- IRM temporal discussion: "cutoff" 0, "temporal" 0, "pre-training data" 0. — [IRM v1 HTML](https://arxiv.org/html/2604.21223v1)
- IRM numbers vs baselines (tool-extracted, reported as "average score" with AUROC and F1 not cleanly separated). Llama-3.2-1B family: IRM 91.77 vs Binoculars 87.67 vs Fast-DetectGPT 59.82. Gemma-2-2B family: IRM 77.05 vs Binoculars 77.81 vs Fast-DetectGPT 67.61, so IRM is below Binoculars here. RAID average row for IRM ends "… 92.10"; DivScore "… 96.61". — [IRM v1 HTML](https://arxiv.org/html/2604.21223v1)
- LAPD's related work groups ReMoDetect, IRM and "dual-network preference divergence" as base-vs-aligned gap methods, so this family is already crowded (tool-extracted). — [LAPD v2 HTML](https://arxiv.org/html/2604.16923v2)

### Inferences
- C4 overlap: **substantial in form.** log p_A − log p_B between two related LMs is published (IRM; LAPD's raw imprint; also Binoculars' perplexity/cross-perplexity ratio). The idea cannot claim the score form as new. It can claim (i) the *axis of contrast* (crawl date with identical architecture and data pipeline, base vs base, no alignment step) and (ii) the lambda-weighted generalization, which interpolates between the ratio (lambda = 1) and plain likelihood (lambda = 0).
- C1, C2 and C3 overlap: none. Neither paper treats the model pair as a measurement instrument for diffusion of style into human text, nor addresses membership or knowledge confounds.
- IRM's documented failure on LLM-polished human text is useful baseline evidence. If the temporal pair responds to polished human text the same way, it inherits that limitation. If it responds differently, that is a distinguishing result.
- LAPD's cross-family finding (AUROC drops 8.73-29.80% when the pair is not same-family) supports the "identical except crawl year" design choice.

### Gaps
- LAPD Appendix F/G tables and the IRM Table 3 per-attack numbers were not transcribed verbatim, so the exact polish-attack AUROC drop for IRM is not recorded.
- Whether DetectRL labels polished human text as human or as AI in IRM's setup was not verified from the IRM text. IRM's wording ("distinction between human-written and LLM-generated texts become less clear") suggests the polished texts are treated as the LLM class.

---

## Q4. Visokay et al., "The uneven rise of AI-generated web content" (arXiv 2610.36631): method; any LM likelihood or vintage models?

### Takeaway
It is a design-based survey: a stratified probability sample of 5.1M pages from 71 Common Crawl crawls (Jan 2019 to Aug 2026), scored with off-the-shelf detectors and a Bayesian misclassification model. The only likelihood-based component is Binoculars (Falcon-7B pair) as a secondary detector on English pages. It uses no vintage or date-split LMs. Pre-2022 crawls serve as a *control period* to calibrate detector specificity. **Partial overlap with C2** (a year-curve of AI-text prevalence with an onset after ChatGPT). No overlap with C1, C3 or C4.

### Cited Findings
- v1 27 Oct 2026; "12 pages, 4 figures"; 13 authors (Visokay … McCormick). The abstract reports a sharp but uneven rise, more AI text in English and .com pages and in less-central pages, and that the "Bayesian hierarchical model … corrects for detector false positives and false negatives". — [arXiv abs](https://arxiv.org/abs/2610.36631)
- Data (tool-extracted from full HTML): "a probability sample of 5.1 million pages from the crawlable web from 2019-2026". 71 crawls from Jan 2019 to Aug 2026; 5,118,347 eligible pages from 505,974 domains; two-stage stratified cluster design over six PageRank strata. — [arXiv HTML](https://arxiv.org/html/2610.36631v1)
- Detectors: Open Pangram (supervised, fine-tuned from Llama-3.2-3B) is primary on all pages. Binoculars (Falcon-7B observer/performer) runs on English pages. EditLens RoBERTa-large, Binoculars-Qwen2.5-7B and commercial Pangram 4 were evaluated or used for validation only. Specificity is calibrated on 30 pre-2022 control crawls; sensitivity on EditLens, RAID, a multilingual corpus and AITDNA. Time is modeled with a penalized B-spline with knots at early 2022 and ChatGPT's launch (tool-extracted). — [arXiv HTML](https://arxiv.org/html/2610.36631v1)
- Key numbers (tool-extracted): about 24% of pages in Aug 2026 contain AI-generated or AI-edited text (95% CrI 22.9-24.4). Growth is 0.2 pp/yr (Jan 2019 to Feb 2023), 4.6 pp/yr (Feb 2023 to Jan 2025) and 8.1 pp/yr (Jan 2025 to Aug 2026). By tier: 11.6% most central vs 27.5% least central; English-only 32.8% vs non-English 17.0%; .com 29.5%, .edu 6.1%, .gov 5.8%. — [arXiv HTML](https://arxiv.org/html/2610.36631v1)
- DatedGPT, ChronoGPT and FineWeb are not mentioned in the portion read. The paper estimates page-level prevalence and does not measure diffusion of AI *style* into human writing (tool-extracted). — [arXiv HTML](https://arxiv.org/html/2610.36631v1)

### Inferences
- C2 overlap: partial. It establishes, with careful survey statistics, that AI text on the web is near zero before 2023 and rises from Feb 2023. This is an *external* validation series for the idea's 2023-crawl jump. The idea's C2 differs in measuring absorption by a *model trained on that web* and the likelihood preference for chat text, not page-level prevalence.
- The idea's paper should cite Visokay et al. as the prevalence ground truth and compare its year-curve against their 2019-2026 trajectory. This is especially useful because FineWeb-Edu up to 2024 corresponds to the 4.6 pp/yr segment.

### Gaps
- About 131k of 231k characters of the HTML (later sections and appendices) were not read in full. Absence of vintage-LM use is confirmed for the method sections, but an appendix mention cannot be fully excluded.

---

## Q5. FineWeb (Penedo et al., NeurIPS 2024 D&B): per-dump marker-phrase analysis (onset around CC-MAIN-2023-14); were the 192 per-dump ablation models released?

### Takeaway
The per-dump marker-phrase analysis and the 192 per-crawl models are in the **FineWeb blog post/tech report on Hugging Face, not in the arXiv/NeurIPS paper** (arXiv v2, 31 Oct 2024, has no "delve", "2023-14", "192" or "each dump" passage). The blog tracked ChatGPT marker phrases per crawl and found them "approximately constant until 2023-14" followed by "a steep increase". It trained two 1.8B models per crawl, 192 in total. The **per-crawl models are not public**: none of the 105 models under HuggingFaceFW is a per-dump model. Only filtering and dedup ablations and dataset-comparison ablation models are released. **Partial overlap with C2** (marker-phrase year-curve, not likelihood). Weak partial overlap with C1, since date-split per-crawl models existed but were used only for benchmark scores.

### Cited Findings
- Blog, synthetic-data section (verbatim): "we measured the frequency of the following words in each crawl: "delve", "as a large language model", "it's important to note", "rich tapestry", "intertwined", "certainly!", "dive into" … While the frequency remained approximately constant until 2023-14 (ChatGPT was released at the end of 2022), we find a steep increase of our proxy metric in recent crawls. While this simple test is not enough to conclude that ChatGPT completions and other synthetic data is improving the quality of the most recent crawl, it at the very least does not seem to drastically harm it." — [FineWeb blog (static page)](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html); [Space](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- The motivation links per-crawl model quality to synthetic data: "We wondered if the strong performance of the last few crawls could be, in part, attributed to the presence of a larger quantity of synthetic data (data generated by LLMs) … Since, to the best of our knowledge, there is no foolproof method to detect synthetic data, we opted to use a proxy metric". — [FineWeb blog](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html)
- Per-crawl models (verbatim): "For each crawl, we trained two 1.8B models on 27 billion tokens randomly sampled from that crawl's data … We trained 192 such models, totaling over 60 thousand H100 GPU-hours." Later: "We expect to continue seeing increasing quantities of synthetic data on new CC crawls. However, while for relatively small trainings this data does not seem to harm performance (and might actually improve it), it is not clear that this holds for much larger trainings." — [FineWeb blog](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html)
- The arXiv paper (v2, 31 Oct 2024) has 0 hits for "2023-14", "delve", "each dump" or "per dump". It states: "We make all models trained for our ablations publicly available on our dataset repository. In total, we trained over 70 models". — [arXiv 2406.17557v2 HTML](https://arxiv.org/html/2406.17557v2)
- The Hugging Face API lists 105 models under HuggingFaceFW. None contains "CC-MAIN", "dump" or "crawl" in its name. Released ablation models include ablation-model-fineweb-v1/-edu/-c4/-dolma-v1_6/-refinedweb/-slimpajama/-the-pile/-redpajama2 and ablation-exp-{textext, dedup, filter}-* (28BT/350BT); the rest are finepdfs classifiers. — [HF API listing](https://huggingface.co/api/models?author=HuggingFaceFW&limit=1000)

### Inferences
- C2 overlap: partial. The "flat until 2023-14, then jump" observation for marker phrases per crawl is published (2024) and should be cited as the lexical precedent. The idea's contribution is the *model-side* curve: how much a model trained on each vintage *prefers* chat text, plus a likelihood-based diffusion measure.
- C1 overlap: weak partial. FineWeb did build a per-crawl (date-split, non-cumulative) model family and asked whether later crawls' synthetic data explain better benchmark scores. It never used the models' likelihoods to measure chat-style preference and never released them. This is the closest "before/after-ChatGPT comparison with date-split models" found, and it is benchmark-based.
- C3 and C4: none.

### Gaps
- Whether the 192 per-crawl models were ever released elsewhere (another org or a private collection) was not checked beyond the HuggingFaceFW listing.
- The blog's plot values (marker frequency per crawl) were not transcribed.

---

## Q6. Final sweep (2025-2026): "knowledge cutoff" / "pre-ChatGPT model" / "time-sliced" / "vintage" combined with AI-text detection / LLM contamination / synthetic text in pretraining

### Takeaway
No 2025-2026 preprint was found that uses date-split or vintage LMs (or a pre-ChatGPT vs post-ChatGPT model pair) to detect AI text or to measure AI-style absorption. Adjacent work falls into three groups, each overlapping at most partially with C2:
- word-frequency or distributional estimates of LLM use over time;
- detector-score pre/post-ChatGPT comparisons on human text;
- page or site prevalence surveys.

The two-model likelihood-ratio detector form (C4) is crowded (Binoculars, IRM, LAPD).

### Cited Findings
- Searches for pre-ChatGPT reference models with likelihood-ratio detection and for knowledge-cutoff model pairs found no work pairing older and newer checkpoints for detection. The search tool's summary: "The search did not find work pairing an older and a newer checkpoint of the same model family to exploit knowledge cutoffs." Results were Binoculars, DetectGPT, Fast-DetectGPT and the EACL 2024 "Smaller Language Models are Better Zero-shot Machine-Generated Text Detectors". That paper also reports that, for checkpoints within a training run, "the final checkpoint is consistently the worst" for detection (snippet only). — [EACL 2024 short paper](https://aclanthology.org/2024.eacl-short.25.pdf); [Binoculars](https://arxiv.org/abs/2401.12070)
- "Style as a Confound: False Positives in AI Detection of Non-Native Academic Writing" (Park, Jeong, Kim; arXiv 2608.26710, 27 Aug 2026) compares detector behavior on human-written academic texts "between the pre-ChatGPT (2018–2022) and post-ChatGPT (2023–2025) periods". It treats pre-ChatGPT documents as "human-written texts with negligible AI involvement". Appendix D.1 is titled "Detector Sensitivity Weakens in the post-ChatGPT Era". Detectors include Binoculars, Fast-DetectGPT and others. No vintage LMs: grep for "vintage" and "knowledge cutoff" is 0. — [arXiv 2608.26710 HTML](https://arxiv.org/html/2608.26710)
- "Token-Level Likelihood-Array Regression for Membership Inference and AI-Generated Text Detection" (Sun, Cai; arXiv 2608.22179, 23 Aug 2026) applies one likelihood-feature method to both membership inference and AI-text detection. The contrast is across context lengths, not model dates (abstract-level). — [arXiv abs](https://arxiv.org/abs/2608.22179)
- "What Improves During Pretraining?" (2610.37631) is the only work found using cross-vintage likelihood as an instrument (knowledge TVS on PIT-4B 2017-2024). See Q2. — [arXiv 2610.37631 HTML](https://arxiv.org/html/2610.37631v1)
- Lexical and distributional diffusion studies (search-level, not full-text-read):
  - Human-LLM Coevolution in arXiv abstracts: "delve" usage dropped after it was publicized. — [arXiv 2502.09606](https://arxiv.org/html/2502.09606v2)
  - Linguistic shifts across 823,798 arXiv abstracts. — [arXiv 2505.12218](https://arxiv.org/abs/2505.12218)
  - Is ChatGPT Transforming Academics' Writing Style? — [arXiv 2404.08627](https://arxiv.org/html/2404.08627)
  - Distributional MLE estimate of AI-modified peer reviews. — [arXiv 2403.07183](https://www.alphaxiv.org/abs/2403.07183)
  - AI-associated lexical shifts across 34 languages. — [arXiv 2605.25358](https://arxiv.org/pdf/2605.25358)
  - Reddit detector-score difference-in-differences before and after ChatGPT. — [arXiv 2609.12447](https://arxiv.org/pdf/2609.12447)
- Web prevalence studies (search-level):
  - About 4.30% of sampled Common Crawl sites from 2020-2025 classed as LLM-dominant. — [arXiv 2507.13933](https://arxiv.org/abs/2507.13933)
  - DeGenTWeb follow-up. — [arXiv 2605.00087](https://arxiv.org/pdf/2605.00087)
- ChronoGPT and ChronoGPT-Instruct searches surfaced no cross-vintage perplexity or style-shift analysis tied to ChatGPT. — [ChronoGPT](https://arxiv.org/abs/2502.21206); [ChronoGPT-Instruct](https://arxiv.org/html/2510.11677)

### Inferences
Overlap matrix (C1 temporal pair as instrument / C2 year-curve of chat-style absorption / C3 membership-confound separation / C4 temporal-pair detector):

| Work | C1 | C2 | C3 | C4 | Verified from |
|---|---|---|---|---|---|
| Kelly et al. PIT (2607.11889 v1-v3) | none | none | none | none | full text, all 3 versions |
| DatedGPT (2603.11838) | none | none | partial-adjacent (perplexity reversal at cutoff = knowledge confound, not separated from style) | none | full text |
| ChronoGPT / -Instruct (2502.21206, 2510.11677) | none | none | none | none | full text (grep) |
| What Improves During Pretraining? (2610.37631) | partial (cross-vintage NLL as instrument, for knowledge) | none | none | none | full text (grep) |
| LAPD (2604.16923 v2) | none | none | none | substantial in form (aligned−base LLR); partial in substance | full text |
| IRM (2604.21223 v1) | none | none | none | substantial in form (instruct−base LLR, polish attack evaluated) | full text |
| Visokay et al. (2610.36631) | none | partial (prevalence year-curve, onset 2023) | none | none | ~43% of full text plus abstract |
| FineWeb blog per-crawl analysis | weak partial (192 per-crawl models, benchmark-only, unreleased) | partial (marker-phrase curve flat until 2023-14) | none | none | blog full text, HF listing |
| Style as a Confound (2608.26710) | none | partial (pre vs post-ChatGPT human text, detector scores) | none | none | full text (grep) |
| LAR (2608.22179) | none | none | tangential (MIA and detection with shared likelihood features) | none | abstract only |
| Lexical-shift studies (2502.09606, 2505.12218, 2403.07183, etc.) | none | partial (word-frequency year-curves) | none | none | search snippets only |

- Bottom line: no work found does C1+C2 with date-split LMs, C3 in the form of separating membership or knowledge from style in a temporal pair, or C4 with a date-split base/base pair. The highest-risk items are:
  - C4's score form, which is already published as base/aligned LLR (IRM, LAPD).
  - C2's onset finding, already published lexically (FineWeb blog: "until 2023-14") and as prevalence (Visokay: 0.2 pp/yr before Feb 2023, 4.6 pp/yr after).
  - The public PIT 2013-2024 checkpoints, whose authors had announced "updated results for 2022–2025". Any group could reproduce C1 and C2 quickly.

### Gaps
- Google Scholar and OpenReview were not queried, and S2 indexing of very recent (Nov 2026) preprints may be incomplete.
- The lexical-shift and prevalence studies listed above were seen only via search summaries and were not read in full.
- No search was run in non-English venues.
