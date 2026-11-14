# Time-locked / temporally split language models and their uses, and whether any were used for AI-text detection or measuring AI-text diffusion (as of 2026-11-14)

Scope note: about 28 searches and fetches (arXiv, ACL Anthology, Hugging Face, GitHub, project sites, news coverage). Items marked **[bg]** are well-known papers cited from background knowledge with their standard arXiv/venue URLs. I did not re-fetch them in this session. Items marked **[secondary]** were seen only through aggregators or news coverage. Everything else was read from a primary page in this session.

---

## Q1. DatedGPT: authors, what it is, data, per-year vs cumulative, experiments, and whether it mentions AI-generated text after 2022

### Takeaway
DatedGPT is a finance/economics paper (CUHK Finance and UCL AI Centre) built to remove lookahead bias. It has twelve 1.3B models trained from scratch on FineWeb-Edu filtered by Common Crawl crawl year (2013–2024), with about 100B tokens each. The paper never discusses AI-generated text, ChatGPT's effect on web text, or post-2022 web change. ChatGPT appears only in the titles of cited references. I could not confirm the "NeurIPS 2026" venue. A footnote refers to ACL guidelines, and the work also appears on the AEA/ASSA 2026 programme.

### Cited Findings
- **Title / ID:** "DatedGPT: Preventing Lookahead Bias in Large Language Models with Time-Aware Pretraining," arXiv:2603.11838. v1 was posted 12 Mar 2026 and v2 on 23 Jul 2026. Categories are cs.CL (primary) and q-fin.GN. The arXiv abstract page shows no venue comment. — [arXiv abs](https://arxiv.org/abs/2603.11838)
- **Authors:** Yutong Yan, Zhenyu Gao and Wenxi Jiang (Department of Finance, CUHK Business School), and Raphael Tang and Yao Lu (UCL Centre for Artificial Intelligence). — [arXiv v2 HTML](https://arxiv.org/html/2603.11838v2)
- **Venue:** No NeurIPS 2026 statement appears on arXiv. A footnote says a concurrent work "qualifies as concurrent work under the ACL guidelines," which points to an ACL-family submission. A DatedGPT paper is listed on the AEA 2026 conference programme. — [arXiv v2 HTML](https://arxiv.org/html/2603.11838v2); [AEA 2026 program listing](https://www.aeaweb.org/conference/2026/program/paper/BG2Tayk7)
- **What it is:** Twelve 1.3B-parameter models, one per annual cutoff from 2013 to 2024. Each is trained from scratch on about 100B tokens. The paper also introduces DatedInstruct, instruction data built from each year's documents. Perplexity probing confirms that each model's knowledge stops at its cutoff year. — [arXiv abs](https://arxiv.org/abs/2603.11838)
- **Data:** FineWeb-Edu filtered by Common Crawl crawl timestamp, "rounded to the year." The paper does not name specific crawl dumps. — [arXiv v2 HTML](https://arxiv.org/html/2603.11838v2)
- **Per-year vs cumulative:** The paper is ambiguous. The introduction says each model uses data "available up to its designated cutoff year," which reads as cumulative. The methods section says "approximately 100 billion tokens per year." — [arXiv v2 HTML](https://arxiv.org/html/2603.11838v2)
- **Architecture:** 1.34B total parameters (0.13B embedding, 1.21B non-embedding), 24 layers, d=2048, FFN 5504, 16 heads, RoPE, SwiGLU, RMSNorm, sequence length 2048, vocabulary 32,000. The tokenizer is not named. Training uses LR 2e-4, batch 2048, 25,000 iterations, about 100B tokens and about 2,000 A100-hours per model. — [arXiv v2 HTML](https://arxiv.org/html/2603.11838v2)
- **Post-training:** DatedInstruct runs for 3 epochs with 10% pretraining data mixed in, about 60K QA pairs per year. In v2 the QA pairs are LLM-generated: DeepSeek-V4-Pro converts each year's documents into QA, with rule-based verification. — [arXiv v2 HTML](https://arxiv.org/html/2603.11838v2)
- **Experiments:**
  - Zero-shot and 5-shot benchmarks: ARC-C/E, HellaSwag, PIQA, IFEval, MMLU, TruthfulQA.
  - AlpacaEval 2.0 win rates against ChronoGPT and PIT.
  - DatedInstruct applied to ChronoGPT and PIT base checkpoints (cross-family tuning).
  - Perplexity probing on quarterly company headlines, 2013–2024.
  - Stock-return prediction on 61,290 firm-day headlines, 2014–2024: panel regressions, long-short Sharpe ratios, and a lookahead-premium test.
  - Source: [arXiv v2 HTML](https://arxiv.org/html/2603.11838v2)
- **Headline results:** DatedGPT-instruct reaches an annualised Sharpe ratio of 3.20 in the bias-free setup. Models whose training covers the outcome period add a "lookahead premium" of 26.4 bp per SD, significant at 1%. — [arXiv abs](https://arxiv.org/abs/2603.11838)
- **v1 and v2 differ:** Instruct average scores are 40.1–42.7 in v1 and 39.4–41.8 in v2. — [arXiv v1](https://arxiv.org/html/2603.11838v1); [arXiv v2](https://arxiv.org/html/2603.11838v2)
- **Release:** Hugging Face organisation `datedGPT` lists 24 models (base and instruct, 2013–2024, shown as "1B"), plus the dataset `datedgpt/datedgpt-instruct-train` (about 765k) organised by year. The org page states no licence and no tokenizer, and describes the organisation as "non-profit." A demo is at datedgpt.com. — [HF org](https://huggingface.co/datedGPT); [arXiv v2](https://arxiv.org/html/2603.11838v2)
- **AI-generated text and contamination:** The paper does not discuss AI-generated or LLM-generated content in the post-2022 crawls, or post-2022 web change. "ChatGPT" appears only in reference titles. "Contamination" or leakage in the paper means future-information leakage. The authors criticise other instruction datasets because responses "written by human annotators or generated by modern LLMs" may carry out-of-cutoff knowledge. The page shows no limitations section. — [arXiv v2 HTML](https://arxiv.org/html/2603.11838v2)
- **Prior circulation:** DatedGPT was already cited in the related work of a 20 Jan 2026 benchmark paper, before the arXiv posting. — [Look-Ahead-Bench, arXiv 2601.13770](https://arxiv.org/html/2601.13770v1)

### Inferences
- The token arithmetic points to single-year data. 25,000 iterations × 2,048 batch × 2,048 tokens ≈ 105B tokens, the same budget every year. FineWeb-Edu as a whole is about 1.3T tokens over 2013–2024 crawls ([FineWeb paper](https://arxiv.org/abs/2406.17557)), roughly 100B tokens per crawl-year. Training on only that year's crawls for one epoch therefore fits "~100B tokens per year" naturally. Each vintage is probably one crawl-year slice, though crawl-year slices also contain older pages (see Q5, "Dated Data"). This is an inference. Confirm with the authors or the model cards.
- DatedInstruct responses are generated by DeepSeek-V4-Pro for every year, so all instruct vintages, including 2013, are exposed to modern-LLM style. Only the base models are suitable as "pre-ChatGPT" references in a likelihood instrument.
- FineWeb-Edu's educational-quality classifier was trained on Llama-3-70B-Instruct annotations ([FineWeb paper](https://arxiv.org/abs/2406.17557)). It may keep LLM-like expository text from post-2022 crawls at a different rate than human text. That would make the 2023 jump partly a filtering effect. This is a possible confound to discuss.
- The paper's own experiments (benchmarks, headline perplexity, returns) never analyse the 2022→2023 transition in style or likelihood. A 12-vintage curve of chat-text preference that jumps at 2023 is not reported in the DatedGPT paper.

### Gaps
- NeurIPS 2026 acceptance could not be verified. arXiv has no comment field, and the paper references ACL guidelines.
- The model licence, tokenizer identity (whether a single shared tokenizer is used and what data it was trained on), and the exact crawl dumps per year are not stated on the pages I read.
- The cumulative vs single-year question is not settled in the text.

---

## Q2. ChronoGPT, TimeMachineGPT, and point-in-time / lookahead-bias-free LLMs for finance and economics

### Takeaway
A dense 2024–2026 literature builds date-locked LMs for finance and economics backtesting: TiMaGPT (2024), ChronoBERT/ChronoGPT (2025), ChronoGPT-Instruct (2025), DatedGPT (2026), PIT models (2026), and commercial Pitinf models, plus benchmarks and adapters. Every one is motivated by lookahead bias in forecasting. None of the papers I read discusses AI-generated text in its post-2022 training data or uses vintages to study it.

### Cited Findings
**Catalogue of time-locked model families**

| Family | Authors / venue / date | Sizes, vintages | Data and training | Tokenizer, release, licence | AI-text mention? |
|---|---|---|---|---|---|
| **DatedGPT** | Yan, Tang, Gao, Jiang, Lu; arXiv 2603.11838 (Mar/Jul 2026) | 1.3B ×12, annual 2013–2024, base and instruct | FineWeb-Edu by crawl year, about 100B tokens each, from scratch | 32k vocabulary (tokenizer not named); HF `datedGPT`; licence not stated | No — [arXiv](https://arxiv.org/html/2603.11838v2) |
| **ChronoBERT / ChronoGPT** | Songrun He, Linying Lv, Asaf Manela, Jimmy Wu (WashU); arXiv 2502.21206 (v3, 6 Jul 2025); AFA session at ASSA 2026 | ChronoBERT 149M; ChronoGPT 1,552M (1.5B), context 1,792; 26 yearly vintages 1999–2024 per series | About 7B tokens of pre-2000 text; ChronoGPT1999 trained 71B tokens (~10 epochs); about 65B incremental tokens 2000–2024; each year continues from the previous checkpoint; FineWeb-edu classifier score >2; CC data added from 2013 | Vocabulary 50,304; HF `manelalab`; licence not stated | No. "ChatGPT" appears only as a 2022 event-prediction target — [arXiv](https://arxiv.org/html/2502.21206v3); [AEA session](https://www.aeaweb.org/conference/2026/program/1904) |
| **ChronoGPT-Instruct** | Same group; "Instruction Tuning Chronologically Consistent Language Models," arXiv 2510.11677 | Instruct versions, e.g. ChronoGPT-Instruct1999 | Instruction-tuned only on time-appropriate data; framed as a "conservative lower bound" on lookahead bias | — | Not found in snippet — [arXiv](https://arxiv.org/pdf/2510.11677) |
| **TimeMachineGPT (TiMaGPT)** | Felix Drinkall, Eghbal Rahimikia, Janet Pierrehumbert, Stefan Zohren; arXiv 2404.18543; Findings of NAACL 2024 | GPT-2 small, 117M; yearly TiMaGPT'11–'22 (cutoffs 2011–2022) | Each trained from scratch on 2.5B tokens; WMT News (5-year window, recency-weighted) and Wikipedia revisions as of 31 Dec of each year; WMT:Wiki 0.6:0.4 | GPT-2-style BPE (not stated as time-restricted); HF `Ti-Ma`; models and data CC0 | No — [arXiv](https://arxiv.org/html/2404.18543v1); [ACL Anthology](https://aclanthology.org/2024.findings-naacl.208) |
| **PIT (Scaling Point-in-Time LMs)** | Bryan Kelly, Semyon Malamud, Johannes Schwab, Teng Andrea Xu; arXiv 2607.11889 (v2, 17 Jul 2026); NBER WP w35247 | PIT-1.5B (170B tokens); PIT-4B (about 4.2B, 1T tokens); **monthly** checkpoints 2013–2024 | FineWeb (96 CC snapshots) as a chronologically ordered stream (implies cumulative, continual training); fine-tuning on Tulu 3 / Magpie filtered by gpt-5-nano; 2022–2025 results deferred to a later version | Vocabulary 50,304; code at GitHub `anonymouspitllm-collab/PIT-LLM`; HF `Diamegs`; licence not stated | No — [arXiv](https://arxiv.org/html/2607.11889v2); [NBER PDF](https://www.nber.org/system/files/working_papers/w35247/w35247.pdf) |
| **Pitinf (PiT-Inference)** | Commercial; benchmarked in Look-Ahead-Bench (Mostapha Benhenda, arXiv 2601.13770, 20 Jan 2026) | Small (<10B), Medium (<100B), Large (>500B); effective cutoff Jan 2020 | Proprietary | Proprietary | No — [arXiv](https://arxiv.org/html/2601.13770v1) |

**Related finance and economics lookahead work**
- **Look-Ahead-Bench** (Benhenda 2026) measures alpha decay. Llama 3.1 8B decays −17.23 pp, 70B −15.25 pp, DeepSeek 3.2 −21.77 pp. Pitinf models show about 0 to +1.3 pp. Only Pitinf models are benchmarked; ChronoGPT, TimeMachineGPT and DatedGPT appear only in related work. The study is a proof of concept on 5 stocks. — [arXiv 2601.13770](https://arxiv.org/html/2601.13770v1)
- **Look-Ahead-Bench also cites:** GPT-4o recalls S&P 500 closing prices with <1% error inside its training window, with errors rising after the cutoff. — [arXiv 2601.13770](https://arxiv.org/html/2601.13770v1)
- **PALM (Point-in-Time Adaptation for Financial LMs):** freezes the base weights and fits a low-rank adapter on text published between the checkpoint cutoff and the decision date. — [arXiv 2609.30316](https://arxiv.org/pdf/2609.30316)
- **HindsightBench:** a black-box behavioural audit for "parametric hindsight" in time-indexed decision tasks. — [arXiv 2607.18867](https://arxiv.org/pdf/2607.18867)
- **Stated cutoffs are not enough:** an argument that stated cutoffs are opaque and do not exclude post-cutoff information, with point-in-time retrieval proposed instead. — ["Evaluating LLMs in Finance Requires Explicit Bias Consideration," arXiv 2602.14233](https://arxiv.org/html/2602.14233v1)
- **Fixes that do not retrain:** "entity neutering" (Engelberg et al. 2025) and the ChronoGPT approach are summarised in University of Chicago FINM course notes. Merchant & Levy's logit-adjustment approach is cited in Look-Ahead-Bench. — [FINM 33200 notes](https://finm-33200.github.io/discussions/llms_and_lookahead_bias.html); [arXiv 2601.13770](https://arxiv.org/html/2601.13770v1)
- **[bg]** Glasserman & Lin, "Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis" (2023). — [arXiv 2309.17322](https://arxiv.org/abs/2309.17322)

### Inferences
- **Different training regimes give different instrument properties:**
  - DatedGPT is from scratch with an equal token budget per vintage, so vintages are independent and comparable.
  - ChronoGPT is continued pretraining. Each new year adds only a few billion tokens on top of a 71B-token 1999 base (about 65B tokens across 25 years), so ChronoGPT-2022 vs -2023 is a small fine-tuning delta, not two independent models.
  - PIT is a single chronological stream with monthly checkpoints. It is the only family with sub-annual resolution around Nov 2022, but recency and curriculum effects are confounded with era.
  - TiMaGPT stops at a 2022 cutoff, so it has no post-ChatGPT vintage.
- In this literature, "contamination" means future-information leakage. Any paper using these families for AI-text work should define its terms explicitly to avoid confusion.

### Gaps
- **FinText** (year-specific financial models by Rahimikia and Drinkall) and **NoLBERT** (a "no lookahead/lookback" BERT) came up in my memory and in course-note context. I did not verify either in this session and give no URL.
- Sarkar & Vafa, "Lookahead Bias in Pretrained Language Models" (2024, SSRN) is unverified. I have no URL for it.
- Licences for ChronoGPT and PIT are not stated in the papers. I did not check the HF model cards.

---

## Q3. Temporal-LM research: temporal knowledge, misalignment, adaptation, alignment, continual pretraining, and checkpoint suites

### Takeaway
Before the finance work, the time-split literature studied temporal generalisation and decay, time-conditioned knowledge, temporal adaptation, and the effective knowledge cutoff. Two items are most relevant here. TiC-LM (Apple, ACL 2025) splits 114 monthly CC dumps (2013–2024). "Dated Data" (COLM 2024) uses perplexity over dated document versions as a measurement instrument. None of these uses vintages to detect or measure AI-generated text. Pythia and OLMo checkpoints are training-step snapshots on a fixed corpus, not data-date vintages.

### Cited Findings
- **TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining** (Li et al., Apple; ACL 2025). TiC-CommonCrawl covers 114 monthly CC dumps from May 2013 to July 2024, 2.9T tokens, revealed one month at a time. Domain evaluations are TiC-Wiki, TiC-StackExchange and TiC-CodeDocs. Autoregressive meta-schedules with fixed-ratio replay match retraining from scratch with 2.6× less compute. Code is at github.com/apple/ml-tic-lm. — [arXiv 2504.02107](https://arxiv.org/html/2504.02107v3); [ACL Anthology](https://aclanthology.org/2025.acl-long.1551)
- **Dated Data: Tracing Knowledge Cutoffs in LLMs** (Cheng, Marone, Weller, Lawrie, Khashabi, Van Durme; JHU; COLM 2024, outstanding paper).
  - The effective cutoff is the minimum of perplexity over time-stamped document versions (Wikipedia histories, NYT), and no training-data access is needed.
  - Effective cutoffs often differ drastically from reported ones, partly because CC dumps contain a lot of old data and deduplication misses near-duplicates.
  - Source: [arXiv 2403.12958](https://arxiv.org/html/2403.12958v2); [Ai2 blog](https://allenai.org/blog/knowledge-cutoffs); [code](https://github.com/nexync/dated_data/)
- **Set the Clock: Temporal Alignment of Pretrained LMs** (Bowen Zhao, Zander Brumbaugh, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith; Findings of ACL 2024). The TAQA dataset covers 2000–2023. LLaMa2 (cutoff 2022) mostly answers with about 2019 knowledge. Aligning it to 2022 improves performance by up to 62%. Code is at yizhongw/llm-temporal-alignment. — [arXiv 2402.16797](https://arxiv.org/abs/2402.16797v2); [ACL Anthology](https://aclanthology.org/2024.findings-acl.892)
- **TimeLMs: Diachronic Language Models from Twitter** (Loureiro et al., ACL 2022 demo). A series of Twitter RoBERTa models updated quarterly. — [arXiv 2202.03829](https://arxiv.org/pdf/2202.03829)
- **[bg]** Dhingra et al., "Time-Aware Language Models as Temporal Knowledge Bases" (TACL 2022). Introduces TempLAMA and time-prefix conditioning of T5. — [arXiv 2106.15110](https://arxiv.org/abs/2106.15110)
- **[bg]** Lazaridou et al., "Mind the Gap: Assessing Temporal Generalization in Neural Language Models" (NeurIPS 2021). Perplexity degrades on future data. — [arXiv 2102.01951](https://arxiv.org/abs/2102.01951)
- **[bg]** Luu et al., "Time Waits for No One! Analysis and Challenges of Temporal Misalignment" (NAACL 2022). — [arXiv 2111.07408](https://arxiv.org/abs/2111.07408)
- **[bg]** Röttger & Pierrehumbert, "Temporal Adaptation of BERT and Performance on Downstream Document Classification" (Findings of EMNLP 2021). — [arXiv 2104.08116](https://arxiv.org/abs/2104.08116)
- **[bg]** Agarwal & Nenkova, "Temporal Effects on Pre-trained Models for Language Processing Tasks" (TACL 2022). — [arXiv 2111.12790](https://arxiv.org/abs/2111.12790)
- **[bg]** Jang et al., "TemporalWiki" (EMNLP 2022). — [arXiv 2204.14211](https://arxiv.org/abs/2204.14211)
- **[bg]** Nylund, Gururangan & Smith, "Time is Encoded in the Weights of Finetuned Language Models" (ACL 2024). Introduces "time vectors" from per-year fine-tunes, which can be interpolated. — [arXiv 2312.13401](https://arxiv.org/abs/2312.13401)
- **[bg]** Pythia (Biderman et al., ICML 2023). 154 checkpoints per model, all trained on the Pile in a fixed order, so the checkpoints index training steps, not data date. — [arXiv 2304.01373](https://arxiv.org/abs/2304.01373). OLMo (Groeneveld et al., ACL 2024) releases intermediate checkpoints of the same kind. — [arXiv 2402.00838](https://arxiv.org/abs/2402.00838)
- **ChronoCulture** (2026). A benchmark showing that LLMs often present a country's evolving culture as largely static across years. — [arXiv 2610.20472](https://arxiv.org/html/2610.20472v1)
- **Naming collisions:** "Time Travel in LLMs: Tracing Data Contamination" (2023) and "Test of Time: Rethinking Temporal Signal of Benchmark Contamination" (2025) are about benchmark contamination, not time-locked models. — [ResearchGate PDF of arXiv 2308.08493](https://www.researchgate.net/publication/373163691_Time_Travel_in_LLMs_Tracing_Data_Contamination_in_Large_Language_Models/fulltext/64dd8e0778e40b48bd4f15e8/Time-Travel-in-LLMs-Tracing-Data-Contamination-in-Large-Language-Models.pdf); [arXiv 2509.00072](https://arxiv.org/pdf/2509.00072)

### Inferences
- "Dated Data" is the closest methodological prior art for using LM likelihood over time as a measuring device. It measures when a model's knowledge ends, not whether text is machine-written.
- Its finding that CC dumps carry much older content implies that a "crawl-year" vintage is a mixture of authoring years. This dilutes any post-ChatGPT signal in the 2023 vintage, so the observed jump is, if anything, a lower bound on the change in newly authored text.
- TiC-LM's monthly CC splits are a ready-made alternative dataset (and possibly a source of checkpoints) for testing whether the 2023 discontinuity appears at monthly resolution.

### Gaps
- I did not check whether TiC-LM released per-timestep checkpoints, or whether the paper comments on LLM-generated text in 2023–2024 dumps.
- ChroKnow (ICLR 2025) and other 2025–2026 temporal-knowledge benchmarks were not checked.

---

## Q4. Historical ("vintage") LMs and 2025–2026 time-stamped releases

### Takeaway
2025–2026 saw a wave of "vintage LLMs" trained only on pre-1900 to pre-1946 text: Ranke-4B, TypewriterLM, talkie-1930, GPT-1900, TimeCapsuleLLM, Mr. Chatterbox, and StoriesLM (yearly 1900–1963). Their stated uses are humanities and social-science "windows into the past," hindsight-free forecasting and invention tests, and contamination-free benchmarks. None targets AI-text detection. Talkie is notable for releasing a same-architecture, same-FLOPs "modern twin" trained on FineWeb for controlled comparisons.

### Cited Findings
- **Ranke-4B / History LLMs**
  - Authors: Daniel Göttlich, Dominik Loibner, Hans-Joachim Voth (University of Zurich) and Guohui Jiang (Cologne).
  - Models: 4B, Qwen3 architecture, trained from scratch on about 80B tokens drawn from about 600B curated time-stamped tokens. Cutoffs are 1913, 1929, 1933, 1939 and 1946.
  - Status: announced 2025-12-14, with artifacts and a working paper "coming soon."
  - Purpose: "windows into the past" and avoiding "hindsight contamination."
  - It does not mention AI-generated text.
  - Source: [GitHub history-llms](https://github.com/DGoettlich/history-llms); [Gigazine](https://gigazine.net/gsc_news/en/20251222-ranke-4b) [secondary]
- **TypewriterLM** ("A Language Model from 1913: Pretraining on Historical Text," Luo, Shinnick, Griesshaber, Wang, Yu, Shi, Torr, Yao Lu and others; Waterloo, Vector, Adelaide, Oxford, UCL; arXiv 2606.02991, v2 2 Oct 2026).
  - 7.24B parameters, 1913 cutoff, trained on the 54B-token TypewriterCorpus.
  - Introduces the History-Event benchmark (2,344 events) and "lexically grounded instruction tuning."
  - Venue not stated. It does not mention AI-generated text.
  - Yao Lu is also a DatedGPT co-author.
  - Source: [alphaXiv](https://www.alphaxiv.org/abs/2606.02991)
- **talkie-1930** (Nick Levine, David Duvenaud, Alec Radford; April 2026).
  - 13B base and instruct, 260B tokens of pre-1931 English (books, newspapers, journals, patents, case law). An earlier 7B version also exists.
  - **Modern twin** `talkie-web-13b-base`: identical architecture and FLOPs, trained on FineWeb, so that differences "trace to training data."
  - The article notes that modern models are trained on the web "directly or indirectly (via distillation and synthetic data)," but runs no AI-text analysis.
  - Licence not stated.
  - Source: [talkie-lm.com](https://talkie-lm.com/)
- **Other vintage models, from the curated "awesome-vintage-llms" list:**
  - **GPT-1900 / Machina Mirabilis** (Michael Hla, 2026): 3.3B, pre-1900, a test of whether the model can rediscover quantum mechanics and relativity.
  - **TimeCapsuleLLM** (Hayk Grigorian, 2025–26): 16M–1.2B, London texts 1800–1875.
  - **Mr. Chatterbox** (Trip Venturella, 2026): about 340M, 1837–1899, about 2.93B tokens.
  - **MonadGPT** (2023): a Mistral fine-tune, so not clean-room.
  - The list does not discuss AI-text detection.
  - Source: [awesome-vintage-llms](https://github.com/entanglr/awesome-vintage-llms)
- **StoriesLM** ("A Family of Language Models With Sequentially-Expanding Pretraining Windows"). Yearly models from 1900 to 1963, each continuing from the previous year's checkpoint, trained on the American Stories historical-newspaper dataset. — [HF StoriesLM-v1-1911](https://huggingface.co/StoriesLM/StoriesLM-v1-1911); [HF v1-1963](https://huggingface.co/StoriesLM/StoriesLM-v1-1963)
- **"Vintage Large Language Models"** talk by Owain Evans (2024), credited with naming the field.
  - Proposed uses: forecasting via backtesting (e.g., a model trained through 2019 tested on 2019–2024), reinvention of later ideas, humanistic dialogue, intellectual history, and gold-standard data for epistemic AI.
  - It does not mention AI-text detection or pre- vs post-ChatGPT models.
  - Source: [transcript](https://owainevans.github.io/talk-transcript.html)
- **Earlier and related:**
  - Hosseini et al. 2021, BERT and word2vec on time-sliced 19th-century corpora. — [arXiv 2105.11321](https://arxiv.org/abs/2105.11321)
  - PNAS 2024, arguing that LLMs trained on historical text can serve behavioural science. — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2407639121)
  - Benjamin Breen's 2026 taxonomy of vintage-LLM uses. — [Res Obscura](https://resobscura.substack.com/p/are-vintage-llms-the-start-of-a-new)

### Inferences
- The vintage-LLM community's "contamination-free" framing concerns hindsight and knowledge leakage. The "pre-AI web as uncontaminated reference" idea appears in data-archiving circles (Q5), not in vintage-LLM papers.
- Talkie's vintage-vs-modern twin is the closest analogue to a controlled "identical except for data era" comparison, but its contrast is 1930 vs present, not pre- vs post-ChatGPT.

### Gaps
- I did not check whether the Ranke-4B working paper or weights have been released since December 2025.
- StoriesLM's author, paper venue and licence were not confirmed in this session.

---

## Q5. Has anyone compared models trained before vs after ChatGPT, or used time-locked models, to study AI-generated content, ChatGPT-isms or style drift on the web?

### Takeaway
I found **no work, through November 2026, that uses a time-locked model family (DatedGPT, ChronoGPT, TiMaGPT, PIT, TiC-LM, or the historical LMs) as an AI-text detector or as an instrument for measuring AI-text diffusion or ChatGPT's effect on web text.** The closest prior art is the FineWeb project (2024). It trained identical 1.8B models on every Common Crawl dump, 2 models × 96 dumps = 192 models, and separately counted ChatGPT-style phrases per dump. The phrase counts were flat until the 2023-14 crawl and rose steeply after it. It did not use model likelihood as the measurement or build a detector. Web-scale diffusion estimates published in 2026 (Pew, Visokay et al., Graphite) all rely on trained AI-text detectors, not on vintage models.

### Cited Findings
- **FineWeb per-dump ablation models.** "For each crawl, two 1.8B-parameter models trained on 27 billion tokens" sampled from that crawl: 192 models, more than 60k H100-hours. Scores varied a lot by crawl, and the most recent crawls performed strongly. — [FineWeb blog (NeurIPS 2024 D&B)](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html)
- **FineWeb ChatGPT-phrase proxy.** The phrases counted were "delve," "as a large language model," "it's important to note," "rich tapestry," "intertwined," "certainly!" and "dive into." Frequency was roughly flat until 2023-14, followed by "a steep increase of our proxy metric in recent crawls," tied to ChatGPT's release. The authors concluded the evidence was "not enough to conclude" that synthetic data improves recent crawls, but that it "does not seem to drastically harm it." — [FineWeb blog](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html)
- **FineWeb paper.** Authors are Penedo, Kydlíček, Ben allal, Lozhkov, Mitchell, Raffel, von Werra, Wolf. FineWeb-Edu's classifier uses Llama-3-70B-Instruct synthetic annotations. — [arXiv 2406.17557](https://arxiv.org/html/2406.17557); [search summary noting the authors expect increasing synthetic data in new crawls](https://arxiv.org/pdf/2406.17557)
- **Pew Research Center (20 Aug 2026).** About 490k English Common Crawl pages scored with a single detector (Open Pangram). About 10% of all pages showed significant signs of AI authorship, rising to about 35% among pages published after ChatGPT. Only 10–15% of pages had usable publication dates. — [Pew methodology](https://www.pewresearch.org/data-labs/2026/08/20/methodology-ai-content/); [The Decoder](https://the-decoder.com/pew-study-shows-ai-written-text-has-surged-across-the-web-since-late-2022/) [secondary]
- **Visokay et al.** (arXiv 2610.36631, 27 Oct 2026) **[secondary; the arXiv paper itself was not reached]**.
  - Scope: 5,118,347 pages on 505,974 domains across 71 CC snapshots, Jan 2019 to Aug 2026, using detector-corrected Bayesian estimates.
  - Prevalence: about 24% AI-generated or AI-edited in Aug 2026 (95% CrI 22.9–24.4).
  - Annual growth: +0.2 pp (2019 to Feb 2023), +4.6 pp (Feb 2023 to Jan 2025), +8.1 pp (2025 to Aug 2026).
  - By domain: .com about 29.5%, .edu about 6.1%, .gov about 5.8%.
  - Sources disagree on the author list.
  - Source: [FourWeekMBA](https://fourweekmba.com/ai-written-text-reaches-24-of-web-pages-up-from-near-zero/); [FourWeekMBA 2](https://fourweekmba.com/ai-text-is-on-24-of-web-pages-up-from-near-zero-pre-2022/)
- **Graphite.** 55,400 English CC articles. Articles classified as mostly AI-generated were an estimated 49.9% in Q1 2026, roughly stable at about 50% since a post-ChatGPT surge. The three detectors used define "AI-generated" differently. — [Search Engine Land](https://searchengineland.com/nearly-half-online-articles-ai-generated-study-478233) [secondary]
- **Distributional estimates (Liang et al.).** "Monitoring AI-Modified Content at Scale" (ICML 2024) applies a distributional maximum-likelihood estimate to peer reviews, not vintage models. — [arXiv 2403.07183](https://arxiv.org/pdf/2403.07183). **[bg]** Liang et al., "Mapping the Increasing Use of LLMs in Scientific Papers" (COLM 2024). — [arXiv 2404.01268](https://arxiv.org/abs/2404.01268). **[bg]** Liang et al. 2025, "The Widespread Adoption of LLM-Assisted Writing Across Society." — [arXiv 2502.09747](https://arxiv.org/abs/2502.09747)
- **Excess vocabulary.** **[bg]** Kobak et al., "Delving into ChatGPT usage in academic writing through excess vocabulary" (2024; Science Advances 2025), a word-frequency method. — [arXiv 2406.07016](https://arxiv.org/abs/2406.07016). Similarly, PubMed term changes after ChatGPT. — [medRxiv 2024.05.14.24307373](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373.full.pdf)
- **Pre-AI corpora as "low-background steel."** John Graham-Cumming's lowbackgroundsteel.ai catalogues pre-2022 human content, using the end of 2022 as the cutoff. The Python `wordfreq` library stopped updating, citing LLM-generated web content. — [CO/AI](https://getcoai.com/news/former-cloudflare-exec-launches-archive-of-pre-ai-human-content/) [secondary]; [New Scientist via AITopics](https://aitopics.org/doc/news:56CC1D19) [secondary]
- **[bg]** Model collapse: Shumailov et al., "AI models collapse when trained on recursively generated data," Nature 2024. — [Nature](https://www.nature.com/articles/s41586-024-07566-y)
- **Searches with no matches.** Searches for detectors using a pre-2022 reference model, for DatedGPT or ChronoGPT combined with AI-generated text, and for perplexity trends of web text across CC snapshots found no matching work. — [search: pre-2022 reference detector](https://arxiv.org/pdf/2310.15264); [search: DatedGPT/ChronoGPT + detector](https://arxiv.org/pdf/2608.22179); [search: web text LLM-like trend](https://arxiv.org/pdf/2406.17557). These URLs are representative of the result sets returned; none matched.
- **Time-locked papers read in full.** DatedGPT, ChronoGPT, PIT, TiMaGPT, Look-Ahead-Bench, TypewriterLM and talkie were each checked, and none discusses AI-generated content in post-2022 training data. — [DatedGPT](https://arxiv.org/html/2603.11838v2); [ChronoGPT](https://arxiv.org/html/2502.21206v3); [PIT](https://arxiv.org/html/2607.11889v2); [TiMaGPT](https://arxiv.org/html/2404.18543v1); [Look-Ahead-Bench](https://arxiv.org/html/2601.13770v1); [TypewriterLM](https://www.alphaxiv.org/abs/2606.02991); [talkie](https://talkie-lm.com/)

### Inferences
- **Novelty assessment.** Using a vintage family as an instrument appears unclaimed. That instrument is the likelihood difference between a post-ChatGPT and a pre-ChatGPT vintage, or a 12-vintage preference curve that is flat through 2022 and jumps at 2023, applied either to measure diffusion or as a training-free detector. The nearest work and what separates it:
  1. **FineWeb (2024):** an identical model per dump, plus a ChatGPT-phrase time series with the same 2023 onset. It uses word counts and benchmark scores, has no likelihood instrument and no detector, and its per-dump models are not (to my knowledge) a released family.
  2. **Dated Data (2024):** perplexity-over-time as an instrument for knowledge cutoffs, not for authorship.
  3. **Detector-based prevalence studies (Pew, Visokay, Graphite, 2026):** they measure the same phenomenon with supervised or commercial detectors.
  4. **Talkie's modern twin:** a controlled data-era comparison, but not around 2022.
- **Corroboration.** FineWeb's flat-then-rising phrase proxy from the 2023-14 crawl, and Visokay's growth break around Feb 2023, are independent evidence consistent with a vintage curve that is flat through 2022 and jumps at 2023. Either can be cited as external validation.
- **Reviewer threats to pre-empt:**
  - Recency and newer-topic effects. Post-2022 vintages also know 2023 events and vocabulary. Use same-title human vs chat pairs, as the context describes, and perhaps date-neutral topics.
  - The FineWeb-Edu classifier, trained on LLM labels, may filter selectively.
  - Crawl-year ≠ authoring-year dilution.
  - Instruct vintages are contaminated with DeepSeek-generated QA.

### Gaps
- The Visokay et al. paper (arXiv 2610.36631) was seen only through secondary write-ups. Its detector list and author list are unverified.
- I did not check whether FineWeb's 192 per-dump ablation models were publicly released. If they were, they would be a second vintage family at monthly-dump resolution.
- Searches cannot rule out very recent (Oct–Nov 2026) preprints or non-English work. The absence finding is based on about 28 searches and fetches, not a systematic review.

---

## Q6. Prior art for the "model vintages as a measurement instrument" framing in social science

### Takeaway
Treating models from different time slices as an instrument has clear precedents. Chained yearly embedding models for semantic change go back to 2014. Diachronic degradation of social-media models dates from 2018. Perplexity-based knowledge-cutoff dating appeared in 2024. Finance uses vintages to quantify a "lookahead premium" by comparing vintages before and after the outcome (DatedGPT, ChronoGPT). None applies the instrument to machine authorship.

### Cited Findings
- **Kim et al. 2014, "Temporal Analysis of Language through Neural Language Models."** One model per year on Google Books, each year initialised from the previous one. Comparing vectors across years flags words that changed between 1900 and 2009. — [ACL W14-2517](https://aclanthology.org/W14-2517.pdf)
- **Hamilton et al. 2016, "Cultural Shift or Linguistic Drift?"** Global measures suit linguistic change, while local measures capture cultural change. — [Stanford PDF](https://nlp.stanford.edu/pubs/hamilton2016cultural.pdf)
- **Jaidka et al. 2018, "Diachronic degradation of language models: Insights from social media."** Compares age and gender models trained on 2011 and 2015 language, and treats drift as a signal. — [ACL P18-2032](https://aclanthology.org/P18-2032.pdf)
- **Dated Data (2024).** Perplexity minima over time used to date effective knowledge cutoffs. — [arXiv 2403.12958](https://arxiv.org/html/2403.12958v2)
- **Lookahead premium as a vintage contrast.**
  - DatedGPT: "models whose training data covers the outcome period" add 26.4 bp per SD. — [arXiv abs](https://arxiv.org/abs/2603.11838)
  - ChronoGPT: real-time vs full-sample Sharpe comparisons. — [arXiv](https://arxiv.org/html/2502.21206v3)
- **Time vectors.** **[bg]** Nylund et al. 2024: per-year fine-tune weight differences encode time. — [arXiv 2312.13401](https://arxiv.org/abs/2312.13401)
- **Vintage-LLM programme.** Proposed uses are behavioural science (PNAS 2024), forecasting backtests (Evans 2024), and "exploring discourse patterns at scale" (History LLMs). — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2407639121); [Evans](https://owainevans.github.io/talk-transcript.html); [history-llms](https://github.com/DGoettlich/history-llms)
- **ChronoCulture (2026).** LLMs' predictions for different years stay close despite real change. This is a caution that vintages may under-react to change. — [arXiv 2610.20472](https://arxiv.org/html/2610.20472v1)

### Inferences
- The instrument framing should cite these precedents (Kim 2014, Jaidka 2018, Dated Data 2024, and the DatedGPT and ChronoGPT lookahead-premium designs). The claim to novelty should rest on the object being measured (machine authorship and AI-text diffusion) and on the specific contrast: pre- vs post-ChatGPT vintages, and a same-title human vs chat preference curve.

### Gaps
- I did not find any economics or sociology paper that uses DatedGPT or ChronoGPT vintages for a non-finance measurement task, but citations of these 2025–2026 papers were not exhaustively traced. A forward-citation check on Google Scholar or Semantic Scholar is recommended.
