# AI-Generated Content in Web Crawls / Pretraining Data, Its Measurement, and Its Absorption by Base Models (status as of 2026-11-14)

Verification legend used on every item:
- **[V]** = checked this session against the primary page (arXiv abstract/HTML, ACL Anthology, publisher/dataset-builder page).
- **[S]** = seen only through search-engine summaries or secondary coverage this session; the primary source was not opened.
- **[M]** = from prior knowledge; NOT re-fetched this session. Treat numbers as unverified until checked.

"Relation:" lines explain how each work bears on a family of time-locked models (identical except for crawl year, 2013-2024, FineWeb-Edu) whose preference for chat-model text over human text is flat through 2022 and jumps with the 2023 crawl. That curve is the claim whose novelty is being checked.

---

## Q1. How much of the web, Common Crawl, and pretraining corpora is AI-generated, by year, and how was it measured?

### Takeaway
Every detector-based time series finds roughly zero AI text (no more than detector false-positive rates) before 2022 and a clear break in early 2023, with growth speeding up through 2025-2026. The most rigorous Common Crawl estimate (Visokay et al., Oct 2026) puts about 24% of crawlable pages as containing AI text in August 2026. The break point is February 2023, and the growth rate rises from 0.2 to 4.6 to 8.1 percentage points per year. Higher-level shares are higher still: about 21% of Common Crawl sites are dominated by machine-generated text (2026), and about half of new English "articles" are AI-generated (Graphite). FineWeb's own per-dump marker-phrase counts were flat through CC-MAIN-2023-14 and then rose steeply. I found no published detector-based AI-share-by-dump estimate specific to FineWeb or FineWeb-Edu.

### Cited Findings

**Common Crawl / whole-web prevalence**
- **[V] Visokay, A., Pan, S., ... McCormick, T. H., "The uneven rise of AI-generated web content," arXiv 2610.36631, submitted 27 Oct 2026 (preprint, not peer reviewed).**
  - Method: 5,118,347 pages from 505,974 domains in 71 Common Crawl snapshots (Jan 2019 to Aug 2026). Each crawl samples 20,000 domains stratified by PageRank tier (about 72k scored pages per crawl). Pages need at least 500 characters.
  - Detectors: open Pangram (QLoRA adapter on Llama-3.2-3B) on all pages, and Binoculars-Falcon-7B on English pages only. A Bayesian hierarchical model corrects for each detector's false-positive and false-negative rates, by group.
  - Control period: the 30 crawls CC-MAIN-2019-04 to CC-MAIN-2021-49 are treated as all human. Pangram control false-positive rate is 4.5% on English-only pages and 28.5% on bilingual pages.
  - Headline: about 24% of pages contained AI-generated or AI-edited text in Aug 2026 (95% credible interval 22.9-24.4).
  - Growth: 0.2 pp/yr (Jan 2019 to Feb 2023), then 4.6 pp/yr (Feb 2023 to Jan 2025), then 8.1 pp/yr (Jan 2025 to Aug 2026).
  - By language (Aug 2026): English-only 32.8%, non-English 17.0%, bilingual 20.2%.
  - By domain (Aug 2026): .com 29.5%, .net 20.4%, .org 14.6%, .edu 6.1%, .gov 5.8%. In early 2024, .edu was 2.1% and .gov 2.5%, versus .com 7.0%.
  - By PageRank tier: top tier 11.6%, tail tier 27.5%. The bottom and tail tiers account for 78.8% of the net increase since Dec 2021.
  - The paper warns that future models could "write increasingly less like humans and more like previous versions of themselves." This is a scenario, not something they measured.
  - Source: https://arxiv.org/abs/2610.36631 ; https://arxiv.org/html/2610.36631
  - Relation: their break point (Feb 2023, the first 2023 crawls) matches a jump that starts with the 2023 crawl. In early 2024 page-level prevalence was still only single-digit percent, so a 2023 crawl carrying about 74% of the 2024 effect implies a strongly nonlinear dose-response, or amplification (see the quality-filter discussion in Q2).
- **[V] He, S. S., Ardi, C., Govindan, R., Madhyastha, H. V., "DeGenTWeb: A First Look at MGT-dominant Websites," arXiv 2605.00087, v1 30 Apr 2026, v2 26 Oct 2026.**
  - Method: page-level machine-generated-text (MGT) detectors adapted to web pages, with results aggregated into a site-level label. Only 0.12% of human sites are mislabeled.
  - Findings: 20.9% of sites archived by Common Crawl in Jan-Jul 2026 are MGT-dominant, as are 13.6% of sites linked from Bing results in 2025. Both are stated as lower bounds. The share is growing, and many MGT-dominant sites share owners.
  - Source: https://arxiv.org/abs/2605.00087
- **[V] Paredes, J. L., Benson, B., Smith, E., Druck, G. (Graphite), "More Articles Are Now Created by AI Than Humans," 14 Oct 2025 (industry report).**
  - Method: random Common Crawl URLs, filtered to English pages with article schema, at least 100 words, published Jan 2020 to May 2025. Final sample is about 43k articles.
  - Detector: Surfer, on 500-word chunks. An article counts as AI if more than 50% of its chunks are flagged. False-positive rate is 4.2% (on 15,894 pre-Nov-2022 articles). False-negative rate is 0.6% (on 6,009 GPT-4o articles; other models untested).
  - Findings: about 39% AI-generated about 12 months after ChatGPT's launch. AI articles outnumbered human articles in Nov 2024. The share has plateaued since about May 2024.
  - The page now says it was superseded by "AI Now Writes as Many Online Articles as Humans," which uses data through Q1 2026 and averages three detectors. That version's shares are on average 3.3 points lower.
  - Source: https://graphite.io/five-percent/more-articles-are-now-created-by-ai-than-humans
- **[S] Graphite follow-up (2026):** about 49.9% of sampled articles were mostly AI-generated in Q1 2026, using three commercial detectors on about 55k pages. The number is from search summaries only.
  - Source: https://searchengineland.com/nearly-half-online-articles-ai-generated-study-478233
- **[S] Ahrefs (Apr 2025):** 74.2% of 900k new English pages contained some AI-generated content. This counts any AI content, not majority-AI pages, and is not a Common Crawl sample.
  - Source: https://ahrefs.com/blog/what-percentage-of-new-content-is-ai-generated/ (URL from memory; figure from search summary)

**Platform- and genre-specific shares**
- **[V] Sun, Z., Zhang, Z., Shen, X., ... He, X., "Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media," ACL 2025 Main (arXiv 2412.18148).**
  - Method: they built AIGTBench (texts from 12 LLMs) and chose the best detector, OSM-Det. They applied it to about 2.4M posts.
  - AI-attribution rate, Jan 2022 to Oct 2024: Medium 1.77% to 37.03%; Quora 2.06% to 38.95%; Reddit 1.31% to 2.45%.
  - Source: https://arxiv.org/abs/2412.18148
- **[S/M] Brooks, C., Eggert, S., Peskoff, D., "The Rise of AI-Generated Content in Wikipedia," WikiNLP workshop 2024 (arXiv 2410.08044).**
  - Method: GPTZero and Binoculars, with thresholds set to a 1% false-positive rate on pre-GPT-3.5 articles, applied to new pages created in Aug 2024.
  - Findings: more than 5% of 2,909 new English articles were flagged, with lower rates for German, French and Italian (stated as lower bounds). Flagged articles have fewer references and are often self-promotional or one-sided.
  - Source: https://arxiv.org/abs/2410.08044
- **[M] Thompson, B., Dhaliwal, M. P., Frisch, P., Domhan, T., Federico, M., "A Shocking Amount of the Web is Machine Translated: Insights from Multi-Way Parallelism," Findings of ACL 2024 (arXiv 2401.05749).**
  - Method: a 6.4B-sentence multi-way parallel corpus spanning 90 languages.
  - Findings: about 57% of sentences belong to translation tuples spanning 3 or more languages. This content is mostly low-quality machine translation and is concentrated in low-resource languages.
  - This is machine translation, not chat-LLM text. It shows "machine text" was already in the web before ChatGPT, which matters for any pre-2022 baseline in non-English data.
  - Source: https://arxiv.org/abs/2401.05749
- **[M] Kobak, D., González-Márquez, R., Horvát, E.-Á., Lause, J., "Delving into LLM-assisted writing in biomedical publications through excess vocabulary," Science Advances 2025 (arXiv 2406.07016).**
  - Method: "excess word" analysis, a detector-free word-marker method.
  - Finding: at least 13.5% of 2024 PubMed abstracts were LLM-processed, with higher rates in some subgroups.
  - Source: https://arxiv.org/abs/2406.07016
- **[M] Liang, W. et al., "The Widespread Adoption of Large Language Model-Assisted Writing Across Society," arXiv 2502.09747 (2025).**
  - Method: distributional maximum-likelihood estimation of the LLM-modified fraction.
  - Findings (late 2024): about 18% of US financial consumer complaints, up to about 24% of corporate press releases, about 10% of job postings, and about 14% of UN press releases.
  - Related work by the same group: ICML 2024 found 6.5-16.9% of AI-conference peer reviews were substantially LLM-modified (arXiv 2403.07183). COLM 2024 found up to about 17.5% of CS papers were LLM-modified (arXiv 2404.01268).
  - Source: https://arxiv.org/abs/2502.09747

**FineWeb-specific temporal signal (dataset builder)**
- **[V] Penedo, G. et al., "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," NeurIPS 2024 Datasets & Benchmarks (arXiv 2406.17557), plus the companion blog.**
  - Method: they say no reliable synthetic-data detector exists, so they counted proxy phrases per crawl: "delve", "as a large language model", "it's important to note", "rich tapestry", "intertwined", "certainly!", "dive into".
  - Finding: the frequency "remained constant until 2023-14 and then increased steeply" in later crawls.
  - They caution that not every such sample is ChatGPT output, and that many ChatGPT outputs contain none of these phrases.
  - Source: https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html ; paper https://arxiv.org/abs/2406.17557
  - Relation: this is the only dataset-builder time series of AI-style markers across Common Crawl dumps that I found. It places the onset within 2023, after the March/April crawl.
- **[V] Satharasi, T., Iyengar, S. S., "Future of AI Models: A Computational perspective on Model collapse," arXiv 2511.05535, Oct 2025.**
  - Method: year-by-year embedding cosine similarity of "English-language Wikipedia (filtered Common Crawl)" text, 2013-2025.
  - Finding: a steady rise in similarity before public LLM adoption, then an "exponential rise" after it.
  - The abstract gives no numbers, and the corpus description is ambiguous. The collapse-date extrapolation (2035) appears only in secondary summaries [S].
  - Source: https://arxiv.org/abs/2511.05535
- **[V] wordfreq (Speer) stopped updating in September 2024**, citing generative-AI pollution of its web sources. The lowbackgroundsteel.ai catalogue lists it as "no longer updated due to AI contamination," accessioned 18 Sep 2024.
  - Sources: https://lowbackgroundsteel.ai/ ; https://github.com/rspeer/wordfreq/blob/master/SUNSET.md (URL [M])

### Inferences
- **Methods fall into three families:**
  - Supervised detectors: Pangram, GPTZero, Surfer, OSM-Det.
  - Zero-shot likelihood detectors: Binoculars.
  - Detector-free statistical methods: marker words (FineWeb phrases, Kobak's excess vocabulary) and distributional MLE (Liang).
  - All of these count documents or words. None measures what a model trained on the crawl absorbs. A model-based absorption curve is therefore a distinct, complementary measure.
- **The timing is consistent across sources.** Visokay's Feb 2023 break and FineWeb's "after 2023-14" marker rise both fit a jump that starts with the 2023 crawl. FineWeb puts the marker rise mostly in the later 2023 dumps. So if a "2023 crawl" pools all 2023 dumps, its effect probably comes mostly from the mid- and late-2023 dumps.
- **Document prevalence in 2023 was low, which makes a large 2023 effect notable.** Visokay's 4.6 pp/yr from Feb 2023 implies only a few percent net AI pages by the end of 2023. A 2023 crawl carrying about 74% of the 2024 effect therefore suggests one or more of:
  - (a) A small dose of distinctive chat text is enough to shift a model's preferences, so the response saturates early.
  - (b) Selection or amplification by the educational-quality filter.
  - (c) Humans adopting ChatGPT-style vocabulary (see Yakura et al. under Q2).
- **Domain mix pulls the other way.** The low AI share on .edu and .gov domains means FineWeb-Edu's educational selection does not obviously enrich for AI pages by domain. Any amplification would have to work at the level of text style.

### Gaps
- No detector-based estimate of AI share per FineWeb or FineWeb-Edu dump was found. Neither was any analysis of whether the FineWeb-Edu score distribution differs for AI-flagged and human pages.
- Spennemann (2025, arXiv 2504.08755, "Delving into: the quantification of AI-generated content on the internet") reportedly estimates at least 30-40% of text on active web pages is AI-derived. I did NOT verify this this session; treat it as unverified.
- No post-Aug-2024 update of the Wikipedia AI-share estimate was found.
- Visokay et al.'s year-by-year series sits in figures that I did not read (the HTML was read only up to character 100k).

---

## Q2. Do newer base models (trained after 2022) absorb chat-model style or behave like instruction-tuned models? Could the FineWeb-Edu classifier amplify this?

### Takeaway
The direct evidence is thin, observational, and compares different model families:
- Base models trained after ChatGPT name "GPT" as their identity, and this rises with training-data cutoff. OLMo-3-32B base names GPT 25.6% of the time.
- Some post-ChatGPT base models refuse requests, possibly because chat outputs leaked into their pretraining data.
- FineWeb's per-dump ablation models scored best on the most recent crawls.

At the same time, commercial detectors still judge base-model text "overwhelmingly human" (Xu et al., May 2026). Document-level detectors therefore do not register the absorption, which leaves room for a more sensitive model-based probe. For the FineWeb-Edu amplifier hypothesis there is only indirect support: Wikipedia-style rewriting flips about 7% of FineWeb-Edu filter decisions, and the DCLM filter uses GPT-4-style instruction data as its positive class. No direct test was found.

### Cited Findings

**Base models vs detectors**
- **[V] Xu, Y. E., Zhong, Z., Raghunathan, A., Fang, F., Kolter, J. Z., "Base Models Look Human To AI Detectors," arXiv 2605.19516, 19 May 2026 (CMU).**
  - Method: GPTZero and Pangram scores for base and instruct continuations of human-written or AI-written prefixes (truncated to the first sentence). Models: Llama-3 and Qwen-3 families, 0.6B-70B.
  - Key numbers: Llama-3-8B base continuations score 96.7% human (GPTZero) and 98.8% (Pangram). The instruct model scores 30.3% and 17.1%.
  - Further language-model fine-tuning of an instruct model on FineWeb-Edu "recovers much of the lost human-likeness."
  - The authors' explanation is that "pretraining is almost entirely human text." They argue detectors track artifacts of instruction tuning rather than "machine-generated" text in general. They also propose HIP, an iterative paraphraser built from a base model.
  - The paper does NOT compare older and newer base models, and does not discuss contamination of pretraining data.
  - Source: https://arxiv.org/abs/2605.19516 ; https://arxiv.org/html/2605.19516
  - Relation: by detector standards, post-2022 base models (Qwen3, Llama-3) still look human. Whatever chat style they absorbed is invisible to document-level detectors. A preference or likelihood probe on time-locked models measures something detectors miss.
  - Caveat [M]: Llama-3-8B's published knowledge cutoff is about March 2023, so its data includes only a few post-ChatGPT months.

**Identity and refusal behaviour in base models**
- **[V] Zhong, Z., "Model self-identification could be subliminally transferred," LessWrong, 30 Jul 2026 (not peer reviewed).**
  - Findings for OLMo-3-32B base (no instruction tuning), on identity questions: names GPT 25.6% of the time, Claude 4.0%, Gemini 2.3%, DeepSeek 4.0%.
  - The OLMo-3 pretraining corpus contains about 62.8M mentions of "ChatGPT."
  - Across 9 base models there is "a clear trend with training-data cutoff," shown in a figure only. Pythia, trained on pre-ChatGPT data, is the exception.
  - Fine-tuning on human-written responses raised GPT identification to 45.5%, and fine-tuning on GPT-4o outputs raised it to 70.5%.
  - The author attributes the effect to associations learned in pretraining between chat-style text and AI identity. Style is the driver: rewriting into "caveman" style removed most of the effect in 2 of 3 models.
  - Source: https://www.lesswrong.com/posts/cb5quszpxCbFDGk68/model-self-identification-could-be-subliminally-transferred
  - Relation: this is the closest existing analog of an absorption-versus-cutoff curve. But it compares heterogeneous model families (different data, sizes and recipes), not a controlled family that varies only crawl year.
- **[S] DeepSeek-V3 identified itself as ChatGPT in tests (TechCrunch, Dec 2024).** An expert attributed this to ChatGPT outputs in the training data.
  - Source: https://techcrunch.com/?p=2936858
- **[S] "Base LLMs refuse too," AI Alignment Forum (2024).** The base models Gemma-2-9B and Qwen1.5-0.5B, trained after ChatGPT, refuse requests. The author suggests leaked ChatGPT outputs in pretraining as a possible cause, and contrasts them with the pre-ChatGPT LLaMA-1-7B. The comparison is uncontrolled.
  - Source: https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too

**Per-dump effects in FineWeb**
- **[V] FineWeb per-dump ablations (Penedo et al. 2024).**
  - Method: 1.8B models trained on 27B tokens per Common Crawl dump (192 models, 2 seeds each).
  - Findings: scores vary across dumps, and there is "strong performance of the last few crawls," but the authors found no conclusive cause. Synthetic data "does not seem to drastically harm" performance at this scale. They say it is "not clear that this holds for much larger trainings."
  - Source: https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html

**Where chat vocabulary comes from**
- **[M] Juzek, T. S., Ward, Z. B., "Why Does ChatGPT 'Delve' So Much? Exploring the Sources of Lexical Overrepresentation in LLMs," COLING 2025 (arXiv 2412.11385).** They attribute focal-word overuse ("delve" and similar) mainly to post-training and human-feedback steps, not to pretraining.
  - Source: https://arxiv.org/abs/2412.11385
  - Relation: if chat-style lexis comes from post-training, its appearance in base models trained on later crawls implies transfer through the web, which is what a time-locked family can test.
- **[M] Yakura, H. et al., "Empirical evidence of Large Language Model's influence on human spoken communication," arXiv 2409.01754 (2024).** ChatGPT-preferred words such as "delve" rose in human speech (YouTube talks and podcasts) after ChatGPT's release.
  - Source: https://arxiv.org/abs/2409.01754
  - Relation: this is a confound. Part of any "absorbed ChatGPT style" in post-2022 crawls may come from humans who adopted that style, not from AI-generated documents.

**Instruction-like data in "base" model pretraining (confounds)**
- **[S/M] DataComp-LM (Li, J. et al., NeurIPS 2024 D&B, arXiv 2406.11794).** Its fastText quality classifier uses LLM-generated instruction data (OpenHermes-2.5, largely GPT-4 outputs, plus top r/ExplainLikeImFive posts) as positives and random web text as negatives.
  - Source: https://arxiv.org/abs/2406.11794
  - Relation: by design, this filter favors text that resembles chat-model output.
- **[S] Phi-1.5 can be prompted like an instruct model because its pretraining data is heavy in question-answer formats.** Many modern "base" models deliberately mix instruction or synthetic data into pretraining or mid-training, so base-model behaviour is not a clean readout of web contamination.
  - Source: https://arxiv.org/pdf/2409.05314

**The FineWeb-Edu amplifier hypothesis**
- **[V] How the FineWeb-Edu classifier was built (Penedo et al. 2024).**
  - Llama-3-70B-Instruct scored 460k pages from CC-MAIN-2024-10 on a 0-5 educational scale.
  - The classifier is Snowflake-arctic-embed-m with a linear regression head, trained on 410k annotations.
  - The threshold is a score of 3, giving 82% binary F1. It was applied to all 15T FineWeb tokens (6,000 H100-hours).
  - Source: https://arxiv.org/abs/2406.17557
  - Relation: the labels come from an instruction-tuned LLM, and the annotation snapshot is a 2024 crawl that already contains AI text. Both are plausible routes for "LLM-annotator taste" to favor chat-like prose.
- **[V] Klimaszewski, M., Andruszkiewicz, P., "Is a Document Educational or Just Wikipedia-Style? Pitfalls of Classifier-Based Quality Filtering," ACL 2026 (Short), pp. 99-108.**
  - Method: rewrite documents into Wikipedia-style prose and re-score them.
  - Finding: the FineWeb-Edu classifier's keep/discard decision flips for about 7% of documents.
  - Source: https://aclanthology.org/2026.acl-short.10/
  - Relation: this is the most direct evidence that the FineWeb-Edu filter rewards surface style rather than content. Chat-LLM prose is typically encyclopedic and expository, so this supports (but does not prove) style-based amplification.
- **[S] Wettig, A. et al., "Organize the Web" (arXiv 2502.10341, 2025).** Quality filters such as DCLM-fastText implicitly amplify particular formats and topics.
  - Source: https://arxiv.org/abs/2502.10341
- **[S] Dense-retrieval "source bias."** Neural retrievers prefer LLM-generated text, and fine-tuning on LLM-generated corpora reinforces the bias. Low perplexity is one proposed cause.
  - Sources: arXiv 2602.10833 (2026), https://arxiv.org/html/2602.10833v1 ; [M] Dai, S. et al., "Neural Retrievers are Biased Towards LLM-Generated Content," KDD 2024, https://arxiv.org/abs/2310.20501
  - Relation: this is an analog in another domain. Training on LLM text produces a learned preference for LLM text, which is the same kind of quantity a preference probe measures.
- **[M] Ferbach, D. et al., "Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences," NeurIPS 2024 (arXiv 2407.09499).** When curation by a reward model sits in the retraining loop, the model drifts toward that reward's preferences.
  - Source: https://arxiv.org/abs/2407.09499
  - Relation: this is the theoretical form of the amplifier hypothesis. The FineWeb-Edu classifier works like a reward or curator trained on LLM judgments.

### Inferences
- **Detectors and absorption probes measure different things.** Xu et al. show base models look human to detectors. Zhong and the FineWeb dumps show post-2022 corpora still shift base-model behaviour. Absorption can therefore be real while still being below detector thresholds. A likelihood or preference probe on models that vary only in crawl year is a more sensitive test than detector scores of base-model generations.
- **The GPT-Neo control matters.** If the jump were a generic "machine text has low perplexity" effect, text from a pre-ChatGPT generator would also show it. It does not, which points to ChatGPT-specific style (post-training artifacts, per Juzek & Ward) entering through the web. That rules out generic machine-text likelihood bias as the explanation (the perplexity mechanism proposed in the retriever source-bias work).
- **The amplifier hypothesis has indirect support only.** The FineWeb-Edu filter is style-sensitive (7% decision flips from Wikipedia-style rewriting), and it was built from LLM labels on a 2024 crawl. Testing it directly would mean comparing absorption in raw FineWeb versus FineWeb-Edu at the same crawl year, or scoring AI-flagged versus human pages with the classifier.

### Gaps
- No controlled study was found showing that post-2022 base models trained on raw web text adopt chat-model style. All the evidence compares different model families (Zhong 2026; the refusals post) or measures benchmarks rather than style (FineWeb).
- No published test was found of whether the FineWeb-Edu or DCLM classifiers score AI-generated pages above matched human pages. Nor was any comparison of AI-share before and after educational filtering.
- Xu et al.'s per-model numbers for Qwen3-Base (newer data, heavy synthetic mid-training) versus Llama-3 base were not in the text I read (they are in figures). So whether newer base models look less human remains unverified.

---

## Q3. Model collapse and synthetic-data contamination: theory, accumulate-versus-replace, and 2025-2026 evidence on real crawls

### Takeaway
Collapse is robustly shown when synthetic data REPLACES real data across generations (Shumailov et al., Nature 2024). It is largely avoided when synthetic data ACCUMULATES alongside real data (Gerstgrasser et al., COLM 2024; Kazdan et al., ICML 2025), with caveats from "strong collapse" theory (Dohmatob et al., ICLR 2025). On real crawls there is no measured case of a model degrading from newer crawls. FineWeb found recent dumps did as well or better at 1.8B scale. The real-crawl signals so far are distributional (rising semantic similarity, marker words), not capability collapse.

### Cited Findings
- **[M] Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., Gal, Y., "AI models collapse when trained on recursively generated data," Nature 631:755-759 (Jul 2024).**
  - Method: recursive fine-tuning of OPT-125m on its own outputs (WikiText-2), plus theory for Gaussian mixture models and VAEs.
  - Findings: tails of the distribution vanish first, and generations converge to low-variance output. Keeping some original data slows the degradation.
  - Source: https://www.nature.com/articles/s41586-024-07566-y
- **[M; S via secondary mention] Gerstgrasser, M., Schaeffer, R., Dey, A., Rafailov, R., et al., "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data," COLM 2024 (arXiv 2404.01413).**
  - Findings: when data accumulates across generations instead of being replaced, test error stays bounded and collapse is avoided. Shown for language models trained on TinyStories, for diffusion models and VAEs, and analytically for linear regression.
  - Source: https://arxiv.org/abs/2404.01413
- **[S] Kazdan, J., Schaeffer, R., et al., "Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World," ICML 2025 (arXiv 2410.16713).** Mixing real and synthetic data contains collapse. The authors also compare replace, accumulate, and fixed-budget accumulate regimes.
  - Sources: https://icml.cc/virtual/2025/poster/44713 ; https://arxiv.org/html/2410.16713
- **[S] Schaeffer, R. et al., "Position: Model Collapse Does Not Mean What You Think," arXiv 2503.03150 (2025).** "Model collapse" has been used for several distinct phenomena. The realistic setting is real data accumulating while the synthetic share grows, and in that setting catastrophic collapse is unlikely.
  - Source: https://arxiv.org/abs/2503.03150
- **[S/M] Dohmatob, E., Feng, Y., Subramonyam, A., Kempe, J., "Strong Model Collapse," ICLR 2025 (arXiv 2410.04840).** In their theoretical and regression settings, even a small synthetic fraction (on the order of 1% of training data) can stop scaling gains.
  - Source: https://arxiv.org/abs/2410.04840
  - [M] Companion work: "A Tale of Tails: Model Collapse as a Change of Scaling Laws," ICML 2024, https://arxiv.org/abs/2402.07043
- **[M] Alemohammad, S. et al., "Self-Consuming Generative Models Go MAD," ICLR 2024 (arXiv 2307.01850).** Without enough fresh real data in each generation of an autophagous loop, quality or diversity degrades. This was shown for image models.
  - Source: https://arxiv.org/abs/2307.01850
- **[M] Guo, Y. et al., "The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text," Findings of NAACL 2024 (arXiv 2311.09807).** Lexical, syntactic and semantic diversity decline over recursive training.
  - Source: https://arxiv.org/abs/2311.09807
- **[S] "Demystifying Synthetic Data in LLM Pre-training," EMNLP 2025 Main (first author reportedly Kang, F.; [M]).** This is a large single-round pretraining study. Rephrased synthetic data mixed with natural web text can speed up training. Pure "textbook-style" synthetic data shows some patterns predicted by model collapse.
  - Source: https://aclanthology.org/2025.emnlp-main.544.pdf
- **[V] Satharasi & Iyengar (arXiv 2511.05535, 2025)** report an "exponential rise" in year-over-year semantic similarity of web text after LLM adoption. This is the closest real-crawl collapse-style signal found; it is abstract-level only (see Q1).
  - Source: https://arxiv.org/abs/2511.05535
- **[V] FineWeb (2024) ablations:** synthetic data in recent crawls "does not seem to drastically harm" small-model performance (see Q2).
  - Source: https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html
- **[S] Search-level summary:** no 2025-2026 primary source was found documenting measured degradation of a frontier or large model caused by newer crawls. Several 2026 claims of "model collapse already here" appeared only in blogs and could not be traced to primary studies.
  - Example blog: https://www.webpronews.com/the-ai-industrys-dirty-secret-model-collapse-is-already-here-and-nobody-wants-to-talk-about-it/

### Inferences
- **The time-locked absorption curve fits the realistic accumulate regime.** Each crawl year adds AI text on top of persistent human text, and the observed effect is a stylistic shift, not a collapse. It is an early, measurable precursor (models moving toward chat-model style) that the collapse literature predicts but rarely measures on real crawls.
- **Collapse experiments usually define their synthetic data by construction.** A crawl-year family measures the naturally occurring dose, so it complements controlled collapse studies.

### Gaps
- No study was found that retrains models on successive real crawls and measures diversity or tail loss attributable to AI text, as opposed to benchmark scores.
- The numbers from Shumailov, Gerstgrasser, Dohmatob and Guo come from prior knowledge and were not re-fetched this session.

---

## Q4. "Low-background" (pre-ChatGPT) data: the framing and archival efforts

### Takeaway
The "low-background steel" metaphor treats late 2022 as the digital equivalent of the 1945 nuclear tests. Since 2023 it has been institutionalized in a curated catalogue (lowbackgroundsteel.ai) and in policy and legal scholarship arguing that pre-2022 human data is a scarce, competition-relevant resource. The catalogue now lists a Common Crawl August 2022 crawl and a 20 Aug 2022 Wikipedia dump. Sources caution that date stamps are an unreliable contamination boundary.

### Cited Findings
- **[V] lowbackgroundsteel.ai (curated by John Graham-Cumming; page updated 14 Nov 2026).**
  - Purpose: a hand-maintained catalogue of sources that "predate the flood of machine-generated text, images, and audio that began in late 2022, or deliberately exclude it."
  - Earliest entries were accessioned 10 Mar 2023.

  | Source | Data date | Accessioned |
  |---|---|---|
  | Arctic Code Vault | 2 Feb 2020 GitHub snapshot | 10 Mar 2023 |
  | English Wikipedia dump | 20 Aug 2022 | 27 Apr 2025 |
  | Common Crawl | August 2022 crawl | 5 May 2026 |
  | Project Gutenberg | no specific date | 10 Mar 2023 |
  | Chronicling America | 1730-1969 newspapers | 5 May 2026 |
  | wordfreq | "no longer updated due to AI contamination" | 18 Sep 2024 |

  - Source: https://lowbackgroundsteel.ai/
- **[S] The Register, "The launch of ChatGPT polluted the world forever, like the first atomic weapons tests," 15 Jun 2025.** This covers the legal and economics argument that firms holding pre-2022 data gain a lock-in advantage over newcomers, whose models are more exposed to collapse.
  - Source: https://www.theregister.com/2025/06/15/ai_model_collapse_pollution
- **[S] Harvard JOLT Digest, "Model Collapse and the Right to Uncontaminated Human-Generated Data."**
  - It summarizes "Legal Aspects of Access to Human-Generated Data and Other Essential Inputs for AI Training" (Dec 2024; co-authors include Podszun, R.; reportedly Chiodo, M. et al. [M]).
  - Argument: pre-2022 data holders may entrench dominance. The digest notes that date stamps cannot reliably separate pre- from post-2022 data.
  - Source: https://jolt.law.harvard.edu/digest/model-collapse-and-the-right-to-uncontaminated-human-generated-data
- **[S] Counterpoint from the Internet Archive.** The Wayback Machine's director reportedly prefers archiving AI outputs, by asking chatbots about 1,000 topical questions per day, rather than only preserving pre-AI data.
  - Source: https://getcoai.com/news/digital-archaeologists-race-to-preserve-pre-ai-internet-content/
- **[V] wordfreq's retirement (Sept 2024)** is widely cited as the first concrete casualty, since post-2021 word frequencies became unreliable.
  - Source: https://lowbackgroundsteel.ai/
- **[M; unverified] FineWeb's last pre-ChatGPT dump is CC-MAIN-2022-49, crawled around late Nov to early Dec 2022.** ChatGPT launched on 30 Nov 2022, so this dump straddles the launch. The first fully post-ChatGPT dump is CC-MAIN-2023-06.
  - Source: Common Crawl index https://index.commoncrawl.org/ (dates not re-checked)

### Inferences
- **A crawl-year model family is an operational use of the low-background idea.** The 2013-2022 models are "low-background instruments," and their flat preference curve is empirical support that pre-2022 crawls are uncontaminated in the style-absorption sense. That is a stronger claim than date-stamp arguments, which the JOLT digest notes are unreliable.
- **The catalogue picked August 2022 for Common Crawl.** That suggests practitioners treat the Aug-Nov 2022 crawls as the last safe ones. A curve that is flat through 2022 and jumps in 2023 agrees with that choice.

### Gaps
- I did not open the primary legal paper (author list and exact arXiv/SSRN ID were not verified).
- No quantitative study was found that measures how contaminated "pre-2022-dated" pages in post-2022 crawls are (re-crawled old URLs whose content was later rewritten with AI). That matters because each crawl contains many old URLs.

---

## Q5. Has anyone measured contamination by what a model trained on a given crawl year absorbs, rather than by counting documents?

### Takeaway
I found no work that uses a controlled model family, varying only crawl year, to date the onset of chat-model style absorption. The closest pieces each lack something:
- FineWeb trained 1.8B models per Common Crawl dump and counted marker phrases, but evaluated benchmarks, not style or preference.
- Zhong (2026) shows GPT self-identification in base models rising with training cutoff, but across uncontrolled model families.
- Several 2024-2026 point-in-time model families exist (including FineWeb-based 2013-2024 monthly checkpoints by Kelly et al., 2026). They are built for lookahead-bias or temporal-knowledge research, and their abstracts do not report ChatGPT-style absorption.

### Cited Findings
- **[V] FineWeb per-dump ablations (Penedo et al. 2024):** one 1.8B model (27B tokens, 2 seeds) per Common Crawl dump, 192 models in total. Marker-phrase frequency is flat until CC-MAIN-2023-14 and then rises steeply. The last few crawls perform strongly on benchmarks, with no conclusive cause.
  - Source: https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html
  - Relation: this is the nearest precedent in design (one model per crawl, same recipe). It never measured whether the models prefer chat-model text, and it did not use FineWeb-Edu.
- **[V] Kelly, B., Malamud, S., Schwab, J., Xu, T. A., "Scaling Point-in-Time Language Models: Enabling Economic Evaluation of Embeddings," arXiv 2607.11889 (v1 24 Apr 2026, v3 12 Oct 2026).**
  - Design: decoder models up to 4B parameters, trained on 1T chronologically filtered FineWeb tokens, with monthly checkpoints across 2013-2024. LoRA instruction tuning is applied. Purpose: removing lookahead bias in finance and social-science work.
  - The abstract does not discuss LLM-generated text in post-2022 slices. I read only the abstract.
  - Source: https://arxiv.org/abs/2607.11889
  - Relation: this is the closest existing artifact (FineWeb, 2013-2024, time-locked) that could be used for the same measurement. It needs to be checked for any analysis of post-2022 chat-style effects, and could serve as replication material.
- **[V] Pilchen, H., Fabre, R., Signe Talla, F., Perez, P., Grave, E., "Understanding Data Temporality Impact on Large Language Models Pre-training," arXiv 2605.22769 (May 2026).** 6B models pretrained on temporally ordered Common Crawl snapshots versus shuffled training, with a benchmark of 7,000+ time-sensitive questions. The abstract makes no mention of AI-generated text.
  - Source: https://arxiv.org/abs/2605.22769
- **[S] TiC-LM (Li, J. et al., ACL 2025, arXiv 2504.02107).** A time-continual pretraining benchmark over 114 Common Crawl dumps. It studies forgetting and knowledge updating, not contamination by AI text. A search snippet mentioned LLM-generated examples, but in an unclear context.
  - Source: https://arxiv.org/abs/2504.02107
- **[M] Drinkall, F. et al., "Time Machine GPT," Findings of NAACL 2024 (arXiv 2404.18543).** A series of point-in-time models trained only on data up to each year. Purpose: avoiding lookahead bias.
  - Source: https://arxiv.org/abs/2404.18543
- **[M] He, S., Lv, L., Manela, A., Wu, J., "Chronologically Consistent Large Language Models" (ChronoBERT/ChronoGPT), arXiv 2502.21206 (2025).** Yearly vintage models for finance research.
  - Source: https://arxiv.org/abs/2502.21206
- **[M] Cheng, J. et al., "Dated Data: Tracing Knowledge Cutoffs in Large Language Models," COLM 2024 (arXiv 2403.12958).** A model's effective knowledge cutoff often differs from its reported cutoff, partly because Common Crawl dumps contain much older, re-crawled content and deduplication does not remove it.
  - Source: https://arxiv.org/abs/2403.12958
  - Relation: "crawl year" is not "content year." A 2023 crawl is mostly pre-2023 pages, so a jump at 2023 is produced by a minority of new pages, which strengthens the dose-sensitivity inference from Q1.
- **[V] Zhong, Z. (LessWrong, Jul 2026):** GPT self-identification in base models is a function of training-data cutoff, across 9 heterogeneous base models.
  - Source: https://www.lesswrong.com/posts/cb5quszpxCbFDGk68/model-self-identification-could-be-subliminally-transferred
- **[S] "Base LLMs refuse too" (AI Alignment Forum, 2024):** a qualitative contrast between pre-ChatGPT and post-ChatGPT base models.
  - Source: https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too

### Inferences
- **The specific combination appears new as of 2026-11-14,** based on searches plus abstract-level checks of the closest families. The combination is:
  - (i) a controlled model family varying only crawl year (2013-2024, FineWeb-Edu);
  - (ii) a preference or likelihood probe pitting chat-model text against same-title human text;
  - (iii) dating the onset to the first post-ChatGPT crawl;
  - (iv) a pre-ChatGPT generator (GPT-Neo) as a negative control.
- **The main overlap risks are:**
  - Kelly et al. 2026: same data source and years; the full text was not read.
  - FineWeb's per-dump models: same design idea, different outcome measure.
  - Zhong 2026: an absorption-versus-cutoff trend, but uncontrolled.

  A careful positioning would cite all three and stress control (identical recipe), the outcome measure (style preference rather than benchmarks or identity strings), and the negative control.
- **Agreement with document-count studies is useful external validation.** Visokay et al. place the break at Feb 2023 and FineWeb's markers rise after 2023-14, both consistent with an absorption jump at the 2023 crawl. A model-based curve can then quantify how strongly a small document share (single-digit percent in 2023) shifts model preferences.

### Gaps
- I read only the abstracts of Kelly et al. 2026 and Pilchen et al. 2026. Their full texts should be checked for any AI-text or ChatGPT-style analysis of post-2022 checkpoints.
- I did not check whether TimeMachineGPT or ChronoGPT report any post-2022 stylistic shift. Their vintages may stop before or around 2023.
- No source was found that compares absorption in raw FineWeb versus FineWeb-Edu at matched crawl years. That would be the direct test of the classifier-amplifier hypothesis.
