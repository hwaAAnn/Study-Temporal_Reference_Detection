# 서론 초안 (영어, 2026-11-14)

논문 서론 초안입니다. 숫자는 00~05의 결과입니다. 04(탐지) 단락과 5번 기여는 로컬 결과(11/15, 작은 기존 탐지기)로 채웠고, Falcon-7B 설정과 비교하기 전입니다. `[CITE: …]`는 선행연구 조사 뒤 확인할 인용입니다.

---

**Working title:** *Time-Locked Language Models as Instruments: Measuring When and How Much AI-Generated Style Entered the Web and Human Writing*

## 1 Introduction

Language models trained on the 2023 web already prefer ChatGPT-style text. We use a family of twelve 1.3B-parameter models that are identical except for the crawl year of their training data, from 2013 to 2024 [CITE: DatedGPT]. For each model we measure how much it prefers a chat model's abstract over the human-written abstract with the same title. This preference stays flat across a decade of crawls: the slope is −0.003 nats per token per year, and the scatter between models is 0.009. Then it jumps by 0.21 nats per token in a single year, with the 2023 crawl, and rises further in 2024. The jump is more than ten times larger than any change between two earlier vintages. Nobody set out to train these models on machine-written text, and the data they were trained on is widely used.

Knowing *when* and *how much* AI-generated text has entered the web and human writing matters to at least three communities.

- **Pretraining.** Anyone who pretrains on web crawls has to decide which crawls to trust. Work on model collapse shows what can go wrong when models learn from model output [CITE: Shumailov et al. 2024], and pre-2022 text is increasingly treated as a scarce, "low-background" resource [CITE]. For these decisions the relevant quantity is not how many documents an LLM wrote. It is what a model trained on a crawl actually absorbs.
- **Monitoring AI use.** Studies of AI use in science, education and everyday language track words that LLMs overuse [CITE: Liang et al. 2024; "delve" studies]. Writers can learn to avoid those words. In our arXiv data, such marker words fall from 7.5 to 4.9 per 1,000 words after 2025, while the model-based measure we propose stays at its 2024 level.
- **Detection.** Detection research needs human-written negatives. We estimate that "human" abstracts written after 2024 behave as if roughly a third of their text had been rewritten by an LLM, so benchmarks built from recent human text carry label noise.

Per-document detectors are a poor instrument for these questions, which are about populations, not single documents. A detector's error rates depend on the base rate and on the period it was built for. A detector also tells us whether a document looks machine-written, not what a model trained on that document has learned. Distributional estimators avoid the first problem [CITE: Liang et al. 2024], but they rely on a vocabulary of marker words fixed in advance.

We propose to use **time-locked language models**, models that differ only in the date of their training data, as a measurement instrument. Suppose AI-generated text entered the web after late 2022. Then a model trained on a later crawl should assign relatively higher likelihood to AI-style text than a model trained on an earlier crawl. The difference Δ = log p_new − log p_old needs no training and no access to the generator.

Δ also picks up other things:

- ordinary language change;
- new knowledge, such as terms coined after the older model's crawl;
- whether the text was in the newer model's training data.

The last of these is exactly what reference-based membership inference measures [CITE: "Machine text detectors are membership inference attacks"]. We separate these sources with five controls:

- **(a)** Same-title pairs, which cancel topic and knowledge.
- **(b)** A difference-in-differences against a model pair whose crawls both predate ChatGPT.
- **(c)** A twelve-vintage baseline that measures model-to-model noise.
- **(d)** Same-text polishing, which changes style without changing content.
- **(e)** Web documents with known crawl dates, which measure the membership effect directly.

Our findings:

- **When.** Web-trained models depart from their decade-long trend starting with the 2023 crawl (z = 17). That first post-ChatGPT crawl already carries 74% of the 2024 effect. The same departure appears for human text that a chat model only polished, which holds content fixed. It does not appear for text from a pre-ChatGPT generator (GPT-Neo). What spread is chat style, not machine text in general.
- **How much, in human writing.** From the same vintage onward, human arXiv abstracts written in 2024 shift toward AI style. The size of the shift is what we would see if about a third of their text had been polished by an LLM (0.40–0.53 when calibrated on commercial LLMs). The estimate holds up when writers avoid marker words.
- **Membership.** When the newer model was trained on a human web document, that document looks AI-written by about a third of the AI effect. This is enough to cut the true-positive rate at 5% false-positive rate to 0.06. Likelihood-difference measurements are therefore valid only for text written after the newer model's crawl.
- **Knowledge filters do not separate knowledge from style.** A common remedy is to drop the tokens the older model finds most surprising. In human text it removes new technical terms, but in AI text it removes the chat-style words themselves. Knowledge has to be controlled by design, not by filtering.
- **Detection.** The same pair detects human text that an LLM has polished. On heavily polished APT-Eval texts, s_λ = log p_new − λ·log p_old (λ fixed on development data) reaches an AUROC of 0.85. Likelihood under the newer model reaches 0.82, and Binoculars and Fast-DetectGPT reach 0.79 and 0.63 at the same 1–3B model budget. The gain over likelihood exists only when the pair straddles ChatGPT; the same formula built from a pre-ChatGPT pair gives none. On text an LLM wrote from scratch, Binoculars remains stronger.

We make the following contributions:

1. **Instrument.** Time-locked language model pairs as a training-free instrument for measuring the diffusion of AI-generated style, with a control design that separates AI exposure from language change, knowledge and membership.
2. **Absorption curve.** A twelve-year curve showing when web-trained models absorbed chat style.
3. **Human writing.** Estimates of AI influence on human scientific writing that hold up when writers avoid marker words.
4. **Scope.** A quantification of the membership confound and of the scope it imposes on likelihood-based measurement and detection.
5. **Detection.** A training-free detector for LLM-polished human text that outperforms likelihood and small-model zero-shot detectors on a public benchmark, with a gain that exists only when the model pair straddles ChatGPT.

---

## 메모 (논문에 넣지 않음)

- 인용 확인: DatedGPT, model collapse, "low-background" 데이터, Liang et al.(논문·리뷰 분포 추정), "delve" 연구, "Machine text detectors are MIAs", Binoculars, Fast-DetectGPT, RAID, APT-Eval. 선행연구 조사(딥리서치) 결과로 빠진 것과 겹치는 것을 확인합니다.
- 숫자 출처:
  - 01: 기울기 −0.003, s 0.009, J(2023) +0.207, r(2023) z 17, 74%
  - 05: 표지어 7.55 → 4.93, β 0.34 (실제 LLM 0.40~0.53)
  - 02: +0.085(약 1/3), TPR@5% 0.063
  - 03: 놀람 필터의 제거 토큰
- 주의:
  - "about a third"는 05의 직선 보정 β(로컬 0.34)입니다. 초록 단위(α)로는 약 절반입니다. 본문에서는 둘을 구분해 씁니다.
  - "stays at its 2024 level"은 추세를 보정한 Δ 기반 β(2024 0.27, 2025 이후 0.28)를 말합니다.
- 04 (11/15): APT 강 s_0.9 0.847, likelihood 0.815, Binoculars(Llama-3.2-1B 쌍) 0.790, Fast-DetectGPT(GPT-Neo-2.7B) 0.626. 시간 경계 이득 +0.039 [+0.025, +0.054]. 결합은 이득이 없어(H4 불성립) "complements"라는 표현을 뺐습니다.
  - Binoculars는 Llama가 다듬은 글에서 s_λ와 비슷하거나 낫고, DeepSeek·GPT-4o가 다듬은 글에서 크게 뒤집니다. 7B 설정에서 APT 강의 순서가 바뀌면 탐지 단락은 "likelihood보다 낫고, 그 이득이 ChatGPT 경계에서 온다"까지로 줄입니다.
