# 00. k=0 신호 확인 (go/no-go)

- 브랜치: `exp/00-k0-signal`
- 마감: 11/15
- 장비: 로컬 GTX 1060 6GB (1.3B fp16 추론, 모델 하나씩 로드)

## 질문

필터 없이(k=0) `Δ(x) = mean_t[log p_2024 − log p_2021]`이 사람 글과 AI 글을 가르는가?

## 가설

- H1: AI 글의 Δ가 사람 글보다 크다 (짝지은 비교에서 AI > 사람인 비율 > 50%).
- H2: Δ 단독 AUROC가 0.70 이상이다.
- H3: Δ는 단일 모델 log-likelihood(`log p_2024`)와 다른 정보를 담는다 (상관 |r| < 0.8).

## 데이터 (`data/pairs.jsonl`)

한 줄에 하나: `{"id": str, "human": str, "ai": str, "generator": str, "domain": str}`

- **사람 글은 2025년 이후 작성분만** 씁니다. 두 모델 모두 학습 때 보지 못했어야 MIA 혼동이 없습니다.
  - 1순위: arXiv cs 초록 2025-06 이후 (제목 + 초록)
  - 2순위: 2025년 이후 뉴스 또는 블로그 글
- **AI 글은 같은 주제로 생성**합니다. 예: arXiv는 제목을 주고 "이 제목의 초록을 써라"
  - 생성기: 로컬 `Qwen2.5-1.5B-Instruct`, `Llama-3.2-1B-Instruct` (+ 예산이 되면 API 모델 1개)
- 규모: 도메인 1개 × 생성기 2개 × 200쌍 = 400쌍
- 길이: 두 글 모두 앞 200 토큰으로 자릅니다(길이 효과 통제).

## 절차

1. `scripts/score_pairs.py`로 사람 글과 AI 글 각각의 토큰 log-prob를 old/new 모델로 계산합니다.
   모델은 하나씩 올리고 내려서 VRAM을 아낍니다.
2. `scripts/evaluate.py`로 Δ, `log p_new`, `log p_old`의 AUROC, TPR@1%FPR, 짝 승률을 표로 냅니다.

## 판단 기준

| 결과 | 다음 단계 |
| --- | --- |
| H1, H2 성립 | 01·02·03 진행 |
| H1만 성립 (AUROC 0.55~0.70) | 03(k 스윕)과 04(보조 신호)를 우선 확인한 뒤 결정 |
| H1 불성립 | 방향 중단, 브랜치에 사유 기록 |

## 산출물

- `outputs/k0.jsonl` (git에 올리지 않음), 결과 표를 이 문서 아래에 추가

## 구현하면서 정한 것 (2026-11-12)

위 명세에서 비어 있던 부분을 아래처럼 채웠습니다.

| 항목 | 결정 | 이유 |
| --- | --- | --- |
| 사람 글 | arXiv API에서 첫 버전 제출일이 2025-07~2026-10이고 primary category가 cs.*인 초록. 월별로 무작위 구간 2개(100편씩)를 읽고 40편씩 추출 (`scripts/fetch_arxiv.py`) | 두 모델 모두 못 본 글이면서, 기간을 고르게 덮도록 |
| 사람 글 필터 | 150~400단어, `$` 4개 이하(수식 위주 초록 제외), 철회 논문 제외 | 200토큰을 채우고, 수식 토큰이 점수를 좌우하지 않도록 |
| 사람 글 배정 | 생성기마다 **서로 다른** 사람 글 200편 (총 400편) | 같은 사람 글이 AUROC에 두 번 들어가지 않도록 |
| AI 글 프롬프트 | `Write the abstract of a computer science research paper titled "{title}". Respond with only the abstract text, as a single paragraph of about 200 words.` (user 메시지 하나, 각 모델의 기본 chat template) | |
| 디코딩 | 각 생성기의 `generation_config` 기본값 (Qwen: T=0.7, top-p 0.8, top-k 20, rep 1.1 / Llama: T=0.6, top-p 0.9), 최대 400토큰 | 사용자가 그대로 쓸 때의 글 |
| 후처리 | "Here is…", "Abstract:", 마크다운, 제목 반복 줄 제거, 한 문단으로 합침. 거절 응답 제외 (`scripts/generate_ai.py`) | |
| 채점 텍스트 | 초록 본문만 (제목 제외). 쌍마다 사람·AI 글을 **짧은 쪽 길이에 맞춰** 자르고(BOS 포함 최대 200토큰 = 채점 토큰 최대 199개), 채점 토큰이 100개 미만인 쌍은 제외 (`score_pairs.py` 기본값) | 짝 안에서는 길이가 항상 같음. causal LM이라 채점 후 잘라도 잘라서 채점한 것과 같음 |
| (변경 이력) | 처음에는 "양쪽 모두 200토큰 이상인 쌍만"으로 정했으나, Qwen이 초록을 짧게 써서(중앙값 121단어, 175토큰) 270개 중 53개만 남았습니다. 긴 출력만 고르면 선택 편향이 생기고 프롬프트를 바꾸면 생성기 간 조건이 달라지므로, **본 결과를 보기 전에** 위 규칙으로 바꾸고 같은 생성 결과를 재사용했습니다(`generate_ai.py --from-raw`) | 결과: Qwen 쌍 채점 토큰 중앙값 175 (최소 106), Llama 199 |
| 정밀도 | GPU fp16. DatedGPT는 bf16 학습이지만 GTX 1060은 bf16 미지원. CPU fp32와 16개 글에서 비교: 토큰 log-prob 최대 차이 0.024, Δ 최대 차이 0.0008 (Δ 범위 −0.21~+0.14), 상관 0.99999 (`scripts/check_precision.py`) | 결론에 영향 없음 |
| 판정 | H1: 짝 승률 > 0.5이고 쌍 bootstrap 95% CI 하한 > 0.5. H2: 두 생성기를 합친 400쌍의 Δ AUROC ≥ 0.70. H3: 합친 데이터에서 corr(Δ, `log p_2024`) 절댓값 < 0.8 | 결과를 보기 전에 고정 |

## 미리 적어둔 예측 (결과를 보기 전, 2026-11-12)

### 데이터가 가진 편향: 둘 다 Δ 분리를 **어렵게** 하는 방향

1. **사람 글 라벨 노이즈.** 2025년 이후 arXiv 초록 중 일부는 LLM으로 다듬은 글입니다(Liang et al., 2024: CS 초록의 LLM 수정 비율 추정치가 2024-02에 이미 최대 17.5%). 이런 초록은 Δ가 AI 쪽으로 올라가서 사람/AI 차이를 줄입니다.
2. **지식 노이즈.** 2025~2026년 초록에는 2021 모델이 모르는 이름(새 모델·벤치마크)이 많습니다. 사람 초록이 AI 초록보다 구체적인 새 이름을 더 많이 쓴다면 사람 쪽 Δ가 올라갑니다. 반대로 AI 초록이 "LLMs" 같은 최신 일반어를 반복하면 AI 쪽 Δ가 올라갑니다. 방향을 미리 알 수 없어서, Δ 기여가 큰 토큰을 직접 읽어 확인합니다.

따라서 00에서 H1이 성립하면 이 편향을 넘어선 신호이고, H1이 불성립하면 위 편향 때문인지 먼저 확인한 뒤 중단을 판단합니다.

### 파일럿: 기존 데이터로 미리 보기

본 데이터를 만드는 동안 이전 프로젝트(Study-AIText_Detection)의 데이터로 같은 점수를 먼저 냅니다(`data/pilot_pairs.jsonl`, 15세트 × 150쌍).

- 세트: XSum·WritingPrompts·PubMedQA × {GPT-3.5-turbo, GPT-4, davinci(GPT-3, ChatGPT 이전), Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct}
- 한계: 사람 글이 2021년 이전 글이라 두 모델이 모두 봤을 수 있습니다. 연도별 모델의 학습량이 같다면 2021 모델이 옛 글을 상대적으로 더 많이 봤을 것이고, 그러면 사람 글의 Δ가 내려가서 AUROC가 **부풀려질** 수 있습니다. 그래서 파일럿 숫자는 go/no-go 판정에 쓰지 않습니다.
- **davinci가 대조군입니다.** davinci 글은 ChatGPT 이전 문체이고 웹에 퍼진 적이 없습니다. 웹의 AI 문체 흡수가 원인이면 davinci의 Δ는 사람 글과 비슷해야 합니다. 반대로 사람 글의 학습 포함 효과나 "기계 글 일반" 효과가 원인이면 davinci도 Δ가 높게 나옵니다.

예측:

| 세트 | Δ 짝 승률 | Δ AUROC | 근거 |
| --- | --- | --- | --- |
| GPT-3.5/4 (6세트) | > 0.6 | 0.65~0.85 | TiC-LM에서 2023-01 → 2024-07 사이 채팅 모델 글의 AI−사람 likelihood 차이가 0.640 → 0.770 nat로 커졌음 |
| Qwen2.5 / Llama-3.2 (6세트) | > 0.6 | 0.65~0.85 | 같은 계열의 채팅 문체 |
| davinci (3세트) | 약 0.5 | 0.45~0.60 | TiC-LM에서 davinci 점수는 컷오프에 따라 변하지 않았음 |

davinci의 Δ AUROC가 채팅 세트와 비슷하게 높으면, 신호가 "AI 문체 흡수"가 아니라 다른 요인이라는 뜻이므로 02(MIA 통제)를 앞당깁니다.

### 본 실험 예측

- H1 성립(짝 승률 0.60~0.75), H2는 경계(AUROC 0.62~0.75), H3 성립(|r| < 0.6)으로 예측합니다. 라벨 노이즈 때문에 파일럿보다 낮게 나올 것으로 봅니다.

## 결과

### 파일럿: 기존 데이터 (2026-11-12)

`data/pilot_pairs.jsonl` 15세트 × 150쌍. 사람 글은 2021년 이전 글이라 판정에는 쓰지 않습니다(위 "파일럿" 참고).

```bash
python scripts/convert_raw_data.py --out data/pilot_pairs.jsonl \
    ../Study-AIText_Detection/data/fastdetectgpt/raw/*.raw_data.json \
    ../Study-AIText_Detection/data/generated/raw/{xsum,writing,pubmed}_{qwen25-1.5b-instruct,llama32-1b-instruct}.raw_data.json
python scripts/score_pairs.py --pairs data/pilot_pairs.jsonl --old datedgpt/datedgpt-2021-base \
    --new datedgpt/datedgpt-2024-base --out outputs/pilot_k0.jsonl --batch-size 8
python scripts/evaluate.py --scores outputs/pilot_k0.jsonl --by generator
python scripts/diag_residual.py --scores outputs/pilot_k0.jsonl --by generator
```

생성기별(3개 도메인 합계 450쌍, 괄호는 쌍 bootstrap 95% CI):

| 생성기 | Δ AUROC | Δ 짝 승률 | Δ 평균 차이 (AI−사람) | `log p_2024` AUROC | `log p_2021` AUROC | 잔차 AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| **davinci** (대조군) | **0.468** [0.439, 0.497] | 0.442 [0.396, 0.487] | −0.013 | 0.587 | 0.595 | 0.472 |
| GPT-3.5-turbo | **0.933** [0.916, 0.949] | 0.962 | +0.226 | 0.936 | 0.872 | 0.934 |
| GPT-4 | **0.906** [0.887, 0.923] | 0.949 | +0.158 | 0.849 | 0.781 | 0.905 |
| Llama-3.2-1B-Instruct | **0.887** [0.868, 0.906] | 0.936 | +0.169 | 0.882 | 0.836 | 0.887 |
| Qwen2.5-1.5B-Instruct | **0.870** [0.848, 0.891] | 0.891 | +0.166 | 0.930 | 0.882 | 0.874 |

도메인별 Δ AUROC (채팅 생성기 4개 / davinci):

| 도메인 | GPT-3.5 | GPT-4 | Llama | Qwen | davinci |
| --- | --- | --- | --- | --- | --- |
| XSum (뉴스) | 0.979 | 0.941 | 0.907 | 0.920 | 0.497 |
| WritingPrompts (소설) | 0.989 | 0.951 | 0.953 | 0.966 | 0.461 |
| PubMedQA (짧은 답변, 약 50단어) | 0.817 | 0.840 | 0.792 | 0.718 | 0.449 |

관찰:

- **예측한 방향 그대로, 크기는 예측보다 큽니다.** 채팅 생성기 4개 모두 Δ AUROC 0.87~0.93입니다(예측 0.65~0.85). davinci는 0.47로 사람 글과 구분되지 않습니다(예측 약 0.5).
- **davinci 대조군이 두 가지 대안 설명을 약하게 만듭니다.** (1) "기계가 쓴 글 일반"이 원인이면 davinci도 Δ가 높아야 하는데 그렇지 않습니다. (2) "옛 사람 글이 2021 모델 학습에 더 많이 들어가서 사람 글의 Δ가 낮다"가 원인이어도 davinci의 Δ가 사람 글보다 높아야 하는데, 실제로는 조금 낮습니다(−0.013).
- **"새 모델이 예측하기 쉬운 글에 확률을 더 준다"도 아닙니다.** 사람 글 안에서 Δ와 `log p_2021`의 상관은 0.00이고, `log p_2021`로 Δ를 보정한 잔차의 AUROC도 원래와 같습니다.
- XSum에서는 Δ가 단일 모델 likelihood보다 좋습니다(GPT-4: 0.941 vs 0.887, Llama: 0.907 vs 0.828). PubMedQA처럼 짧고 질문이 양쪽에 공통으로 붙은 글에서는 likelihood가 더 좋습니다.
- 상관: corr(Δ, `log p_2024`)는 전체 +0.53, 사람 글 안에서 +0.17, AI 글 안에서 +0.43입니다.

토큰 단위(`scripts/token_contrib.py`, GPT-3.5/4 6세트):

- 차이가 몇몇 단어에 몰려 있지 않습니다. 기여 상위 10개 토큰이 전체 |기여|의 10%, 상위 200개가 30%입니다.
- 가장 큰 기여는 기능어입니다. 쉼표(AI 글에서 토큰당 Δ +0.233, 사람 글 +0.017), `and`(+0.226 vs −0.041), `a`, `the`, `This`가 그렇습니다. 같은 토큰이라도 AI 글의 문맥에 있을 때 2024 모델이 더 잘 맞춥니다. 나열("X, Y, and Z") 같은 AI 글의 흐름을 배운 것으로 보입니다.
- 전형적인 AI 단어도 보입니다. `significant`(+1.34 vs −0.15), `these`, `once` 등입니다.
- **데이터 결함:** Fast-DetectGPT 데이터는 사람 글에서만 줄바꿈이 지워져 있습니다(줄바꿈 토큰 AI 2,305개, 사람 0개). 줄바꿈 토큰의 Δ는 AI 글에서 −0.23이라 파일럿 AUROC를 **낮추는** 방향입니다. 본 실험 데이터는 양쪽 모두 한 문단입니다.

판단: 파일럿은 사람 글 시기 문제 때문에 판정에 쓰지 않지만, 신호의 방향과 대조군은 가설과 맞습니다. 본 실험으로 넘어갑니다.

### 본 실험: arXiv cs 초록 400쌍 (2026-11-12)

```bash
python scripts/fetch_arxiv.py --start 2025-07 --end 2026-10 --per-month 40 --out data/human_arxiv.jsonl   # 640편
python scripts/generate_ai.py --human data/human_arxiv.jsonl --out data/pairs.jsonl --per-generator 200
python scripts/score_pairs.py --pairs data/pairs.jsonl --old datedgpt/datedgpt-2021-base \
    --new datedgpt/datedgpt-2024-base --out outputs/k0.jsonl --batch-size 8
python scripts/evaluate.py --scores outputs/k0.jsonl --by generator --bootstrap 2000 --json outputs/k0_eval.json
python scripts/diag_residual.py --scores outputs/k0.jsonl --by generator
python scripts/token_contrib.py --scores outputs/k0.jsonl --pairs data/pairs.jsonl --min-count 10
```

환경: GTX 1060 6GB, torch 2.14.1+cu126, transformers 5.22.2, fp16, batch 8. 채점 800개 글에 약 2분, AI 초록 생성 540개에 약 15분.

데이터: 사람 초록 640편 수집(첫 제출 2025-07-02 ~ 2026-10, 월 40편, cs.CV 133 · cs.LG 119 · cs.CL 79 · cs.AI 62 · cs.RO 49 · 기타). 그중 생성기별로 서로 다른 200편씩 400편 사용. AI 초록은 거절 0건이고 정리 후 남은 마크다운이나 머리말도 0건입니다.

| | Δ AUROC [95% CI] | TPR@1%FPR | Δ 짝 승률 [95% CI] | Δ 평균 차이 | `log p_2024` AUROC | `log p_2021` AUROC | 잔차 AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **합계 400쌍** | **0.920** [0.902, 0.937] | 0.318 | **0.948** [0.925, 0.968] | +0.263 | 0.993 | 0.965 | 0.924 |
| Qwen2.5-1.5B-Instruct | 0.974 [0.962, 0.986] | 0.620 | 0.990 | +0.355 | 0.988 | 0.938 | 0.974 |
| Llama-3.2-1B-Instruct | 0.873 [0.837, 0.904] | 0.020 | 0.905 | +0.171 | 1.000 | 0.998 | 0.883 |

상관: corr(Δ, `log p_2024`)는 전체 +0.62, 사람 글 안에서 +0.23, AI 글 안에서 −0.19입니다.

**가설 판정 (미리 정한 기준)**

| 가설 | 기준 | 결과 | 판정 |
| --- | --- | --- | --- |
| H1 | 짝 승률 > 0.5, CI 하한 > 0.5 | 0.948 [0.925, 0.968] | 성립 |
| H2 | Δ AUROC ≥ 0.70 | 0.920 | 성립 |
| H3 | \|corr(Δ, `log p_2024`)\| < 0.8 | +0.62 | 성립 |

예측(짝 승률 0.60~0.75, AUROC 0.62~0.75)보다 훨씬 크게 나왔습니다. 라벨 노이즈의 영향을 과대평가했습니다.

**Δ 차이를 만드는 토큰** (기여 상위 200개가 전체 |기여|의 47%):

- AI 글 쪽: ChatGPT 문체로 알려진 단어가 상위를 채웁니다. `By`(AI 3.36회/1k토큰 vs 사람 0.35회, 토큰당 Δ +2.34 vs +1.07), `leverage`(3.46 vs 0.46), `various`(2.60 vs 0.12), `enhance`, `ensure`, `integrate`, `advanced`, `Through`, `significant`, `traditional`. 기능어도 같은 방향입니다(`and`: Δ +0.435 vs −0.047). 2024 모델은 이런 단어를 사람 글에서도 더 기대하지만(`various` 사람 글 Δ +1.83), AI 글이 훨씬 자주 씁니다.
- 사람 글 쪽: 2022년 이후 LLM 분야 용어입니다. `LM`, `prompt`, `Language`, `Large`, `hall`(ucination), `foundation`, `reasoning`, `tokens`. 두 모델의 차이가 크고(토큰당 Δ +2~6) 사람 초록에 더 자주 나옵니다. **지식 노이즈는 가설에 불리한 방향**으로 작용했습니다.

**강건성 확인**

- 놀람 필터(03 실험의 k): k = 0 / 0.05 / 0.1 / 0.2에서 합계 AUROC 0.920 / 0.937 / 0.937 / 0.928입니다. 옛 모델이 모르는 토큰을 빼도 신호가 유지되고 조금 좋아집니다.
- "새 모델이 예측 쉬운 글을 일괄 우대" 설명: 사람 글 안에서 corr(Δ, `log p_2021`) = −0.02이고, 보정 후 잔차 AUROC가 0.924로 그대로입니다. 파일럿과 같은 결론입니다.

**Δ가 가장 높은 사람 초록**(상위 8편)은 모두 LLM·멀티모달·RAG·프롬프트 주제였습니다. 그중 여럿은 AI로 다듬은 듯한 문장입니다("have demonstrated remarkable proficiency across diverse tasks", "plays a crucial role", "leveraging self-attention mechanisms to excel in various tasks", "To bridge this gap"). 사람 글 Δ 분포의 오른쪽 꼬리(p99 = 0.50, AI 글 p10 = 0.26)는 이 두 가지가 섞인 것으로 보입니다. 이 꼬리 때문에 TPR@1%FPR이 낮습니다(특히 Llama 0.02). 둘을 분리하려면 같은 주제의 2022년 이전 사람 글(ChatGPT 이전)과 비교해야 하고, 이는 02에서 다룹니다.

탐색(사전 계획 없음): 사람 초록의 평균 Δ를 제출 분기별로 보면 2025Q3~2026Q2는 +0.18~0.19로 평평하고, 2026Q3 +0.13, 2026Q4 +0.10(24편)으로 낮아집니다. 원인은 아직 모릅니다.

### 해석과 한계

1. **go 기준은 넘었지만, 이 데이터에서 Δ는 단일 모델 likelihood보다 약합니다.** `log p_2024` AUROC가 0.993입니다. 1B 모델이 기본 디코딩(T = 0.6~0.7, top-p)으로 쓴 초록은 likelihood만으로 거의 다 잡힙니다. 00이 보여준 것은 "Δ에 신호가 있다"이지 "Δ가 기존 방법보다 낫다"가 아닙니다. 파일럿에서는 XSum의 GPT-4·Llama·GPT-3.5 글에서 Δ가 likelihood보다 좋았습니다(0.941 vs 0.887 등). 그래서 Δ의 가치는 likelihood가 약해지는 조건에서 확인해야 합니다. 그런 조건은 강한 생성기, 높은 temperature, 다시 쓰기(paraphrase)입니다.
2. **생성기 두 개가 모두 2024년에 공개된 1B급 모델입니다.** 웹에 퍼진 AI 문체의 주류는 ChatGPT(GPT-3.5/4) 계열입니다. 파일럿의 GPT-3.5/4 결과는 신호가 그쪽에도 있음을 보여 주지만, 사람 글 시기 문제가 있습니다.
3. **MIA 혼동은 아직 직접 재지 않았습니다.** 사람 글을 두 모델 모두 못 본 2025년 이후 글로 골라 피했을 뿐입니다. 02에서 2024년 글과 2025년 글을 비교해 크기를 잽니다.

### 판단

H1·H2·H3가 모두 성립하므로 계획대로 **01·02·03을 진행합니다.** 다만 위 한계 1 때문에 우선순위를 아래처럼 조정할 것을 제안합니다.

- **02 (MIA 통제)를 먼저 합니다.** 지금 결과를 가장 크게 위협하는 설명이고, Δ가 높은 사람 글 꼬리(LLM 주제·AI로 다듬은 글)의 원인을 가르는 데에도 필요합니다.
- **04에 쓸 어려운 데이터를 앞당겨 만듭니다.** API 생성기 1개(계획의 "예산이 되면"), temperature 1.0 샘플링, paraphrase 버전입니다. likelihood가 무너지는 곳에서 Δ가 버티는지가 논문의 핵심 주장이 됩니다.
- 01(연도 곡선)은 모델 10개(약 27GB)를 더 받아야 하지만 채점은 빠릅니다(모델당 약 1분). 02와 병행할 수 있습니다.
