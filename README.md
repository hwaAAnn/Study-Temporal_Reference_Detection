# Study-Temporal_Reference_Detection: 시간 축으로 웹에 퍼진 AI 글 재기

> 데이터 연도만 다른 언어모델 쌍은 같은 글을 다르게 볼까?
> 그 차이는 AI 문체가 웹에 퍼진 흔적일까, 그냥 시간이 흐른 결과일까? 그리고 그걸로 AI 글을 가려낼 수 있을까?

가제: *Can Temporal Differences in Language Model Likelihoods Reveal the Diffusion of AI-Generated Text Across the Web?*

ACL 2027 (ARR 2027-01-04 마감) 목표로 하는 연구 레포입니다.
**main 브랜치에는 연구 전체의 방향과 공통 코드**가 있고, **각 실험은 `exp/*` 브랜치**에서 진행합니다.

## 연구 한눈에 보기

2022년 말 이후 웹에는 LLM이 쓴 글이 섞였습니다. 그러면 2023년 이후 웹으로 학습한 base 모델은 AI 문체를 조금 "먹은" 상태일 것입니다.
구조·학습량·토크나이저가 같고 **데이터 연도만 다른** 모델 쌍(DatedGPT, 1.3B × 2013~2024)으로

```
Δ_{new,old}(x) = (1/T) Σ_t [ log p_new(x_t | x_<t) − log p_old(x_t | x_<t) ]
```

를 재면, AI 글은 Δ가 크고 사람 글은 작을 것이라는 가설입니다.

그런데 Δ에는 **AI 노출 말고도 일반적인 언어 변화, 지식 변화, 학습 포함 여부(MIA)**가 섞여 있습니다.
이 연구의 핵심은 이것들을 분리하는 것이고, 주 도구는 **이중차분(DiD)**입니다.

```
DiD = [Δ_{2024,2021}(AI) − Δ_{2024,2021}(사람)] − [Δ_{2021,2018}(AI) − Δ_{2021,2018}(사람)]
       └ AI 이후 구간: 일반 변화 + AI 노출 ┘      └ AI 이전 구간: 일반 변화만 ┘
```

- **학습이 필요 없습니다.** 공개된 시간 잘린 모델만 씁니다.
- **생성 모델을 몰라도 됩니다.** 특정 LLM이 아니라 웹 전체의 AI 문체 침투를 신호로 씁니다.
- **집 PC(GTX 1060 6GB)에서 돌아갑니다.** 1.3B fp16 추론, 모델을 하나씩 올립니다.

자세한 근거·데이터 그룹(A/B/C/D′)·선행연구·위험 요소는 [`docs/plan.md`](docs/plan.md)를 보세요.

## 실험 목록

| ID | 실험 | 브랜치 | 중요도 | 마감 | 상태 |
| --- | --- | --- | --- | --- | --- |
| 00 | [k=0 신호 확인 + 모델 점검 + 첫 DiD](docs/experiments/00-k0-signal.md) | `exp/00-k0-signal` | 필수 | 11/15 | ✅ go: H1~H4 성립. Δ AUROC 0.899, DiD +0.274 [0.254, 0.294], AI 이전 쌍 AUROC 0.42. 단 단일 모델 likelihood가 0.990 |
| 02 | [MIA·시간 변화 분리: 학습 이전/이후 사람 글, 이후 AI 글](docs/experiments/02-mia-control.md) | `exp/02-mia-control` | **핵심 기여** | 11/22 | ✅ H1 불성립: 새 모델이 학습한 사람 웹 글은 Δ가 AI 쪽으로 밀림(+0.085, AI 효과의 약 1/3, TPR@5% 0.06) → 탐지 주장은 새 모델 크롤 이후 글로 한정. H2 성립. arXiv는 학습 풀 밖이라 00 유지. ChatGPT 이후 사람 초록에서만 Δ 상승(+0.074) |
| 05 | [AI 다듬기 보정: 사람 초록의 Δ 상승을 AI가 고친 글의 비중으로 환산](docs/experiments/05-polish-calibration.md) | `exp/05-polish-calibration` | 핵심 기여 (02 후속) | 11/22 | ✅ H1~H4 성립: 다듬기는 AI 이후 쌍 Δ만 올림(위약 기울기 −0.01). 반응은 편집 비율의 제곱꼴이라 가벼운 교정은 거의 안 보임. 2024년 사람 초록은 글의 약 1/3(실제 LLM 기준 0.40~0.53), 초록으로는 약 절반이 다듬어진 것과 같음. 표지어 기준 0.21 ≈ Liang et al. 17.5%. 다듬은 글 탐지는 Δ가 likelihood보다 좋음(0.875 vs 0.792) |
| 01 | [연도 곡선: 12개 vintage의 AI/사람 gap, 변화점 사후 확인](docs/experiments/01-year-curve.md) | `exp/01-year-curve` | 핵심 그림 | 11/29 | ✅ H1~H4 성립: AI 글 선호는 10년간 평평(−0.003/년, 모델 잡음 0.009)하다가 2023 vintage부터 이탈(r 2023 +0.20, 2024 +0.27, z 17·22). 같은 글을 다듬기만 해도 같이 이탈, GPT-Neo 글은 안 오름, 사람 초록의 AI 쪽 이동도 2023부터. vintage 데이터는 누적 아님(그해 크롤 + 이월, f 0.26). 탐지 쌍은 2024−2022가 최선(AUROC 0.950) |
| 04 | [기존 탐지기와 비교: 공개 벤치마크(RAID 초록, APT-Eval)에서 Δ·s_λ vs Binoculars·Fast-DetectGPT, 상보성](docs/experiments/04-complement.md) | `exp/04-complement` | 중요 | 12/11 | ✅ H2·H3·H5 성립, H1·H4 불성립(1~3B 탐지기 기준). AI가 다듬은 글(APT 강)에서 s_0.9 0.847 > likelihood 0.815 > Binoculars(1B) 0.790 > Fast-DetectGPT 0.626. AI가 처음부터 쓴 글은 Binoculars가 우세(0.91~1.00). 이득은 ChatGPT 경계에서 옴(AI 이전 쌍은 이득 없음). Δ는 paraphrase에 가장 약함. Falcon-7B 비교(Runpod)는 후속 결정 대기 |
| 03 | [놀람 기반 지식 토큰 필터 (k 스윕) + 제거 토큰 유형 분석](docs/experiments/03-k-sweep.md) | `exp/03-k-sweep` | 후순위 | 12/11 | ✅ H1·H2·H4 성립, H3 (b) 불성립. 토큰 10%를 빼도 DiD 95% 유지, 위약 그대로. 단 필터는 지식 필터가 아님: 사람 글에서는 LLM 분야 용어를, AI 글에서는 채팅 문체 단어(leverage, comprehensive)를 뺌 → "지식 통제"로 쓰지 않음. 탐지에는 도움(사람 쪽 오탐 꼬리를 깎음): k = 0.05에서 Δ2024−2022 AUROC 0.959, TPR@5% 0.65 → 0.73 |
| 07 | [TiC-LM Oracle: 품질 분류기 없이 처음부터 학습한 계열(2013-05~2024-07, 7개)에서의 재현](docs/experiments/07-tic-oracle.md) | `exp/07-tic-oracle` | 핵심 그림 (분류기 반론) | 11/29 | ✅ H1~H4 성립. AI 선호가 2013-05~2023-01 평평(−0.003/년, 잡음 0.009)하다가 2024-07에서 이탈(r +0.191, z 16). 다듬기도 이탈(z 21), GPT-Neo 글은 안 오름, 사람 초록 이동도 작게 이탈(z 3.2). 분류기 없는 데이터·다른 구조·토크나이저에서 01 재현 → FineWeb-Edu 분류기 반론에 답함. 7개 시점 모두 fp16 점검 통과 |
| 08 | [탐지기의 두 자리: 채점 모델과 기준 모델의 학습 연도(DatedGPT 5×5, Fast-DetectGPT·Binoculars)](docs/experiments/08-detector-slots.md) | `exp/08-detector-slots` | 핵심 (위치 잡기 판정) | 11/22 | ✅ H1~H4 성립(두 탐지기·세 묶음) → 판정 "채택". 채점 자리에는 ChatGPT 이후 모델이 돕고 기준 자리에서는 해침: 기준 2022 + 채점 2024는 F 0.9998, 단일 모델 2024는 0.921로 단일 모델 2022(0.968)보다 나쁨. 2024 모델에서는 likelihood(0.996)가 F·Bi를 앞서 순위가 뒤집힘(사후). 효과는 Qwen 글에서 나옴(Llama 글은 천장). 배경: 07 출력의 R1 판정 성립(z_J +26.6) |
| 09 | [탐지기의 두 자리를 공개 벤치마크에서: RAID 초록·APT-Eval에 08의 기준×채점 칸(DatedGPT 2018·2021·2022·2024)](docs/experiments/09-slots-benchmarks.md) | `exp/09-slots-benchmarks` | 핵심 (탐지 주장) | 12/11 | ✅ 일반화 성립: H1~H4 성립(두 탐지기·두 주 묶음), H5 불성립. 기준 2022 + 채점 2024는 RAID 채팅 F 0.981·Bi 0.987(단일 모델 2024는 0.885), APT 강 0.751·0.768(단일 모델 2024는 0.584). "최신 단일 모델이 더 나쁘다"를 미리 세운 가설로 재현(H4). 최신 모델의 손해와 두 자리 효과는 채팅 문체의 글에만 있음(단일 모델 2022 → 2024: 채팅 이전 생성기 −0.004, 채팅 생성기 −0.073). 그러나 APT 강에서 Binoculars-1B(0.790)·s_0.9(0.847)·likelihood(0.815)보다 낮음 → 탐지 방법은 s_λ, 두 자리 결과는 분석으로. 원인: 다듬은 글은 log p가 오르는 만큼 엔트로피도 내려가 F·Bi의 정규화가 신호를 지움(추가 예측 E1~E3로 확인) |
| 10 | [TiC-LM에서 탐지기의 두 자리: 08의 기준×채점 격자를 다른 계열(TiC-LM Oracle 1B, 다섯 시점)에서 재현](docs/experiments/10-tic-slots.md) | `exp/10-tic-slots` | 핵심 (계열 재현) | 11/22 | ✅ 재현 판정 "재현". H1~H3 성립(두 탐지기, A 부분): 기준 202301 + 채점 202407 0.996 vs 단일 0.961/0.953(H19/C19 F), 효과는 DatedGPT의 약 절반. H4는 P1에서만 분명. S1(TiC s_0.9의 likelihood 대비 이득, P2) 불성립 +0.020 [−0.003, +0.043], S2 성립. TiC likelihood도 다듬은 글에 약함(0.701, DatedGPT 0.815) |
| 11 | [원 논문 설정의 두 자리: Falcon-7B·Falcon-7B-instruct 2×2 (Binoculars·Fast-DetectGPT 기본 쌍)](docs/experiments/11-falcon-slots.md) | `exp/11-falcon-slots` | 중요 (표준 설정, 04 Runpod 규칙) | 11/22 | ✅ H1~H4 성립(두 탐지기, 네 묶음) → 두 자리 판정 "성립". 원래 설정(기준 base + 채점 instruct)이 최선이고 뒤집으면 최악(APT 강 F 0.394), instruct 하나는 base 하나보다 나쁨. H5 성립: 다듬은 글에서 s_0.9 0.847 > 원래 Binoculars-7B 0.646(+0.20). 04의 Falcon 규칙 이행. RTX 4090 약 17분, 약 $0.25 |
| 12 | [탐지기 모델보다 새로운 생성기: 2025~2026년 공개 모델(gpt-oss-120b, GLM-4.7-Flash, Qwen3-30B-A3B Instruct/Thinking)이 쓴 초록에서 08의 두 자리](docs/experiments/12-recent-generators.md) | `exp/12-recent-generators` | 중요 (생성기 시기) | 11/22 | ✅ 판정 "이어진다". H2 성립(두 탐지기, H19·B): 기준 2022 + 채점 2024가 단일 모델보다 +0.35(H19, 0.537 → 0.891). 새 생성기는 훨씬 어려움(단일 2022 0.620, 08에선 0.968), gpt-oss-120b는 최고 0.764. H4는 H19만, H5(추론형이 더 어려움)·H6(DatedGPT 쌍 > Falcon 원래) 성립. s_0.9가 H19 최고 0.922. A100 약 48분, 약 $1.75 |
| 13 | [문턱값 이전: H19 사람 초록에서 한 번 정한 5% 문턱을 다른 출처·도메인·시기의 글에 그대로](docs/experiments/13-threshold-transfer.md) | `exp/13-threshold-transfer` | 중요 (실사용 조건) | 11/22 | ✅ T1~T3 성립. H19에서 한 번 정한 5% 문턱을 RAID에 그대로 써도 섞은 칸(기준 2022 + 채점 2024)이 단일 모델보다 나음(J 0.869 vs 0.570/0.792, RAID 사람 오탐 2.5%). 그러나 두 시점을 대비하는 점수는 ChatGPT 이후 사람 초록을 많이 AI로 판정(섞은 칸 15~27%, s_0.9 47%; 단일 모델·Falcon은 1~7%). 같은 시기 사람 글로 문턱을 다시 정하면 분리는 거의 완벽 → 시간 대비 탐지기는 같은 시기 보정이 조건. 추가 분석: 그 오탐은 LLM 주제 글(판정 42% vs 그 밖 20%)과 AI 표지어가 많은 글(판정된 글 1.7배)에서 비슷한 크기로 옴(표준화 계수 각 +0.28) → 내용(지식) 혼동과 실제 다듬기가 섞임. 같은 제목 짝 설계의 근거 |
| 14 | [용량-반응: 01·07에서 잰 채팅 흡수량 G가 기준 자리의 손해 크기를 예측하나 (DatedGPT 2023 중간 용량, 세 계열의 척도)](docs/experiments/14-dose-response.md) | `exp/14-dose-response` | 중요 (측정과 탐지를 한 양으로) | 11/22 | ✅ A 판정 "정량": 기준 2023은 기준 2022 이득의 33~49%를 남김(흡수량 예측 25%, 2배 범위 안, 선형보다 덜함). 흡수량 차이 하나가 격자 25칸 d 분산의 80~90% 설명(DatedGPT·TiC). 틀림: 가장 오래된 기준(2013)은 흡수량과 따로 손해 → 권고는 "가장 최근의 ChatGPT 이전 모델". B(12의 2025~2026년 생성기) "이어진다(단조)"지만 f 0.72~0.90(예측 0.3~0.6 틀림). C(09의 RAID 채팅·APT 강을 2023 포함 다섯 vintage로 로컬 재채점) "상용·대형 생성기로 이어진다": RAID F f 0.32로 A와 같음(ChatGPT·GPT-4·Llama-2-70B-chat 포함), Bi 0.54, APT 0.55~0.66. 틀림: B를 보고 찾은 생성기별 용량은 C에서 표본 밖 corr 0.18(F)·−0.05(Bi)로 실패(묶음 평균 수준만 맞음), 계열 척도(TiC·Falcon)는 글 묶음마다 흔들림. AUROC로는 천장이라 안 보임 → d로 보고 |
| 15 | [RAID 여덟 도메인의 연도 곡선: 2023 vintage의 이탈이 초록 밖에서도, 같은 모델의 기본판 대 채팅판(Mistral·MPT)에서도 나오나](docs/experiments/15-raid-domains.md) | `exp/15-raid-domains` | 핵심 (학습 시기가 숨은 변수) | 11/20 | ✅ "일반화 (채팅 문체)": D1·D3·D4 8/8 도메인, D2 6/8(news·wiki는 τ 2021, 근소), D5 성립. G_chat r(2024) 합침 +0.176 [0.170, 0.181] z 39.5, 도메인별 +0.111~+0.249. 같은 모델의 기본판 대 채팅판 K +0.185 z 22.1: 기본판 글은 이탈 0(Mistral +0.004, MPT −0.000), 채팅판은 +0.18~+0.19. 지시 튜닝(gpt3 +0.046, cohere +0.021)은 사이. 도메인 크기 순서 예측은 틀림 |
| 16 | [크롤 연도별 AI 문체 비율: DatedGPT의 학습 데이터(FineWeb-Edu)에서 AI 표지가 2023 크롤부터 늘었나, 교육 필터 전(FineWeb)과 비교](docs/experiments/16-crawl-ai-share.md) | `exp/16-crawl-ai-share` | 핵심 (학습 시기 → 크롤 속 AI 글) | 11/20 | ✅ 판정 "첫째 고리 성립": DatedGPT의 학습 데이터(FineWeb-Edu)에서 ChatGPT 특유 구가 2013~2022년 1,000단어당 약 0.02로 평평하다가 2023 크롤 0.060(약 3배), 2024 크롤 0.195(약 9배). 드문 표지어도 2022년까지 평평, 2024년 3.4배. 표지 곡선과 01의 G 상관 0.82~0.89. 교육 필터가 AI 문체를 더 남김(C4). 틀림: 상승 크기(예측의 수 배), C3(데이터 쪽 2023 몫 0.20 vs 모델 0.74 → 적은 노출로 모델이 크게 움직임, 사후). AI 글 비율 환산은 보정에 따라 8~94%라 쓰지 않음 |
| 17 | [문헌 감사: zero-shot 탐지 논문은 채점 모델의 학습 시기를 보고하나, 비교가 ChatGPT 경계에 걸쳐 섞여 있나 (무작위 30편)](docs/experiments/17-literature-audit.md) | `exp/17-literature-audit` | 중요 ("숨은"의 근거) | 11/20 | ✅ 판정(미리 정한 기준) "보고 안 됨, 교란은 드묾": 무작위 30편 중 열린 채점 모델의 학습 시기를 적은 논문 0편(R1 1편은 OpenAI 모델 각주), 학습 데이터 속 AI 글을 요인으로 다룬 논문 1편. 예상 밖: 3분의 1(10편)은 채점 모델 이름도 적지 않음. 모델을 적은 논문의 3분의 2가 ChatGPT 이후 모델 사용. 섞인 비교 17%(모름 제외 29%) → 예측(30% 이상)과 판정 예측("숨은 변수")은 틀림 |

상태 표시: 🟢 실행 가능 · 🔵 진행 중 · ✅ 완료(main에 merge) · ⏸ 대기 · ❌ 중단(브랜치에 사유 기록) · ⚪ 예정

```
11/15  00 ──┬─ H1·H2·H4 성립 → 02·01 진행
            └─ 미달 → 1주 진단(길이·도메인 통제, 생성기별, 토큰 유형별) → 11/22 결정
11/22  02 MIA·시간 변화 분리 (핵심 기여)
11/29  01 연도 곡선 (논문의 대표 그림)
12/11  04 상보성 + 03 k 스윕 → 12/17 arXiv → 2027-01-04 ARR 제출
```

## 빠른 시작 (00 실험)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # 로컬 GTX 1060
pip install -r requirements.txt
pip install -e .

# 0) 모델 점검: 토크나이저 동일성, fp16 안전성
python scripts/check_models.py --models datedgpt/datedgpt-2018-base datedgpt/datedgpt-2021-base datedgpt/datedgpt-2024-base --fp-check datedgpt/datedgpt-2021-base
# 1) 데이터: data/pairs.jsonl (B/C 짝), data/extra.jsonl (D′) — 형식은 docs/experiments/00-k0-signal.md
python scripts/fetch_arxiv.py --start 2025-07 --end 2026-10 --per-month 40 --out data/human_arxiv.jsonl   # B: arXiv API, 약 3분
python scripts/generate_ai.py --human data/human_arxiv.jsonl --out data/pairs.jsonl --per-generator 200    # C: 같은 제목의 AI 초록
python scripts/generate_old_lm.py --pairs data/pairs.jsonl --out data/extra.jsonl                         # D′: GPT-Neo-1.3B
cat data/pairs.jsonl data/extra.jsonl > data/all.jsonl
# 2) 점수 계산 (모델을 하나씩 올려서 VRAM 5GB 안에서 동작, 같은 pair_id의 글은 같은 길이로 맞춤)
python scripts/score_texts.py --input data/all.jsonl --out outputs/k0.jsonl --models datedgpt/datedgpt-2018-base datedgpt/datedgpt-2021-base datedgpt/datedgpt-2024-base --batch-size 8
# 3) 평가: 탐지 지표 + 그룹별 Δ + DiD + H1~H4 판정, 그리고 읽기용 진단
python scripts/evaluate.py --scores outputs/k0.jsonl --new 2024 --old 2021 --pre-new 2021 --pre-old 2018 --by-generator --json outputs/k0_eval.json
python scripts/token_contrib.py --scores outputs/k0.jsonl --input data/all.jsonl
python scripts/plot_delta_hist.py --scores outputs/k0.jsonl --out docs/experiments/figs/00-delta-hist.png
```

## 레포 구조

```
docs/plan.md              연구 질문, Δ의 원인 분리, 데이터 그룹, DiD, 확인 항목, 선행연구
docs/experiments/         실험별 목적·가설·판단 기준
src/trd/scoring.py        토큰 log-prob, Δ 점수, 놀람 기반 필터
src/trd/metrics.py        AUROC·AUPRC·TPR@FPR, Cohen's d, 짝 단위 bootstrap CI
src/trd/data.py           입력 형식 통일(짝/그룹), 짝 단위 공통 길이, 모델 연도 키
scripts/check_models.py   토크나이저 동일성, fp16 vs fp32 점검
scripts/fetch_arxiv.py    B: arXiv API → 2025-07 이후 cs 초록
scripts/generate_ai.py    C: 같은 제목으로 채팅 모델 초록 생성 → pairs.jsonl
scripts/generate_old_lm.py  D′: 같은 제목으로 ChatGPT 이전 세대 모델(GPT-Neo) 초록 생성 → extra.jsonl
scripts/convert_raw_data.py Fast-DetectGPT 형식 데이터 → 짝 행 (파일럿: 사람 A, davinci D′, 채팅 모델 C)
scripts/score_texts.py    여러 vintage로 점수 계산 → 점수 파일
scripts/evaluate.py       탐지 지표, 그룹별 Δ, DiD, H1~H4 판정
scripts/token_contrib.py  Δ 차이를 만드는 토큰과 극단 예시 읽기
scripts/diag_residual.py  "새 모델이 예측 쉬운 글을 우대"하는지 확인 (log p_old로 보정한 잔차)
scripts/plot_delta_hist.py  그룹별 Δ 분포 그림
scripts/fineweb_presence.py  arXiv 초록 페이지가 FineWeb-Edu 덤프에 있는지 URL 열로 확인 (02)
scripts/sample_fineweb.py    FineWeb-Edu 덤프에서 날짜 URL로 사람 웹 글 뽑기 (02-b)
scripts/generate_continuation.py  제목 없는 문서의 AI 기준점: 앞 30단어 이어쓰기 (02-b)
scripts/mia_contrast.py      시기별 C−H gap과 시기 간 대비(학습 포함, AI 보조), 시기 층화 bootstrap (02)
scripts/plot_period_means.py 시기별 사람·AI 평균 Δ 그림 (02)
scripts/polish_texts.py      로컬 채팅 모델로 사람 초록을 세 강도(교정·다듬기·다시 쓰기)로 다듬기 (05)
scripts/convert_apt_eval.py  APT-Eval(실제 LLM 5종이 다듬은 글) → 채점 입력: 논문 초록(05-b), 6개 도메인 전체(04)
scripts/polish_calibration.py 편집 비율당 Δ 상승, 실제 LLM 전이, AI가 고친 글의 비중 β, 표지어·탐지·분포 모양 (05)
scripts/plot_polish_calibration.py 편집 비율별 Δ 상승, 실제 LLM 전이, β 그림 (05)
scripts/year_curve.py        12개 vintage 곡선, 2013~2022 추세 이탈(r, z), 변화점, vintage 데이터 점검 f, 탐지 (01)
scripts/plot_year_curve.py   연도 곡선 그림 (01)
scripts/fetch_raid.py        RAID 벤치마크를 받으면서 한 도메인만 남기기 (04)
scripts/convert_raid.py      RAID 초록 표본 → 채점 입력 (04)
scripts/baseline_scores.py   기존 zero-shot 탐지기: Binoculars, Fast-DetectGPT (6GB용 작은 모델) (04)
scripts/det04.py             공개 벤치마크 탐지 비교: 조건별 AUROC·TPR, 상보성(구제율·결합), 시간 경계 대조 (04)
scripts/plot_det04.py        탐지기별·조건별 AUROC 그림 (04)
scripts/web_doc_curves.py    사람 글만의 vintage 곡선: 크롤 데이터가 누적인지 탐색 (01)
scripts/k_sweep.py           놀람 필터 k 스윕: DiD·위약·D′·polish_24·다듬기 기울기·탐지, 무작위 제거 대조, 제거 토큰 유형 (03)
scripts/plot_k_sweep.py      k 스윕 그림 (03)
src/trd/tic.py               TiC-LM Oracle(open_lm) 불러오기·변환·채점 (07)
scripts/tic_check.py         TiC-LM 체크포인트 fp16 변환, CPU fp32 기준값, fp16 규칙 비교 (07)
scripts/score_tic.py         TiC-LM 시점 하나로 점수 계산 → 점수 파일 (07)
scripts/run_07_tic.sh        TiC-LM 1B 일곱 시점 받기·점검·채점 체인 (07)
scripts/tic_curve.py         TiC-LM 곡선, ChatGPT 이전 여섯 시점 추세 이탈(r, z), 단계 크기(J, z_J), 탐지 (07)
third_party/open_lm/         open_lm 모델 코드(커밋 9bb92ef, MIT), TiC-LM용 (07)
scripts/score_slots.py       DatedGPT 다섯 연도의 기준×채점 25칸, Fast-DetectGPT·Binoculars (08)
scripts/slots_check.py       08 파이프라인 점검: 대각선 ll vs 01(09는 --yc-prefix yc04_로 04), 직접 계산 vs 묶음 계산 (08, 09)
scripts/slots_grid.py        칸별 AUROC, H1~H4 대비, 방향 판정 (08)
scripts/tic_r1_gate.py       07 출력에 대한 탐지 판정 R1: 단일 모델 AUROC의 2024-07 도약 (08 배경)
scripts/plot_slots.py        기준×채점 AUROC 격자 그림 (08)
scripts/run_09_slots.sh      04 시험 글에 기준×채점 칸 채점: 주 묶음 네 vintage, 나머지 두 vintage (09)
scripts/det09.py             공개 벤치마크에서 칸별 AUROC, H1~H5 대비, 04 탐지기와 비교 (09)
scripts/slots_decomp.py      칸별 ll·X(엔트로피) 분해, F·Bi − likelihood와 두 자리 대비를 04의 모든 묶음에서 bootstrap, 추가 예측 E1~E3 (09)
scripts/score_slots_family.py 08의 기준×채점 격자를 다른 계열에: TiC-LM, Hugging Face 모델 (10, 11, 12)
scripts/slots_check_family.py 위 스크립트의 점검: 직접 fp32 계산 vs 묶음 계산, 07 대각선 (10, 11, 12)
scripts/run_10_tic.sh        10 채점 체인: 체크포인트 받기·SHA 대조 → 점검 → A(25칸) → B(9칸)
scripts/build_11_input.py    11 입력: 08 초록 + 09 주 묶음 7,994편 (pod에 올릴 파일)
scripts/pack_11.sh           11 pod 묶음: 커밋된 코드(LF) + 입력 파일
scripts/run_11_pod.sh        11 pod 실행: 설치 → Falcon 두 모델 → 12편 점검 → 2×2 격자
scripts/build_12_titles.py   12 제목: 08의 H19·B 사람 초록 800편
scripts/generate_recent.py   12 생성 (vLLM): 2025~2026년 생성기, 01의 프롬프트, 최종 답만
scripts/build_12_input.py    12 채점 입력: 01의 거르기, 버린 이유 기록
scripts/run_12_pod.sh        12 pod 실행: 생성 4개 → 입력 → DatedGPT 25칸 → Falcon 4칸 (각 12편 점검)
scripts/thr13.py             문턱값 이전: H19 5% 문턱을 RAID·APT·H24·B에 그대로, FPR·TPR·J와 bootstrap (13)
scripts/thr13_why.py         최근 사람 초록 오탐의 원인: LLM 주제(지식 시점) vs AI 표지어(다듬기) (13)
scripts/dose14.py            14 용량-반응: 01·07의 흡수량 G로 기준 자리 손해 예측, D1~D5 판정 (부분 A)
scripts/dose14_posthoc.py    14 사후: 달력 거리 항, 두 자리를 따로 둔 적합
scripts/dose14_ext.py        14 부분 B(12의 생성기)·C(09 묶음 다섯 vintage 재채점)
scripts/dose14_gen_posthoc.py 14 생성기별 용량(사후, C에서 사전 등록 검증)
scripts/run_14_C.sh          14 부분 C 채점: 09 main 묶음을 다섯 vintage로
scripts/fetch_raid_domains.py RAID를 받으면서 여덟 도메인의 공격 없음·sampling 행만 남기기 (15)
scripts/build_15_input.py    15 입력: 도메인마다 원문 150개 × 사람 + 생성기 11개
scripts/check_15.py          15 파이프라인 점검: 04의 RAID 초록 점수와 대조
scripts/run_15.sh            15 채점: 점검 → 12 vintage
scripts/domains15.py         15 도메인별 연도 곡선(G_chat, K, G_old), 01의 이탈 통계, D1~D5 판정
scripts/sample_dumps16.py    16 FineWeb(-Edu) 크롤 연도마다 row group 16개, 문서별 표지 개수만 저장
scripts/crawl16.py           16 연도별 표지 비율, 01의 이탈 통계, C1~C4 판정, AI 글 비율 환산
scripts/fetch_17_candidates.py 17 감사 후보: arXiv 질의 하나, seed 0 순서
tests/                    모델 없이 도는 단위 테스트
```

## 라이선스

- 코드(`src/`, `scripts/`, `tests/`)는 MIT입니다. [LICENSE](LICENSE)를 보세요.
- 글과 그림(`README.md`, `docs/`)은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)입니다. 가져다 쓸 때는 출처(이 레포)를 밝혀 주세요.
- `third_party/open_lm/`은 원래 라이선스를 따릅니다([third_party/open_lm/LICENSE](third_party/open_lm/LICENSE)).
- 데이터와 모델 가중치는 이 레포에 없습니다. RAID, APT-Eval, DatedGPT, TiC-LM, Falcon 등은 각자의 원래 라이선스를 따릅니다. TiC-LM 가중치는 연구용 라이선스입니다.
