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
| 05 | [AI 다듬기 보정: 사람 초록의 Δ 상승을 AI가 고친 글의 비중으로 환산](docs/experiments/05-polish-calibration.md) | `exp/05-polish-calibration` | 핵심 기여 (02 후속) | 11/22 | 🔵 결과 기록: H1~H4 성립. 다듬기는 AI 이후 쌍 Δ만 올림(위약 기울기 −0.01). 반응은 편집 비율의 제곱꼴이라 가벼운 교정은 거의 안 보임. 2024년 사람 초록은 글의 약 1/3(실제 LLM 기준 0.40~0.53), 초록으로는 약 절반이 다듬어진 것과 같음. 표지어 기준 0.21 ≈ Liang et al. 17.5%. 다듬은 글 탐지는 Δ가 likelihood보다 좋음(0.875 vs 0.792) |
| 01 | 연도 곡선: 12개 vintage의 AI/사람 gap, 변화점 사후 확인 | `exp/01-year-curve` | 핵심 그림 | 11/29 | 🟢 실행 가능 (12개 vintage 받음, 토크나이저 동일 확인) |
| 04 | 기존 탐지기(Binoculars, Fast-DetectGPT)와 상보성: 놓친 글을 Δ가 잡는가 | `exp/04-complement` | 중요 | 12/11 | ⚪ 예정 |
| 03 | 놀람 기반 지식 토큰 필터 (k 스윕) + 제거 토큰 유형 분석 | `exp/03-k-sweep` | 후순위 | 12/11 | 🟢 실행 가능 (00에서 k = 0~0.2 예비 확인: 결론 불변) |

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
scripts/convert_apt_eval.py  APT-Eval(실제 LLM 5종이 다듬은 글)의 논문 초록 → 채점 입력 (05-b)
scripts/polish_calibration.py 편집 비율당 Δ 상승, 실제 LLM 전이, AI가 고친 글의 비중 β, 표지어·탐지·분포 모양 (05)
scripts/plot_polish_calibration.py 편집 비율별 Δ 상승, 실제 LLM 전이, β 그림 (05)
tests/                    모델 없이 도는 단위 테스트
```
