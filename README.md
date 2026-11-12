# Study-Temporal_Reference_Detection: 시간 축으로 AI 생성 글 잡기

> ChatGPT 이전 데이터로만 학습한 언어모델과 이후 데이터까지 학습한 언어모델은
> 같은 글을 다르게 볼까? 그 차이로 AI가 쓴 글을 가려낼 수 있을까?

ACL 2027 (ARR 2027-01-04 마감) 목표로 하는 연구 레포입니다.
**main 브랜치에는 연구 전체의 방향과 공통 코드**가 있고, **각 실험은 `exp/*` 브랜치**에서 진행합니다.

## 연구 한눈에 보기

2022년 말 이후 웹에는 LLM이 쓴 글이 빠르게 섞였습니다. 그러면 **2023년 이후 웹으로 학습한 base 모델은
AI 문체를 조금 "먹은" 상태**이고, 2021년까지의 웹으로만 학습한 모델은 그렇지 않을 것입니다.
같은 구조·같은 학습량·같은 토크나이저인데 **데이터 연도만 다른 모델 쌍**이 있다면, 두 모델의 확률 차이

```
Δ(x) = (1/T) Σ_t [ log p_new(x_t | x_<t) − log p_old(x_t | x_<t) ]
```

는 "AI 문체에 노출된 정도"를 재는 신호가 될 수 있습니다. AI 글은 Δ가 크고, 사람 글은 작을 것이라는 가설입니다.

- **학습이 필요 없습니다.** 공개된 시간 잘린 모델(DatedGPT, 1.3B × 2013~2024)만 씁니다.
- **생성 모델을 몰라도 됩니다.** 특정 LLM이 아니라 "웹 전체의 AI 문체 침투"를 신호로 씁니다.
- **집 PC(GTX 1060 6GB)에서 돌아갑니다.** 1.3B fp16 추론, 모델을 하나씩 올려 점수를 계산합니다.

자세한 근거·선행연구·위험 요소는 [`docs/plan.md`](docs/plan.md)를 보세요.

## 실험 목록

| ID | 실험 | 브랜치 | 마감 | 상태 |
| --- | --- | --- | --- | --- |
| 00 | [k=0 신호 확인 (go/no-go)](docs/experiments/00-k0-signal.md) | `exp/00-k0-signal` | 11/15 | ✅ go: arXiv 400쌍 Δ AUROC 0.920, H1~H3 성립. 단 단일 모델 likelihood가 0.993 |
| 01 | 연도 곡선: 12개 vintage에서 Δ가 2023 전후로 꺾이는가 | `exp/01-year-curve` | 11/22 | 🟢 실행 가능 |
| 02 | MIA 혼동 통제: 학습 포함 여부 vs AI 문체 | `exp/02-mia-control` | 11/22 | 🟢 실행 가능 (우선 권장) |
| 03 | 놀람 기반 지식 토큰 필터 (k 스윕) | `exp/03-k-sweep` | 12/04 | 🟢 실행 가능 |
| 04 | 기존 탐지기(Binoculars, Fast-DetectGPT, IRM)에 Δ를 더했을 때 | `exp/04-complement` | 12/11 | ⚪ 예정 |

상태 표시: 🟢 실행 가능 · 🔵 진행 중 · ✅ 완료(main에 merge) · ⏸ 대기 · ❌ 중단(브랜치에 사유 기록) · ⚪ 예정

```
11/15  00 k=0 확인 ──┬─ Δ가 사람/AI를 가름 (AUROC ≥ 0.70) → 01·02·03 진행
                    └─ 안 가름 → 04(보조 신호)로만 축소하거나 방향 중단
11/22  01 연도 곡선 + 02 MIA 통제 → 논문 핵심 그림 2개
12/04  03 k 스윕 → 지식 토큰 노이즈 제거 효과
12/11  04 보완 실험 → 12/17 arXiv → 2027-01-04 ARR 제출
```

## 빠른 시작 (00 실험)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # 로컬 GTX 1060
pip install -r requirements.txt
pip install -e .

# 1) 데이터 준비 (docs/experiments/00-k0-signal.md 참고)
python scripts/fetch_arxiv.py --start 2025-07 --end 2026-10 --per-month 40 --out data/human_arxiv.jsonl   # 사람 글 (arXiv API, 약 3분)
python scripts/generate_ai.py --human data/human_arxiv.jsonl --out data/pairs.jsonl --per-generator 200    # 같은 제목의 AI 글
# 2) 점수 계산 (모델을 하나씩 올려서 VRAM 5GB 안에서 동작)
python scripts/score_pairs.py --pairs data/pairs.jsonl --old datedgpt/datedgpt-2021-base --new datedgpt/datedgpt-2024-base --out outputs/k0.jsonl
# 3) 평가 (생성기별 표, 쌍 bootstrap 95% CI, H1~H3 판정) 와 토큰 단위 분석
python scripts/evaluate.py --scores outputs/k0.jsonl --by generator --json outputs/k0_eval.json
python scripts/token_contrib.py --scores outputs/k0.jsonl --pairs data/pairs.jsonl
```

## 레포 구조

```
docs/plan.md              연구 방향, 선행연구 비교, 위험 요소
docs/experiments/         실험별 목적·가설·기각 기준
src/trd/scoring.py        토큰 log-prob 계산, Δ 점수, 놀람 기반 필터
src/trd/metrics.py        AUROC, TPR@FPR, 쌍 단위 bootstrap CI
scripts/fetch_arxiv.py    arXiv API → 사람 글 (2025-07 이후 cs 초록)
scripts/generate_ai.py    같은 제목으로 AI 글 생성 → pairs.jsonl
scripts/convert_raw_data.py  Fast-DetectGPT 형식 데이터 → pairs.jsonl (파일럿용)
scripts/score_pairs.py    pairs.jsonl → 점수 파일
scripts/evaluate.py       점수 파일 → 지표 표
scripts/token_contrib.py  Δ 차이를 만드는 토큰과 극단 예시 읽기
scripts/check_precision.py  GPU fp16 vs CPU fp32 점수 차이 확인
tests/                    모델 없이 도는 단위 테스트
```
