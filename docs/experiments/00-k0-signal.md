# 00. k=0 신호 확인

- 브랜치: `exp/00-k0-signal`
- 마감: 11/15 (진단까지 포함하면 11/22)
- 장비: 로컬 GTX 1060 6GB (1.3B fp16 추론, 모델 하나씩 로드, batch 1부터)

## 질문

필터 없이(k=0) `Δ_{2024,2021}`이 같은 주제의 사람 글(B)과 AI 글(C)을 가르는가?
그 차이는 AI 이전 구간 `Δ_{2021,2018}`의 차이보다 큰가(DiD)?

## 0단계: 모델 점검 (데이터 만들기 전에)

```bash
python scripts/check_models.py \
  --models datedgpt/datedgpt-2018-base datedgpt/datedgpt-2021-base datedgpt/datedgpt-2024-base \
  --fp-check datedgpt/datedgpt-2021-base
```

- 토크나이저: 세 모델의 어휘와 샘플 토큰화가 완전히 같아야 함. 다르면 중단하고 대체 계열 검토.
- fp16: inf/nan 없음, CPU fp32 대비 평균 log-prob 차이 < 0.05. 넘으면 fp32를 CPU에서 돌리거나(느림) 클라우드 bf16 사용.
- 학습 스텝·데이터 양은 모델 카드와 논문 값을 이 문서 아래에 기록.

## 가설

- H1: C의 Δ가 B보다 크다 (짝 승률 > 50%, Cohen's d > 0).
- H2: Δ 단독 AUROC ≥ 0.70, 95% CI 하한 > 0.5.
- H3: Δ는 단일 모델 log-likelihood와 다른 정보를 담는다 (|r| < 0.8).
- H4: DiD > 0 이고 95% CI 하한 > 0.

## 데이터

`data/pairs.jsonl` (B/C 짝): `{"id", "human", "ai", "generator", "domain"}`
`data/extra.jsonl` (D′ 등 추가): `{"id", "text", "group": "D", "pair_id": <B의 id>, "generator"}`

- **B (사람)**: 2025년 이후 작성분만. 두 모델 모두 학습하지 않았어야 함.
  1순위 arXiv cs 초록 2025-06 이후 (제목 + 초록). 출처 선정 기준을 기록.
- **C (AI)**: 같은 제목을 주고 초록 생성. 생성기 `Qwen2.5-1.5B-Instruct`, `Llama-3.2-1B-Instruct` (+ 예산이 되면 API 모델 1개).
- **D′ (옛 세대 AI)**: 같은 제목으로 `gpt2-xl` 또는 `EleutherAI/gpt-neo-1.3B`가 이어 쓴 초록.
- 규모: 도메인 1개 × B 200개, C는 생성기마다 200개, D′ 200개
- 길이: 모든 글을 앞 200 토큰으로 자름(길이 효과 통제). 200 토큰 미만은 제외하거나 길이를 기록.

## 절차

```bash
cat data/pairs.jsonl data/extra.jsonl > data/all.jsonl
python scripts/score_texts.py --input data/all.jsonl --out outputs/k0.jsonl \
  --models datedgpt/datedgpt-2018-base datedgpt/datedgpt-2021-base datedgpt/datedgpt-2024-base
python scripts/evaluate.py --scores outputs/k0.jsonl --new 2024 --old 2021 --pre-new 2021 --pre-old 2018
```

보고 항목: AUROC [95% CI], AUPRC, TPR@1%·5% FPR, 짝 승률, 그룹별 평균 Δ [95% CI], Cohen's d,
Δ 분포 히스토그램(B/C/D′), DiD [95% CI]. 신뢰구간은 짝(pair_id) 단위 cluster bootstrap 1000회.

학습 없는 방법이라 00에는 학습/평가 분할이 없음. 04에서 결합 분류기를 학습할 때는 **원문(pair_id) 단위로 분할**.

## 판단 기준

| 결과 | 다음 단계 |
| --- | --- |
| H1·H2·H4 성립 | 02·01 진행 |
| H1 성립, H2 미달 (AUROC 0.55~0.70) | 진단 1주 후 결정 (아래) |
| H1 불성립 | 진단 1주 후에도 신호가 없으면 축소하거나 중단, 부정적 결과로 기록 |

진단 (1주 이내): Δ 분포가 실제로 다른가 / 길이·도메인 통제 후 달라지는가 / 특정 생성기에서만 효과가 있는가 /
전체 평균에서는 차이가 없지만 특정 토큰 유형에서는 차이가 있는가(03 일부를 앞당김).

## 산출물

- `outputs/k0.jsonl` (git에 올리지 않음), 결과 표와 그림을 이 문서 아래에 추가
