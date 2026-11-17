# 16. 크롤 연도별 AI 문체 비율: DatedGPT의 학습 데이터에서 AI 문체가 2023 크롤부터 늘었나

- 브랜치: `exp/16-crawl-ai-share`
- 상태: 명세·예측 (2026-11-17, 표본 전)
- 위치: 논문의 주장은 "모델의 학습 시기는 보고되지 않는 숨은 변수이고, 그 시기의 크롤에 섞인 AI 글을 통해 결과를 바꾼다"입니다. 사슬은 학습 시기 → 크롤 속 AI 글 → 모델의 AI 문체 선호 → 도구의 결과입니다. 이 실험은 **첫째 고리**를 처음으로 직접 잽니다. 지금까지는 가정이었습니다.

## 왜 하나

01·07은 2023 vintage부터 모델이 AI 글을 더 좋아한다는 것을 보였습니다. 그 원인이 "그 크롤에 AI 글이 섞였다"라는 것은 추론일 뿐, 데이터에서 잰 적이 없습니다. 심사자는 "시기는 대리 변수일 뿐, 크롤 크기·주제·언어가 같이 바뀐다"고 반박할 수 있습니다. DatedGPT가 학습한 FineWeb-Edu를 크롤 연도별로 뽑아, AI 문체 표지가 같은 시점에 늘었는지 봅니다.

## 데이터 (`scripts/sample_dumps16.py`)

- **FineWeb-Edu** (DatedGPT의 학습 데이터 기반)와 **FineWeb** (교육 품질 필터 전)를 씁니다. HF `HuggingFaceFW/fineweb-edu`, `HuggingFaceFW/fineweb`입니다. 둘 다 Common Crawl 덤프 110개(2013-20 ~ 2025-26)로 나뉘어 있습니다.
- **표본**: 크롤 연도마다 row group 16개(row group 하나 = 문서 1,000편)를 뽑습니다.
  - 그해 덤프들을 돌아가며 고르고, 덤프 안의 parquet 파일과 row group은 무작위입니다(seed 0).
  - 2013~2025년, 연도당 16,000편, 두 데이터셋 합쳐 416,000편입니다.
  - 글은 저장하지 않고 개수만 남깁니다. 표지 구가 나온 문서 일부(최대 300개)만 URL과 앞 300자를 따로 남겨 읽어 봅니다.
- DatedGPT vintage y는 대체로 그해 크롤입니다(01 Q5). 정확히 어떤 덤프를 어떻게 섞었는지는 모르므로, 크롤 연도 = 덤프 이름의 연도로 둡니다.

## 측정 (`scripts/crawl16.py`)

크롤 연도 y마다 m(y) = 1000 × 표지 개수 합 / 단어 수 합입니다. 표지 세 가지는 표본을 보기 전에 정했습니다.

| 표지 | 정의 | 쓰임 |
| --- | --- | --- |
| **mark** | 05의 표지어 18개 어간: delve, intricate, showcas, pivotal, meticulous, commendable, notably, realm, underscore, leverag, enhanc, comprehensive, crucial, various, seamless, innovative, robust, ensur | 주 측정 (05와 같음) |
| rare | 그중 드문 9개: delve, intricate, showcas, pivotal, meticulous, commendable, notably, realm, underscore | 보조 |
| phrase | ChatGPT 특유 구 18개: delve(s/ing) into, a testament to, in today's fast-paced, navigat(e/ing) the complexities, play(s) a crucial role, in the realm of, embark on a journey, it's/it is important to note, unlock the potential, rich tapestry, a pivotal role, as an AI language model, foster a sense of | 보조 |

- 2013~2024 곡선에 01의 통계(`year_curve.curve_stats`)를 그대로 씁니다. 2013~2022 직선에서 2023, 2024의 이탈 r, z(자유도 8), 꺾이는 시점입니다.
- 95% CI: 해마다 row group을 다시 뽑는 bootstrap 2,000회입니다.
- 2025는 DatedGPT vintage가 없어 보고만 합니다.

## 가설과 판단 기준 (표본 전, 2026-11-17)

FineWeb-Edu, mark 기준:
- **C1 (오름)**: r(2024)의 CI > 0 그리고 z(2024) > 2.306.
- **C2 (2023부터)**: r(2023)의 CI > 0 그리고 z(2023) > 2.306.
- **C3 (모양)**: r(2023)/r(2024)가 [0.44, 1.04] 안입니다. 01의 G가 0.74이므로 ±0.3입니다.
- **C4 (필터)**: 상대 상승 rel = r(2024) / (2019~2022 평균)이 FineWeb-Edu에서 FineWeb보다 큽니다(차이의 CI > 0).

| 결과 | 판정 | 논문 |
| --- | --- | --- |
| C1, C2 성립 | **"첫째 고리 성립"** | DatedGPT의 학습 데이터에서 AI 문체가 2023 크롤부터 늘었습니다. 모델의 이탈 시점과 같습니다 |
| C1만 성립 | "2024에만 보임" | 데이터에서는 2024 크롤부터 보입니다. 2023 vintage의 이탈은 표지로 잡히지 않을 만큼 적은 노출로도 생긴다고 씁니다 |
| C1 불성립 | "보이지 않음" | 표지로는 첫째 고리를 잴 수 없습니다. 첫째 고리는 가정으로 남기고, 주입 실험(별도 결정)이 필요합니다 |

C3과 C4는 판정과 따로 보고합니다.

**기술 (판정 없음)**
- 표지 비율에 해당하는 AI 글 비율: r(y) / (m_AI − m_사람). 보정은 15 입력의 RAID 채팅 생성기 대 사람 글, 그리고 00의 B/C 초록입니다.
- 12개 연도에서 m(y)와 01의 G(v)의 상관
- 2025년 값

## 미리 적어둔 예측 (2026-11-17, 표본 전)

본 것: 표본 점검으로 2013년 row group 하나(문서 1,000편)를 읽었습니다(개수가 기록되는지만 확인, 비율은 보지 않음).

- **C1**: 성립합니다(약 85%). mark의 rel2024는 +2%~+20%입니다. rare와 phrase는 바탕이 낮아 상대 상승이 더 큽니다(+10%~+100%).
- **C2**: 성립합니다(약 60%). 2023 덤프는 5개뿐이고, 그해 안에서도 AI 글이 늘어나는 중이었습니다.
- **C3**: 성립하지 않을 쪽입니다(성립 약 40%). r(2023)/r(2024)를 0.2~0.6으로 예측합니다. 14에서 손해가 흡수량에 대해 오목했던 것처럼, 적은 노출로도 모델이 크게 움직였다면 데이터 쪽 비율은 G의 0.74보다 작습니다.
- **C4**: 성립합니다(약 55%, 확신 낮음). 교육 품질 분류기가 설명문 같은 AI 글을 더 남겼을 것이기 때문입니다.
- AI 글 비율 환산(FineWeb-Edu, mark): 2023 0.5%~6%, 2024 2%~15%.
- m(y)와 G(v)의 상관: C1이 성립하면 0.8 이상입니다.
- 2025: rare와 phrase는 계속 오릅니다(확신 낮음). mark는 평평해질 수 있습니다(05에서 2025년 사람 초록의 표지어가 줄었음).
- **판정**: "첫째 고리 성립" 약 55%, "2024에만 보임" 약 30%.

## 한계 (미리)

- **표지는 대리 측정입니다.**
  - 사람이 LLM 단어를 따라 쓰는 간접 확산도 같이 잡습니다. 모델이 보는 것은 문체이므로 주장은 "크롤 속 AI 문체"이지 "AI가 쓴 문서 수"가 아닙니다.
  - 2023년의 AI 관련 글 급증(사람이 쓴 것 포함)이 일부 표지어를 늘릴 수 있습니다. 2022년까지의 추세는 느린 변화만 걷어냅니다. 그래서 phrase를 함께 봅니다.
  - various, ensure, enhance 같은 흔한 어간이 mark의 대부분을 차지해서 신호가 묽어집니다. 그래서 rare를 함께 봅니다.
- **크롤 ≠ 학습 데이터**: DatedGPT가 vintage마다 쓴 덤프 목록, 중복 제거, 표집은 정확히 모릅니다.
- **비율 환산은 거칩니다**: 생성기마다 표지어를 쓰는 정도가 다르고, 최신 모델은 덜 씁니다(05). 그래서 범위로만 씁니다.
- 2023은 덤프가 5개뿐이고, 연도당 row group 16개라 CI가 거칩니다.

## 실행

```bash
python scripts/sample_dumps16.py --repo fineweb-edu --out outputs/s16_edu.jsonl --examples outputs/s16_edu_examples.jsonl
python scripts/sample_dumps16.py --repo fineweb --out outputs/s16_fw.jsonl --examples outputs/s16_fw_examples.jsonl
python scripts/crawl16.py --edu outputs/s16_edu.jsonl --fw outputs/s16_fw.jsonl --g outputs/yc_eval.json \
    --calib-raid data/15_input.jsonl --calib-bc data/pairs.jsonl --json outputs/s16_eval.json > outputs/s16_eval.txt
```
