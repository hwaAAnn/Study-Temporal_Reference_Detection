# 시간 쌍 계측기, 부품은 선점됐고 조합은 비어 있다

2026년 11월 14일 기준으로, 크롤 연도만 다른 동일 레시피 언어모델 가족(DatedGPT, PIT, ChronoGPT 등)을 (1) LLM 문체가 웹과 인간 글쓰기로 퍼진 정도를 재는 계측기로 쓰거나 (2) 학습 없는 AI 생성·AI 연마 텍스트 탐지기로 쓴 연구는 찾지 못했다. 이 판단은 가장 가까운 논문 10여 편의 전문 확인과, 세 시간 고정 모델 가족의 Semantic Scholar 피인용 기록 50여 건 전수 검토에 근거한다. 하지만 아이디어를 이루는 부품은 이미 모두 공개돼 있다. 점수 형태 `log p_A − log p_B`는 IRM과 LAPD가 instruct−base 쌍으로 발표했고, MINT가 이를 멤버십 추론(MIA)과 같은 이론 틀로 묶었다. "2023년 크롤에서 뛴다"는 시점도 FineWeb 블로그의 마커 구문 곡선과 Visokay et al.의 웹 유병률 곡선이 먼저 보였다. 빈티지 간 우도를 계측기로 쓰는 형식도 Dated Data와 "What Improves During Pretraining?"이 지식 측정에 이미 썼다. 따라서 새로움은 공식이나 개시 시점이 아니라 **대비 축(크롤 연도, base 대 base), 측정 대상(채팅 문체의 흡수와 확산), 그리고 이를 믿을 수 있게 만드는 통제(사전-ChatGPT 쌍 대비 이중차분, 멤버십 교란 분리, 연마 보정)의 조합**에 있다. 선점 위험은 2013–2024 연도별 체크포인트 36개를 공개하고 "2022–2025 결과를 다음 버전에 싣겠다"고 예고했던 PIT 가족에서 가장 크다. 논문은 계측기(C1–C3)를 주 기여로, 탐지기(C4)를 보조 기여로 세워야 심사를 견딘다.

## 가장 가까운 선행연구는 모두 축이나 대상 하나가 다르다

이 보고서는 네 주장을 다음과 같이 부른다. **C1**은 시간 쌍을 확산 계측기로 쓰는 것(`Δ = log p_new − log p_old`, 사전-ChatGPT 쌍 대비 이중차분)이다. **C2**는 12개 빈티지에 걸친 채팅 문체 흡수의 연도 곡선과 LLM 연마 보정이다. **C3**는 멤버십 추론·지식 교란의 분리다. **C4**는 시간 쌍 탐지기 `s_λ = log p_new − λ·log p_old`와 그 연마 텍스트 초점이다.

가까운 선행연구는 네 무리로 나뉜다. 첫째는 금융·경제의 lookahead bias를 없애려고 만든 시간 고정 모델 가족이다. 계측기는 갖췄지만 AI 텍스트에 겨눈 적이 없다. 둘째는 빈티지 간 우도나 크롤별 모델을 계측에 쓴 연구다. 형식은 같지만 재는 대상이 지식이나 벤치마크다. 셋째는 집단 수준의 LLM 사용 측정이다. 같은 현상을 재지만 도구가 단어 빈도나 탐지기다. 넷째는 두 모델 탐지기다. 같은 대수를 쓰지만 쌍을 정렬 여부로 고른다.

| 연구 | 무엇을 했나 | 이 아이디어와 다른 점 | 겹치는 주장 |
|---|---|---|---|
| DatedGPT (Yan et al., 2026) | 1.3B 모델 12개를 FineWeb-Edu 크롤 연도별(2013–2024)로 각 약 100B 토큰씩 처음부터 학습. 주가 예측의 lookahead bias 제거 ([arXiv](https://arxiv.org/html/2603.11838v2)) | "AI-generated", "LLM-generated", "synthetic" 0회. "ChatGPT"는 참고문헌 제목에만 나옴 ([arXiv](https://arxiv.org/html/2603.11838)) | C3 인접(컷오프 perplexity 반전) |
| PIT (Kelly et al., 2026) | 원본 FineWeb로 1B/4B 모델을 시간순 연속 학습. 연도별 체크포인트 36개 공개 ([arXiv v3](https://arxiv.org/html/2607.11889v3)) | 2022년 이후 분석은 IFEval 전후 비교 하나뿐 | 없음(자원 수준 선점 위험) |
| ChronoGPT / -Instruct (He et al., 2025) | 1.5B, 1999–2024 연도별, 이전 해 체크포인트에서 이어 학습 ([arXiv](https://arxiv.org/html/2502.21206v3)) | AI 텍스트 언급 0회. "최신 데이터일수록 지시 수행이 좋아진다"는 결과를 설명하지 않음 ([arXiv](https://arxiv.org/html/2510.11677)) | 없음 |
| FineWeb 블로그 (Penedo et al., 2024) | 크롤마다 1.8B 모델 2개, 총 192개 학습. ChatGPT 마커 구문 빈도를 크롤별로 셈 ([블로그](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html)) | 모델은 벤치마크로만 평가. 우도·선호 측정 없음. 모델 미공개 | C1 약한 부분, C2 부분(어휘) |
| Dated Data (Cheng et al., COLM 2024) | 날짜 붙은 문서 버전의 perplexity 최소점으로 실효 지식 컷오프 추정 ([arXiv](https://arxiv.org/html/2403.12958v2)) | 재는 대상이 지식 컷오프 | C1 형식만 |
| What Improves During Pretraining? (Su et al., AACL 2026) | PIT-4B 연도별 8개 모델의 NLL 변동계수(TVS)를 지식 계측기로 씀 ([arXiv](https://arxiv.org/html/2610.37631v1)) | "AI-generated", "ChatGPT" 0회 | C1 형식 부분 |
| Liang et al. 분포 MLE (2024–2025) | 사전-ChatGPT 인간 단어 분포와 LLM 생성 단어 분포의 혼합 비율을 말뭉치 수준에서 추정 ([ICML 2024](https://arxiv.org/abs/2403.07183), [COLM 2024](https://arxiv.org/abs/2404.01268)) | 단어 빈도 기반. 문서별 점수 없음. AI 기준을 만들 생성기를 골라야 함 | C2 인접 |
| Visokay et al. (2026) | Common Crawl 71개 스냅샷의 510만 페이지에 Pangram·Binoculars를 적용하고 베이지안 모형으로 오류 보정 ([arXiv](https://arxiv.org/abs/2610.36631)) | 페이지 유병률이지 모델 흡수가 아님. 빈티지 모델 없음 | C2 부분 |
| Zhong (LessWrong, 2026) / Xu et al. (2026) | base 모델의 GPT 자기 식별이 학습 컷오프와 함께 오름 ([LessWrong](https://www.lesswrong.com/posts/cb5quszpxCbFDGk68/model-self-identification-could-be-subliminally-transferred)) / base 모델 출력은 탐지기에 사람처럼 보임 ([arXiv](https://arxiv.org/abs/2605.19516)) | 서로 다른 모델 가족 비교(비통제). 우도 측정 아님 | C2 부분 |
| IRM / LAPD (2026) | instruct(정렬) − base 로그우도비를 학습 없는 탐지 점수로 씀 ([IRM](https://arxiv.org/html/2604.21223v1), [LAPD](https://arxiv.org/html/2604.16923v2)) | 쌍을 정렬 여부로 고름. "cutoff" 0회 | C4 형식 상당 |
| MINT (Koike et al., ICML 2026) | 탐지기와 MIA의 최적 통계량이 같은 우도비 `P_M/P_Q`임을 증명하고 기존 방법을 `P_Q` 근사법으로 분류 ([arXiv](https://arxiv.org/html/2510.19492v2)) | 참조 모델을 크기·가족으로 고름. 시간 분할 언급 없음 | C3·C4 이론 틀 |
| ImBD (Chen et al., AAAI 2025) | 연마 쌍 500–1,000개로 `β·log(p_θ/p_ref)`를 학습해 기계 연마 텍스트 탐지 ([arXiv](https://arxiv.org/html/2412.10432v2)) | 학습 필요. 최종 점수는 정렬된 단일 모델의 곡률 | C4 연마 초점 부분 |
| Machines in the Crowd? (2025) | Reddit에서 2021년 1월–2022년 11월을 기준 기간으로 두고 기성 탐지기 점수에 이중차분 ([arXiv](https://arxiv.org/pdf/2510.07226)) | 텍스트 기간 간 이중차분이지 모델 쌍 간 이중차분이 아님 | C1 설계 인접 |

시간 고정 모델 논문들은 계측기를 만들고도 AI 텍스트를 보지 않았다. DatedGPT는 학습 데이터가 결과 기간을 포함하면 **표준편차당 26.4bp의 lookahead 프리미엄**이 붙는다는 빈티지 대비를 보였지만, 2022→2023 전환의 문체나 우도는 분석하지 않았다 ([arXiv](https://arxiv.org/abs/2603.11838)). DatedGPT를 인용할 때 주의할 점도 있다. 노트는 "NeurIPS 2026" 채택을 확인하지 못했고, 본문 각주는 ACL 지침을 언급하며, AEA 2026 프로그램에도 실렸다 ([arXiv v2](https://arxiv.org/html/2603.11838v2), [AEA](https://www.aeaweb.org/conference/2026/program/paper/BG2Tayk7)). 각 빈티지가 그해 크롤만 쓴 것인지 누적인지도 본문이 모호하다. 25,000 반복 × 배치 2,048 × 길이 2,048 ≈ 105B 토큰이라는 계산은 단일 연도 분할을 시사하지만 저자 확인이 필요하다. v2의 DatedInstruct는 DeepSeek-V4-Pro가 만든 QA로 학습했으므로 instruct 빈티지는 2013년판도 현대 LLM 문체에 노출돼 있다. 사전-ChatGPT 기준으로는 base 빈티지만 쓸 수 있다 ([arXiv v2](https://arxiv.org/html/2603.11838v2)). 흥미롭게도 이 계열 논문들에는 AI 텍스트 가설로 설명할 수 있는 미해명 규칙성이 있다. ChronoGPT-Instruct는 2024 빈티지가 지시 수행에서 가장 좋다고 보고했고 ([arXiv](https://arxiv.org/html/2510.11677)), PIT v3는 2021–2024 미세조정 체크포인트의 IFEval이 2013–2020보다 3–4%p 높다고 보고했다 ([arXiv v3](https://arxiv.org/html/2607.11889v3)). 어느 쪽도 원인을 논하지 않는다. 선점이 아니라 인용할 기회다.

설계상 가장 가까운 선례는 FineWeb 블로그다. 저자들은 "최근 크롤의 강한 성능이 부분적으로 LLM 생성 데이터 때문인가"라는 바로 그 질문을 던졌고, 크롤마다 동일한 1.8B 모델을 학습했다(**192개, H100 6만 시간 이상**). 그러나 답은 모델 우도가 아니라 단어 수로 냈고, 이 분석은 arXiv 논문이 아니라 블로그에만 있다 ([블로그](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html), [arXiv v2](https://arxiv.org/html/2406.17557v2)). Hugging Face의 HuggingFaceFW 조직에 있는 모델 105개 중 크롤별 모델은 없다 ([HF API](https://huggingface.co/api/models?author=HuggingFaceFW&limit=1000)). 탐지 쪽에서는 LAPD가 "base 모델은 인간 언어 분포를 모델링한다"를 가정 1로 명시한다 ([LAPD](https://arxiv.org/html/2604.16923)). 사전-ChatGPT base 모델은 학습 데이터에 LLM 출력이 거의 없으므로 이 가정을 글자 그대로 만족한다. 시간 쌍은 바로 이 가정이 성립하는 참조 모델을 고르는 방법이라고 내세울 수 있다.

## 계측기 주장은 대부분 비어 있지만, 2023년 개시 자체는 이미 알려졌다

| 주장 | 판정 | 이미 있는 것 | 새로운 것 |
|---|---|---|---|
| C1 시간 쌍을 확산 계측기로 | **새로움(대상·축 기준)** | 빈티지 간 우도 계측(Dated Data, TVS), 사전-ChatGPT 기준 이중차분(탐지기 점수) | 크롤 연도만 다른 base 쌍의 우도 차로 기계 문체 확산 측정, 모델 쌍 수준 이중차분 |
| C2 12빈티지 연도 곡선 | **부분적으로 새로움** | 2023년 개시(FineWeb 마커, Visokay 유병률), 컷오프별 흡수 경향(Zhong, 비통제) | 동일 레시피 가족에서 모델 측 흡수 곡선, 사전 빈티지 10개를 위약 시점으로 쓰는 설계 |
| C2 연마 보정 | **부분적으로 새로움** | 사전-ChatGPT 텍스트를 LLM에 통과시켜 AI 기준 분포 구축(Liang) | 모델 우도 계측기를 "연마 등가" 단위로 보정, 마커 회피 강건성 증거 |
| C3 멤버십 교란 분리 | **새로움(단, 필수 통제로 읽힘)** | 시간 이동이 MIA를 부풀린다는 지적(Duan, Das, Meeus), 컷오프 perplexity 반전(DatedGPT) | 시간 쌍 안에서 멤버십·지식·문체 분해 |
| C4 점수 형태 | **선점됨** | `log p_A − log p_B`(IRM, LAPD, Reference attack), 비율(Binoculars), 통합 이론(MINT) | 없음 |
| C4 λ 혼합 | **기여로 보기 어려움** | `log p − λ·log p_ref`(contrastive decoding, PMI 보정) | 분석 도구로서의 λ 스윕 |
| C4 시간 축 + 연마 텍스트 | **부분적으로 새로움** | 연마 탐지(ImBD, DetectAnyLLM, 학습 필요), IRM의 연마 공격 약점 | 학습 없이 "웹이 해 준 정렬", 편집률 스케일링, 가벼운 교정에 대한 둔감성 |

### C1: 빈티지 계측기를 기계 저작에 처음 겨눴다는 점이 새롭다

C1의 조합, 즉 크롤 연도만 다른 두 base 모델의 우도 차로 기계 문체의 확산을 잰 연구는 없다. 그러나 "시간대별 모델을 계측기로 쓴다"는 형식은 오래됐다. Kim et al.은 2014년에 연도별로 이어 학습한 모델로 1900–2009년 단어 의미 변화를 찾았고 ([ACL W14-2517](https://aclanthology.org/W14-2517.pdf)), Jaidka et al.은 2011년과 2015년 언어로 학습한 모델의 성능 저하를 신호로 썼다 ([ACL P18-2032](https://aclanthology.org/P18-2032.pdf)). Dated Data는 perplexity로 지식 컷오프를 쟀다 ([arXiv](https://arxiv.org/html/2403.12958v2)). 가장 가까운 것은 2026년 10월의 "What Improves During Pretraining?"으로, PIT-4B 연도별 8개 모델의 NLL 변동을 지식 계측기로 썼다 ([arXiv](https://arxiv.org/html/2610.37631v1)). 따라서 C1은 "최초의 빈티지 계측기"가 아니라 **"빈티지 계측기를 기계 저작이라는 새 대상에 처음 겨눈 것"**으로 써야 한다.

사전-ChatGPT 기준 이중차분도 설계 자체는 선례가 있다. Reddit 연구는 ChatGPT 이전 기간을 기준으로 탐지기 점수에 이중차분을 적용했고 2022년 11월에 첫 증가를 찾았다 ([arXiv](https://arxiv.org/pdf/2510.07226)). 아이디어의 이중차분은 텍스트의 기간이 아니라 **모델 쌍을 차분**한다. 사후 쌍의 Δ에서 간격이 같은 사전 쌍의 Δ를 빼서 크롤 간 일반 표류를 지운다. 이 모델 측 이중차분은 새롭다.

C1에는 이론 틀도 붙일 수 있다. `p_old ≈ P_H`(인간 분포), `p_new ≈ (1−α)·P_H + α·P_AI`로 두면 `Δ = log(1 − α + α·P_AI/P_H)`가 된다. 이는 Neyman–Pearson 최적 통계량 `P_AI/P_H`의 단조 증가 함수다. 이 식은 Liang et al.의 말뭉치 혼합 모형을 문서별·모델 기반으로 옮긴 것이다. 이 유도는 문헌에 없으며 이 보고서의 제안이다. 이 틀을 쓰면 Liang 방식의 약점인 "AI 기준 분포를 만들 생성기를 하나 골라야 한다"는 문제를 피하고, 2023–2024 크롤에 실제로 들어온 생성기 혼합을 기준으로 삼는다고 주장할 수 있다. 대가는 대상이 덜 특정적이라는 점이다. 크롤 간 AI 외 변화도 p_new에 함께 흡수된다.

### C2: 새로움은 "언제"가 아니라 "모델이 얼마나 흡수했나"에 있다

"2013–2022년은 평평하고 2023년 크롤에서 뛴다"는 시점은 새 발견이 아니다. FineWeb 저자들은 "delve", "as a large language model" 같은 구문 빈도가 **2023-14 크롤까지 거의 일정하다가 이후 급증**했다고 썼다 ([블로그](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html)). Visokay et al.은 AI 텍스트 포함 페이지가 2023년 2월까지 연 0.2%p, 이후 연 4.6%p, 2025년부터 연 8.1%p 늘어 **2026년 8월 약 24%**(95% 신용구간 22.9–24.4)에 이르렀다고 추정했다 ([arXiv](https://arxiv.org/html/2610.36631v1)). Medium의 AI 귀속률은 2022년 1월 1.77%에서 2024년 10월 37.03%로 올랐다 ([arXiv](https://arxiv.org/html/2412.18148v3)). 모델 측 흡수도 비통제 형태로는 보고됐다. Zhong은 OLMo-3-32B base가 정체성 질문에 **25.6%** GPT라고 답하고, base 모델 9개에서 이 비율이 학습 컷오프와 함께 오르며 ChatGPT 이전 데이터의 Pythia만 예외라고 보였다 ([LessWrong](https://www.lesswrong.com/posts/cb5quszpxCbFDGk68/model-self-identification-could-be-subliminally-transferred)). 다만 데이터·크기·레시피가 모두 다른 모델을 비교했다.

새로운 것은 두 가지다. 첫째, 크롤 연도 외에는 동일한 12개 빈티지로 모델이 채팅 문체를 얼마나 흡수했는지 그린 곡선은 없다. 둘째, 2013–2022년의 사전 빈티지 10개는 설계상 위약 시점(placebo breakpoint)을 여럿 준다. Hubarava & Ineichen이 텍스트 시계열에서 2018–2021년 위약 시점을 쓴 것처럼 ([alphaXiv](https://www.alphaxiv.org/abs/2610.26008)), 사전 빈티지 쌍들로 "우연한 점프"의 분포를 만들면 2023년 점프를 사건 연구 방식으로 검정할 수 있다. 이 곡선이 탐지기와 다른 정보를 준다는 근거도 있다. Xu et al.은 Llama-3-8B base의 이어쓰기가 GPTZero 기준 **96.7%** 사람으로 판정되고 instruct는 30.3%에 그친다고 보였다 ([arXiv](https://arxiv.org/abs/2605.19516)). base 모델이 흡수한 채팅 문체는 문서 단위 탐지기에 보이지 않는다. 우도 계측기는 그 보이지 않는 흡수를 잰다고 내세울 수 있다. 같은 맥락에서 "저배경 강철" 담론은 2022년 말을 오염 경계로 쓰고, lowbackgroundsteel.ai는 Common Crawl 2022년 8월 크롤을 안전한 자료로 등재했다 ([lowbackgroundsteel.ai](https://lowbackgroundsteel.ai/)). 평평한 사전 구간은 이 날짜 기반 주장을 모델 흡수 기준으로 처음 실증한다.

곡선에는 설명해야 할 수수께끼도 있다. Visokay의 성장률대로라면 2023년 말 AI 페이지의 순증은 몇 %에 불과하다. 그런데 2023년 빈티지가 큰 효과를 보인다면(프로젝트 내부 결과로는 2024년 효과의 약 74%), 세 설명 중 하나가 필요하다. 적은 양의 독특한 채팅 텍스트로도 선호가 빨리 포화하거나, FineWeb-Edu 분류기가 채팅풍 글을 더 남겼거나, 사람이 ChatGPT 어휘를 받아들였다. 둘째 설명은 실재하는 위험이다. FineWeb-Edu 분류기는 Llama-3-70B-Instruct가 CC-MAIN-2024-10 페이지에 매긴 점수로 학습했고 ([arXiv](https://arxiv.org/abs/2406.17557)), 문서를 위키백과 문체로 바꾸기만 해도 이 분류기의 채택 결정이 **약 7%** 뒤집힌다 ([ACL 2026](https://aclanthology.org/2026.acl-short.10/)).

연마 보정은 발상만으로는 새롭지 않다. Liang et al.은 사전-ChatGPT 초록과 서론을 LLM으로 요약한 뒤 다시 확장해 AI 기준 분포를 만들었다 ([Nature Human Behaviour](https://www.nature.com/nature-index/article/10.1038/s41562-025-02273-8)). 새로운 것은 모델 우도 계측기를 연마 비율을 아는 텍스트로 보정해 "연마 등가" 단위로 읽는 점이다. 이 보정은 현재 논쟁의 빈칸을 채운다. Topaz & Bahl은 어휘 변화가 "LLM 유병률의 보정된 척도가 아니다"라고 PNAS에서 비판했고 ([PNAS](https://www.pnas.org/doi/10.1073/pnas.2620928123)), Siler도 답변에서 자신의 통계가 문서 수가 아니라 "분포 이동"이라고 인정했다 ([PNAS reply](https://www.pnas.org/doi/10.1073/pnas.2621834123)). 마커 회피 강건성은 더 분명한 빈자리다. arXiv에서 "delve", "intricate" 같은 마커는 2024년 3–4월 이후 줄었지만 "significant" 같은 덜 눈에 띄는 단어는 계속 늘었다 ([arXiv](https://arxiv.org/html/2502.09606v2)). 모델 기반 척도가 마커 회피에 강건하다고 주장한 논문은 없다. 프로젝트의 관찰(2025년 이후 초록에서 마커가 1,000단어당 7.5개에서 4.9개로 줄었지만 Δ는 줄지 않음)은 새 유형의 증거다. 다만 마커 감소는 사람의 회피, 새 세대 LLM의 다른 어휘, 사용 감소를 구별하지 못한다. Lee는 한국어 KCI 초록 연구에서 LLM 제공사별 대조 초록의 마커가 모델 세대에 따라 바뀐다고 보였다 ([arXiv](https://arxiv.org/abs/2609.07447)). 2024년 LLM과 2025–2026년 LLM으로 각각 만든 연마 기준을 비교해야 앞의 두 원인을 가를 수 있다.

"인간 글쓰기로의 확산"이라는 표현은 조심해야 한다. 이 계측기는 직접 연마와 간접 문체 전파의 합을 잰다. Yakura et al.은 비대본 팟캐스트 824,634회분에서 ChatGPT 선호 단어가 출시 직후 급증했음을 합성 통제법으로 보였고, 사전 등록 실험(N=496)에서 짧은 챗봇 대화만으로 사람이 그 단어를 받아들인다고 보였다 ([alphaXiv](https://alphaxiv.org/overview/2409.01754v3)). 서면 텍스트에서 직접 사용과 간접 확산을 정량 분해한 연구는 없다. 붙여넣기가 불가능한 채널(비대본 발화 전사)이나 같은 저자의 전후 비교(Kusumegi et al.의 저자 수준 채택 시점 설계, [Cornell Chronicle](https://news.cornell.edu/stories/2025/12/ai-gives-scientists-boost-cost-too-many-mediocre-papers))를 써야 둘을 가를 수 있다.

### C3: 멤버십 교란 분리는 비어 있지만, 심사자는 이를 필수 통제로 본다

C3는 Δ가 높은 이유가 새 모델이 그 텍스트나 주제를 이미 봤기 때문(멤버십·지식)인지, 채팅 문체 때문인지를 가르는 일이다. 교란 자체는 잘 알려져 있다. Duan et al.은 MIA의 겉보기 성공이 회원과 비회원의 시간 범위 차이에서 온다고 지적했다 ([arXiv](https://arxiv.org/html/2402.07841v2)). Das et al.은 본문에서 날짜만 뽑아 임계값을 거는 "눈 가린 기준선"이 BookMIA에서 **AUROC 90.5%**로 기존 MIA를 이긴다고 보였다 ([arXiv](https://arxiv.org/html/2406.16201v2)). DatedGPT 자체도 분기별 헤드라인 perplexity가 컷오프에서 반전한 뒤 꾸준히 오른다고 보고했다 ([arXiv](https://arxiv.org/html/2603.11838)). 새 빈티지는 컷오프 이후 내용에 낮은 perplexity를 준다. 이것이 Δ의 지식 교란을 직접 보여 주는 그림이다. 여기에 MINT가 탐지기와 MIA는 같은 우도비를 근사한다고 증명했으므로 ([arXiv](https://arxiv.org/html/2510.19492v2)), 심사자는 "Δ는 결국 MIA 아니냐"고 물을 것이다.

시간 쌍 안에서 정확한 텍스트 멤버십, 주제 지식, 문체를 나눈 연구는 없다. 이 점에서 C3는 새롭다. 그러나 심사자는 이를 독립 기여보다 필수 통제로 읽을 가능성이 높다. 위상을 높이는 방법은 두 가지다. 하나는 분해를 정량화하는 것이다. 각 크롤과의 n-gram 중복 검사로 정확한 멤버십을, 같은 제목의 인간·채팅 쌍으로 주제 지식을, 사전-ChatGPT 생성기(GPT-Neo) 텍스트로 "기계 텍스트 일반"을 각각 떼어 내고 남는 몫을 보고한다. 다른 하나는 MINT를 긍정적으로 쓰는 것이다. p_new는 "LLM 생성 텍스트라는 부류"를 학습 데이터로 본 모델이고 p_old는 보지 않은 모델이다. 그러면 Δ는 개별 문서가 아니라 **분포 수준의 멤버십 검정**이 되고, MIA에서 회원 데이터를 빼고 학습한 참조 모델과 같은 구조가 된다. 이 해석은 MINT와 충돌하지 않고 그 위에 선다.

## 탐지기 주장은 점수 형태가 선점돼 시간 축과 연마 분석으로 승부해야 한다

C4의 점수 형태는 이미 발표됐다. IRM은 instruct와 base의 토큰별 로그비 합을 정규화 없이 탐지 점수로 쓰고, Llama-3.2-1B 쌍에서 평균 91.77로 Binoculars(87.67)를 넘었다. 다만 Gemma-2-2B 쌍에서는 77.05로 Binoculars(77.81)보다 낮았다 ([IRM](https://arxiv.org/html/2604.21223v1)). LAPD는 같은 차이를 "정렬 각인"이라 부르고 정보량 가중과 표준화를 더해 평균 AUROC **92.37**을 보고했다. 같은 표에서 Binoculars는 84.03, Fast-DetectGPT는 82.26이었다 ([LAPD](https://arxiv.org/html/2604.16923v2)). MIA의 Reference attack은 `log p_target − log p_ref`를 쓰고, Binoculars는 perplexity와 교차 perplexity의 비율을 쓴다. MINT는 이 모두를 "외부 참조로 인간 분포를 근사하는 방법"으로 묶었다 ([MINT](https://arxiv.org/html/2510.19492v2)). 따라서 "두 모델 로그비 탐지기"나 "스타일 정렬 탐지의 첫 학습 없는 버전"이라는 주장은 과장이다. 인용 시 주의할 점도 있다. IRM의 arXiv 코멘트는 "NeurIPS 2025"인데 게시일은 2026년 4월 23일이라 서로 맞지 않는다 ([arXiv](https://arxiv.org/abs/2604.21223)).

λ 혼합도 그 자체로는 기여가 되기 어렵다. IRM·LAPD·Reference attack은 λ=1, 단일 모델 우도 기준선은 λ=0에 해당하며, λ를 쓰는 탐지기는 찾지 못했다. 그러나 `log p − λ·log p_ref` 꼴은 contrastive decoding과 PMI식 보정에서 흔하다 ([Contrastive decoding](https://arxiv.org/abs/2210.15097)). λ는 최적점이 어디인지 보여 주는 분석 도구로 쓰는 편이 낫다. 위의 혼합 모형은 p_old가 인간 분포에 가까우면 λ=1이 최적이라고 예측한다. 실제 최적 λ가 1보다 작으면 p_old도 인간 분포에서 벗어나 있다는(예: 크롤 간 일반 표류) 진단 신호로 읽을 수 있다.

새로 남는 것은 세 가지다. 첫째는 대비 축이다. 쌍의 차이가 SFT나 RLHF가 아니라 크롤 연도이고, 두 모델 모두 base다. ImBD의 SPO 목적함수는 `r(x) = β·log(p_θ(x)/p_ref(x))`를 기계 연마본과 원문 사이에서 최대화한다 ([ImBD](https://arxiv.org/html/2412.10432v2)). 이는 `log p_new − log p_old`와 정확히 같은 꼴이다. ImBD가 연마 쌍 500–1,000개로 학습한 정렬을 시간 쌍에서는 웹 오염이 기계 텍스트 0개로 대신 해 준 셈이다. 둘째는 편집률 스케일링 분석이다. ImBD, DetectAnyLLM, IRM, LAPD 어느 것도 연마 정도에 따른 반응 곡선을 연구하지 않았다. 셋째는 가벼운 교정에 둔감하다는 성질이다. 이는 정책상 장점이 될 수 있다. GPT 탐지기들은 TOEFL 에세이에 평균 **61.22%** 거짓 양성을 냈고 ([arXiv](https://arxiv.org/pdf/2304.02819)), APT-Eval은 단어 1%만 연마해도 GLTR가 26.85%를 AI로 판정한다고 보였다 ([APT-Eval](https://arxiv.org/html/2502.15666v2)).

| 연마 텍스트 AUROC | ImBD GPT-4o 연마 | MIRAGE-DIG 연마 |
|---|---|---|
| ImBD (학습형) | 0.8899 | 0.7888 |
| DetectAnyLLM (학습형) | 미보고 | 0.9297 |
| Fast-DetectGPT | 0.6931 | 0.5720 |
| Likelihood | 0.5690 | 0.4653 |
| Binoculars | 미보고 | 미보고 |

출처: [ImBD](https://arxiv.org/html/2412.10432v2), [DetectAnyLLM](https://arxiv.org/html/2509.14268v1).

이 표에 비추면 프로젝트 내부 결과(국소 연마 초록에서 시간 점수 AUROC 약 0.88–0.90, 우도 약 0.79)는 학습형 탐지기 범위에 든다. 그러나 평가 세트가 다르다. MIRAGE에서는 우도가 우연 수준 아래(0.47)인데 프로젝트에서는 0.79다. 프로젝트의 "강한 국소 연마" 프로토콜이 MIRAGE의 "세부 보존" 연마보다 쉽다는 뜻일 수 있다. 정직한 주장은 "ImBD를 이긴다"가 아니라 "학습 없는 탐지기와 스타일 정렬 학습형 탐지기의 격차를 도메인 안에서 상당 부분 줄인다"다. Binoculars의 연마 텍스트 AUROC는 ImBD와 MIRAGE 표에 없고 APT-Eval에는 그림으로만 있다. 이 빈칸을 채우는 것만으로도 작은 기여다. IRM은 연마 공격이 자신의 가장 약한 지점이라고 밝혔다. 연마된 텍스트가 원문보다 높은 보상을 받아 인간과 LLM의 구분이 흐려진다는 것이다 ([IRM](https://arxiv.org/html/2604.21223v1)). 시간 쌍이 같은 텍스트에 다르게 반응하면 그것이 차별점이다. 설계를 지지하는 근거도 있다. LAPD는 다른 가족끼리 쌍을 만들면 AUROC가 **8.73–29.80%** 떨어진다고 보고했다 ([LAPD](https://arxiv.org/html/2604.16923v2)). "크롤 연도 외에는 동일"이라는 조건이 중요하다는 뜻이다.

가장 가까운 "체크포인트 대비" 선례는 Mireshghallah et al.이다. 이들은 작은 모델과 부분 학습 체크포인트가 더 나은 범용 탐지기라고 보고했고, 한 학습 실행 안에서는 최종 체크포인트가 늘 가장 나빴다 ([EACL 2024](https://aclanthology.org/2024.eacl-short.25.pdf)). 그러나 체크포인트의 차이는 학습 단계이지 데이터 날짜가 아니다. OpAI-Bench는 AI 편집 비율이 중간인 혼합본이 양 끝보다 탐지하기 어렵다는 비단조성을 보고했다 ([arXiv](https://arxiv.org/abs/2606.06481)). 프로젝트가 관찰한 매끈한(편집률 제곱에 비례하는) 반응이 이 벤치마크에서도 유지되는지가 좋은 시험이다. 종합하면 C4는 부분적으로 새롭고, 탐지기만으로 쓴 논문은 Findings나 워크숍 수준으로 판단된다.

## 선점 위험은 PIT 체크포인트에서, 심사 반론은 시간 표류에서 가장 크다

### 인접 그룹 다섯 곳이 한두 걸음 거리에 있다

가장 큰 위험은 PIT다. Kelly et al.은 PIT-1B, PIT-4B, PIT-4B-FT의 2013–2024 연도별 체크포인트 **36개**를 Hugging Face에 공개했다 ([HF API](https://huggingface.co/api/models?author=Diamegs)). v1과 v2는 "2022–2025년 결과를 다음 버전에서 보고하겠다"고 적었고, v3(2026년 10월 12일)는 이 문장을 지우고 IFEval 부록 하나만 더했다 ([v1](https://arxiv.org/html/2607.11889v1), [v3](https://arxiv.org/html/2607.11889v3)). NBER 판(w35247)에 그 결과가 들어갔는지는 확인하지 못했다. 누구든 PIT 2022 대 2023 쌍으로 같은 우도 분석을 곧바로 돌릴 수 있다. 차이는 설계에 있다. PIT 논문은 "시간순으로 정렬된 토큰 흐름"으로 연속 학습했다고 쓴다. 그러면 PIT 2023 대 2022는 한 해 더 이어 학습한 모델과 원래 모델의 대비이고, DatedGPT의 독립 학습 쌍과는 다르다. 그런데 다른 논문은 PIT-4B를 "FineWeb 연도별 분할로 학습한 모델"이라고 묘사해 설명이 엇갈린다 ([arXiv](https://arxiv.org/html/2610.37631v1)). 논문은 어느 설계에 기대는지 밝혀야 한다.

둘째, "What Improves During Pretraining?"은 이미 PIT-4B 빈티지 간 NLL을 계측기로 쓴다 ([arXiv](https://arxiv.org/html/2610.37631v1)). 대상을 지식에서 AI 문체로 바꾸는 데 한 걸음이면 된다. 셋째, DatedGPT 연구진은 빈티지 모델 연구를 이어 가고 있다. DatedGPT 공저자 Yao Lu는 1913년 컷오프 모델 TypewriterLM에도 참여했다 ([alphaXiv](https://www.alphaxiv.org/abs/2606.02991)). 넷째, CMU의 Xu et al.은 base 모델이 탐지기에 사람처럼 보인다는 논문을 냈고 ([arXiv](https://arxiv.org/abs/2605.19516)), 같은 이름의 저자 Zhong은 base 모델의 GPT 자기 식별이 컷오프와 함께 오른다는 글을 썼다 ([LessWrong](https://www.lesswrong.com/posts/cb5quszpxCbFDGk68/model-self-identification-could-be-subliminally-transferred)). 이 흐름은 "통제된 빈티지로 흡수를 재자"는 다음 단계로 곧장 이어진다. 다섯째, 정렬−base 탐지기 분야는 붐빈다. LAPD는 ReMoDetect, IRM, 이중 네트워크 선호 발산(ESWA 2026)을 같은 계열로 묶는다 ([LAPD](https://arxiv.org/html/2604.16923v2)). "옛 base 대 새 base"는 이들에게 자연스러운 다음 절제 실험이다. FineWeb 팀도 크롤별 모델 192개를 갖고 있지만 공개하지 않았다.

검색에는 한계가 있다. 노트는 arXiv, ACL Anthology, Hugging Face, Semantic Scholar 피인용 목록(DatedGPT 15건, ChronoGPT 28건, PIT 10건)을 썼고, Google Scholar와 OpenReview는 조회하지 않았다 ([S2 DatedGPT 피인용](https://api.semanticscholar.org/graph/v1/paper/arXiv:2603.11838/citations?fields=title,externalIds,year,abstract&limit=100)). Semantic Scholar는 2026년 11월 게시물을 늦게 색인할 수 있고, 비영어권 학회는 검색하지 않았다. Visokay et al.은 본문의 약 43%만 읽었다. 따라서 "없다"는 결론은 체계적 문헌 고찰이 아니라 집중 검색에 근거한다. 투고 직전에 "temporal reference model", "knowledge cutoff" + "machine-generated text detection", "pre-ChatGPT model" + "Binoculars", "point-in-time language model" + "AI-generated" 같은 질의로 다시 확인하고, 가능한 한 빨리 arXiv에 올려 우선권을 확보해야 한다.

### 예상 반론 11개와 이를 막는 대조 실험

| 예상 반론 | 반론이 실재한다는 근거 | 막는 실험·통제 |
|---|---|---|
| Δ는 AI 문체가 아니라 새 주제·어휘의 시간 표류를 잰다 | 시간 이동만으로 MIA 성능이 부풀려진다 ([Duan](https://arxiv.org/html/2402.07841v2), [Das](https://arxiv.org/html/2406.16201v2)) | 연도 간격을 맞춘 사전/사전 쌍(예: 2020 대 2021)과 사후/사후 쌍(2023 대 2024)의 이중차분, 같은 제목의 인간·채팅 쌍, 날짜 중립 주제, 옛 모델 컷오프 이후·ChatGPT 이전에 쓴 인간 글 |
| 새 모델이 그 글이나 주제를 이미 봤다 | DatedGPT의 컷오프 perplexity 반전 ([arXiv](https://arxiv.org/html/2603.11838)) | 평가 텍스트와 각 크롤의 n-gram 중복 검사, 어느 크롤에도 없는 텍스트, GPT-Neo 생성 텍스트 음성 대조, 평가 세트에 눈 가린 기준선 적용 |
| 점프는 FineWeb-Edu 분류기의 선택 효과다 | 분류기 라벨이 Llama-3-70B-Instruct 판정, 문체 변환만으로 결정 7% 반전 ([ACL 2026](https://aclanthology.org/2026.acl-short.10/)) | 원본 FineWeb로 학습한 PIT에서 곡선 재현, 같은 연도 AI·인간 페이지의 edu 점수 비교 |
| 크롤 연도는 작성 연도가 아니다 | CC 덤프에 옛 페이지가 많다 ([Dated Data](https://arxiv.org/html/2403.12958v2), [TiMoE](https://arxiv.org/pdf/2508.08827)) | 효과를 하한으로 해석, 2022년 마지막 덤프와 ChatGPT 출시일(2022-11-30)의 겹침 확인, 월 단위 분해([TiC-LM](https://arxiv.org/html/2504.02107v3) 114개 덤프) |
| 날짜로 고른 Reference attack이나 IRM일 뿐이다 | 같은 점수 형태가 이미 발표됨 ([MINT](https://arxiv.org/html/2510.19492v2), [IRM](https://arxiv.org/html/2604.21223v1)) | 혼합 모형 유도, 같은 레시피의 작은 모델·이른 체크포인트를 참조로 쓴 비교, 비슷한 규모의 IRM(Llama-3.2-1B 쌍)·LAPD·Binoculars와 직접 비교 |
| 자체 데이터로만 평가했다 | MIRAGE에서 우도 0.47 대 프로젝트 0.79 ([DetectAnyLLM](https://arxiv.org/html/2509.14268v1)) | APT-Eval ArXiv-10 부분집합, ImBD 연마 분할, MIRAGE-polish, OpAI-Bench, DetectRL 연마 공격 |
| Δ는 GPT-3.5/4 시대 문체만 잡는다 | 모델 세대에 따른 마커 교체 ([Lee](https://arxiv.org/abs/2609.07447)) | 2024년 LLM과 2025–2026년 LLM의 연마 기준 비교, 생성기별 반응 보고 |
| "인간 글쓰기로의 확산"은 직접 사용과 섞여 있다 | 간접 확산의 존재 ([Yakura](https://arxiv.org/abs/2409.01754)), 사람의 문체 모방이 추정을 부풀림 ([Holzwarth](https://arxiv.org/html/2608.10715v1)) | 비대본 발화 전사, 같은 저자의 전후 패널 비교 |
| 비원어민 글을 오탐한다 | TOEFL 61.22% 거짓 양성 ([Liang 2023](https://arxiv.org/pdf/2304.02819)), 사전/사후 학술 글의 탐지기 행동 차이 ([Style as a Confound](https://arxiv.org/html/2608.26710)) | TOEFL 세트와 2023년 이전 비원어민 초록에서 거짓 양성률 보고 |
| 2023년 이후 "인간" 라벨이 오염됐다 | CS 초록의 17.5%가 2024년 2월까지 LLM 수정 ([Liang COLM](https://arxiv.org/abs/2404.01268)) | 인간 라벨은 2023년 이전 텍스트만 사용 |
| 모델 자체가 모호하다 | DatedGPT instruct는 DeepSeek 생성 QA로 학습, 단일 연도/누적 불명확, 토크나이저 미공개 ([arXiv v2](https://arxiv.org/html/2603.11838v2)) | base 빈티지만 사용, 저자와 모델 카드로 확인해 명시 |

## 반드시 인용할 문헌

| 역할 | 문헌 | 인용 이유 |
|---|---|---|
| 계측 도구 | [DatedGPT](https://arxiv.org/abs/2603.11838) (Yan et al., 2026) | 주 계측 가족. 학회명은 미확인이므로 arXiv로 인용 |
| 계측 도구 | [PIT](https://arxiv.org/abs/2607.11889) (Kelly et al., 2026; NBER w35247) | 재현용 두 번째 가족, 원본 FineWeb, 선점 위험 |
| 계측 도구 | [ChronoGPT](https://arxiv.org/abs/2502.21206) / [ChronoGPT-Instruct](https://arxiv.org/html/2510.11677) (He et al., 2025) | 연속 학습형 빈티지, 2024 빈티지의 지시 수행 향상 |
| 계측 도구 | [TimeMachineGPT](https://aclanthology.org/2024.findings-naacl.208) (Drinkall et al., Findings NAACL 2024) | 2022 컷오프에서 멈춘 빈티지 가족 |
| 계측 도구 | [FineWeb 논문](https://arxiv.org/abs/2406.17557) + [블로그](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html) (Penedo et al., 2024) | 데이터 출처, Edu 분류기 구성, 크롤별 마커 곡선, 크롤별 모델 192개 |
| 계측 도구 | [TiC-LM](https://aclanthology.org/2025.acl-long.1551) (Li et al., ACL 2025) | 월 단위 Common Crawl 분할 |
| 계측 선례 | [Dated Data](https://arxiv.org/abs/2403.12958) (Cheng et al., COLM 2024) | perplexity 계측기, 크롤 연도와 작성 연도의 불일치 |
| 계측 선례 | [What Improves During Pretraining?](https://arxiv.org/abs/2610.37631) (Su et al., AACL 2026) | 빈티지 NLL을 지식 계측기로 사용 |
| 계측 선례 | [Kim et al. 2014](https://aclanthology.org/W14-2517.pdf), [Jaidka et al. 2018](https://aclanthology.org/P18-2032.pdf) | 시간대별 모델 계측의 전통 |
| 확산 측정 | Liang et al. [ICML 2024](https://arxiv.org/abs/2403.07183), [COLM 2024](https://arxiv.org/abs/2404.01268), [NHB 2025](https://www.nature.com/nature-index/article/10.1038/s41562-025-02273-8), [2502.09747](https://arxiv.org/abs/2502.09747) | 분포 MLE, 사전-ChatGPT 텍스트로 AI 기준을 만드는 발상 |
| 확산 측정 | [Kobak et al.](https://arxiv.org/abs/2406.07016) (Sci. Adv. 2025), [Holzwarth et al.](https://arxiv.org/abs/2608.10715) (2026) | 초과 어휘 방법과 그 한계 |
| 확산 측정 | [Geng & Trotta](https://aclanthology.org/2025.findings-acl.657/) (Findings ACL 2025), [Hubarava & Ineichen](https://www.alphaxiv.org/abs/2610.26008) (2026) | 마커 감소, 위약 시점 시계열 |
| 확산 측정 | [Topaz & Bahl](https://www.pnas.org/doi/10.1073/pnas.2620928123) (PNAS 2026) | 어휘 척도가 보정되지 않았다는 비판, 연마 보정의 동기 |
| 확산 측정 | [Visokay et al.](https://arxiv.org/abs/2610.36631) (2026), [Sun et al.](https://arxiv.org/abs/2412.18148) (ACL 2025) | 웹·소셜 미디어 유병률, 외부 검증 곡선 |
| 확산 측정 | [Yakura et al.](https://arxiv.org/abs/2409.01754), [Anderson et al.](https://arxiv.org/html/2508.00238v1) (2025) | 발화에서의 간접 확산 |
| 흡수 | [Xu et al.](https://arxiv.org/abs/2605.19516) (2026), [Zhong](https://www.lesswrong.com/posts/cb5quszpxCbFDGk68/model-self-identification-could-be-subliminally-transferred) (2026) | base 모델의 흡수, 탐지기에 보이지 않음 |
| 흡수 | [Klimaszewski & Andruszkiewicz](https://aclanthology.org/2026.acl-short.10/) (ACL 2026) | FineWeb-Edu 분류기의 문체 민감성 |
| 흡수 | [Juzek & Ward](https://aclanthology.org/2025.coling-main.426/) (COLING 2025) | 채팅 어휘의 출처, 웹을 통한 전이 논거 |
| 흡수 | [Shumailov et al.](https://www.nature.com/articles/s41586-024-07566-y) (Nature 2024), [Gerstgrasser et al.](https://arxiv.org/abs/2404.01413) (COLM 2024) | 모델 붕괴와 누적 체제 |
| 탐지 이론 | [MINT](https://arxiv.org/abs/2510.19492) (Koike et al., ICML 2026) | 탐지기 = MIA, 외부 참조 틀 |
| 탐지 | [IRM](https://arxiv.org/abs/2604.21223), [LAPD](https://arxiv.org/abs/2604.16923) (2026) | 같은 점수 형태, 직접 비교 대상 |
| 탐지 | [Binoculars](https://arxiv.org/abs/2401.12070) (ICML 2024), [Fast-DetectGPT](https://arxiv.org/abs/2310.05130) (ICLR 2024), [DetectGPT](https://arxiv.org/abs/2301.11305) (ICML 2023) | 표준 기준선 |
| 탐지 | [Carlini et al.](https://arxiv.org/abs/2012.07805) (USENIX Security 2021) | Reference attack |
| 탐지 | [Mireshghallah et al.](https://aclanthology.org/2024.eacl-short.25.pdf) (EACL 2024) | 체크포인트 대비 선례 |
| 탐지 | [Contrastive decoding](https://arxiv.org/abs/2210.15097) (Li et al., ACL 2023) | λ 꼴의 선례 |
| 연마 탐지 | [ImBD](https://arxiv.org/abs/2412.10432) (AAAI 2025), [DetectAnyLLM](https://arxiv.org/abs/2509.14268) (2025) | 학습형 연마 탐지, DPO 암묵 보상과 같은 꼴 |
| 연마 탐지 | [APT-Eval](https://arxiv.org/abs/2502.15666) (Findings ACL 2025), [EditLens](https://arxiv.org/abs/2510.03154) (ICLR 2026), [OpAI-Bench](https://arxiv.org/abs/2606.06481) (2026) | 연마 벤치마크, 연속 편집량, 비단조성 |
| 연마 탐지 | [Liang et al. 2023](https://arxiv.org/abs/2304.02819) (Patterns) | 비원어민 편향 |
| MIA 교란 | [Duan et al.](https://arxiv.org/abs/2402.07841), [Das et al.](https://arxiv.org/abs/2406.16201), [Meeus et al. SoK](https://arxiv.org/abs/2406.17975) (2024) | 시간 이동 교란 |
| 설계 선례 | [Machines in the Crowd?](https://arxiv.org/pdf/2510.07226) (2025), [Style as a Confound](https://arxiv.org/html/2608.26710) (2026) | 사전-ChatGPT 기준 이중차분, 전후 비교 |
| 경계 담론 | [lowbackgroundsteel.ai](https://lowbackgroundsteel.ai/) | 2022년 말을 오염 경계로 보는 관행 |

## 결론

이번 검토가 바꾼 이해는 이렇다. 탐지 문헌에서 채점 모델의 학습 데이터 날짜는 지금까지 통제되지 않은 변수였다. 2023년 무렵 탐지기들은 우연히 Pile 시대 모델을 채점기로 썼고, 이후 탐지기들은 Llama-3나 Qwen 같은 ChatGPT 이후 모델을 썼다. 두 결과는 이 축에서 비교할 수 없다. 이 아이디어에서 가장 방어하기 쉬운 기여는 새 점수 공식이 아니라, **이 숨은 변수를 처음으로 통제된 실험 변수로 바꾼 것**이다. 같은 이유로 계측기 쪽이 탐지기 쪽보다 강하다. 단어 빈도와 탐지기는 "문서가 얼마나 AI처럼 보이나"만 잰다. 시간 쌍은 "웹으로 학습한 모델이 무엇을 흡수했나"를 재며, 그 흡수는 탐지기에 보이지 않는다. 이 차이가 논문의 중심 주장이 돼야 한다.

빈자리는 아직 열려 있지만 좁아지고 있다. 가장 효율적인 다음 수는 PIT에서 연도 곡선을 재현하는 것이다. 이 실험 하나로 세 가지가 해결된다. 두 번째 가족에서의 재현성, 연속 학습과 독립 학습의 비교, 그리고 원본 FineWeb과 FineWeb-Edu를 대비한 분류기 증폭 검정이다. 동시에 선점 위험이 가장 큰 자원을 먼저 점유하게 된다. 탐지기 주장은 그다음에 공개 연마 벤치마크에서 IRM·LAPD·Binoculars와 직접 비교한 결과를 붙여 보조 기여로 두면 된다.
