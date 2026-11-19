# 17. 논문별 부호화

`17-candidates.jsonl`의 순서(seed 0)대로 1~82위를 읽어 30편을 채웠습니다. 읽기는 arXiv HTML(없으면 PDF)입니다. 긴 논문은 앞 약 8만 자를 그대로 읽고, 뒤쪽 부록은 요약으로 읽은 경우가 있습니다(표의 "부록 요약"). 모델 시기 분류는 `17-models.md`에 있습니다.

## 포함 30편

| 순위 | arXiv | 논문 (줄임) | 채점 모델 | R1 | R2 | P | M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2310.13606 | MULTITuDE | mGPT | 아니오 | 아니오 | 아니오 | 아니오 |
| 8 | 2608.25478 | VietAIDetector | PhoGPT-4B / -Chat | 아니오 | 아니오 | 모름 | 아니오 |
| 10 | 2405.16133 | 코드 다시 쓰기 탐지 | CodeLlama, StarChat-Alpha | 아니오 | **예** ("StarChat is fine-tuned on an instruction dataset distilled from GPT-3.5-Turbo and GPT-4, leading to a closer distribution") | 예 | 모름 |
| 14 | 2508.08096 | 교육 분야 탐지기의 한계 | **명시 안 함** (DetectGPT, Fast-DetectGPT) | 아니오 | 아니오 | 모름 | 모름 |
| 18 | 2505.15261 | AGENT-X | GPT-Neo-2.7B, GPT-J (부록 요약에 Phi-2, Qwen2.5-7B, Llama3-8B) | 아니오 | 아니오 | 아니오 | 아니오 |
| 20 | 2503.20421 | TempTest | Llama-3.1-8B, GPT2-XL, GPT-Neo-2.7B, GPT-J, OPT-2.7B | 아니오 | 아니오 | 예 | **예** |
| 22 | 2602.01240 | 프로토타입 라우팅 | LLaMA3-8B, gpt-oss-20B, Falcon-7B, DeepSeek-7B, Qwen-3B, GPT-J, NeoX, GPT-Neo, GPT2-XL, OPT | 아니오 (말뭉치 이름만) | 아니오 | 예 | **예** |
| 23 | 2611.00359 | 휴머나이저 감사 | Falcon-7B / -Instruct | 아니오 | 아니오 | 예 | 아니오 |
| 24 | 2306.05540 | DetectLLM | GPT2-XL, GPT-Neo, OPT-2.7/13B, GPT-J, LLaMA-13B, NeoX | 아니오 (말뭉치 이름만) | 아니오 | 아니오 | 아니오 |
| 26 | 2402.00263 | Pecola | GPT-2 small, GPT-J, GPT-Neo-2.7B | 아니오 | 아니오 | 아니오 | 아니오 |
| 28 | 2508.11933 | CAMF | **명시 안 함** (DetectGPT) | 아니오 | 아니오 | 모름 | 모름 |
| 29 | 2305.16617 | 베이즈 대리 모델 | GPT-2, Llama-2-7B/13B, Vicuna-7B/13B, GPT-J, GPT-Neo | 아니오 | 아니오 | 예 | **예** |
| 35 | 2610.00883 | DeBERTa-ConPara | GPT-Neo-2.7B (부록의 Fast-DetectGPT) | 아니오 | 아니오 | 아니오 | 아니오 |
| 36 | 2608.10698 | EVIL-Detect | Qwen3.5-4B, Qwen2.5-1.5B; Binoculars는 명시 안 함 | 아니오 | 아니오 | 예 | 모름 |
| 38 | 2609.32770 | C-HAT-Bench | **명시 안 함** (탐지기 10개) | 아니오 | 아니오 | 모름 | 모름 |
| 50 | 2503.17965 | RLHF와 탐지 | **명시 안 함** (Fast-DetectGPT) | 아니오 | 아니오 | 모름 | 모름 |
| 51 | 2412.11506 | Glimpse | GPT-Neo-2.7B, GPT-J, Phi-2, Qwen2.5-7B, Llama3-8B, babbage-002, davinci-002 | 아니오 | 아니오 | 예 | **예** |
| 52 | 2305.18149 | MPU | **명시 안 함** (GLTR, PPL, DetectGPT) | 아니오 | 아니오 | 모름 | 모름 |
| 53 | 2305.17359 | DNA-GPT | text-davinci-003, NeoX-20B, LLaMA-13B, OPT-125M, GPT-2 | **예 (약함)**: "OpenAI ... claimed training data is truncated up to September 2021" | 아니오 | 아니오 | 아니오 |
| 57 | 2410.20964 | DeTeCtive | **명시 안 함** (Binoculars "default settings", DetectGPT, GLTR) | 아니오 | 아니오 | 모름 | 모름 |
| 60 | 2504.02873 | Short-PHD | LLaMA-3-8B | 아니오 | 아니오 | 예 | 아니오 |
| 61 | 2607.04061 | Telescope | Gemma 2 2B/9B, Llama-3.1-8B, Falcon-7B, SmolLM/SmolLM2, GPT-J, GPT-Neo | 아니오 | 경계 (자기 학습 데이터의 합성/사람 글 구분을 다룸) | 예 | **예** |
| 62 | 2604.26328 | DSIPA | **명시 안 함** (LogRank, DetectGPT, Fast-DetectGPT, Binoculars) | 아니오 | 아니오 | 모름 | 모름 |
| 65 | 2611.05385 | 같은 출처, 다른 판정 (러시아어) | **명시 안 함** ("an open Russian-language model"; 뒤 3.5만 자는 읽지 못함) | 아니오 | 아니오 | 모름 | 모름 |
| 67 | 2509.18880 | DivEye | **명시 안 함** (기준선 LogRank, Fast-DetectGPT, Binoculars 등; DivEye 자체는 학습한 분류기) | 아니오 | 아니오 | 모름 | 모름 |
| 71 | 2509.26189 | VietBinoculars | PhoGPT-4B / -Chat, Falcon-7B / -Instruct, Qwen2.5-7B, Phi-3-mini, Sailor-7B | 아니오 | 아니오 | 예 | 아니오 |
| 72 | 2601.04833 | 후반 안정성 | Llama-3-8B-Instruct (부록에 GPT-J, Falcon-7B) | 아니오 | 아니오 | 예 | 아니오 |
| 74 | 2310.14479 | DetectGPT-SC | **명시 안 함** (GLTR, PPL, DetectGPT) | 아니오 | 아니오 | 모름 | 모름 |
| 77 | 2410.03856 | 잠재 공간 변수 | GPT-Neo-2.7B (주 표), gpt2-medium/large, Llama3-8B | 아니오 | 아니오 | 예 | 아니오 |
| 82 | 2610.33767 | 국소 엔트로피 요동 | Qwen3.5-2B (+Instruct), Gemma 4, Ministral 3, Falcon3, Falcon-7B, OLMo-3 | 아니오 | 아니오 | 모름 | 모름 |

## 대표 묶음 (따로)

| arXiv | 논문 | 채점 모델 | R1 | R2 | P | M |
| --- | --- | --- | --- | --- | --- | --- |
| 2301.11305 | DetectGPT | GPT-2, OPT-2.7B, GPT-Neo-2.7B, GPT-J, NeoX (+ GPT-3, Jurassic API) | 아니오 | 아니오 | 아니오 | 아니오 |
| 2305.17359 | DNA-GPT | (위와 같음) | 예 (약함) | 아니오 | 아니오 | 아니오 |
| 2306.05540 | DetectLLM | (위와 같음) | 아니오 | 아니오 | 아니오 | 아니오 |
| 2310.05130 | Fast-DetectGPT | GPT-2, OPT, GPT-Neo, GPT-J, NeoX, BLOOM, LLaMA-13B, Llama-2-13B | 아니오 (표 4에 말뭉치 이름만, 날짜 없음) | 아니오 | 아니오 | 아니오 |
| 2401.12070 | Binoculars | Falcon-7B / -Instruct, LLaMA-2-7B/13B(-Chat), LLaMA-13B | 아니오 (경계: "trained in large part on Common Crawl data from similar time periods") | 아니오 | 예 | 예 |

## 뺀 논문 (1~82위)

- **제목으로 뺌 (39편)**: 이미지·영상·음악(3, 4, 7, 9, 17, 31, 32, 42, 43, 45, 47, 49, 54, 58, 59, 63, 69, 70, 73, 76, 80, 81), 기계 생성 글 탐지가 아닌 과제(2, 5, 11, 12, 13, 19, 21, 25, 27, 33, 37, 39, 41, 56, 66, 68, 79)
- **읽고 뺌 (12편)**
  - 1: 학습한 RoBERTa만
  - 15: FLAN-T5 프롬프팅과 미세조정
  - 16: 미세조정 인코더와 프롬프팅
  - 30: 채팅 모델의 자기 판정
  - 34: NLI와 임베딩 유사도
  - 40: GPT-2 perplexity를 학습한 XGBoost의 특징으로만 씀. "a pre-trained model independent of the generative systems ... rather than artifacts of data overlap"이라는 문장이 있음
  - 44: 특징 분류기와 프롬프팅
  - 46: 어휘 범주 지표
  - 48: ChatGPT 프롬프팅
  - 55: 채팅 모델 프롬프팅. 이 논문은 GPT-3.5의 cutoff를 적었지만 채점 모델이 아님
  - 75: 오류 추론 분류
  - 78: 학습한 ELECTRA
- **읽지 못함 (1편)**: 64 (HTML 없음, PDF 추출 실패)
- 83위(2604.21223, IRM)는 읽었지만 30편이 찬 뒤라 표본에 넣지 않았습니다(R1·R2 아니오, Llama-3.2-1B·Gemma-2-2B).
