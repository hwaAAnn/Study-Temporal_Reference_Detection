# 17. 채점 모델의 학습 시기 분류

규칙(명세): 모델 카드나 논문에 적힌 사전 학습 데이터의 끝이 2022-11-30 이전이면 **이전**입니다. 이후 크롤을 포함하거나, ChatGPT류 모델이 만든 글로 학습(미세조정 포함)했다고 적혀 있으면 **이후**입니다. 적힌 것이 없으면 **모름**입니다. 공개 연도만으로는 정하지 않았습니다.

| 모델 | 분류 | 근거 |
| --- | --- | --- |
| GPT-2 (small~XL) | 이전 | WebText (2019) |
| GPT-Neo-2.7B, GPT-J-6B, GPT-NeoX-20B | 이전 | The Pile (2020) |
| OPT-125M~13B | 이전 | OPT 논문의 데이터(2021년까지의 공개 말뭉치) |
| mGPT | 이전 | Wikipedia + mC4 (논문의 설명) |
| LLaMA-1 13B | 이전 | 2022년 여름까지의 공개 데이터(논문) |
| Llama-2 7B/13B (base, chat) | 이전 | 모델 카드: 사전 학습 데이터 cutoff 2022-09. chat의 미세조정 데이터는 Meta의 사람 주석 |
| BLOOM-7.1B | 이전 | ROOTS (2022) |
| OpenAI babbage-002, davinci-002, text-davinci-003 | 이전 | OpenAI: 학습 데이터 2021-09까지 |
| Llama-3-8B (+Instruct) | 이후 | 모델 카드: cutoff 2023-03 |
| Llama-3.1-8B, Llama-3.2-1B/3B | 이후 | 모델 카드: cutoff 2023-12 |
| Vicuna-7B/13B | 이후 | ShareGPT(ChatGPT 대화)로 미세조정 |
| Falcon-7B-Instruct | 이후 | 모델 카드: Baize, GPT4All, GPTeacher(ChatGPT·GPT-4가 만든 글)로 미세조정 |
| StarChat-Alpha | 이후 | 해당 논문(2405.16133): GPT-3.5/4에서 증류한 지시 데이터로 미세조정 |
| Phi-2, Phi-3-mini | 이후 | 모델 카드: GPT-3.5 등이 만든 합성 데이터 |
| SmolLM, SmolLM2 | 이후 | 모델 카드: Cosmopedia(Mixtral이 만든 합성 글) 포함 |
| Qwen2.5 (0.5B~7B, Instruct) | 이후 | 기술 보고서: Qwen2 지시 모델이 만든 합성 데이터 포함 |
| gpt-oss-20B | 이후 | 모델 카드: cutoff 2024-06 |
| GPT-3.5/GPT-4 (API) | 이후 | gpt-4-1106 cutoff 2023-04 (API 모델은 열린 채점 모델이 아니어서 P에는 보조로만 씀) |
| Falcon-7B (base) | 모름 | RefinedWeb의 크롤 끝 시점이 모델 카드에 없음 |
| CodeLlama | 모름 | Llama-2 위에 코드 데이터를 더 학습, 코드 데이터의 시기 없음 |
| PhoGPT-4B (+Chat), Sailor-7B, DeepSeek-7B | 모름 | 시기 미기재 |
| Gemma 2 (2B/9B), Gemma 4, Qwen3.5, Ministral 3, Falcon3, OLMo-3 | 모름 | 확인 가능한 cutoff나 합성 데이터 기재를 찾지 못함(이 감사의 범위에서) |

- 한 논문 안에서 "이후" 모델이 하나라도 있으면 P = 예입니다.
- "이전"과 "이후"가 같은 주 비교표에 함께 있으면 M = 예입니다.
- "모름"만 있거나 모델을 적지 않은 논문은 P·M을 "모름"으로 둡니다.
