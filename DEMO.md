# 개인화 Audio-LM 데이터셋 파이프라인 데모

화자(speaker)의 페르소나에 맞춰 **같은 질문이라도 다른 답변**이 선호되어야 한다는 가설을 검증하기 위한 엔드투엔드 데모.
한 명의 사용자가 던진 질문을 여러 캐릭터의 목소리로 합성한 뒤, Audio-LM이 화자 정체를 듣기만 하고도 그에 맞는 답변을 내놓는지 평가한다.

```
캐릭터 페르소나 생성  →  맞춤형 답변 생성  →  음성 생성  →  평가
   (characters.py)       (responses.py)    (voices.py)   (evaluate.py)
```

각 단계는 디스크에 캐싱되며, 다시 실행해도 이미 끝난 작업은 건너뛴다.

---

## 1. 캐릭터 페르소나 생성

**파일:** `demo/characters.py` → `data/characters.json`

로컬 LLM(Ollama)에게 다양성 있는 가상 인물 N명을 JSON 배열로 만들게 한다. 각 캐릭터는 답변 스타일이 서로 분명히 달라야 한다.

생성되는 필드:
- `name`, `age`, `gender`, `occupation`, `background`
- `response_style`: `length`(short/medium/long), `tone`(formal/casual/warm/witty/analytical/blunt), `uses_examples`, `expertise_level`
- `voice_id`: `VOICES` 목록 중 하나 (edge-tts neural voice)
- `id`: 자동 부여 (`char_001`, `char_002` …)

핵심 의도는 **"같은 질문에 진짜로 다른 답변이 나올 만큼"** 캐릭터를 분산시키는 것이다. `voice_id`가 잘못된 값이면 인덱스 기반으로 fallback 매핑한다.

---

## 2. 맞춤형 답변 생성

**파일:** `demo/responses.py` → `data/triplets.json`

(캐릭터 × 질문) 모든 쌍에 대해, 그 캐릭터가 **선호하는 형태의 답변**을 LLM이 롤플레이로 생성한다.

- 입력 질문은 `demo/queries.py`의 `QUERIES`(스트레스, 책 추천, 머신러닝 설명 등 8개).
- 프롬프트는 캐릭터 프로필을 통째로 넣고 그 인물처럼 1~4문장으로 답변하라고 지시.
- `voice_id`는 답변 생성에 영향 주지 않도록 프롬프트에서 제거 (스타일은 텍스트 페르소나만으로 결정).
- 결과는 triplet 형태:
  ```json
  {
    "character_id": "...",
    "query_id": "...",
    "query_text": "...",
    "preferred_response": "...",
    "voice_id": "..."
  }
  ```
- (character_id, query_id)를 키로 증분 캐싱 — 도중에 끊겨도 다음 실행에 이어서 생성.

여기서 만들어진 `preferred_response`가 평가 단계의 **gold**가 된다.

---

## 3. 음성 생성

**파일:** `demo/voices.py` → `data/audio/{character_id}_{query_id}.mp3`

각 triplet에 대해 **질문 텍스트**(답변이 아니라)를 그 캐릭터에게 할당된 `voice_id`로 TTS 합성한다.

- `edge-tts` 사용 — 무료, API 키 불필요, 네트워크만 있으면 동작.
- 비동기 합성(`asyncio.run`)을 동기 인터페이스로 감쌈.
- 파일이 이미 존재하면 건너뜀.
- 합성 후 triplet에 `voice_audio_path` 필드 추가.

이 단계의 출력이 다음 단계 Audio-LM의 입력 오디오다. 즉, 모델은 **"누가" 묻는지를 목소리로만 듣고** 답하게 된다.

---

## 4. 평가

**파일:** `demo/evaluate.py` → `data/eval_results.json` (선택 단계, `--evaluate` 플래그 필요)

연구 가설을 직접 시험하는 단계.

- 모델: `Qwen/Qwen2-Audio-7B-Instruct`.
- 입력: 합성된 오디오(질문) + `"Listen to the speaker and answer their question."`라는 중립 지시문.
- **모델에게 캐릭터 정보는 절대 알려주지 않는다.** 오직 음성에서 추론해야 함.
- 출력: `model_response`를 triplet에 합쳐 저장. 비교 대상은 `preferred_response`(gold).

의존성(`transformers`, `torch`, `librosa`, `soundfile`)이 없으면 친절히 스킵. GPU가 있으면 fp16, 없으면 fp32로 동작.

```
gold:  캐릭터 페르소나가 선호하는 답변
model: 음성만 듣고 모델이 생성한 답변
```

이 둘이 얼마나 일치하는지가 곧 "Audio-LM이 화자 스타일에 맞춰 개인화하는 능력" 의 측정값이다.

---

## 실행

```bash
ollama serve
ollama pull llama3.1:8b

pip install -r requirements.txt

python run_demo.py --characters 4 --queries 5
python run_demo.py --characters 4 --queries 5 --evaluate --eval-limit 4
```

환경 변수:
- `OLLAMA_HOST` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `llama3.1:8b`)

## 산출물

```
data/
├── characters.json     # 단계 1
├── triplets.json       # 단계 2 + 단계 3 (voice_audio_path 추가)
├── audio/              # 단계 3
│   └── char_XXX_q_YYY.mp3
└── eval_results.json   # 단계 4 (옵션)
```
