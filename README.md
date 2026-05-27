# 🐦 Twitter NER Lab — NLTK vs spaCy

> **자연어처리(NLP) 실습 | 소셜 미디어 개체명 인식(Named Entity Recognition)**  
> 전통 NLP(NLTK)와 현대 NLP(spaCy)가 트위터 노이즈 앞에서 어떻게 동작하는지 비교·분석합니다.

> ⚠️ **데이터 출처**: 이 실습의 트윗·위키피디아 문장은 실제 서비스에서 수집한 것이 아닌,
> NER 특성 연구를 위해 직접 작성한 **합성 데이터(Synthetic / Simulated Data)**입니다.
> Twitter API 접근 없이 소셜 미디어 노이즈 패턴(소문자, 해시태그, 이모티콘, 멘션)을 재현합니다.

---

## 📁 프로젝트 구조

```
KG-NIE/
├── twitter_ner_lab.py   # 전체 실습 코드 (연습문제 1~7)
├── figures/
│   ├── fig1_entity_count.png   # 트윗 vs 위키 개체 인식 수
│   ├── fig2_prec_rec_f1.png    # Precision / Recall / F1 전체 비교
│   ├── fig3_f1_trend.png       # 전처리 단계별 F1 추이
│   ├── fig4_type_heatmap.png   # 개체 유형 분포 히트맵
│   └── fig5_openie_vs_ner.png  # 간이 Open IE vs NER 비교
└── README.md
```

---

## ⚙️ 환경 설정

```bash
pip install nltk spacy matplotlib numpy
python -m spacy download en_core_web_sm

python -c "
import nltk
for pkg in ['punkt_tab','averaged_perceptron_tagger_eng',
            'maxent_ne_chunker_tab','words']:
    nltk.download(pkg)
"
```

---

## 🔬 연습문제 1 & 2 — 트윗 vs 위키피디아 Raw NER 비교

### 합성 데이터셋

| # | 트윗 (Synthetic) |
|---|-----------------|
| 1 | `omg @elonmusk just announced starship launch tmrw 🚀🚀 #SpaceX cant believe it lol` |
| 2 | `flight delayed again at jfk airport... thx american airlines 🙄 #travelproblems` |
| 3 | `just saw taylor swift in nyc!!!! no way she was at central park 😱😱 #swifties` |
| 4 | `why is amazon prime shipping so slow now?? used to be 2 days smh #disappointed` |
| 5 | `congrats to korea on winning the world cup qualifier vs japan 🔥🔥 #KFA #football` |

| # | 위키피디아 문장 (Synthetic) |
|---|--------------------------|
| 1 | `Elon Musk, the CEO of SpaceX and Tesla, announced a new rocket launch from Cape Canaveral.` |
| 2 | `American Airlines flight AA123 was delayed at John F. Kennedy International Airport in New York.` |
| 3 | `Taylor Swift performed a concert at Madison Square Garden in New York City last Friday.` |
| 4 | `Amazon's founder Jeff Bezos launched Blue Origin, a private aerospace company based in Kent, Washington.` |
| 5 | `South Korea defeated Japan 2-1 in the FIFA World Cup qualifier held in Seoul.` |

### Raw NER 실행 결과

```
트윗 1: omg @elonmusk just announced starship launch tmrw 🚀🚀 #SpaceX ...
  NLTK  → [('SpaceX', 'ORG')]
  spaCy → [('omg @elonmusk', 'PERSON'), ('🚀🚀', 'NUM'), ('SpaceX', 'PERSON')]

트윗 2: flight delayed again at jfk airport... thx american airlines ...
  NLTK  → (없음)
  spaCy → [('jfk airport', 'LOC'), ('american', 'NORP'), ('#', 'NUM')]

트윗 3: just saw taylor swift in nyc!!!! no way she was at central park ...
  NLTK  → (없음)
  spaCy → [('taylor', 'PERSON'), ('#', 'NUM')]

트윗 4: why is amazon prime shipping so slow now?? used to be 2 days smh ...
  NLTK  → (없음)
  spaCy → [('2 days', 'DATE'), ('#', 'NUM')]

트윗 5: congrats to korea on winning the world cup qualifier vs japan ...
  NLTK  → [('KFA', 'ORG')]
  spaCy → [('korea', 'LOC'), ('japan', 'LOC'), ('#KFA #', 'NUM')]
```

### 개체 인식 수 비교

![Figure 1 - Entity Count](figures/fig1_entity_count.png)

| 데이터 | NLTK | spaCy |
|--------|:----:|:-----:|
| **트윗 (5개)**       |  2개  | 13개  |
| **위키피디아 (5개)** | 25개  | 20개  |

- **트윗**: spaCy가 13개로 더 많이 인식하지만 노이즈(NUM, NORP 오분류)가 다수 포함됨
- **위키**: NLTK가 25개로 더 많이 인식하지만 라벨 오류(Musk→LOC, Korea→PERSON 등)가 심함

### 실제 결과 기반 이슈 분석

| 이슈 | NLTK 실제 동작 | spaCy 실제 동작 |
|------|--------------|----------------|
| `@elonmusk` 멘션 | ❌ 미인식 | ⚠️ `omg @elonmusk` 전체를 PERSON (범위 오류) |
| `#SpaceX` 해시태그 | ✅ **ORG 인식** | ❌ PERSON으로 **오분류** |
| `🚀🚀` 이모티콘 | ❌ 파싱 무시 | ⚠️ NUM으로 오분류 |
| `taylor swift` 소문자 인물명 | ❌ 미인식 | ⚠️ `taylor`만 부분 인식 |
| `korea`, `japan` 소문자 국가명 | ❌ 미인식 | ✅ LOC 인식 |
| 위키피디아 표준 문장 | ⚠️ 다수 인식, 라벨 오류 다수 | ✅ 정확 인식 |

> **주목 포인트**: `#SpaceX`에서 **NLTK(ORG)가 spaCy(PERSON)보다 더 정확**합니다.  
> spaCy는 문맥 기반 신경망이라 해시태그 직후 단어를 사람 이름으로 혼동하는 경향이 있습니다.
>
> **핵심 결론**: NLTK는 대문자 규칙에 전적으로 의존해 소문자 트윗에서 재현율이 급락합니다.  
> spaCy는 문맥 기반이라 소문자도 일부 인식하지만, 소셜 미디어 특수문자(@ # 🚀)에서 오분류가 많습니다.

---

## 💡 연습문제 3 — 트윗 특화 Feature 5가지

| # | Feature | 설명 | 구현 방법 |
|---|---------|------|----------|
| 1 | **@멘션 처리** | `@` 기호 제거 → 붙여쓰기 유지 → 가제터로 복원 | `re.sub(r'@(\w+)', r'\1', text)` |
| 2 | **#해시태그 분리** | `#` 기호만 제거, 단어는 유지 | `re.sub(r'#(\w+)', r'\1', text)` |
| 3 | **이모티콘·URL 제거** | 비ASCII 제거로 파서 혼란 방지 | `re.sub(r'[^\x00-\x7F]', '', text)` |
| 4 | **가제터 정규화** | 소문자·멘션형 → 표준 표기 복원 | `nyc→New York City`, `elonmusk→Elon Musk` |
| 5 | **최소 슬랭 처리** | NER 해석 불가 축약어 변환 | `tmrw→tomorrow`, `cant→can't` |

### 전처리 파이프라인

```python
# V1: 기본 노이즈 제거
def preprocess_v1(text):
    text = re.sub(r'http\S+', '', text)          # URL 제거
    text = re.sub(r'[^\x00-\x7F]', '', text)     # 이모티콘 제거
    text = re.sub(r'[!?]{2,}', '.', text)         # 중복 구두점 정리
    return text.strip()

# V2: V1 + @·# 처리 + 가제터 정규화
def preprocess_v2(text):
    text = preprocess_v1(text)
    text = re.sub(r'@(\w+)', r'\1', text)         # @ 제거, 단어 유지
    text = re.sub(r'#(\w+)', r'\1', text)         # # 제거
    words = [SLANG.get(w.lower(), w) for w in text.split()]  # 슬랭 변환
    text = ' '.join(words)
    for k in sorted(GAZETTEER, key=len, reverse=True):       # 가제터 (긴 패턴 우선)
        text = re.sub(re.escape(k), GAZETTEER[k], text, flags=re.IGNORECASE)
    return text.strip()
```

**V2 전처리 결과 예시:**

```
트윗 1 RAW: omg @elonmusk just announced starship launch tmrw 🚀🚀 #SpaceX cant believe it lol
트윗 1 V2 : omg Elon Musk just announced starship launch tomorrow SpaceX can't believe it lol

트윗 3 RAW: just saw taylor swift in nyc!!!! no way she was at central park 😱😱 #swifties
트윗 3 V2 : just saw Taylor Swift in New York City. no way she was at Central Park swifties
```

---

## 📊 연습문제 4 & 5 — Gold Truth 기반 정량 평가

### Gold Truth 구축 기준

| 항목 | 내용 |
|------|------|
| **텍스트** | TWEETS 원본 5개 + 추가 합성 트윗 5개, 총 10개 |
| **라벨 기준** | V2 전처리(가제터 복원) 후 NER이 인식 가능한 표준 표기 형태 |
| **라벨 체계** | PERSON / ORG / LOC (FAC·GPE는 LOC로 통일) |
| **평가 방식** | Exact-match F1 (대소문자 무관, lower() 비교) |

> **Gold 셋의 한계**: 수작업 annotation 없이 합성 텍스트에 정답 레이블을 설계한 것으로,
> 실제 학술 NER 벤치마크(WNUT, CoNLL 등)의 전문가 annotation과 다릅니다.

**Gold Truth 데이터 (10개):**

```
1. omg @elonmusk just announced ...  → Elon Musk(PERSON), SpaceX(ORG)
2. flight delayed again at jfk ...   → JFK Airport(LOC), American Airlines(ORG)
3. just saw taylor swift in nyc ...  → Taylor Swift(PERSON), New York City(LOC), Central Park(LOC)
4. why is amazon prime so slow ...   → Amazon(ORG)
5. congrats to korea vs japan ...    → South Korea(LOC), Japan(LOC), KFA(ORG)
6. elon musk just tweeted about ...  → Elon Musk(PERSON), Tesla(ORG)
7. breaking google ceo sundar ...    → Google(ORG), Sundar Pichai(PERSON), Silicon Valley(LOC)
8. united airlines cancelled ...     → United Airlines(ORG), Chicago(LOC), London(LOC)
9. omg kim kardashian spotted ...    → Kim Kardashian(PERSON), Paris(LOC)
10. microsoft and openai announce ... → Microsoft(ORG), OpenAI(ORG), San Francisco(LOC)
```

### Precision / Recall / F1 결과

![Figure 2 - Precision Recall F1](figures/fig2_prec_rec_f1.png)

| 시스템 | Precision | Recall | **F1** | 특이사항 |
|--------|:---------:|:------:|:------:|---------|
| NLTK Raw        | **1.0000** | 0.0833 | 0.1538 | 맞추면 100% 정확하지만 거의 못 잡음 |
| NLTK + V1       | 1.0000 | 0.0833 | 0.1538 | V1(기본 노이즈 제거)만으론 변화 없음 |
| NLTK + V2       | 0.4800 | 0.5000 | 0.4898 | 가제터로 재현율 대폭 향상 |
| spaCy Raw       | 0.3636 | 0.3333 | 0.3478 | 노이즈 오분류로 정밀도 낮음 |
| spaCy + V1      | 0.3810 | 0.3333 | 0.3556 | 소폭 향상 |
| **spaCy + V2**  | **0.7727** | **0.7083** | **0.7391** | 최고 성능 |

> **핵심 수치**: spaCy + V2 F1(0.74)이 NLTK Raw F1(0.15)의 **4.8배**입니다.  
> NLTK Raw의 Precision=1.0은 "잡은 건 다 맞지만 거의 아무것도 안 잡는다"는 의미입니다.

---

## 🔧 연습문제 6 — 전처리(Preprocessing)의 가치 증명

### F1 Score 단계별 추이

![Figure 3 - F1 Trend](figures/fig3_f1_trend.png)

| 전처리 단계 | NLTK F1 | spaCy F1 | spaCy 향상폭 |
|------------|:-------:|:--------:|:-----------:|
| Raw (없음)          | 0.1538 | 0.3478 | 기준 |
| V1: 기본 노이즈 제거 | 0.1538 | 0.3556 | **+0.008** |
| V2: V1 + 가제터 정규화 | 0.4898 | **0.7391** | **+0.391** |

> **결론**:  
> - **V1**(이모티콘·URL 제거)만으론 거의 효과 없음 → 구조적 노이즈 제거의 한계  
> - **V2 가제터**가 핵심 → `nyc→New York City`, `elonmusk→Elon Musk` 복원으로 F1 2배 이상 향상  
> - 트위터 NER에서는 **"어떤 모델이냐"보다 "텍스트를 어떻게 정규화하느냐"** 가 더 중요합니다

---

## 🗺️ 개체 유형 분포 히트맵

![Figure 4 - Entity Type Heatmap](figures/fig4_type_heatmap.png)

| 관찰 | 내용 |
|------|------|
| **NLTK (Tweet)** | ORG 2건만 인식 — 재현율 참혹 |
| **spaCy (Tweet)** | LOC·PERSON 인식하지만 NUM(이모티콘) 노이즈 다수 |
| **NLTK (Wiki)** | PERSON·ORG·LOC 다수 인식하나 Musk→LOC, Korea→PERSON 등 라벨 오류 |
| **spaCy (Wiki)** | PERSON·ORG·LOC 정확하게 인식, EVENT(World Cup) 추가 |

---

## 🔗 연습문제 7 — 간이 Open IE vs 전통 NER

> ⚠️ **주의**: 본 구현은 POS 태깅 기반 규칙 Open IE로, Stanford Open IE·OpenIE5 등 전문 시스템과 무관합니다.

![Figure 5 - Open IE vs NER](figures/fig5_openie_vs_ner.png)

### 비교 요약

| 방법 | 출력 | 특징 |
|------|------|------|
| **NLTK NER** | `[New York](LOC)`, `[American Airlines](LOC)` | 개체 단편 추출, 관계 맥락 없음. 라벨 오류 있음 |
| **spaCy NER** | `[American Airlines](ORG)`, `[Chicago](LOC)` | 더 많은 개체, 일부 오분류 (SpaceX→LOC) |
| **간이 Open IE** | `(flight, delayed, New York)` | 사전 정의 없이 동적 관계 추출 (품질은 제한적) |

### Open IE 추출 결과 예시

```
문장   : Elon Musk, the CEO of SpaceX, announced a new Starship launch from Cape Canaveral
NER    : [Elon Musk](PERSON), [SpaceX](LOC), [Starship](PROD), [Cape Canaveral](LOC)
OpenIE : (SpaceX, announced, a new Starship launch)

문장   : Jeff Bezos founded Amazon in Seattle before launching Blue Origin
NER    : [Jeff Bezos](PERSON), [Amazon](ORG), [Seattle](LOC), [Blue Origin](ORG)
OpenIE : (Jeff Bezos, founded, Amazon)
         (Seattle, launching, Blue Origin)    ← 주어 오류 (간이 구현 한계)

문장   : Microsoft acquired OpenAI partnership to integrate GPT-4 into Office products
NER    : [Microsoft](ORG), [OpenAI](ORG), [Office](ORG)    ← Office 오분류
OpenIE : (Microsoft, acquired, OpenAI partnership)
         (OpenAI partnership, integrate, GPT-4)
```

### Open IE vs NER 핵심 차이

| 관점 | Open IE | NER |
|------|---------|-----|
| **출력** | (주어, 동사, 목적어) 삼중쌍 | 개체명 + 타입 |
| **사전 정의** | 관계 유형 불필요 (비지도) | 타입 목록 필요 |
| **활용** | KG 구축, 이벤트 추적, 관계 발견 | 개체 인식, 분류, 검색 |
| **소셜 미디어 강점** | 동적 이벤트("지연", "취소") 자동 발견 | 개체 분류 정확도 |

---

## 📈 종합 비교 요약

| 관점 | NLTK | spaCy |
|------|------|-------|
| **트윗 Raw F1** | 0.1538 (Precision 1.0, Recall 0.08) | 0.3478 |
| **트윗 + V2 F1** | 0.4898 | **0.7391** ✅ |
| **위키 개체 수** | 25개 (라벨 오류 다수) | 20개 (정확) |
| **방식** | 규칙 기반 (대문자 의존) | 신경망 (문맥 기반) |
| **해시태그 처리** | ✅ ORG 인식 (의외의 강점) | ❌ PERSON 오분류 |
| **소문자 인물명** | ❌ 미인식 | ⚠️ 부분 인식 |
| **전처리 효과** | 가제터로 F1 3.2배 향상 | 가제터로 F1 2.1배 향상 |
| **트위터 추천** | V2 전처리 필수 | V2 전처리 시 best |

---

## 📝 사용한 문장 목록

### 합성 트윗 (TWEETS, 5개)

```
1. omg @elonmusk just announced starship launch tmrw 🚀🚀 #SpaceX cant believe it lol
2. flight delayed again at jfk airport... thx american airlines 🙄 #travelproblems
3. just saw taylor swift in nyc!!!! no way she was at central park 😱😱 #swifties
4. why is amazon prime shipping so slow now?? used to be 2 days smh #disappointed
5. congrats to korea on winning the world cup qualifier vs japan 🔥🔥 #KFA #football
```

### 합성 위키피디아 문장 (5개)

```
1. Elon Musk, the CEO of SpaceX and Tesla, announced a new rocket launch from Cape Canaveral.
2. American Airlines flight AA123 was delayed at John F. Kennedy International Airport in New York.
3. Taylor Swift performed a concert at Madison Square Garden in New York City last Friday.
4. Amazon's founder Jeff Bezos launched Blue Origin, a private aerospace company based in Kent, Washington.
5. South Korea defeated Japan 2-1 in the FIFA World Cup qualifier held in Seoul.
```

### Gold Truth 데이터셋 (10개)

```
1. omg @elonmusk just announced starship launch tmrw 🚀🚀 #SpaceX cant believe it lol
   → Elon Musk(PERSON), SpaceX(ORG)

2. flight delayed again at jfk airport... thx american airlines 🙄 #travelproblems
   → JFK Airport(LOC), American Airlines(ORG)

3. just saw taylor swift in nyc!!!! no way she was at central park 😱😱 #swifties
   → Taylor Swift(PERSON), New York City(LOC), Central Park(LOC)

4. why is amazon prime shipping so slow now?? used to be 2 days smh #disappointed
   → Amazon(ORG)

5. congrats to korea on winning the world cup qualifier vs japan 🔥🔥 #KFA #football
   → South Korea(LOC), Japan(LOC), KFA(ORG)

6. elon musk just tweeted about tesla stock again lol someone stop him
   → Elon Musk(PERSON), Tesla(ORG)

7. breaking google ceo sundar pichai confirms layoffs in silicon valley #tech
   → Google(ORG), Sundar Pichai(PERSON), Silicon Valley(LOC)

8. united airlines cancelled flight from chicago to london due to snowstorm smh
   → United Airlines(ORG), Chicago(LOC), London(LOC)

9. omg kim kardashian spotted in paris during fashion week 👀
   → Kim Kardashian(PERSON), Paris(LOC)

10. microsoft and openai announce partnership from san francisco #AI
    → Microsoft(ORG), OpenAI(ORG), San Francisco(LOC)
```

### Open IE 평가 문장 (6개)

```
1. flight was delayed in New York because of American Airlines issues
2. elon musk announced starship launch at cape canaveral
3. united airlines lost my luggage at chicago ohare airport
4. Elon Musk, the CEO of SpaceX, announced a new Starship launch from Cape Canaveral.
5. Jeff Bezos founded Amazon in Seattle before launching Blue Origin.
6. Microsoft acquired OpenAI partnership to integrate GPT-4 into Office products.
```

---

## 🚀 실행 방법

```bash
python twitter_ner_lab.py
```

실행 결과:
- 콘솔: 연습문제 1~7 전체 출력 (실제 NER 결과 + 이슈 분석 포함)
- `figures/`: 시각화 PNG 5종 자동 생성

---

*실습 날짜: 2026-05-27 | Python 3.9 | NLTK 3.9 | spaCy 3.8 (en_core_web_sm)*  
*데이터: 합성(Synthetic) — 실제 Twitter/Wikipedia 수집 데이터 아님*
