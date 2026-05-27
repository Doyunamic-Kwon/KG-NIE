# 🐦 Twitter NER Lab — NLTK vs spaCy

> **자연어처리(NLP) 실습 | 소셜 미디어 개체명 인식(Named Entity Recognition)**  
> 전통 NLP(NLTK)와 현대 NLP(spaCy)가 트위터 노이즈 앞에서 어떻게 동작하는지 비교·분석합니다.

---

## 📁 프로젝트 구조

```
KG-NIE/
├── twitter_ner_lab.py     # 전체 실습 코드 (연습문제 1~7)
├── Tweets.csv             # Kaggle Twitter US Airline Sentiment (14,640 실제 트윗)
├── figures/
│   ├── fig1_entity_count.png
│   ├── fig2_prec_rec_f1.png
│   ├── fig3_f1_trend.png
│   ├── fig4_type_heatmap.png
│   └── fig5_openie_vs_ner.png
└── README.md
```

---

## ⚙️ 환경 설정

```bash
pip install nltk spacy matplotlib numpy kaggle
python -m spacy download en_core_web_sm

python -c "
import nltk
for pkg in ['punkt_tab','averaged_perceptron_tagger_eng',
            'maxent_ne_chunker_tab','words']:
    nltk.download(pkg)
"

# Tweets.csv 다운로드 (Kaggle 계정 필요)
kaggle datasets download -d crowdflower/twitter-airline-sentiment --unzip
```

---

## 📦 데이터 출처

| 데이터 | 출처 | 설명 |
|--------|------|------|
| **트윗 1~3** (연습문제 1·2) | Twitter #CaliforniaFires (2019.11 스크린샷) | @SPN_Alameda, @CFACT, @UMengineering |
| **트윗 4~5** (연습문제 1·2) | Kaggle `crowdflower/twitter-airline-sentiment` | 실제 항공사 관련 트윗 |
| **Tweets.csv** (연습문제 4~6) | Kaggle `crowdflower/twitter-airline-sentiment`, CC BY 4.0 | 14,640개 실제 트윗 |

---

## 🔬 연습문제 1 & 2 — 실제 트윗 5개 vs 위키피디아 Raw NER

### 사용 데이터

**실제 트윗 5개** (주제 다양성 — CaliforniaFires + Airline):

| # | 출처 | 트윗 |
|---|------|------|
| 1 | @SPN_Alameda | `Bake sale to help #CaliforniaFires victims sponsored by the Stident Leadership Council at SPN school - come on down!!!` |
| 2 | @CFACT | `Hey Gov. Newsom, the wildfires are mainly due to prohibiting controlled burns and not clearing enough dead trees, not because of utilities! #CaliforniaFires #TuesdayThoughts` |
| 3 | @UMengineering | `In 2018, 1.8 million acres burned in #CaliforniaFires—a record high. Now it could be months before #Australia has more than *100 million acres* of bushfire under control.` |
| 4 | Tweets.csv | `@united 374 ORD to ROC. Fam came to see me at SNA. I'm a member, so is my dad. He used his miles for me` |
| 5 | Tweets.csv | `@AmericanAir booked EWR-FLL in first for two on 3-8. Let's see how you compare to the garbage @united` |

### 실제 NER 출력

```
트윗 1 (@SPN_Alameda):
  NLTK  → [('Bake', 'LOC'), ('CaliforniaFires', 'ORG'), ('Stident Leadership Council', 'ORG'), ('SPN', 'ORG')]
  spaCy → [('Bake', 'ORG'), ('CaliforniaFires', 'ORG'), ('the Stident Leadership Council', 'ORG')]

트윗 2 (@CFACT):
  NLTK  → [('Hey', 'PERSON'), ('Newsom', 'LOC'), ('CaliforniaFires', 'ORG')]
  spaCy → [('Newsom', 'PERSON'), ('#CaliforniaFires #', 'NUM'), ('TuesdayThoughts', 'PERSON')]

트윗 3 (@UMengineering):
  NLTK  → [('Australia', 'LOC')]
  spaCy → [('2018', 'DATE'), ('1.8 million acres', 'NUM'), ('CaliforniaFires', 'PERSON'),
            ('Australia', 'LOC'), ('100 million acres', 'NUM')]

트윗 4 (Airline):
  NLTK  → [('ROC', 'ORG'), ('Fam', 'PERSON'), ('SNA', 'ORG')]
  spaCy → [('374', 'NUM'), ('ORD', 'ORG'), ('ROC', 'LOC'), ('SNA', 'ORG')]

트윗 5 (Airline):
  NLTK  → [('AmericanAir', 'ORG')]
  spaCy → [('@AmericanAir', 'LOC'), ('EWR', 'ORG'), ('first', 'NUM'), ('two', 'NUM')]
```

### 개체 인식 수 비교

![Figure 1 - Entity Count](figures/fig1_entity_count.png)

| 데이터 | NLTK | spaCy |
|--------|:----:|:-----:|
| **트윗 (5개)** | 12개 | 20개 |
| **위키피디아 (5개)** | 24개 | 23개 |

### 관찰된 주요 이슈 — Gold 정답 포함

> **컬럼 설명**  
> `Gold` = 정답 레이블 / `NLTK Raw` = 전처리 없이 실행한 결과 / `spaCy Raw` = 동일  
> `V2 후` = URL·이모티콘 제거 + @·# 처리 + 공항 코드·항공사명 가제터 정규화 적용 결과

| 이슈 토큰 | **Gold 정답** | NLTK Raw | spaCy Raw | V2 후 변화 |
|-----------|:------------:|----------|----------|-----------|
| `#CaliforniaFires` | **EVENT** (산불 이벤트) | ⚠️ ORG 오분류 | ⚠️ ORG 오분류 (트윗3에서 PERSON) | # 제거 → `CaliforniaFires` 그대로, 여전히 오분류 |
| `#TuesdayThoughts` | **비개체** (인식 불필요) | ✅ 미인식 (정답) | ❌ PERSON 오분류 | # 제거 → `TuesdayThoughts`; spaCy 오분류 지속 |
| `Bake` (문장 첫 단어) | **비개체** ("Bake sale"은 활동) | ❌ LOC **False Positive** | ❌ ORG **False Positive** | V1에서 제거 불가 — 문맥 판단이어야 해결 |
| `Gov. Newsom` | **PERSON** (정치인) | ❌ `Hey`→PERSON, `Newsom`→LOC (경계 오류) | ✅ `Newsom`→PERSON | 직함 처리 없음, 동일 |
| `@AmericanAir` 멘션 | **ORG** (항공사) | ✅ ORG 인식 | ❌ LOC 오분류 | V2: `@AmericanAir→American Airlines` → spaCy ORG 정확 |
| 공항 코드 `ORD`, `EWR` | **LOC** (도시/공항) | ⚠️ ORG 오분류 | ⚠️ ORG·LOC 혼용 | V2: `ORD→Chicago O'Hare` → LOC 정확도 향상 |
| `Stident` (오타) | **ORG** ("Student Leadership Council") | ✅ `Stident Leadership Council`→ORG | ✅ `the Stident Leadership Council`→ORG | 두 모델 모두 오타에도 강건 |

**V1 vs V2 처리 내용 요약:**

| 전처리 단계 | 처리 내용 | 위 이슈에 대한 효과 |
|------------|----------|-------------------|
| **V1** | URL 제거, 이모티콘 제거, `!!!`·`???` → `.` | 노이즈 제거만, 개체 인식 정확도 개선 미미 |
| **V2** | V1 + `@(\w+)→단어` + `#(\w+)→단어` + 가제터(항공사명, IATA 코드 정규화) | `@AmericanAir→American Airlines`, `ORD→Chicago O'Hare` 등 복원 → spaCy F1 13배 향상 |

> **핵심 관찰**:
> - `#TuesdayThoughts`는 **개체가 아님** — NLTK의 미인식이 오히려 정답, spaCy가 PERSON 오분류
> - `#CaliforniaFires`는 ORG가 아닌 **EVENT**가 적절하나, 표준 NER 온톨로지(PERSON/ORG/LOC)에서는 EVENT 클래스가 없어 분류 자체가 한계
> - `Bake`는 "Bake sale"(자선 행사)의 일부 — 독립 개체가 아닌데 인식하는 **False Positive**이며, 전처리로 해결 불가 (문맥 이해 필요)
> - `Stident`(오타)도 ORG로 정확 인식 → 두 모델 모두 형태 기반 패턴에 강건함을 보여줌

---

## 💡 연습문제 3 — 트윗 특화 Feature 5가지

| # | Feature | 설명 | 트윗 예시 |
|---|---------|------|----------|
| 1 | **@멘션 처리** | `@` 제거 → 항공사 가제터 대조 → 정식명 복원 | `@united` → `United Airlines` |
| 2 | **#해시태그 분리** | `#` 제거, 단어만 유지 → NER 후보 처리 | `#CaliforniaFires` → `CaliforniaFires` → EVENT (산불); `#TuesdayThoughts`는 비개체이므로 인식 불필요 |
| 3 | **이모티콘·URL 제거** | `[^\x00-\x7F]` 패턴으로 비ASCII 제거 | `🔥`, `http://…` → 제거 |
| 4 | **공항 코드 정규화** | IATA 코드 → 공항/도시 전체명 치환 | `ORD` → `Chicago O'Hare`, `JFK` → `JFK Airport` |
| 5 | **중복 구두점 정리** | `!!!`, `???`, `...` → 단일 구두점 | `come on down!!!` → `come on down.` |

---

## 📊 연습문제 4 — Kaggle Airline Sentiment 데이터셋 준비

```
데이터셋: crowdflower/twitter-airline-sentiment (Kaggle)
총 트윗 수: 14,640개
항공사: United, American, Delta, Southwest, Virgin America, US Airways
```

**NLTK & spaCy NER 샘플 출력 (10개):**

```
[Virgin America / neutral]  @VirginAmerica What @dhepburn said.
  NLTK  → [('VirginAmerica', 'ORG')]        spaCy → (없음)

[Virgin America / negative] @VirginAmerica it's really aggressive...
  NLTK  → (없음)                             spaCy → (없음)

[Virgin America / neutral]  @VirginAmerica Really missed a prime opportunity for Men Without Hats
  NLTK  → [('VirginAmerica Really', 'ORG'), ('Men Without Hats', 'ORG')]
  spaCy → [('Men Without Hats', 'ORG')]
```

> **관찰**: 항공사 트윗은 개체명 밀도가 낮고, `@VirginAmerica Really`처럼 멘션과 단어가 합쳐서 ORG로 오인식되는 사례가 빈번합니다.

---

## 📊 연습문제 5 — 100개 Gold Truth → Precision / Recall / F1

### Gold Truth 구축 방법

```
샘플링: random.seed(42), n=100

· ORG   : @airline 멘션 → 항공사 정식명 (가제터 기반 자동)
· LOC   : IATA 공항 코드(정규식) + 주요 미국 도시명
· PERSON: 영숫자 혼합 @username (일반 사용자 계정으로 판단)
· 온톨로지: PERSON / ORG / LOC (과제 지침: '관대하게 정의')

Gold 클래스 분포: ORG 103건, LOC 11건, PERSON 1건
```

### Precision / Recall / F1 결과

![Figure 2 - Precision Recall F1](figures/fig2_prec_rec_f1.png)

| 시스템 | Precision | Recall | **F1** | 해석 |
|--------|:---------:|:------:|:------:|-----|
| NLTK Raw       | 0.0948 | 0.0957 | 0.0952 | @mention 처리 못해 낮음 |
| NLTK + V1      | 0.1026 | 0.1043 | 0.1034 | URL 제거로 소폭 향상 |
| NLTK + V2      | 0.0476 | 0.1043 | 0.0654 | 가제터 변환 후 NLTK가 재분리 → FP 증가 |
| spaCy Raw      | 0.0382 | 0.0435 | 0.0407 | 멘션·코드 오분류 많음 |
| spaCy + V1     | 0.0388 | 0.0435 | 0.0410 | 미미한 향상 |
| **spaCy + V2** | **0.4286** | **0.7304** | **0.5402** | 가제터 정규화로 대폭 향상 |

> **핵심 수치**: spaCy + V2 F1(0.54)이 spaCy Raw(0.04)의 **13배**입니다.  
> NLTK + V2는 오히려 F1이 하락 → 가제터로 "United Airlines"로 복원해도 NLTK가 "United(LOC) Airlines(ORG)"로 재분리하기 때문.

---

## 🔧 연습문제 6 — 전처리(Preprocessing)의 가치 증명

### F1 추이

![Figure 3 - F1 Trend](figures/fig3_f1_trend.png)

| 전처리 단계 | NLTK F1 | spaCy F1 | 해석 |
|------------|:-------:|:--------:|-----|
| Raw (없음)          | 0.0952 | 0.0407 | 멘션·코드 인식 실패 |
| V1: 기본 노이즈 제거 | 0.1034 | 0.0410 | URL 제거로 미미한 향상 |
| V2: V1 + 가제터      | 0.0654 | **0.5402** | spaCy에만 큰 효과 |

> **결론**:  
> - **NLTK + V2**: 가제터가 역효과 — 복원된 "United Airlines"를 NLTK가 다시 분리 인식  
> - **spaCy + V2**: 가제터 정규화가 핵심 — `@united→United Airlines`로 복원 후 정확히 ORG 인식  
> - **전처리는 만능이 아님**: 모델의 NER 방식(규칙 기반 vs 신경망)과 맞아야 효과

---

## 🗺️ 개체 유형 분포 히트맵

![Figure 4 - Entity Type Heatmap](figures/fig4_type_heatmap.png)

| 관찰 | 내용 |
|------|------|
| **NLTK (Tweet)** | ORG 위주, NUM·DATE 거의 없음 |
| **spaCy (Tweet)** | NUM(숫자·측정값) 많음, LOC 다양 |
| **NLTK (Wiki)** | LOC·ORG·PERSON 고루 인식, 일부 라벨 오류 |
| **spaCy (Wiki)** | NUM·DATE까지 인식, 전반적으로 더 다양한 타입 |

---

## 🔗 연습문제 7 — 간이 Open IE vs 전통 NER

> ⚠️ **주의**: 본 구현은 POS 태깅 기반 규칙 Open IE입니다.  
> Stanford CoreNLP Open IE (https://stanfordnlp.github.io/CoreNLP/openie.html)는 Java 설치 + 서버 실행이 필요하며 동일 개념을 더 정밀하게 구현합니다.

```
문장  : United Airlines cancelled my flight from Chicago to Boston due to weather.
NLTK  : [United(LOC), Airlines(ORG), Chicago(LOC), Boston(LOC)]   ← United/Airlines 분리 오류
spaCy : [United Airlines(ORG), Chicago(LOC), Boston(LOC)]          ← 정확
OpenIE: (United Airlines, cancelled, flight)                        ← 관계 추출 성공

문장  : PG&E caused the Camp Fire that destroyed Paradise in Butte County.
NLTK  : [E(PERSON), Camp Fire(ORG), Paradise(ORG), Butte County(LOC)]  ← PG&E 분리 오류
spaCy : [PG&E(ORG), Camp Fire(LOC), Paradise(LOC), Butte County(LOC)]
OpenIE: (E, caused, Camp Fire), (Camp Fire, destroyed, Paradise)        ← 관계 2개 추출

문장  : Southwest Airlines operates more than 4,000 flights daily across the United States.
NLTK  : [Southwest(LOC), Airlines(ORG), United States(LOC)]        ← Southwest/Airlines 분리
spaCy : [Southwest Airlines(ORG), 4,000(NUM), United States(LOC)]  ← 정확
OpenIE: (Southwest Airlines, operates, flights)                     ← 관계 추출 성공
```

### Open IE vs NER 비교

| 관점 | 규칙 Open IE | NER |
|------|-------------|-----|
| **출력 형식** | (주어, 동사, 목적어) 삼중쌍 | 개체명 + 타입 |
| **사전 정의** | 관계 유형 불필요 | 타입 목록 필요 |
| **발견한 새 정보** | "취소", "원인" 같은 이벤트 관계 | 없음 (개체 분류만) |
| **소셜 미디어 한계** | 동사 없는 짧은 트윗에서 튜플 추출 실패 | 개체는 추출 가능 |

---

## 📈 종합 비교 요약

| 관점 | NLTK | spaCy |
|------|------|-------|
| **트윗 Raw** | 12개 인식 (라벨 오류 다수) | 20개 인식 (NUM·노이즈 포함) |
| **F1 Raw** | 0.0952 | 0.0407 |
| **F1 + V2** | 0.0654 (악화) | **0.5402** (13배 향상) |
| **해시태그** | ✅ ORG 인식 경향 | ⚠️ PERSON·NUM 오분류 |
| **@멘션** | ✅ ORG 인식 (`VirginAmerica`) | ❌ LOC 오분류 |
| **"United Airlines"** | ❌ 항상 분리 (`United`+`Airlines`) | ✅ 하나로 인식 |
| **방식** | 규칙 기반 (대문자 의존) | 신경망 (문맥 의존) |
| **전처리 궁합** | 가제터 역효과 (분리 재발) | 가제터 효과 큼 |

---

## 📝 실행 방법

```bash
# Tweets.csv 다운로드 (최초 1회)
kaggle datasets download -d crowdflower/twitter-airline-sentiment \
    -p /path/to/KG-NIE/ --unzip

# 전체 실습 실행
python twitter_ner_lab.py
```

실행 결과:
- 콘솔: 연습문제 1~7 전체 출력
- `figures/`: 시각화 PNG 5종 자동 생성

---

*실습 날짜: 2026-05-27 | Python 3.9 | NLTK 3.9 | spaCy 3.8 (en_core_web_sm)*  
*트윗 데이터: 실제 Twitter 데이터 — CaliforniaFires(스크린샷) + Kaggle Airline Sentiment (CC BY 4.0)*
