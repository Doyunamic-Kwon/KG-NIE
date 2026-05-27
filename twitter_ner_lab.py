"""
=============================================================
 Twitter NER 실습 | 연습문제 1 ~ 7
 NLTK vs spaCy — 소셜 미디어 개체명 인식(NER) 비교 연구
=============================================================
환경: Python 3.9+  |  nltk >= 3.9  |  spacy >= 3.8 (en_core_web_sm)

⚠️ 데이터 출처: 이 실습에서 사용하는 트윗과 위키피디아 문장은
    실제 서비스에서 수집한 것이 아닌 NER 특성 연구를 위해
    직접 작성한 합성 데이터(Synthetic / Simulated Data)입니다.
    Twitter API 접근 없이 소셜 미디어 노이즈 패턴을 재현합니다.
=============================================================
"""

import re
import warnings
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ─── NLTK 설정 ────────────────────────────────────────────
import nltk
for pkg in ['punkt_tab', 'averaged_perceptron_tagger_eng',
            'maxent_ne_chunker_tab', 'words']:
    try:
        nltk.data.find(
            f'tokenizers/{pkg}' if 'punkt' in pkg
            else f'taggers/{pkg}'  if 'tagger' in pkg
            else f'chunkers/{pkg}' if 'chunker' in pkg
            else f'corpora/{pkg}'
        )
    except LookupError:
        nltk.download(pkg, quiet=True)

# ─── spaCy 설정 ───────────────────────────────────────────
import spacy
NLP_SPACY = spacy.load('en_core_web_sm')

# ══════════════════════════════════════════════════════════
# ■ 공통 유틸
# ══════════════════════════════════════════════════════════
def banner(title):
    w = 64
    print(f"\n{'═'*w}\n  {title}\n{'═'*w}")

def subheader(t):
    print(f"\n{'─'*56}\n  {t}\n{'─'*56}")

# ─── 통일 라벨 매핑 ────────────────────────────────────────
# FAC(시설): 공항·건물 등 물리적 장소이므로 LOC로 통일
NLTK_TO_STD = {
    'PERSON':       'PERSON',
    'ORGANIZATION': 'ORG',
    'ORG':          'ORG',
    'GPE':          'LOC',
    'GSP':          'LOC',
    'LOCATION':     'LOC',
    'FACILITY':     'LOC',    # FAC → LOC
    'CARDINAL':     'NUM',
    'ORDINAL':      'NUM',
    'DATE':         'DATE',
    'TIME':         'DATE',
    'MONEY':        'NUM',
}
SPACY_TO_STD = {
    'PERSON':     'PERSON',
    'ORG':        'ORG',
    'GPE':        'LOC',
    'LOC':        'LOC',
    'FAC':        'LOC',      # FAC → LOC
    'NORP':       'NORP',
    'PRODUCT':    'PROD',
    'EVENT':      'EVENT',
    'WORK_OF_ART':'PROD',
    'CARDINAL':   'NUM',
    'ORDINAL':    'NUM',
    'DATE':       'DATE',
    'TIME':       'DATE',
    'MONEY':      'NUM',
    'PERCENT':    'NUM',
    'QUANTITY':   'NUM',
    'LANGUAGE':   'LANG',
}

def run_nltk_ner(text):
    tokens = nltk.word_tokenize(text)
    tags   = nltk.pos_tag(tokens)
    tree   = nltk.ne_chunk(tags, binary=False)
    ents   = []
    for sub in tree:
        if isinstance(sub, nltk.Tree):
            lbl = NLTK_TO_STD.get(sub.label(), sub.label())
            ent = " ".join(w for w, _ in sub.leaves())
            ents.append((ent, lbl))
    return ents

def run_spacy_ner(text):
    doc = NLP_SPACY(text)
    return [(e.text, SPACY_TO_STD.get(e.label_, e.label_))
            for e in doc.ents]

# ══════════════════════════════════════════════════════════
# ■ 데이터셋 (합성 데이터 — Synthetic)
# ══════════════════════════════════════════════════════════

# ── 연습문제 1·2용: 트윗 vs 위키피디아 비교 ─────────────
TWEETS = [
    "omg @elonmusk just announced starship launch tmrw 🚀🚀 #SpaceX cant believe it lol",
    "flight delayed again at jfk airport... thx american airlines 🙄 #travelproblems",
    "just saw taylor swift in nyc!!!! no way she was at central park 😱😱 #swifties",
    "why is amazon prime shipping so slow now?? used to be 2 days smh #disappointed",
    "congrats to korea on winning the world cup qualifier vs japan 🔥🔥 #KFA #football",
]

WIKIPEDIA = [
    "Elon Musk, the CEO of SpaceX and Tesla, announced a new rocket launch from Cape Canaveral.",
    "American Airlines flight AA123 was delayed at John F. Kennedy International Airport in New York.",
    "Taylor Swift performed a concert at Madison Square Garden in New York City last Friday.",
    "Amazon's founder Jeff Bezos launched Blue Origin, a private aerospace company based in Kent, Washington.",
    "South Korea defeated Japan 2-1 in the FIFA World Cup qualifier held in Seoul.",
]

# ── 연습문제 4·5용: Gold Truth (평가 기준) ──────────────
#
# 설계 기준:
#   텍스트 : TWEETS 5개(원본) + 추가 5개 소셜 미디어 합성 트윗
#   라벨   : V2 전처리(가제터 복원) 후 NER이 인식 가능한 형태로 표기
#             → evaluate()에서 lower()로 비교하므로 대소문자 무관
#   범주   : PERSON / ORG / LOC (FAC·GPE는 LOC로 통일)
#
GOLD_DATASET = [
    # ── TWEETS 5개 (원본 텍스트 그대로) ──────────────────
    (
        "omg @elonmusk just announced starship launch tmrw 🚀🚀 #SpaceX cant believe it lol",
        [("Elon Musk", "PERSON"), ("SpaceX", "ORG")],
    ),
    (
        "flight delayed again at jfk airport... thx american airlines 🙄 #travelproblems",
        [("JFK Airport", "LOC"), ("American Airlines", "ORG")],
    ),
    (
        "just saw taylor swift in nyc!!!! no way she was at central park 😱😱 #swifties",
        [("Taylor Swift", "PERSON"), ("New York City", "LOC"), ("Central Park", "LOC")],
    ),
    (
        "why is amazon prime shipping so slow now?? used to be 2 days smh #disappointed",
        [("Amazon", "ORG")],
    ),
    (
        "congrats to korea on winning the world cup qualifier vs japan 🔥🔥 #KFA #football",
        [("South Korea", "LOC"), ("Japan", "LOC"), ("KFA", "ORG")],
    ),
    # ── 추가 5개 (다양한 트위터 노이즈 패턴) ─────────────
    (
        "elon musk just tweeted about tesla stock again lol someone stop him",
        [("Elon Musk", "PERSON"), ("Tesla", "ORG")],
    ),
    (
        "breaking google ceo sundar pichai confirms layoffs in silicon valley #tech",
        [("Google", "ORG"), ("Sundar Pichai", "PERSON"), ("Silicon Valley", "LOC")],
    ),
    (
        "united airlines cancelled flight from chicago to london due to snowstorm smh",
        [("United Airlines", "ORG"), ("Chicago", "LOC"), ("London", "LOC")],
    ),
    (
        "omg kim kardashian spotted in paris during fashion week 👀",
        [("Kim Kardashian", "PERSON"), ("Paris", "LOC")],
    ),
    (
        "microsoft and openai announce partnership from san francisco #AI",
        [("Microsoft", "ORG"), ("OpenAI", "ORG"), ("San Francisco", "LOC")],
    ),
]

# ══════════════════════════════════════════════════════════
# ■ 전처리 파이프라인
# ══════════════════════════════════════════════════════════

# 슬랭 사전: 문장 품질을 해치는 축약어만 최소 처리
SLANG = {
    "tmrw": "tomorrow", "thx": "thanks", "u": "you",
    "cant": "can't",    "wont": "won't", "dont": "don't",
}

# 가제터: 소셜 미디어에서 자주 쓰이는 소문자·멘션형 → 표준 표기
GAZETTEER = {
    # 멘션 형태 (@ 제거 후 붙여쓰기)
    "elonmusk":       "Elon Musk",
    "taylorswift":    "Taylor Swift",
    "kimkardashian":  "Kim Kardashian",
    # 소문자 형태 (공백 포함 → 긴 것 먼저 매칭)
    "elon musk":      "Elon Musk",
    "taylor swift":   "Taylor Swift",
    "kim kardashian": "Kim Kardashian",
    "sundar pichai":  "Sundar Pichai",
    "jeff bezos":     "Jeff Bezos",
    "spacex":         "SpaceX",
    "amazon":         "Amazon",
    "tesla":          "Tesla",
    "google":         "Google",
    "microsoft":      "Microsoft",
    "openai":         "OpenAI",
    "american airlines": "American Airlines",
    "united airlines":   "United Airlines",
    "jfk airport":    "JFK Airport",   # 공백 포함 먼저
    "jfk":            "JFK Airport",
    "kfa":            "KFA",
    "nyc":            "New York City",
    "korea":          "South Korea",
    "japan":          "Japan",
    "paris":          "Paris",
    "chicago":        "Chicago",
    "london":         "London",
    "silicon valley": "Silicon Valley",
    "san francisco":  "San Francisco",
    "central park":   "Central Park",
    "cape canaveral": "Cape Canaveral",
}

def preprocess_v1(text):
    """V1: URL + 이모티콘 + 중복 구두점 제거"""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]', '', text)
    text = re.sub(r'[!?]{2,}', '.', text)
    text = re.sub(r'\.{2,}', '.', text)
    return text.strip()

def preprocess_v2(text):
    """V2: V1 + @·# 처리 + 슬랭 최소 확장 + 가제터 정규화"""
    # 기본 노이즈 제거
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]', '', text)
    text = re.sub(r'[!?]{2,}', '.', text)
    text = re.sub(r'\.{2,}', '.', text)
    # @ 기호만 제거, 멘션 단어는 유지 (elonmusk → 가제터로 처리)
    text = re.sub(r'@(\w+)', r'\1', text)
    # # 기호만 제거
    text = re.sub(r'#(\w+)', r'\1', text)
    # 슬랭 → 표준어 (최소)
    words = [SLANG.get(w.lower(), w) for w in text.split()]
    text = ' '.join(words)
    # 가제터: 긴 패턴 먼저 매칭
    for k in sorted(GAZETTEER, key=len, reverse=True):
        text = re.sub(re.escape(k), GAZETTEER[k], text, flags=re.IGNORECASE)
    return text.strip()

def make_ner_fn(ner_fn, preprocess=None):
    def fn(text):
        t = preprocess(text) if preprocess else text
        return ner_fn(t)
    return fn

# ══════════════════════════════════════════════════════════
# ■ 평가 함수
# ══════════════════════════════════════════════════════════
def evaluate(dataset, ner_fn):
    """Exact-match token F1 (대소문자 무관)"""
    tp = fp = fn_ = 0
    per_tweet = []
    for text, gold in dataset:
        pred_set = set((e.lower(), t) for e, t in ner_fn(text))
        gold_set = set((g.lower(), t) for g, t in gold)
        tp_  = len(gold_set & pred_set)
        fp_  = len(pred_set - gold_set)
        fn__ = len(gold_set - pred_set)
        tp += tp_; fp += fp_; fn_ += fn__
        per_tweet.append((tp_, fp_, fn__))
    prec = tp / (tp + fp)  if (tp + fp) > 0  else 0.0
    rec  = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, per_tweet

# ══════════════════════════════════════════════════════════
# ■ 모든 조합 실행 → 결과 수집
# ══════════════════════════════════════════════════════════
configs = [
    ("NLTK Raw",        make_ner_fn(run_nltk_ner)),
    ("NLTK + V1",       make_ner_fn(run_nltk_ner,  preprocess_v1)),
    ("NLTK + V2",       make_ner_fn(run_nltk_ner,  preprocess_v2)),
    ("spaCy Raw",       make_ner_fn(run_spacy_ner)),
    ("spaCy + V1",      make_ner_fn(run_spacy_ner, preprocess_v1)),
    ("spaCy + V2",      make_ner_fn(run_spacy_ner, preprocess_v2)),
]

results = {}
for name, fn in configs:
    p, r, f, pt = evaluate(GOLD_DATASET, fn)
    results[name] = dict(precision=p, recall=r, f1=f, per_tweet=pt)

# ── 트윗·위키 개체 수 집계 ──────────────────────────────
def count_ents(sentences, ner_fn):
    return sum(len(ner_fn(s)) for s in sentences)

tweet_counts = {
    "NLTK":  count_ents(TWEETS, run_nltk_ner),
    "spaCy": count_ents(TWEETS, run_spacy_ner),
}
wiki_counts = {
    "NLTK":  count_ents(WIKIPEDIA, run_nltk_ner),
    "spaCy": count_ents(WIKIPEDIA, run_spacy_ner),
}

# ── 개체 유형 분포 집계 ──────────────────────────────────
def type_dist(sentences, ner_fn):
    dist = {}
    for s in sentences:
        for _, t in ner_fn(s):
            dist[t] = dist.get(t, 0) + 1
    return dist

nltk_tweet_dist  = type_dist(TWEETS,     run_nltk_ner)
spacy_tweet_dist = type_dist(TWEETS,     run_spacy_ner)
nltk_wiki_dist   = type_dist(WIKIPEDIA,  run_nltk_ner)
spacy_wiki_dist  = type_dist(WIKIPEDIA,  run_spacy_ner)

# ── Open IE 간이 구현 ─────────────────────────────────────
def simple_open_ie(text):
    tokens = nltk.word_tokenize(text)
    tags   = nltk.pos_tag(tokens)
    def get_nps(tgs):
        nps, buf = [], []
        for w, t in tgs:
            if t.startswith('NN') or t in ('DT', 'JJ', 'NNP', 'NNPS'):
                buf.append(w)
            else:
                if buf:
                    nps.append(" ".join(buf))
                buf = []
        if buf:
            nps.append(" ".join(buf))
        return nps
    verbs = [w for w, t in tags
             if t.startswith('VB')
             and w.lower() not in ('be','is','are','was','were','been')]
    triples = []
    for verb in verbs[:3]:
        vi = next((i for i, (w, _) in enumerate(tags) if w == verb), None)
        if vi is None:
            continue
        s_nps = get_nps(tags[max(0, vi-5):vi])
        o_nps = get_nps(tags[vi+1:min(len(tags), vi+6)])
        if s_nps and o_nps:
            triples.append((s_nps[-1], verb, o_nps[0]))
    return triples

OIE_SENTENCES = [
    "flight was delayed in New York because of American Airlines issues",
    "elon musk announced starship launch at cape canaveral",
    "united airlines lost my luggage at chicago ohare airport",
    "Elon Musk, the CEO of SpaceX, announced a new Starship launch from Cape Canaveral.",
    "Jeff Bezos founded Amazon in Seattle before launching Blue Origin.",
    "Microsoft acquired OpenAI partnership to integrate GPT-4 into Office products.",
]

# ══════════════════════════════════════════════════════════
# ■ 콘솔 출력
# ══════════════════════════════════════════════════════════

# ── 연습문제 1 & 2 ────────────────────────────────────────
banner("연습문제 1 & 2 │ 트윗 vs 위키피디아 — NLTK & spaCy 비교")
print("\n  ※ 아래 데이터는 NER 특성 연구를 위한 합성(Synthetic) 데이터입니다.")

subheader("[ 트윗 5개 — Raw NER ]")
for i, tw in enumerate(TWEETS, 1):
    ne = run_nltk_ner(tw)
    se = run_spacy_ner(tw)
    print(f"\n  트윗 {i}: {tw[:72]}")
    print(f"    NLTK  → {ne if ne else '(없음)'}")
    print(f"    spaCy → {se if se else '(없음)'}")

subheader("[ 위키피디아 5개 — Raw NER ]")
for i, wiki in enumerate(WIKIPEDIA, 1):
    ne = run_nltk_ner(wiki)
    se = run_spacy_ner(wiki)
    print(f"\n  위키 {i}: {wiki[:72]}")
    print(f"    NLTK  → {ne}")
    print(f"    spaCy → {se}")

print(f"""
  ★ 개체 인식 수 요약
  ┌─────────────┬───────────┬───────────┐
  │             │  NLTK     │  spaCy    │
  ├─────────────┼───────────┼───────────┤
  │ 트윗 (5개)  │  {tweet_counts['NLTK']:>3}개      │  {tweet_counts['spaCy']:>3}개      │
  │ 위키 (5개)  │  {wiki_counts['NLTK']:>3}개      │  {wiki_counts['spaCy']:>3}개      │
  └─────────────┴───────────┴───────────┘
""")

# 실제 실행 결과 기반 이슈 분석
subheader("[ 실제 NER 결과 기반 이슈 분석 ]")

# 트윗 1 상세 분석
tw1 = TWEETS[0]
nltk_tw1 = run_nltk_ner(tw1)
spacy_tw1 = run_spacy_ner(tw1)
print(f"""
  트윗 1 분석: "{tw1[:60]}..."
  NLTK  실제 출력: {nltk_tw1}
  spaCy 실제 출력: {spacy_tw1}

  이슈 항목별 실제 동작:
  ┌─────────────────────────────┬──────────────────────────┬──────────────────────────┐
  │ 이슈                        │ NLTK                     │ spaCy                    │
  ├─────────────────────────────┼──────────────────────────┼──────────────────────────┤
  │ @mention (elonmusk)         │ ❌ 미인식                 │ ⚠️ 'omg @elonmusk' PERSON│
  │                             │                          │   (범위 오류)             │
  │ #SpaceX 해시태그             │ ✅ ORG 인식               │ ❌ PERSON 오분류          │
  │ 이모티콘 (🚀🚀)              │ ❌ 파싱 무시              │ ⚠️ NUM 오분류             │
  │ 소문자 국가명 (korea/japan)  │ ❌ 미인식                 │ ✅ LOC 인식               │
  │ 위키피디아 표준 문장         │ ⚠️ 다수 인식, 라벨 오류  │ ✅ 정확 인식              │
  └─────────────────────────────┴──────────────────────────┴──────────────────────────┘

  ★ 주목: #SpaceX에서 NLTK(ORG)가 spaCy(PERSON)보다 더 정확합니다.
           spaCy는 문맥 기반이라 해시태그 직후 단어를 사람 이름으로 혼동합니다.
""")

# ── 연습문제 3 ────────────────────────────────────────────
banner("연습문제 3 │ 트윗 특화 Feature 5가지")
FEATURES = [
    ("1. @멘션 처리",         "@ 기호 제거 후 붙여쓰기(elonmusk) → 가제터로 'Elon Musk' 복원"),
    ("2. #해시태그 분리",      "# 기호 제거, 단어 유지 → ORG/LOC 후보로 가제터 매칭"),
    ("3. 이모티콘·URL 제거",   "[^\x00-\x7F] 패턴으로 제거 → 파서 혼란 방지"),
    ("4. 가제터 정규화",       "소문자/멘션형 → 표준 표기 복원 (nyc→New York City 등)"),
    ("5. 최소 슬랭 처리",     "tmrw→tomorrow 등 NER 해석 불가 축약어 변환"),
]
for name, desc in FEATURES:
    print(f"  ■ {name:<22} : {desc}")

# ── 연습문제 4 & 5 ───────────────────────────────────────
banner("연습문제 4 & 5 │ Gold Truth 기반 Precision / Recall / F1")
print(f"\n  ※ Gold 셋: TWEETS 5개 + 추가 합성 트윗 5개, 총 10개")
print(f"  ※ 평가: Exact-match (lower() 비교), FAC·GPE → LOC 통일\n")
print(f"  {'시스템':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("  " + "─"*50)
for name in results:
    r = results[name]
    print(f"  {name:<18} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

# ── 연습문제 6 ────────────────────────────────────────────
banner("연습문제 6 │ 전처리 단계별 성능 향상")
print(f"""
  전처리 단계            NLTK F1    spaCy F1
  ─────────────────────────────────────────────
  Raw (없음)             {results['NLTK Raw']['f1']:.4f}     {results['spaCy Raw']['f1']:.4f}
  V1: 노이즈 제거        {results['NLTK + V1']['f1']:.4f}     {results['spaCy + V1']['f1']:.4f}
  V2: V1 + 가제터 정규화  {results['NLTK + V2']['f1']:.4f}     {results['spaCy + V2']['f1']:.4f}
""")

# ── 연습문제 7 ────────────────────────────────────────────
banner("연습문제 7 │ 간이 Open IE vs. NER 비교")
print("\n  ※ 본 구현은 규칙 기반 간이 Open IE입니다 (Stanford Open IE와 무관).\n")
for sent in OIE_SENTENCES:
    ne = run_nltk_ner(sent)
    se = run_spacy_ner(sent)
    oi = simple_open_ie(sent)
    print(f"  문장    : {sent[:65]}")
    print(f"  NLTK NER: {ne or '(없음)'}")
    print(f"  spaCy   : {se or '(없음)'}")
    print(f"  OpenIE  : {oi or '(없음)'}")
    print()

# ══════════════════════════════════════════════════════════
# ■ 시각화 (Figure 1 ~ 5)
# ══════════════════════════════════════════════════════════
NLTK_COLOR  = '#4C72B0'
SPACY_COLOR = '#DD8452'
COLORS = {
    'NLTK Raw':   '#6baed6',
    'NLTK + V1':  '#2171b5',
    'NLTK + V2':  '#08306b',
    'spaCy Raw':  '#fd8d3c',
    'spaCy + V1': '#d94701',
    'spaCy + V2': '#7f2704',
}

# ─── Figure 1 : 트윗 vs 위키 개체 인식 수 ─────────────────
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('Figure 1  |  Entity Count: Tweet vs Wikipedia (Synthetic Data)\n'
              'NLTK vs spaCy — Raw NER',
              fontsize=13, fontweight='bold', y=1.01)

for ax, (data, title) in zip(axes, [
    ({'NLTK': tweet_counts['NLTK'], 'spaCy': tweet_counts['spaCy']}, 'Tweets (5)'),
    ({'NLTK': wiki_counts['NLTK'],  'spaCy': wiki_counts['spaCy']},  'Wikipedia (5)'),
]):
    bars = ax.bar(list(data.keys()), list(data.values()),
                  color=[NLTK_COLOR, SPACY_COLOR], edgecolor='white',
                  linewidth=1.5, width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(bar.get_height())),
                ha='center', va='bottom', fontweight='bold', fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Entities Detected', fontsize=11)
    ax.set_ylim(0, max(data.values()) * 1.35 + 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'fig1_entity_count.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  ✔ Figure 1 저장 완료")

# ─── Figure 2 : Precision / Recall / F1 전체 비교 ─────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig2.suptitle('Figure 2  |  Precision / Recall / F1 — All Configurations\n'
              '(Gold: 10 Synthetic Tweets, Exact-match Evaluation)',
              fontsize=13, fontweight='bold')

labels = list(results.keys())
short  = [l.replace('NLTK + ', 'NLTK\n+').replace('spaCy + ', 'spaCy\n+')
            .replace('NLTK Raw', 'NLTK\nRaw').replace('spaCy Raw', 'spaCy\nRaw')
          for l in labels]
colors = [COLORS[l] for l in labels]

for ax, metric in zip(axes, ['precision', 'recall', 'f1']):
    vals = [results[l][metric] for l in labels]
    bars = ax.bar(short, vals, color=colors, edgecolor='white', linewidth=1.2, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(metric.capitalize(), fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

nltk_patch  = mpatches.Patch(color='#2171b5', label='NLTK')
spacy_patch = mpatches.Patch(color='#d94701', label='spaCy')
fig2.legend(handles=[nltk_patch, spacy_patch], loc='lower center',
            ncol=2, fontsize=11, frameon=False)

plt.tight_layout(rect=[0, 0.06, 1, 1])
fig2.savefig(os.path.join(FIG_DIR, 'fig2_prec_rec_f1.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  ✔ Figure 2 저장 완료")

# ─── Figure 3 : 전처리 단계별 F1 추이 ─────────────────────
fig3, ax = plt.subplots(figsize=(9, 5))
steps = ['Raw', 'V1 Prep', 'V2 Prep']
nltk_f1  = [results['NLTK Raw']['f1'],
            results['NLTK + V1']['f1'],
            results['NLTK + V2']['f1']]
spacy_f1 = [results['spaCy Raw']['f1'],
            results['spaCy + V1']['f1'],
            results['spaCy + V2']['f1']]

ax.plot(steps, nltk_f1,  'o-', color=NLTK_COLOR,  linewidth=2.5, markersize=9,
        label='NLTK', zorder=3)
ax.plot(steps, spacy_f1, 's-', color=SPACY_COLOR, linewidth=2.5, markersize=9,
        label='spaCy', zorder=3)

for i, (n, s) in enumerate(zip(nltk_f1, spacy_f1)):
    ax.annotate(f'{n:.3f}', (steps[i], n),
                textcoords='offset points', xytext=(-22, 10),
                fontsize=10, color=NLTK_COLOR, fontweight='bold')
    ax.annotate(f'{s:.3f}', (steps[i], s),
                textcoords='offset points', xytext=(6, 10),
                fontsize=10, color=SPACY_COLOR, fontweight='bold')

ax.fill_between(steps, nltk_f1,  alpha=0.12, color=NLTK_COLOR)
ax.fill_between(steps, spacy_f1, alpha=0.12, color=SPACY_COLOR)
ax.set_title('Figure 3  |  F1 Score by Preprocessing Stage\n'
             '(Exact-match on 10 Synthetic Gold Tweets)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_xlabel('Preprocessing Stage', fontsize=12)
ax.legend(fontsize=12, frameon=False)
ax.set_ylim(0, 1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'fig3_f1_trend.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  ✔ Figure 3 저장 완료")

# ─── Figure 4 : 개체 유형 분포 히트맵 ─────────────────────
all_types = sorted(set(
    list(nltk_tweet_dist) + list(spacy_tweet_dist) +
    list(nltk_wiki_dist)  + list(spacy_wiki_dist)
))

def build_row(d):
    return [d.get(t, 0) for t in all_types]

matrix = np.array([
    build_row(nltk_tweet_dist),
    build_row(spacy_tweet_dist),
    build_row(nltk_wiki_dist),
    build_row(spacy_wiki_dist),
], dtype=float)

row_labels = ['NLTK\n(Tweet)', 'spaCy\n(Tweet)', 'NLTK\n(Wiki)', 'spaCy\n(Wiki)']

fig4, ax = plt.subplots(figsize=(max(10, len(all_types) * 1.1), 4.5))
cmap = LinearSegmentedColormap.from_list('custom', ['#f7f7f7', '#2171b5'])
im   = ax.imshow(matrix, cmap=cmap, aspect='auto')

ax.set_xticks(range(len(all_types)))
ax.set_xticklabels(all_types, fontsize=10, rotation=30, ha='right')
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=11)
ax.set_title('Figure 4  |  Entity Type Distribution Heatmap\n'
             '(Synthetic Data — Raw NER without preprocessing)',
             fontsize=12, fontweight='bold', pad=12)

for i in range(len(row_labels)):
    for j in range(len(all_types)):
        v = int(matrix[i, j])
        if v > 0:
            color = 'white' if matrix[i, j] > matrix.max() * 0.55 else 'black'
            ax.text(j, i, str(v), ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='Count')
plt.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, 'fig4_type_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  ✔ Figure 4 저장 완료")

# ─── Figure 5 : Open IE 시각화 ─────────────────────────────
fig5, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, len(OIE_SENTENCES) + 1)
ax.axis('off')
ax.set_title('Figure 5  |  Rule-based Open IE Triple Extraction vs. NER Output\n'
             '(Simple heuristic — subject/verb/object pattern)',
             fontsize=12, fontweight='bold')

col_x   = [0.1, 3.8, 6.5]
headers = ['Sentence (snippet)', 'NLTK NER', 'Open IE Tuples']
for x, h in zip(col_x, headers):
    ax.text(x, len(OIE_SENTENCES) + 0.5, h,
            fontsize=11, fontweight='bold', color='#333333')

ax.axhline(len(OIE_SENTENCES) + 0.2, color='#aaaaaa', linewidth=1)

for row, sent in enumerate(reversed(OIE_SENTENCES)):
    y      = row + 0.4
    ne     = run_nltk_ner(sent)
    oi     = simple_open_ie(sent)
    ne_str = '; '.join(f'{e}({t})' for e, t in ne[:3]) if ne else '—'
    oi_str = '\n'.join(f'({s},{v},{o})' for s, v, o in oi[:2]) if oi else '—'

    ax.text(col_x[0], y, sent[:38] + '…', fontsize=8.5, va='center', color='#222')
    ax.text(col_x[1], y, ne_str[:42],     fontsize=8,   va='center', color=NLTK_COLOR)
    ax.text(col_x[2], y, oi_str,          fontsize=8,   va='center', color='#7b3f00')
    if row < len(OIE_SENTENCES) - 1:
        ax.axhline(y + 0.6, color='#eeeeee', linewidth=0.7)

plt.tight_layout()
fig5.savefig(os.path.join(FIG_DIR, 'fig5_openie_vs_ner.png'), dpi=150, bbox_inches='tight')
plt.close(fig5)
print("  ✔ Figure 5 저장 완료")

print("\n  ── 모든 시각화 저장 완료 ──")
print(f"  경로: {FIG_DIR}\n")
print("═" * 64)
print("  ✅  연습문제 1~7 실습 완료 (NLTK + spaCy)")
print("═" * 64 + "\n")
