"""
Test script for Phase 2 (and Phase 1 -> Phase 2 transition)
Validates:
1. Sentence Transformer embeddings + normalization
2. FAISS candidate retrieval
3. Ensemble candidate merging (TF-IDF + FAISS)
4. Cross-encoder pair scoring
5. Extended Phase 2 feature extraction and LightGBM model training
6. Test set inference and output file generation
7. Official validate_submission.py validation of Phase 2 outputs
"""
import os, sys, re, unicodedata, subprocess, warnings, random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import lightgbm as lgb
import joblib
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────
TRAIN_DIR  = "dataset/train"
TEST_DIR   = "dataset/test"
OUTPUT_DIR = "output"
MODEL_DIR  = "models"
for d in [TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_SEED         = 42
MAX_TEXT_LEN        = 512
P1_TOP_K_CANDIDATES = 50
P1_MATCH_THRESHOLD  = 0.45
P2_TOP_K_FAISS      = 50
P2_MATCH_THRESHOLD  = 0.40
EMBED_MODEL_NAME    = 'all-MiniLM-L6-v2'
CROSS_MODEL_NAME    = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# ─── Abbreviation dicts ───────────────────────────────────────
NAME_ABBREVS = {
    r'\bcorp\.?\b': 'corporation', r'\bco\.?\b': 'company',
    r'\binc\.?\b': 'incorporated', r'\bltd\.?\b': 'limited',
    r'\bllc\.?\b': 'limited liability company',
    r'\bpvt\.?\b': 'private', r'\bpriv\.?\b': 'private',
    r'\bint\'?l\.?\b': 'international', r'\bintl\.?\b': 'international',
    r'\bsvc\.?s?\b': 'services', r'\bmfg\.?\b': 'manufacturing',
    r'\bmgmt\.?\b': 'management', r'\bassoc\.?s?\b': 'associates',
    r'\bgrp\.?\b': 'group', r'\binds\.?\b': 'industries',
    r'&': 'and',
    r'\bsarl\.?\b': 'societe a responsabilite limitee',
    r'\bsa\.?\b': 'societe anonyme', r'\bsas\.?\b': 'societe par actions simplifiee',
}
ADDR_ABBREVS = {
    r'\bst\.?\b': 'street', r'\brd\.?\b': 'road', r'\bave\.?\b': 'avenue',
    r'\bblvd\.?\b': 'boulevard', r'\bdr\.?\b': 'drive', r'\bln\.?\b': 'lane',
    r'\bste\.?\b': 'suite', r'\bapt\.?\b': 'apartment',
    r'\bflr\.?\b': 'floor', r'\bsq\.?\b': 'square',
    r'\bopp\.?\b': 'opposite', r'\bdist\.?\b': 'district',
}
_NAME_PATTERNS = [(re.compile(k, re.IGNORECASE), v) for k, v in NAME_ABBREVS.items()]
_ADDR_PATTERNS = [(re.compile(k, re.IGNORECASE), v) for k, v in ADDR_ABBREVS.items()]
_URL_RE  = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_JUNK_RE = re.compile(r'[\uFFFD\uFFFE\uFFFF\uD800-\uDFFF]')

def preprocess_name(raw):
    if not isinstance(raw, str) or not raw.strip(): return ''
    text = unicodedata.normalize('NFC', raw)
    text = _JUNK_RE.sub(' ', _URL_RE.sub(' ', text))
    text = text.lower()
    for pat, repl in _NAME_PATTERNS:
        text = pat.sub(repl, text)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()[:MAX_TEXT_LEN]

def preprocess_address(raw):
    if not isinstance(raw, str) or not raw.strip(): return ''
    text = unicodedata.normalize('NFC', raw)
    text = _JUNK_RE.sub(' ', _URL_RE.sub(' ', text))
    text = text.lower()
    for pat, repl in _ADDR_PATTERNS:
        text = pat.sub(repl, text)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()[:MAX_TEXT_LEN]

def preprocess_df(df):
    df = df.copy()
    df['business_name']    = df['business_name'].fillna('').astype(str)
    df['business_address'] = df['business_address'].fillna('').astype(str)
    df['country']          = df['country'].fillna('').astype(str).str.upper().str.strip()
    df['clean_name']       = df['business_name'].apply(preprocess_name)
    df['clean_address']    = df['business_address'].apply(preprocess_address)
    df['combined']         = df.apply(lambda r: (r['clean_name']+' '+r['clean_address']).strip(), axis=1)
    return df

# ─── Load Dataset (dummy files already created by test_phase1) ─
print("[INFO] Loading datasets...")
tr_s1 = preprocess_df(pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t'))
tr_s2 = preprocess_df(pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t'))
tr_s3 = preprocess_df(pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t'))
gt_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t')

te_s1 = preprocess_df(pd.read_csv(os.path.join(TEST_DIR, 'test_source1.tsv'), sep='\t'))
te_s2 = preprocess_df(pd.read_csv(os.path.join(TEST_DIR, 'test_source2.tsv'), sep='\t'))
te_s3 = preprocess_df(pd.read_csv(os.path.join(TEST_DIR, 'test_source3.tsv'), sep='\t'))

tr_s23_idx = {r['entity_id']: dict(r) for _, r in pd.concat([tr_s2, tr_s3], ignore_index=True).iterrows()}
te_s23_idx = {r['entity_id']: dict(r) for _, r in pd.concat([te_s2, te_s3], ignore_index=True).iterrows()}

gt_dict = {}
for _, r in gt_df.iterrows():
    s1_id = r['source1_entity_id'] if 'source1_entity_id' in r else r['entity_id']
    m = r['matched_entity_ids']
    gt_dict[s1_id] = set(m.split(',')) if pd.notna(m) and str(m).strip() else set()

print(f"[OK] Loaded Train S1={len(tr_s1)}, S2={len(tr_s2)}, S3={len(tr_s3)} | Test S1={len(te_s1)}")

# ─── Feature Engineering ─────────────────────────────────────
try:
    from rapidfuzz import fuzz
    def compute_features(s1_r, s23_r):
        n1, n2 = s1_r.get('clean_name', ''), s23_r.get('clean_name', '')
        a1, a2 = s1_r.get('clean_address', ''), s23_r.get('clean_address', '')
        c1, c2 = s1_r.get('country', ''), s23_r.get('country', '')
        exact_n = 1.0 if n1 and n1 == n2 else 0.0
        exact_a = 1.0 if a1 and a1 == a2 else 0.0
        exact_c = 1.0 if c1 and c1 == c2 else 0.0
        return [
            exact_n,
            fuzz.ratio(n1, n2) / 100.0,
            fuzz.partial_ratio(n1, n2) / 100.0,
            fuzz.token_sort_ratio(n1, n2) / 100.0,
            fuzz.token_set_ratio(n1, n2) / 100.0,
            exact_a,
            fuzz.ratio(a1, a2) / 100.0,
            fuzz.token_sort_ratio(a1, a2) / 100.0,
            exact_c,
            float(abs(len(n1) - len(n2))),
            float(abs(len(a1) - len(a2))),
        ]
except ImportError:
    from difflib import SequenceMatcher
    def compute_features(s1_r, s23_r):
        n1, n2 = s1_r.get('clean_name', ''), s23_r.get('clean_name', '')
        a1, a2 = s1_r.get('clean_address', ''), s23_r.get('clean_address', '')
        c1, c2 = s1_r.get('country', ''), s23_r.get('country', '')
        return [
            1.0 if n1 == n2 else 0.0,
            SequenceMatcher(None, n1, n2).ratio(),
            SequenceMatcher(None, n1, n2).ratio(),
            SequenceMatcher(None, n1, n2).ratio(),
            SequenceMatcher(None, n1, n2).ratio(),
            1.0 if a1 == a2 else 0.0,
            SequenceMatcher(None, a1, a2).ratio(),
            SequenceMatcher(None, a1, a2).ratio(),
            1.0 if c1 == c2 else 0.0,
            float(abs(len(n1) - len(n2))),
            float(abs(len(a1) - len(a2))),
        ]

BASE_FEATURE_NAMES = [
    'name_exact', 'name_ratio', 'name_partial', 'name_token_sort', 'name_token_set',
    'addr_exact', 'addr_ratio', 'addr_token_sort',
    'country_exact', 'name_len_diff', 'addr_len_diff'
]

# ─── TF-IDF Blocker ───────────────────────────────────────────
class TFIDFBlocker:
    def __init__(self, max_features=50000, top_k=50):
        self.top_k = top_k
        self.vec = TfidfVectorizer(ngram_range=(1,2), max_features=max_features, sublinear_tf=True)
        self.matrix = None
        self.ids = None

    def fit_transform(self, df):
        self.ids = list(df['entity_id'])
        self.matrix = self.vec.fit_transform(df['combined'].fillna(''))
        return self

    def search(self, s1_df):
        q = self.vec.transform(s1_df['combined'].fillna(''))
        sim = q.dot(self.matrix.T).tocsr()
        results = {}
        for i, s1_id in enumerate(s1_df['entity_id']):
            row = sim.getrow(i)
            cands = []
            if row.nnz > 0:
                top_idx = np.argsort(row.data)[::-1][:self.top_k]
                for col_idx, score in zip(row.indices[top_idx], row.data[top_idx]):
                    cands.append((self.ids[col_idx], float(score)))
            results[s1_id] = cands
        return results

# Train TF-IDF Blocker
tr_s23_df = pd.concat([tr_s2, tr_s3], ignore_index=True)
tfidf_blocker = TFIDFBlocker(top_k=P1_TOP_K_CANDIDATES).fit_transform(tr_s23_df)
tr_tfidf_cands = tfidf_blocker.search(tr_s1)

# Test TF-IDF Blocker
te_s23_df = pd.concat([te_s2, te_s3], ignore_index=True)
te_tfidf_blocker = TFIDFBlocker(top_k=P1_TOP_K_CANDIDATES).fit_transform(te_s23_df)
te_tfidf_cands = te_tfidf_blocker.search(te_s1)

print("[OK] TF-IDF Candidate generation complete")

# ─── Phase 2: Dense Retrieval with FAISS ──────────────────────
print(f"[INFO] Loading Sentence Transformer ({EMBED_MODEL_NAME})...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
embed_dim = embed_model.get_sentence_embedding_dimension()

def encode_df(df, model, batch_size=64):
    texts = list(df['combined'].fillna(''))
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return embs.astype(np.float32)

print("[INFO] Encoding Train text with Sentence Transformer...")
tr_s23_embs = encode_df(tr_s23_df, embed_model)
tr_s1_embs = encode_df(tr_s1, embed_model)

class FAISSBlocker:
    def __init__(self, dim, top_k=50):
        self.dim = dim
        self.top_k = top_k
        self.index = faiss.IndexFlatIP(self.dim)
        self.ids = []

    def build(self, df, embs):
        self.ids = list(df['entity_id'])
        self.index.reset()
        self.index.add(embs)

    def search(self, query_embs, s1_ids):
        k = min(self.top_k, self.index.ntotal)
        D, I = self.index.search(query_embs, k)
        res = {}
        for i, s1_id in enumerate(s1_ids):
            cands = []
            for j in range(k):
                idx = I[i, j]
                if idx >= 0:
                    cands.append((self.ids[idx], float(D[i, j])))
            res[s1_id] = cands
        return res

print("[INFO] Building FAISS index for train...")
faiss_blocker_tr = FAISSBlocker(dim=embed_dim, top_k=P2_TOP_K_FAISS)
faiss_blocker_tr.build(tr_s23_df, tr_s23_embs)
tr_faiss_cands = faiss_blocker_tr.search(tr_s1_embs, list(tr_s1['entity_id']))

def merge_candidates(cands_a, cands_b, top_k=100):
    all_s1 = set(cands_a) | set(cands_b)
    merged = {}
    for s1_id in all_s1:
        score_map = {}
        for s23_id, score in cands_a.get(s1_id, []):
            score_map[s23_id] = max(score_map.get(s23_id, -1), score)
        for s23_id, score in cands_b.get(s1_id, []):
            score_map[s23_id] = max(score_map.get(s23_id, -1), score)
        merged[s1_id] = sorted(score_map.items(), key=lambda x: -x[1])[:top_k]
    return merged

p2_tr_merged = merge_candidates(tr_tfidf_cands, tr_faiss_cands, top_k=100)
print("[OK] Train candidate merge complete")

# ─── Cross-Encoder Pair Scoring ──────────────────────────────
print(f"[INFO] Loading Cross-Encoder ({CROSS_MODEL_NAME})...")
cross_encoder = CrossEncoder(CROSS_MODEL_NAME, max_length=256)

def get_cross_encoder_scores(s1_df, s23_dict, candidates, model, batch_size=32):
    pairs_flat, pair_meta = [], []
    s1_lookup = {r['entity_id']: dict(r) for _, r in s1_df.iterrows()}

    for s1_id, cands in candidates.items():
        row_a = s1_lookup.get(s1_id)
        if not row_a: continue
        text_a = row_a.get('combined', '')
        for s23_id, _ in cands:
            row_b = s23_dict.get(s23_id)
            if not row_b: continue
            text_b = row_b.get('combined', '')
            pairs_flat.append((text_a, text_b))
            pair_meta.append((s1_id, s23_id))

    if not pairs_flat:
        return {}

    scores_flat = model.predict(pairs_flat, batch_size=batch_size, show_progress_bar=False)
    res = defaultdict(dict)
    for (s1_id, s23_id), sc in zip(pair_meta, scores_flat):
        res[s1_id][s23_id] = float(sc)
    return dict(res)

print("[INFO] Scoring train candidate pairs with cross-encoder...")
p2_ce_scores_tr = get_cross_encoder_scores(tr_s1, tr_s23_idx, p2_tr_merged, cross_encoder)
print(f"[OK] Scored pairs across {len(p2_ce_scores_tr)} S1 entities")

# ─── Build Phase 2 Training Pairs ────────────────────────────
P2_FEATURE_NAMES = BASE_FEATURE_NAMES + ['faiss_score', 'tfidf_score', 'cross_encoder_score']

def build_p2_training_pairs(s1_df, s23_dict, gt, merged_cands, tfidf_cands, faiss_cands, ce_scores, neg_ratio=3.0):
    random.seed(42)
    rows_X, rows_y = [], []
    tfidf_map = {k: dict(v) for k, v in tfidf_cands.items()}
    faiss_map = {k: dict(v) for k, v in faiss_cands.items()}

    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row['entity_id']
        gold = gt.get(s1_id, set())
        cands = [c[0] for c in merged_cands.get(s1_id, [])]
        positives = gold
        negatives = set(cands) - gold
        max_neg = int(len(positives) * neg_ratio) if positives else 3
        neg_sample = random.sample(list(negatives), min(max_neg, len(negatives)))

        for s23_id, label in [(i, 1) for i in positives] + [(i, 0) for i in neg_sample]:
            if s23_id not in s23_dict: continue
            base = compute_features(dict(s1_row), s23_dict[s23_id])
            f_sc = faiss_map.get(s1_id, {}).get(s23_id, 0.0)
            t_sc = tfidf_map.get(s1_id, {}).get(s23_id, 0.0)
            c_sc = ce_scores.get(s1_id, {}).get(s23_id, 0.0)
            rows_X.append(base + [f_sc, t_sc, c_sc])
            rows_y.append(label)

    return pd.DataFrame(rows_X, columns=P2_FEATURE_NAMES), np.array(rows_y)

print("[INFO] Building Phase 2 feature matrix...")
X2_df, y2 = build_p2_training_pairs(tr_s1, tr_s23_idx, gt_dict, p2_tr_merged, tr_tfidf_cands, tr_faiss_cands, p2_ce_scores_tr)
print(f"[OK] Training matrix shape: {X2_df.shape}, positives: {sum(y2)} / {len(y2)}")

# Train LightGBM Phase 2 model
p2_lgbm = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42,
    verbose=-1
)
p2_lgbm.fit(X2_df.values, y2)
joblib.dump(p2_lgbm, os.path.join(MODEL_DIR, 'p2_lgbm.pkl'))
print("[OK] Phase 2 LightGBM trained & saved to models/p2_lgbm.pkl")

# ─── Phase 2 Test Set Inference ──────────────────────────────
print("[INFO] Running Phase 2 test set inference...")
te_s23_embs = encode_df(te_s23_df, embed_model)
te_s1_embs = encode_df(te_s1, embed_model)

faiss_blocker_te = FAISSBlocker(dim=embed_dim, top_k=P2_TOP_K_FAISS)
faiss_blocker_te.build(te_s23_df, te_s23_embs)
te_faiss_cands = faiss_blocker_te.search(te_s1_embs, list(te_s1['entity_id']))

p2_te_merged = merge_candidates(te_tfidf_cands, te_faiss_cands, top_k=100)
p2_ce_scores_te = get_cross_encoder_scores(te_s1, te_s23_idx, p2_te_merged, cross_encoder)

tfidf_te_map = {k: dict(v) for k, v in te_tfidf_cands.items()}
faiss_te_map = {k: dict(v) for k, v in te_faiss_cands.items()}

te_matched_p2 = {}
te_all_cands_p2 = {}
TE_S1_IDS = list(te_s1['entity_id'])

for _, s1_row in te_s1.iterrows():
    s1_id = s1_row['entity_id']
    cands = p2_te_merged.get(s1_id, [])
    te_all_cands_p2[s1_id] = [c[0] for c in cands]

    if not cands:
        te_matched_p2[s1_id] = set()
        continue

    feats_list, valid_ids = [], []
    for s23_id, _ in cands:
        if s23_id not in te_s23_idx: continue
        base = compute_features(dict(s1_row), te_s23_idx[s23_id])
        f_sc = faiss_te_map.get(s1_id, {}).get(s23_id, 0.0)
        t_sc = tfidf_te_map.get(s1_id, {}).get(s23_id, 0.0)
        c_sc = p2_ce_scores_te.get(s1_id, {}).get(s23_id, 0.0)
        feats_list.append(base + [f_sc, t_sc, c_sc])
        valid_ids.append(s23_id)

    if not feats_list:
        te_matched_p2[s1_id] = set()
        continue

    X_test = np.array(feats_list)
    probs = p2_lgbm.predict_proba(X_test)[:, 1]
    te_matched_p2[s1_id] = {s23_id for s23_id, p in zip(valid_ids, probs) if p >= P2_MATCH_THRESHOLD}

print(f"[OK] Test inference completed for {len(TE_S1_IDS)} S1 entities")

# ─── Write Phase 2 Output TSVs ───────────────────────────────
p2_out_matching = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
p2_out_candidate = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

rows_match = []
for s1_id in TE_S1_IDS:
    m_ids = sorted(te_matched_p2.get(s1_id, set()))
    rows_match.append({
        'source1_entity_id': s1_id,
        'matched_entity_ids': ','.join(m_ids) if m_ids else ''
    })
pd.DataFrame(rows_match)[['source1_entity_id', 'matched_entity_ids']].to_csv(
    p2_out_matching, sep='\t', index=False
)

rows_cand = []
for s1_id in TE_S1_IDS:
    c_ids = sorted(set(te_all_cands_p2.get(s1_id, [])))
    rows_cand.append({
        'source1_entity_id': s1_id,
        'candidate_entity_ids': ','.join(c_ids) if c_ids else ''
    })
pd.DataFrame(rows_cand)[['source1_entity_id', 'candidate_entity_ids']].to_csv(
    p2_out_candidate, sep='\t', index=False
)

print(f"[OK] Wrote Phase 2 output to:\n  {p2_out_matching}\n  {p2_out_candidate}")

# ─── Official validate_submission.py Execution ───────────────
print("\n" + "=" * 60)
print("RUNNING OFFICIAL VALIDATOR ON PHASE 2 OUTPUT:")
print("=" * 60)
cmd = [
    sys.executable,
    'validate_submission.py',
    '--matching', p2_out_matching,
    '--candidate', p2_out_candidate,
    '--test-dir', TEST_DIR,
]
ret = subprocess.run(cmd, capture_output=True, text=True)
print(ret.stdout)
if ret.stderr:
    print("STDERR:\n", ret.stderr)

assert ret.returncode == 0, f"Validator failed with exit code {ret.returncode}"
print("[PASS] PHASE 2 VALIDATION PASSED -- Ready for leaderboard submission!")
