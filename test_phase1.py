"""
Standalone Phase 1 test — mirrors solution.ipynb logic exactly.
Run: python test_phase1.py
Expected: PASS on all assertions + validator prints PASS
"""
import os, re, unicodedata, sys, subprocess, warnings, time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm
import lightgbm as lgb
import joblib

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
VAL_FRACTION        = 0.20

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

# ─── Preprocessing ────────────────────────────────────────────
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
    df['clean_name']    = df['business_name'].apply(preprocess_name)
    df['clean_address'] = df['business_address'].apply(preprocess_address)
    df['combined']      = df.apply(lambda r: (r['clean_name']+' '+r['clean_address']).strip(), axis=1)
    return df

# Assertion check
assert preprocess_name('Acme Corp. & Sons Ltd.') == 'acme corporation and sons limited'
assert 'street' in preprocess_address('123 Main St., Ste 4B')
print("[OK] Preprocessing assertions pass")

# ─── Dummy dataset ────────────────────────────────────────────
import random; random.seed(42)

TRUE_ENTITIES = [
    ('Acme Corporation',              '123 Main Street, Springfield, IL 62701', 'US'),
    ('Global Tech Solutions Inc',     '500 Silicon Ave, San Jose, CA 95101',    'US'),
    ('Springfield Bakery & Cafe',     '45 Baker Road, Springfield, IL 62702',   'US'),
    ('Sunrise Hotels Limited',        'Near Airport, Andheri East, Mumbai 400069', 'India'),
    ('Sharma Enterprises Pvt Ltd',    'Plot 12, Okhla Industrial Area, New Delhi 110020', 'India'),
    ('Café de Paris SARL',            '10 Rue de Rivoli, 75001 Paris, France',  'France'),
    ('TechVision International Corp', '200 Innovation Blvd, Austin, TX 78701',  'US'),
    ('Mumbai Spice Traders Private',  'Shop 5, Crawford Market, Mumbai 400001', 'India'),
    ('Lyon Textiles SAS',             '42 Avenue Jean Jaurès, 69007 Lyon, France', 'France'),
    ('DataStream Analytics LLC',      '300 Market Street Suite 1200, San Francisco, CA 94105', 'US'),
    ('Unique Solutions Corp',         '999 Nowhere Lane, Remote, WY 82001', 'US'),  # singleton
    ('Paris Boutique SNC',            '7 Rue du Faubourg, 75008 Paris, France', 'France'),  # singleton
]

SINGLETON_IDX = {10, 11}
TEST_ENTITIES = [
    ('Pinnacle Tech Group Inc',     '777 Innovation Dr, Boston, MA 02101', 'US'),
    ('Rajasthan Handicrafts Pvt',   'Johari Bazaar, Jaipur 302001',        'India'),
    ('Bordeaux Vignobles SAS',      '15 Route du Médoc, 33000 Bordeaux',   'France'),
    ('CloudNine Logistics LLC',     '88 Harbor View Blvd, Seattle WA 98101', 'US'),
    ('Standalone Business Co',      '1 Isolated Road, Nowhere, MT 59001',  'US'),  # singleton
]
SINGLETON_TEST = {4}

def perturb_name(name):
    ops = [
        lambda s: s.replace('Corporation','Corp.').replace('Limited','Ltd').replace('Private','Pvt'),
        lambda s: s.replace('International',"Int'l").replace('Solutions','Solns'),
        lambda s: s.replace(' and ',' & ').replace('Incorporated','Inc'),
        lambda s: s.upper(), lambda s: s.lower(),
        lambda s: s+' (Branch)',
        lambda s: s,
    ]
    return random.choice(ops)(name)

def perturb_address(addr):
    ops = [
        lambda s: re.sub(r'\bStreet\b','St.',s).replace('Avenue','Ave'),
        lambda s: re.sub(r',.*$','',s),
        lambda s: re.sub(r'\d{5,6}','',s).strip(),
        lambda s: s+'  (Near Main Market)',
        lambda s: s.upper(), lambda s: '',
        lambda s: s,
    ]
    return random.choice(ops)(addr)

def make_s23(entities, id_prefix, singleton_indices):
    rows, mapping = [], {}
    rid = 1
    for i,(name,addr,country) in enumerate(entities):
        if i in singleton_indices: continue
        n_matches = random.randint(1,3)
        mids = []
        for _ in range(n_matches):
            eid = f'{id_prefix}{rid:05d}'
            rows.append({'entity_id':eid,'business_name':perturb_name(name),
                         'business_address':perturb_address(addr),'country':country})
            mids.append(eid); rid+=1
        # hard negative
        hn = random.choice([e for j,e in enumerate(entities) if j!=i])
        rows.append({'entity_id':f'{id_prefix}{rid:05d}','business_name':perturb_name(hn[0]),
                     'business_address':perturb_address(hn[1]),'country':country})
        rid+=1
        mapping[f'S1-{i+1:05d}'] = mids
    return pd.DataFrame(rows), mapping

N = len(TRUE_ENTITIES)
s1_train = pd.DataFrame([{'entity_id':f'S1-{i+1:05d}','business_name':n,'business_address':a,'country':c}
                          for i,(n,a,c) in enumerate(TRUE_ENTITIES)])
s2_train, gt2 = make_s23(TRUE_ENTITIES,'S2-',SINGLETON_IDX)
s3_train, gt3 = make_s23(TRUE_ENTITIES,'S3-',SINGLETON_IDX)

# Ground truth
gt_rows = []
for s1_id in s1_train['entity_id']:
    m2 = gt2.get(s1_id,[]); m3 = gt3.get(s1_id,[])
    gt_rows.append({'source1_entity_id':s1_id,'matched_entity_ids':','.join(m2+m3)})
gt_train = pd.DataFrame(gt_rows)

s1_test = pd.DataFrame([{'entity_id':f'S1-{i+1:05d}','business_name':n,'business_address':a,'country':c}
                         for i,(n,a,c) in enumerate(TEST_ENTITIES)])
s2_test, _ = make_s23(TEST_ENTITIES,'S2-',SINGLETON_TEST)
s3_test, _ = make_s23(TEST_ENTITIES,'S3-',SINGLETON_TEST)

for df,path in [(s1_train,f'{TRAIN_DIR}/train_source1.tsv'),(s2_train,f'{TRAIN_DIR}/train_source2.tsv'),
                (s3_train,f'{TRAIN_DIR}/train_source3.tsv'),(gt_train,f'{TRAIN_DIR}/train_ground_truth.tsv'),
                (s1_test, f'{TEST_DIR}/test_source1.tsv'), (s2_test,f'{TEST_DIR}/test_source2.tsv'),
                (s3_test, f'{TEST_DIR}/test_source3.tsv')]:
    df.to_csv(path, sep='\t', index=False, encoding='utf-8')

print(f"[OK] Dummy dataset: S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}")

# ─── Preprocess ───────────────────────────────────────────────
tr_s1 = preprocess_df(s1_train); tr_s2 = preprocess_df(s2_train); tr_s3 = preprocess_df(s3_train)
te_s1 = preprocess_df(s1_test);  te_s2 = preprocess_df(s2_test);  te_s3 = preprocess_df(s3_test)

def build_index(df): return df.set_index('entity_id').to_dict('index')
tr_s1_idx = build_index(tr_s1); tr_s23_idx = {**build_index(tr_s2),**build_index(tr_s3)}
te_s23_idx = {**build_index(te_s2),**build_index(te_s3)}

def parse_gt(df):
    d={}
    for _,r in df.iterrows():
        s=str(r.get('matched_entity_ids','')).strip()
        d[r['source1_entity_id']] = set(s.split(','))-{''} if s else set()
    return d
gt_dict = parse_gt(gt_train)
print(f"[OK] GT: {len(gt_dict)} entities, {sum(len(v) for v in gt_dict.values())} matches")

# ─── Metric ───────────────────────────────────────────────────
def f_beta(p,r,b=0.5):
    if p+r==0: return 0.0
    return (1+b**2)*p*r/(b**2*p+r)

def score_predictions(pred,gold):
    scores=[]
    for s1_id,g in gold.items():
        p=pred.get(s1_id,set())
        if not g and not p: scores.append(1.0)
        elif not g: scores.append(0.0)
        elif not p: scores.append(0.0)
        else:
            tp=len(g&p); scores.append(f_beta(tp/len(p),tp/len(g)))
    return np.mean(scores)

# ─── Features ─────────────────────────────────────────────────
from rapidfuzz import fuzz
FEATURE_NAMES=['name_jaccard','name_token_sort','name_token_set','name_char3','name_len_ratio',
               'name_common_tok','addr_jaccard','addr_token_sort','addr_token_set','addr_char3',
               'addr_len_ratio','addr_num_overlap','comb_jaccard','comb_token_sort','country_match',
               'name_empty_a','name_empty_b','addr_empty_a','addr_empty_b']

def jaccard(a,b):
    sa,sb=set(a.split()),set(b.split())
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa&sb)/len(sa|sb)

def char3(a,b):
    if len(a)<3 and len(b)<3: return 1.0
    sa=set(a[i:i+3] for i in range(len(a)-2))
    sb=set(b[i:i+3] for i in range(len(b)-2))
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa&sb)/len(sa|sb)

def num_overlap(a,b):
    na=set(re.findall(r'\b\d+\b',a)); nb=set(re.findall(r'\b\d+\b',b))
    if not na and not nb: return 1.0
    if not na or not nb: return 0.0
    return len(na&nb)/len(na|nb)

def compute_features(ra,rb):
    na,nb=ra.get('clean_name',''),rb.get('clean_name','')
    aa,ab=ra.get('clean_address',''),rb.get('clean_address','')
    ca,cb=ra.get('country',''),rb.get('country','')
    ca_=na+' '+aa; cb_=nb+' '+ab
    lr=lambda a,b: min(len(a),len(b))/max(len(a),len(b)) if max(len(a),len(b))>0 else 1.0
    return [jaccard(na,nb),fuzz.token_sort_ratio(na,nb)/100,fuzz.token_set_ratio(na,nb)/100,
            char3(na,nb),lr(na,nb),float(len(set(na.split())&set(nb.split()))),
            jaccard(aa,ab),fuzz.token_sort_ratio(aa,ab)/100,fuzz.token_set_ratio(aa,ab)/100,
            char3(aa,ab),lr(aa,ab),num_overlap(aa,ab),
            jaccard(ca_,cb_),fuzz.token_sort_ratio(ca_,cb_)/100,
            float(ca==cb and ca!=''),
            float(na==''),float(nb==''),float(aa==''),float(ab=='')]

assert len(compute_features({'clean_name':'acme','clean_address':'123 main','country':'US'},
                             {'clean_name':'acme corp','clean_address':'123 main st','country':'US'}))==len(FEATURE_NAMES)
print("[OK] Feature shape correct")

# ─── TF-IDF Blocker ───────────────────────────────────────────
tr_s23_df = pd.concat([tr_s2,tr_s3],ignore_index=True)
vectorizer = TfidfVectorizer(max_features=50000,ngram_range=(1,2),sublinear_tf=True)
vectorizer.fit(list(tr_s23_df['combined'].fillna('')))
s23_matrix = vectorizer.transform(tr_s23_df['combined'].fillna(''))
s23_ids = list(tr_s23_df['entity_id'])

def get_candidates(s1_df, k=P1_TOP_K_CANDIDATES):
    results={}
    s1_ids=list(s1_df['entity_id']); corpus=list(s1_df['combined'].fillna(''))
    q_mat = normalize(vectorizer.transform(corpus))
    d_mat = normalize(s23_matrix)
    sims = (q_mat @ d_mat.T).toarray()
    kk = min(k,sims.shape[1])
    top_idx = np.argpartition(sims,-kk,axis=1)[:,-kk:]
    for i,s1_id in enumerate(s1_ids):
        row=[(s23_ids[j],float(sims[i,j])) for j in top_idx[i]]
        row.sort(key=lambda x:-x[1])
        results[s1_id]=row
    return results

tr_cands = get_candidates(tr_s1)
hit=total=0
for s1_id,gold in gt_dict.items():
    if not gold: continue
    cs=set(c[0] for c in tr_cands.get(s1_id,[]))
    hit+=len(gold&cs); total+=len(gold)
blk_recall = hit/total if total else 0
print(f"[OK] Blocking recall: {blk_recall:.3f}")

# ─── Build training pairs ─────────────────────────────────────
import random; random.seed(RANDOM_SEED)
rows_X,rows_y=[],[]
for _,s1_row in tr_s1.iterrows():
    s1_id=s1_row['entity_id']; gold=gt_dict.get(s1_id,set())
    cands=tr_cands.get(s1_id,[])
    for s23_id in gold:
        if s23_id in tr_s23_idx:
            rows_X.append(compute_features(dict(s1_row),tr_s23_idx[s23_id])); rows_y.append(1)
    neg=[c[0] for c in cands if c[0] not in gold]
    random.shuffle(neg)
    for s23_id in neg[:max(1,3*len(gold))]:
        if s23_id in tr_s23_idx:
            rows_X.append(compute_features(dict(s1_row),tr_s23_idx[s23_id])); rows_y.append(0)

X=np.array(rows_X); y=np.array(rows_y)
print(f"[OK] Pairs: {len(X)} total, {y.sum()} positives ({y.mean():.2f} rate)")

# ─── Train LightGBM ───────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_tr,X_va,y_tr,y_va = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
model = lgb.LGBMClassifier(objective='binary',num_leaves=63,learning_rate=0.05,
                            n_estimators=300,min_child_samples=3,verbose=-1,random_state=42)
model.fit(X_tr,y_tr,eval_set=[(X_va,y_va)],
          callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(100)])
print("[OK] LightGBM trained")

# ─── Test inference ───────────────────────────────────────────
te_s23_df_all = pd.concat([te_s2,te_s3],ignore_index=True)
te_s23_ids = list(te_s23_df_all['entity_id'])
te_vec = TfidfVectorizer(max_features=50000,ngram_range=(1,2),sublinear_tf=True)
te_vec.fit(list(te_s23_df_all['combined'].fillna('')))
te_s23_mat = te_vec.transform(te_s23_df_all['combined'].fillna(''))
te_s23_idx_local = build_index(te_s23_df_all)

def get_te_candidates(s1_df, k=P1_TOP_K_CANDIDATES):
    results={}
    s1_ids=list(s1_df['entity_id']); corpus=list(s1_df['combined'].fillna(''))
    q_mat = normalize(te_vec.transform(corpus))
    d_mat = normalize(te_s23_mat)
    sims = (q_mat @ d_mat.T).toarray()
    kk = min(k,sims.shape[1])
    top_idx = np.argpartition(sims,-kk,axis=1)[:,-kk:]
    for i,s1_id in enumerate(s1_ids):
        row=[(te_s23_ids[j],float(sims[i,j])) for j in top_idx[i]]
        row.sort(key=lambda x:-x[1]); results[s1_id]=row
    return results

te_cands = get_te_candidates(te_s1)
matched_out={}; cands_out={}

for _,s1_row in te_s1.iterrows():
    s1_id=s1_row['entity_id']; cands=te_cands.get(s1_id,[])
    cands_out[s1_id]=[c[0] for c in cands]
    if not cands: matched_out[s1_id]=set(); continue
    feats=[compute_features(dict(s1_row),te_s23_idx_local[s23_id])
           for s23_id,_ in cands if s23_id in te_s23_idx_local]
    valid=[s23_id for s23_id,_ in cands if s23_id in te_s23_idx_local]
    if not feats: matched_out[s1_id]=set(); continue
    probs=model.predict_proba(np.array(feats))[:,1]
    matched_out[s1_id]={s23_id for s23_id,p in zip(valid,probs) if p>=P1_MATCH_THRESHOLD}

# ─── Write output ─────────────────────────────────────────────
te_s1_ids=list(te_s1['entity_id'])
rows=[{'source1_entity_id':s1_id,'matched_entity_ids':','.join(sorted(matched_out.get(s1_id,set())))}
      for s1_id in te_s1_ids]
pd.DataFrame(rows).to_csv(f'{OUTPUT_DIR}/matching_results.tsv',sep='\t',index=False,encoding='utf-8')
rows2=[{'source1_entity_id':s1_id,'candidate_entity_ids':','.join(list(dict.fromkeys(cands_out.get(s1_id,[]))))}
       for s1_id in te_s1_ids]
pd.DataFrame(rows2).to_csv(f'{OUTPUT_DIR}/candidate_pairs.tsv',sep='\t',index=False,encoding='utf-8')
print("[OK] Output files written")

# ─── Validate ─────────────────────────────────────────────────
result = subprocess.run(
    [sys.executable, 'validate_submission.py',
     '--matching', f'{OUTPUT_DIR}/matching_results.tsv',
     '--candidate', f'{OUTPUT_DIR}/candidate_pairs.tsv',
     '--test-dir', TEST_DIR],
    capture_output=True, text=True
)
print("\n" + "="*55)
print("VALIDATOR OUTPUT:")
print(result.stdout)
if result.stderr: print("STDERR:", result.stderr[:200])
print("="*55)
if result.returncode == 0:
    print("[PASS] VALIDATION PASSED -- All Phase 1 tests pass!")
else:
    print("❌ VALIDATION FAILED")
    sys.exit(1)

# ─── Quick metric check on train (sanity) ────────────────────
tr_cands_val=get_candidates(tr_s1)
tr_matched={}
for _,s1_row in tr_s1.iterrows():
    s1_id=s1_row['entity_id']; cands=tr_cands_val.get(s1_id,[])
    if not cands: tr_matched[s1_id]=set(); continue
    feats=[compute_features(dict(s1_row),tr_s23_idx[s23_id]) for s23_id,_ in cands if s23_id in tr_s23_idx]
    valid=[s23_id for s23_id,_ in cands if s23_id in tr_s23_idx]
    if not feats: tr_matched[s1_id]=set(); continue
    probs=model.predict_proba(np.array(feats))[:,1]
    tr_matched[s1_id]={s23_id for s23_id,p in zip(valid,probs) if p>=P1_MATCH_THRESHOLD}
train_f05 = score_predictions(tr_matched, gt_dict)
print(f"\n[OK] Train F_0.5 (in-sample, sanity check): {train_f05:.4f}")
print("\n[DONE] All tests passed successfully!")
