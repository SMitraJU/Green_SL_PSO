
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from skrebate import ReliefF
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display
import os
from scipy.special import j0

# ---------------------------
# GLOBAL PARAMETERS
# ---------------------------
MAX_ITER = 200 # PSO iterations
SWARM_SIZE = 30    # EBPSO swarm size
CGPOP_SIZE = SWARM_SIZE   # CGPSO population
TEST_SIZE = 0.25   # train/test split for final evaluation
BASE_RANDOM_STATE =42
PATIENCE = 15
N_RUNS = 5

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def show_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1, cm

def safe_stratified_split(X, y, test_size=TEST_SIZE, random_state=BASE_RANDOM_STATE):
    try:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_solution(X_train, X_test, y_train, y_test, mask):
    mask = np.array(mask, dtype=int)
    if mask.sum() == 0:
        y_pred = np.zeros_like(y_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        return acc, f1, confusion_matrix(y_test, y_pred), mask
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train[:, mask==1], y_train)
    y_pred = clf.predict(X_test[:, mask==1])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return acc, f1, confusion_matrix(y_test, y_pred), mask

# ============================================================
# PART A — DATA PREPROCESSING
# ============================================================
datasets = {}


# ============================================================
# 1) BREAST CANCER
# ============================================================
bc_path = r'data.csv'
bc = pd.read_csv(bc_path, header=None).drop(index=0).reset_index(drop=True)
if 0 in bc.columns:
    bc.drop(columns=[0], inplace=True)
if bc.columns[-1] == 32 or (bc.columns[-1] == bc.shape[1]-1 and bc.iloc[:, -1].isnull().all()):
    try: bc.drop(columns=[32], inplace=True)
    except: pass
first_col = bc.iloc[:,0]
bc = bc.drop(bc.columns[0], axis=1)
bc['label'] = first_col.map({'M': 1, 'B':0}).astype(int)
datasets['BreastCancer'] = bc

# ============================================================
# 2) HEART
# ============================================================
hd_path = r'heart.csv'
hd = pd.read_csv(hd_path, header=None).drop(index=0).reset_index(drop=True)
hd.rename(columns={hd.columns[-1]: 'label'}, inplace=True)
hd['label'] = hd['label'].astype(int)
datasets['Heart'] = hd

# ============================================================
# 3) IONOSPHERE
# ============================================================
iono_path = r'ionosphere.csv'
iono = pd.read_csv(iono_path, header=None)
iono.columns = list(range(iono.shape[1]-1)) + ['label']
iono['label'] = iono['label'].map({'g':1,'b':0}).astype(int)
datasets['Ionosphere'] = iono

# ============================================================
# 4) CONGRESSIONAL VOTING
# ============================================================
cong_path = r'house-votes-84.csv'
cong = pd.read_csv(cong_path, header=None)
cong.replace('?', np.nan, inplace=True)
cong.dropna(inplace=True)
cols = cong.columns.tolist()
cong = cong[cols[1:] + [cols[0]]].iloc[1:].reset_index(drop=True)
cong.columns = [*cong.columns[:-1], 'label']
cong.replace({'democrat':1,'republican':0,'y':1,'n':0}, inplace=True)
cong['label'] = cong['label'].astype(int)
datasets['Congress'] = cong

# ============================================================
# 5) SPECT HEART
# ============================================================
spect_path = r'spect_train.csv'
spect = pd.read_csv(spect_path, header=None)
spect = spect.iloc[1:].reset_index(drop=True)
spect = spect[[*spect.columns[1:], spect.columns[0]]]
spect.columns = [*spect.columns[:-1], 'label']
spect['label'] = spect['label'].astype(int)
datasets['SPECT'] = spect

# ============================================================
# 6) TIC-TAC-TOE
# ============================================================
ttt_path = r'tic-tac-toe.data.csv'
ttt = pd.read_csv(ttt_path, header=None).iloc[1:].reset_index(drop=True)
ttt.columns = [*ttt.columns[:-1], 'label']
ttt = ttt[~(ttt == 'b').any(axis=1)].reset_index(drop=True)
ttt.replace({'x':1, 'o':0,'positive':1,'negative':0}, inplace=True)
ttt['label'] = ttt['label'].astype(int)
datasets['TicTacToe'] = ttt

# ============================================================
# 7) SONAR
# ============================================================
sonar_path = r'sonar data.csv'
sonar = pd.read_csv(sonar_path, header=None)
sonar.columns = [*sonar.columns[:-1], 'label']
sonar.replace({'R':1,'M':0}, inplace=True)
sonar['label'] = sonar['label'].astype(int)
datasets['Sonar'] = sonar

# ============================================================
# 8) WINE
# ============================================================
wine_path = r'WineQT.csv'
wine = pd.read_csv(wine_path, header=None).iloc[1:].reset_index(drop=True)
wine.columns = list(range(wine.shape[1]))
wine['label'] = pd.to_numeric(wine.iloc[:, -1], errors='coerce')
wine['label'] = (wine['label'] >= 6).astype(int)
if wine.shape[1] >= 2 and wine.columns[-2] != 'label':
    wine.drop(columns=[wine.shape[1]-2], inplace=True)
datasets['Wine'] = wine

# ============================================================
# 9) VOTE 114
# ============================================================
vote_path = r"114_congress.csv"
vote = pd.read_csv(vote_path, header=None).iloc[1:].reset_index(drop=True)
vote.drop(columns=[0,2], inplace=True)
vote = vote[[c for c in vote.columns if c != 1] + [1]]
vote.rename(columns={vote.columns[-1]: 'label'}, inplace=True)
vote.rename(columns=dict(zip(vote.columns[:-1], range(len(vote.columns[:-1])))), inplace=True)
vote['label'] = vote['label'].map({'D': 0, 'R':1}).fillna(2).astype(int)
datasets['Vote114'] = vote

# ============================================================
# 10) TITANIC
# ============================================================
titanic_path = r"Titanic-Dataset.csv"
titanic = pd.read_csv(titanic_path, header=None).iloc[1:].reset_index(drop=True)
titanic.rename(columns={1:'label'}, inplace=True)
titanic.drop(columns=[0,3,8,9,10], inplace=True)
le_sex = LabelEncoder()
titanic[4] = le_sex.fit_transform(titanic[4])
titanic[11] = titanic[11].fillna('Missing')
le_embarked = LabelEncoder()
titanic[11] = le_embarked.fit_transform(titanic[11])
titanic.dropna(inplace=True)
cols = [c for c in titanic.columns if c != 'label'] + ['label']
titanic = titanic[cols]
titanic.rename(columns=dict(zip(titanic.columns[:-1], range(titanic.shape[1]-1))), inplace=True)
datasets['Titanic'] = titanic

# FINAL CLEANUP for all loaded datasets
for name, df in datasets.items():
    df = df.copy()
    if df.columns[-1] != 'label':
        df.rename(columns={df.columns[-1]:'label'}, inplace=True)
    feat_cols = [c for c in df.columns if c != 'label']
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors='coerce')
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype(int)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    datasets[name] = df
    print(f"{name}: shape={df.shape}, labels={np.unique(df['label'])}")


# ============================================================
# PART B — EBPSO STANDARD
# ========================================================# ============================================================
# PART B & C — EBPSO FUNCTIONAL IMPLEMENTATIONS 
# (Replaces class BPSO, EBParticleSimple, and their runners)
# ============================================================

def ebpso_internal_fitness(X_train, y_train, mask, seed,alpha=0.9):
    """Internal fitness evaluation using a 20% validation split to prevent data leakage."""
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0 or len(np.unique(y_train)) < 2:
        return 0.0, 0.0
    num_sel = mask_bool.sum()
    if num_sel == 0 or len(np.unique(y_train)) < 2:
        return 0.0, 0.0
    
    X_masked = X_train[:, mask_bool]
    X_t, X_v, y_t, y_v = safe_stratified_split(X_masked, y_train, test_size=0.2, random_state=seed)
    
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_t, y_t)
    ypr = clf.predict(X_v)
    
    acc = accuracy_score(y_v, ypr)
    penalty = num_sel / X_train.shape[1]
    f1 = f1_score(y_v, ypr, average='macro', zero_division=0)
    score = alpha * acc + (1 - alpha) * (1 - penalty)
    return score, f1


def EBPSO_standard(X_train, y_train, X_test, y_test, seed, iterations=MAX_ITER, swarm_size=SWARM_SIZE, patience=PATIENCE, tol=1e-4):
    np.random.seed(seed)
    N, D = swarm_size, X_train.shape[1]
    w, c1, c2 = 0.6, 1.2, 1.2

    # Initialize Swarm (Vectorized)
    pos_act = np.random.uniform(0, 1, (N, D))
    position = (pos_act > 0.5).astype(int)
    velocity = np.random.uniform(-1, 1, (N, D))
    
    pbest = position.copy()
    pbest_scores = np.array([ebpso_internal_fitness(X_train, y_train, pbest[i], seed) for i in range(N)])
    
    best_idx = np.argmax(pbest_scores[:, 0])
    gbest = pbest[best_idx].copy()
    gbest_acc = pbest_scores[best_idx, 0]

    last_improve_iter = 0
    stagnation_iter = None

    for t in range(1, iterations + 1):
        # Update Velocity and Position (Vectorized)
        r1, r2 = np.random.rand(N, D), np.random.rand(N, D)
        velocity = w * velocity + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
        
        pos_act = np.clip(pos_act + velocity, 0.0, 0.9)
        position = (pos_act > 0.5).astype(int)

        # Evaluate and Update PBest
        for i in range(N):
            acc, f1 = ebpso_internal_fitness(X_train, y_train, position[i], seed)
            if (acc > pbest_scores[i, 0]) or (acc == pbest_scores[i, 0] and f1 > pbest_scores[i, 1]):
                pbest[i] = position[i].copy()
                pbest_scores[i] = (acc, f1)

        # Update GBest
        best_idx = np.argmax(pbest_scores[:, 0])
        cand_acc = pbest_scores[best_idx, 0]
        
        if cand_acc > gbest_acc + tol:
            gbest_acc = cand_acc
            gbest = pbest[best_idx].copy()
            last_improve_iter = t

        if t - last_improve_iter >= patience:
            stagnation_iter = last_improve_iter
            break

    # Final Fair Evaluation on UNSEEN test set
    acc, f1, _, mask = evaluate_solution(X_train, X_test, y_train, y_test, gbest)
    return int(mask.sum()), acc, f1, mask, stagnation_iter


def EBPSO_SL(X_train, y_train, X_test, y_test, seed, iterations=MAX_ITER, swarm_size=SWARM_SIZE, patience=PATIENCE, tol=1e-4):
    np.random.seed(seed)
    N, D = swarm_size, X_train.shape[1]
    c1, c2 = 1.2, 1.2
    low_b, up_b = 0.0, 0.9

    # Initialize Swarm
    pos_act = np.random.uniform(0, 1, (N, D))
    position = (pos_act > 0.5).astype(int)
    
    pbest = position.copy()
    pbest_scores = np.array([ebpso_internal_fitness(X_train, y_train, pbest[i], seed) for i in range(N)])
    
    best_idx = np.argmax(pbest_scores[:, 0])
    gbest = pbest[best_idx].copy()
    gbest_acc = pbest_scores[best_idx, 0]

    last_improve_iter = 0
    stagnation_iter = None

    for t in range(1, iterations + 1):
        for i in range(N):
            # --- Cosine Vector (Bessel J0) Update ---
            phi1, phi2 = c1 * np.random.rand(), c2 * np.random.rand()
            phi_t = max(phi1 + phi2, 1e-6)
            mu_t = (phi1 * pbest[i] + phi2 * gbest) / phi_t
            
            z = math.sqrt(phi_t) * float(t)
            u = np.random.randn(D)
            norm = np.linalg.norm(u)
            u = u / norm if norm > 1e-12 else np.ones(D) / math.sqrt(D)
            
            pos_act[i] = mu_t + j0(z) * u
            pos_act[i] = np.clip(pos_act[i], low_b, up_b)
            position[i] = (pos_act[i] > 0.5).astype(int)

            # Evaluate and Update PBest
            acc, f1 = ebpso_internal_fitness(X_train, y_train, position[i], seed)
            if (acc > pbest_scores[i, 0]) or (acc == pbest_scores[i, 0] and f1 > pbest_scores[i, 1]):
                pbest[i] = position[i].copy()
                pbest_scores[i] = (acc, f1)

        # Update GBest
        best_idx = np.argmax(pbest_scores[:, 0])
        cand_acc = pbest_scores[best_idx, 0]

        if cand_acc > gbest_acc + tol:
            gbest_acc = cand_acc
            gbest = pbest[best_idx].copy()
            last_improve_iter = t

        if t - last_improve_iter >= patience:
            stagnation_iter = last_improve_iter
            break

    acc, f1, _, mask = evaluate_solution(X_train, X_test, y_train, y_test, gbest)
    return int(mask.sum()), acc, f1, mask, stagnation_iter


def EBPSO_green(X_train, y_train, X_test, y_test, seed, iterations=MAX_ITER, swarm_size=SWARM_SIZE, patience=PATIENCE, tol=1e-4):
    np.random.seed(seed)
    N, D = swarm_size, X_train.shape[1]
    w_const, c1, c2 = 0.6, 1.2, 1.2
    low_b, up_b = 0.0, 0.9

    # Initialize Swarm
    pos_act = np.random.uniform(0, 1, (N, D))
    position = (pos_act > 0.5).astype(int)
    
    pbest = position.copy()
    pbest_scores = np.array([ebpso_internal_fitness(X_train, y_train, pbest[i], seed) for i in range(N)])
    
    best_idx = np.argmax(pbest_scores[:, 0])
    gbest = pbest[best_idx].copy()
    gbest_acc = pbest_scores[best_idx, 0]

    last_improve_iter = 0
    stagnation_iter = None

    # Green's function particle histories
    x0_list = pos_act.copy()
    mu_histories = [[] for _ in range(N)]
    history_F = {0: 1.0, 1: 1.0}

    for t in range(1, iterations + 1):
        for i in range(N):
            # --- Green Vector Update ---
            phi1, phi2 = c1 * np.random.uniform(0, 1), c2 * np.random.uniform(0, 1)
            phi_t = max(phi1 + phi2, 1e-6)
            
            a1 = float(phi_t - 1.0 - w_const)
            a2 = float(w_const)
            disc = (a1 ** 2) - (4.0 * a2)
            
            if disc < 0:
                real_part = -a1 / 2.0
                imag = math.sqrt(max(0.0, -disc)) / 2.0
                r_mag = math.hypot(real_part, imag)
                omega = math.atan2(imag, real_part) if r_mag > 1e-12 else 0.0
                root_type = "complex"
            else:
                lam1 = (-a1 + math.sqrt(disc)) / 2.0
                lam2 = (-a1 - math.sqrt(disc)) / 2.0
                root_type = "real"

            def f0(n):
                return (r_mag ** n) * math.sin(omega * n) if root_type == "complex" else float(lam1 ** n)
            def f1(n):
                return (r_mag ** n) * math.cos(omega * n) if root_type == "complex" else float(lam2 ** n)
            def Green_G(n, m):
                denom = float(f0(m-1) * f1(m) - f1(m-1) * f0(m))
                if abs(denom) < 1e-16: return 0.0
                return float((f0(n) * f1(m-1) - f1(n) * f0(m-1)) / denom)

            # Record mu trajectory
            mu_val = (phi1 * pbest[i] + phi2 * gbest) / phi_t
            mu_histories[i].append(mu_val)
            current_idx = len(mu_histories[i]) - 1

            if current_idx >= 2:
                Pn = np.zeros(D)
                for m in range(2, current_idx + 1):
                    rn = (mu_histories[i][m] - mu_histories[i][m-1]) - w_const * (mu_histories[i][m-1] - mu_histories[i][m-2])
                    Pn += Green_G(current_idx, m) * rn
                
                A = np.array([[f0(0), f1(0)], [f0(1), f1(1)]], dtype=float)
                b_arr = np.array([history_F[0], history_F[1]], dtype=float)
                try:
                    C = np.linalg.solve(A, b_arr)
                except np.linalg.LinAlgError:
                    C = np.linalg.lstsq(A, b_arr, rcond=None)[0]
                
                pos_act[i] = ((C[0] * f0(current_idx) + C[1] * f1(current_idx)) * x0_list[i]) + Pn

            pos_act[i] = np.clip(pos_act[i], low_b, up_b)
            position[i] = (pos_act[i] > 0.5).astype(int)

            # Evaluate and Update PBest
            acc, f1_score_val = ebpso_internal_fitness(X_train, y_train, position[i], seed)
            if (acc > pbest_scores[i, 0]) or (acc == pbest_scores[i, 0] and f1_score_val > pbest_scores[i, 1]):
                pbest[i] = position[i].copy()
                pbest_scores[i] = (acc, f1_score_val)

        # Update GBest
        best_idx = np.argmax(pbest_scores[:, 0])
        cand_acc = pbest_scores[best_idx, 0]

        if cand_acc > gbest_acc + tol:
            gbest_acc = cand_acc
            gbest = pbest[best_idx].copy()
            last_improve_iter = t

        if t - last_improve_iter >= patience:
            stagnation_iter = last_improve_iter
            break

    acc, f1_score_val, _, mask = evaluate_solution(X_train, X_test, y_train, y_test, gbest)
    return int(mask.sum()), acc, f1_score_val, mask, stagnation_iter


# ============================================================
# PART D — CGPSO IMPLEMENTATIONS
# ============================================================
def corr_guided_init(X, y, N):
    D = X.shape[1]
    relief = ReliefF(n_neighbors=10, n_features_to_select=D)
    relief.fit(X, y)
    w = relief.feature_importances_
    norm_w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    Pop = (np.random.rand(N, D) < norm_w).astype(int)
    return Pop, w

def cg_fitness(X, y, particle):
    sel = np.where(particle==1)[0]
    if len(sel)==0 or len(np.unique(y))<2:
        return 0.0
    clf = KNeighborsClassifier(n_neighbors=3)
    try:
        score = np.mean(cross_val_score(clf, X[:,sel], y, cv=5))
    except Exception:
        score = np.mean(cross_val_score(clf, X[:,sel], y, cv=3))
    penalty = len(sel)/X.shape[1]
    return score - 0.15 * penalty

def CGPSO_Standard(X, y, Pop, feat_w, max_iter=MAX_ITER, patience=PATIENCE, tol=1e-4):
    N, D = Pop.shape
    V = np.zeros((N, D))
    pbest = Pop.copy()
    pbest_f = np.array([cg_fitness(X, y, pbest[i]) for i in range(N)])
    gbest = pbest[np.argmax(pbest_f)].copy()
    gbest_f = np.max(pbest_f)

    norm_w = (feat_w - feat_w.min()) / (feat_w.max() - feat_w.min() + 1e-12)
    last_improve_iter = 0
    stagnation_iter = None

    for t in range(1, max_iter + 1):
        for i in range(N):
            w, c1, c2 = 0.6, 1.2, 1.2
            r1, r2 = np.random.rand(), np.random.rand()
            V[i] = w*V[i] + c1*r1*(pbest[i]-Pop[i]) + c2*r2*(gbest-Pop[i])
            sigmoid_v = 1/(1+np.exp(-V[i]))
            TF = 0.5*sigmoid_v + 0.5*norm_w
            Pop[i] = (np.random.rand() < TF).astype(int)

            f = cg_fitness(X, y, Pop[i])
            if f > pbest_f[i]:
                pbest[i] = Pop[i].copy()
                pbest_f[i] = f
                if f > gbest_f + tol:
                    gbest = Pop[i].copy()
                    gbest_f = f
                    last_improve_iter = t

        if t - last_improve_iter >= patience:
            stagnation_iter = last_improve_iter
            break

    return gbest, stagnation_iter

def CGPSO_green(X, y, Pop, feat_w, max_iter=MAX_ITER, patience=PATIENCE, tol=1e-4):
    N, D = Pop.shape
    w_const, c1, c2 = 0.6, 1.2, 1.2
    low_b, up_b = -2.0, 2.0  # pre-sigmoid state bounds

    pbest = Pop.copy()
    pbest_f = np.array([cg_fitness(X, y, pbest[i]) for i in range(N)])
    gbest = pbest[np.argmax(pbest_f)].copy()
    gbest_f = np.max(pbest_f)

    norm_w = (feat_w - feat_w.min()) / (feat_w.max() - feat_w.min() + 1e-12)
    last_improve_iter = 0
    stagnation_iter = None

    # Green's function particle histories
    state = np.zeros((N, D))
    x0_list = state.copy()
    mu_histories = [[] for _ in range(N)]
    history_F = {0: 1.0, 1: 1.0}

    for t in range(1, max_iter + 1):
        for i in range(N):
            # --- Green Vector Update ---
            phi1, phi2 = c1 * np.random.uniform(0, 1), c2 * np.random.uniform(0, 1)
            phi_t = max(phi1 + phi2, 1e-6)
            
            a1 = float(phi_t - 1.0 - w_const)
            a2 = float(w_const)
            disc = (a1 ** 2) - (4.0 * a2)
            
            if disc < 0:
                real_part = -a1 / 2.0
                imag = math.sqrt(max(0.0, -disc)) / 2.0
                r_mag = math.hypot(real_part, imag)
                omega = math.atan2(imag, real_part) if r_mag > 1e-12 else 0.0
                root_type = "complex"
            else:
                lam1 = (-a1 + math.sqrt(disc)) / 2.0
                lam2 = (-a1 - math.sqrt(disc)) / 2.0
                root_type = "real"

            def f0(n):
                return (r_mag ** n) * math.sin(omega * n) if root_type == "complex" else float(lam1 ** n)
            def f1(n):
                return (r_mag ** n) * math.cos(omega * n) if root_type == "complex" else float(lam2 ** n)
            def Green_G(n, m):
                denom = float(f0(m-1) * f1(m) - f1(m-1) * f0(m))
                if abs(denom) < 1e-16: return 0.0
                return float((f0(n) * f1(m-1) - f1(n) * f0(m-1)) / denom)

            # Record mu trajectory
            mu_val = (phi1 * pbest[i] + phi2 * gbest) / phi_t
            mu_histories[i].append(mu_val)
            current_idx = len(mu_histories[i]) - 1

            if current_idx >= 2:
                Pn = np.zeros(D)
                for m in range(2, current_idx + 1):
                    rn = (mu_histories[i][m] - mu_histories[i][m-1]) - w_const * (mu_histories[i][m-1] - mu_histories[i][m-2])
                    Pn += Green_G(current_idx, m) * rn
                
                A = np.array([[f0(0), f1(0)], [f0(1), f1(1)]], dtype=float)
                b_arr = np.array([history_F[0], history_F[1]], dtype=float)
                try:
                    C = np.linalg.solve(A, b_arr)
                except np.linalg.LinAlgError:
                    C = np.linalg.lstsq(A, b_arr, rcond=None)[0]
                
                state[i] = ((C[0] * f0(current_idx) + C[1] * f1(current_idx)) * x0_list[i]) + Pn

            state[i] = np.clip(state[i], low_b, up_b)

            sigmoid_v = 1 / (1 + np.exp(-state[i]))
            TF = 0.5 * sigmoid_v + 0.5 * norm_w
            Pop[i] = (np.random.rand(D) < TF).astype(int)

            f = cg_fitness(X, y, Pop[i])
            if f > pbest_f[i]:
                pbest[i] = Pop[i].copy()
                pbest_f[i] = f
                if f > gbest_f + tol:
                    gbest = Pop[i].copy()
                    gbest_f = f
                    last_improve_iter = t

        if t - last_improve_iter >= patience:
            stagnation_iter = last_improve_iter
            break

    return gbest, stagnation_iter

def CGPSO_SL(X, y, Pop, feat_w, max_iter=MAX_ITER, patience=PATIENCE, tol=1e-4):
    N, D = Pop.shape
    c1, c2 = 1.2, 1.2
    low_b, up_b = -2.0, 2.0  # pre-sigmoid state bounds

    state = np.zeros((N, D))
    pbest = Pop.copy()
    pbest_f = np.array([cg_fitness(X, y, pbest[i]) for i in range(N)])
    gbest = pbest[np.argmax(pbest_f)].copy()
    gbest_f = np.max(pbest_f)

    norm_w = (feat_w - feat_w.min()) / (feat_w.max() - feat_w.min() + 1e-12)
    last_improve_iter = 0
    stagnation_iter = None

    for t in range(1, max_iter + 1):
        for i in range(N):
            # --- Cosine Vector (Bessel J0) Update ---
            phi1, phi2 = c1 * np.random.rand(), c2 * np.random.rand()
            phi_t = max(phi1 + phi2, 1e-6)
            mu_t = (phi1 * pbest[i] + phi2 * gbest) / phi_t

            z = math.sqrt(phi_t) * float(t)
            u = np.random.randn(D)
            norm = np.linalg.norm(u)
            u = u / norm if norm > 1e-12 else np.ones(D) / math.sqrt(D)

            state[i] = mu_t + j0(z) * u
            state[i] = np.clip(state[i], low_b, up_b)

            sigmoid_v = 1 / (1 + np.exp(-state[i]))
            TF = 0.6 * sigmoid_v + 0.4 * norm_w
            Pop[i] = (np.random.rand(D) < TF).astype(int)

            f = cg_fitness(X, y, Pop[i])
            if f > pbest_f[i]:
                pbest[i] = Pop[i].copy()
                pbest_f[i] = f
                if f > gbest_f + tol:
                    gbest = Pop[i].copy()
                    gbest_f = f
                    last_improve_iter = t

        if t - last_improve_iter >= patience:
            stagnation_iter = last_improve_iter
            break

    return gbest, stagnation_iter


# ============================================================
# PART E — RUN ALL APPROACHES (WITH CONSISTENT SPLITS & TIMING)
# ============================================================
approach_names = ["EBPSO_Standard", "EBPSO_SL", "EBPSO_Green", "CGPSO_Standard", "CGPSO_Green", "CGPSO_SL"]

results_features_all = pd.DataFrame()
results_acc_all = pd.DataFrame()
results_f1_all = pd.DataFrame()
results_stag_all = pd.DataFrame()
results_time_all = pd.DataFrame()

final_masks = {}

for name, df in datasets.items():
    print("\n" + "="*80)
    print(f"RUNNING DATASET: {name} | shape={df.shape}")
    print("="*80)

    X = df.drop(columns=['label']).values.astype(float)
    y = df['label'].values.astype(int)

    features_runs = {a: [] for a in approach_names}
    acc_runs = {a: [] for a in approach_names}
    f1_runs = {a: [] for a in approach_names}
    stag_runs = {a: [] for a in approach_names}
    time_runs = {a: [] for a in approach_names}

    for run_idx in range(1, N_RUNS+1):
        print(f"\n--- Run {run_idx} ---")
        run_seed = BASE_RANDOM_STATE + run_idx*7 # Ensure same split for all algos this run
        # ---------------- CGPSO Setup ----------------
        np.random.seed(run_seed)
        Xtr, Xte, ytr, yte = safe_stratified_split(X, y, test_size=TEST_SIZE, random_state=run_seed)
        Pop, feat_w = corr_guided_init(Xtr, ytr, N=CGPOP_SIZE)
        # ---------------- EBPSO ----------------
        t0 = time.perf_counter()
        sel_std, acc_std, f1_std, mask_std, stag_std = EBPSO_standard(Xtr, ytr,Xte,yte, seed=run_seed)
        time_runs["EBPSO_Standard"].append(time.perf_counter() - t0)
        features_runs["EBPSO_Standard"].append(sel_std); acc_runs["EBPSO_Standard"].append(acc_std)
        f1_runs["EBPSO_Standard"].append(f1_std); stag_runs["EBPSO_Standard"].append(stag_std or MAX_ITER)

        t0 = time.perf_counter()
        sel_sl, acc_sl, f1_sl, mask_sl, stag_sl = EBPSO_SL(Xtr, ytr,Xte,yte, seed=run_seed)
        time_runs["EBPSO_SL"].append(time.perf_counter() - t0)
        features_runs["EBPSO_SL"].append(sel_sl); acc_runs["EBPSO_SL"].append(acc_sl)
        f1_runs["EBPSO_SL"].append(f1_sl); stag_runs["EBPSO_SL"].append(stag_sl or MAX_ITER)

        t0 = time.perf_counter()
        sel_gr, acc_gr, f1_gr, mask_gr, stag_gr = EBPSO_green(Xtr, ytr,Xte,yte, seed=run_seed)
        time_runs["EBPSO_Green"].append(time.perf_counter() - t0)
        features_runs["EBPSO_Green"].append(sel_gr); acc_runs["EBPSO_Green"].append(acc_gr)
        f1_runs["EBPSO_Green"].append(f1_gr); stag_runs["EBPSO_Green"].append(stag_gr or MAX_ITER)

        if run_idx == N_RUNS:
            final_masks[(name, "EBPSO_Standard")] = mask_std
            final_masks[(name, "EBPSO_SL")] = mask_sl
            final_masks[(name, "EBPSO_Green")] = mask_gr

       

        # ---------------- CGPSO Standard ----------------
        t0 = time.perf_counter()
        best_Standard, stag_cp = CGPSO_Standard(Xtr, ytr, Pop.copy(), feat_w)
        acc_cp, f1_cp, cm_cp, sel_cp = evaluate_solution(Xtr, Xte, ytr, yte, best_Standard)
        time_runs["CGPSO_Standard"].append(time.perf_counter() - t0)
        features_runs["CGPSO_Standard"].append(np.sum(sel_cp)); acc_runs["CGPSO_Standard"].append(acc_cp)
        f1_runs["CGPSO_Standard"].append(f1_cp); stag_runs["CGPSO_Standard"].append(stag_cp or MAX_ITER)

        
        # ---------------- CGPSO SL ----------------
        t0 = time.perf_counter()
        best_sl, stag_cs = CGPSO_SL(Xtr, ytr, Pop.copy(), feat_w)
        acc_cs, f1_cs, cm_cs, sel_cs = evaluate_solution(Xtr, Xte, ytr, yte, best_sl)
        time_runs["CGPSO_SL"].append(time.perf_counter() - t0)
        features_runs["CGPSO_SL"].append(np.sum(sel_cs)); acc_runs["CGPSO_SL"].append(acc_cs)
        f1_runs["CGPSO_SL"].append(f1_cs); stag_runs["CGPSO_SL"].append(stag_cs or MAX_ITER)

      # ---------------- CGPSO Green ----------------
        t0 = time.perf_counter()
        best_green, stag_cg = CGPSO_green(Xtr, ytr, Pop.copy(), feat_w)
        acc_cg, f1_cg, cm_cg, sel_cg = evaluate_solution(Xtr, Xte, ytr, yte, best_green)
        time_runs["CGPSO_Green"].append(time.perf_counter() - t0)
        features_runs["CGPSO_Green"].append(np.sum(sel_cg)); acc_runs["CGPSO_Green"].append(acc_cg)
        f1_runs["CGPSO_Green"].append(f1_cg); stag_runs["CGPSO_Green"].append(stag_cg or MAX_ITER)


        

        if run_idx == N_RUNS:
            final_masks[(name, "CGPSO_Standard")] = best_Standard
            final_masks[(name, "CGPSO_SL")] = best_sl
            final_masks[(name, "CGPSO_Green")] = best_green

    # Store results per dataset
    for a in approach_names:
        def build_row(metrics_dict):
            return pd.DataFrame({"Dataset": name, "Approach": a, **{f"Run_{i+1}": [metrics_dict[a][i]] for i in range(N_RUNS)}})
        
        results_features_all = pd.concat([results_features_all, build_row(features_runs)], ignore_index=True)
        results_acc_all = pd.concat([results_acc_all, build_row(acc_runs)], ignore_index=True)
        results_f1_all = pd.concat([results_f1_all, build_row(f1_runs)], ignore_index=True)
        results_stag_all = pd.concat([results_stag_all, build_row(stag_runs)], ignore_index=True)
        results_time_all = pd.concat([results_time_all, build_row(time_runs)], ignore_index=True)

excel_path = r"EBPSO_CGPSO_results_Ionesphere.xlsx"
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    results_acc_all.to_excel(writer, sheet_name='Accuracy', index=False)
    results_f1_all.to_excel(writer, sheet_name='F1_Score', index=False)
    results_features_all.to_excel(writer, sheet_name='Selected_Features', index=False)
    results_stag_all.to_excel(writer, sheet_name='Stagnation_Iteration', index=False)
    results_time_all.to_excel(writer, sheet_name='Runtime_Seconds', index=False)
print(f"\n✅ Results exported successfully to Excel: {excel_path}")


# ============================================================
# PART F, G, H — STATS, TOST, COMPLEXITY (Unchanged Math Logic)
# ============================================================
EBPSO_FAMILY = ["EBPSO_Standard", "EBPSO_SL", "EBPSO_Green"]
CGPSO_FAMILY = ["CGPSO_Standard", "CGPSO_SL", "CGPSO_Green"]
FAMILIES = {"EBPSO": EBPSO_FAMILY, "CGPSO": CGPSO_FAMILY}

def build_wide(results_df, approach_names_sub, n_runs):
    run_cols = [f"Run_{i+1}" for i in range(n_runs)]
    wide = {}
    for dataset in results_df['Dataset'].unique():
        sub = results_df[results_df['Dataset'] == dataset].set_index('Approach')
        wide[dataset] = sub.reindex(approach_names_sub)[run_cols].values.astype(float)
    return wide

def holm_bonferroni(pvals):
    pvals = np.array(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj

def friedman_and_posthoc(results_df, metric_name, approach_names_sub, n_runs, family_name):
    wide_dict = build_wide(results_df, approach_names_sub, n_runs)
    friedman_rows, posthoc_rows = [], []
    for dataset, mat in wide_dict.items():
        samples = [mat[i, :] for i in range(mat.shape[0])]
        try:
            stat_, p = stats.friedmanchisquare(*samples)
        except Exception:
            stat_, p = np.nan, np.nan
        friedman_rows.append({
            "Family": family_name, "Dataset": dataset, "Metric": metric_name,
            "Friedman_stat": stat_, "p_value": p, "Significant(p<0.05)": bool(p < 0.05) if not np.isnan(p) else False
        })
        pairs = list(combinations(range(len(approach_names_sub)), 2))
        raw_p = []
        for (i, j) in pairs:
            a, b = mat[i, :], mat[j, :]
            try:
                w_p = 1.0 if np.allclose(a, b) else stats.wilcoxon(a, b)[1]
            except Exception:
                w_p = np.nan
            raw_p.append(w_p if not np.isnan(w_p) else 1.0)
        adj_p = holm_bonferroni(raw_p)
        for (i, j), rp, ap in zip(pairs, raw_p, adj_p):
            posthoc_rows.append({
                "Family": family_name, "Dataset": dataset, "Metric": metric_name,
                "Approach_A": approach_names_sub[i], "Approach_B": approach_names_sub[j],
                "Wilcoxon_p_raw": rp, "Wilcoxon_p_holm": ap, "Significant(p_holm<0.05)": bool(ap < 0.05)
            })
    return pd.DataFrame(friedman_rows), pd.DataFrame(posthoc_rows)

friedman_acc_parts, posthoc_acc_parts, friedman_f1_parts, posthoc_f1_parts = [], [], [], []
for family_name, family_approaches in FAMILIES.items():
    f_acc, p_acc = friedman_and_posthoc(results_acc_all, "Accuracy", family_approaches, N_RUNS, family_name)
    f_f1, p_f1 = friedman_and_posthoc(results_f1_all, "F1_Score", family_approaches, N_RUNS, family_name)
    friedman_acc_parts.append(f_acc); posthoc_acc_parts.append(p_acc)
    friedman_f1_parts.append(f_f1); posthoc_f1_parts.append(p_f1)

friedman_acc = pd.concat(friedman_acc_parts, ignore_index=True)
posthoc_acc = pd.concat(posthoc_acc_parts, ignore_index=True)
friedman_f1 = pd.concat(friedman_f1_parts, ignore_index=True)
posthoc_f1 = pd.concat(posthoc_f1_parts, ignore_index=True)

with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    friedman_acc.to_excel(writer, sheet_name='Friedman_Accuracy', index=False)
    friedman_f1.to_excel(writer, sheet_name='Friedman_F1', index=False)
    posthoc_acc.to_excel(writer, sheet_name='Posthoc_Accuracy', index=False)
    posthoc_f1.to_excel(writer, sheet_name='Posthoc_F1', index=False)

print("\n✅ Statistical test results appended to Excel.")
# ============================================================
# PART G — TOST (Two One-Sided Tests) EQUIVALENCE ANALYSIS (FAMILY-SPLIT)
# (Append this after the Friedman/Wilcoxon block, Part F)
#
# WHY TOST:
# A non-significant Wilcoxon/Friedman result only tells you that
# you FAILED to detect a difference -- it never proves two
# approaches perform "the same". TOST flips the null hypothesis:
# H0 = "the true difference is at least as large as +/- eps"
# H1 = "the true difference lies within (-eps, +eps)"
# Rejecting H0 (TOST_p < alpha) lets you positively claim
# statistical equivalence within a margin eps you define.
#
# Same family split as Part F: TOST runs within EBPSO's 3 approaches
# and within CGPSO's 3 approaches separately (3 pairs each), never
# comparing across families.
# ============================================================

TOST_EPS = 0.02
TOST_ALPHA = 0.05


def tost_paired(a, b, eps=TOST_EPS, alpha=TOST_ALPHA):
    """
    Paired TOST for equivalence between two related samples (same runs,
    same dataset, different approach) -- analogous to a paired t-test
    but testing for equivalence instead of difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    se = sd_diff / np.sqrt(n) if sd_diff > 0 else 1e-12
    df = n - 1

    t_lower = (mean_diff - (-eps)) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    t_upper = (mean_diff - eps) / se
    p_upper = stats.t.cdf(t_upper, df)

    tost_p = max(p_lower, p_upper)
    equivalent = tost_p < alpha

    ci_low = mean_diff - stats.t.ppf(1 - alpha, df) * se
    ci_high = mean_diff + stats.t.ppf(1 - alpha, df) * se

    cohens_dz = mean_diff / sd_diff if sd_diff > 0 else 0.0

    return {
        "Mean_Diff": mean_diff,
        "SD_Diff": sd_diff,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "TOST_p": tost_p,
        "CI90_Low": ci_low,
        "CI90_High": ci_high,
        "Cohens_dz": cohens_dz,
        "Equivalence_Margin": eps,
        "Equivalent(TOST_p<alpha)": bool(equivalent)
    }


def run_tost_all(results_df, metric_name, approach_names_sub, N_RUNS, family_name, eps=TOST_EPS, alpha=TOST_ALPHA):
    """
    Run pairwise TOST equivalence tests for every approach pair WITHIN
    one family, per dataset, for the given metric.
    """
    wide_dict = build_wide(results_df, approach_names_sub, N_RUNS)
    rows = []

    for dataset, mat in wide_dict.items():
        pairs = list(combinations(range(len(approach_names_sub)), 2))
        for (i, j) in pairs:
            a, b = mat[i, :], mat[j, :]
            res = tost_paired(a, b, eps=eps, alpha=alpha)
            rows.append({
                "Family": family_name,
                "Dataset": dataset,
                "Metric": metric_name,
                "Approach_A": approach_names_sub[i],
                "Approach_B": approach_names_sub[j],
                **res
            })

    return pd.DataFrame(rows)


# ---------------- Run TOST for Accuracy and F1, PER FAMILY ----------------
tost_acc_parts, tost_f1_parts = [], []
for family_name, family_approaches in FAMILIES.items():
    tost_acc_parts.append(run_tost_all(results_acc_all, "Accuracy", family_approaches, N_RUNS, family_name))
    tost_f1_parts.append(run_tost_all(results_f1_all, "F1_Score", family_approaches, N_RUNS, family_name))

tost_acc = pd.concat(tost_acc_parts, ignore_index=True)
tost_f1 = pd.concat(tost_f1_parts, ignore_index=True)

# print(f"\n--- TOST Equivalence Test (Accuracy, margin=+/-{TOST_EPS}) — by Family ---")
# display(tost_acc)

# print(f"\n--- TOST Equivalence Test (F1 Score, margin=+/-{TOST_EPS}) — by Family ---")
# display(tost_f1)

# print("\n--- Pairs found EQUIVALENT (Accuracy) ---")
# display(tost_acc[tost_acc["Equivalent(TOST_p<alpha)"]][
#     ["Family", "Dataset", "Approach_A", "Approach_B", "Mean_Diff", "TOST_p", "Cohens_dz"]
# ])

# print("\n--- Pairs found EQUIVALENT (F1 Score) ---")
# display(tost_f1[tost_f1["Equivalent(TOST_p<alpha)"]][
#     ["Family", "Dataset", "Approach_A", "Approach_B", "Mean_Diff", "TOST_p", "Cohens_dz"]
# ])

# ---------------- Append TOST results into the existing Excel file ----------------
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    tost_acc.to_excel(writer, sheet_name='TOST_Accuracy', index=False)
    tost_f1.to_excel(writer, sheet_name='TOST_F1', index=False)

print(f"\n✅ TOST equivalence results appended to Excel: {excel_path}")
