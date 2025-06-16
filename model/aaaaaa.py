import os, pickle, optuna
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- 1. 데이터 로드 & 전처리 ---
BASE_DIR = "../data"
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

for df in [train_df, test_df]:
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["시간"] = df["측정일시"].dt.hour
    df["요일"] = df["측정일시"].dt.weekday
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["sin_시간"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_시간"] = np.cos(2 * np.pi * df["시간"] / 24)

def get_season(month):
    if month in [6,7,8]: return "여름"
    if month in [3,4,5,9,10]: return "봄가을"
    return "겨울"

def get_time_zone(hour, season):
    if season in ["여름","봄가을"]:
        if hour>=22 or hour<8: return "경부하"
        if (8<=hour<11) or (12<=hour<13) or (18<=hour<22): return "중간부하"
        return "최대부하"
    else:
        if hour>=22 or hour<8: return "경부하"
        if (8<=hour<9) or (12<=hour<16) or (19<=hour<22): return "중간부하"
        return "최대부하"

RATE = {
    "before": {"여름": {"경부하":93.1,"중간부하":146.3,"최대부하":216.6},
               "봄가을": {"경부하":93.1,"중간부하":115.2,"최대부하":138.9},
               "겨울": {"경부하":100.4,"중간부하":146.5,"최대부하":193.4}},
    "after": {"여름": {"경부하":110.0,"중간부하":163.2,"최대부하":233.5},
              "봄가을":{"경부하":110.0,"중간부하":132.1,"최대부하":155.8},
              "겨울":{"경부하":117.3,"중간부하":163.4,"최대부하":210.3}}
}
CUT = datetime(2024,10,24)
for df in [train_df, test_df]:
    df["계절"] = df["월"].apply(get_season)
    df["시점"] = df["측정일시"].apply(lambda x:"before" if x<CUT else "after")
    df["시간대"] = df.apply(lambda r:get_time_zone(r["시간"], r["계절"]),axis=1)
    df["요금단가"] = df.apply(lambda r: RATE[r["시점"]][r["계절"]][r["시간대"]], axis=1)

le = LabelEncoder()
train_df["작업유형_encoded"] = le.fit_transform(train_df["작업유형"])
test_df["작업유형_encoded"] = le.transform(test_df["작업유형"])

def tgt_enc(df_tr, df_te, col, target):
    gm = df_tr[target].mean()
    ag = df_tr.groupby(col)[target].agg(["mean","count"])
    sw = 1/(1+np.exp(-(ag["count"]-10)))
    en = gm*(1-sw) + ag["mean"]*sw
    mp = en.to_dict()
    df_tr[col+"_te"] = df_tr[col].map(mp)
    df_te[col+"_te"] = df_te[col].map(mp)

for c in ["시간","요일","시간대","작업유형"]:
    tgt_enc(train_df, test_df, c, "전기요금(원)")

# 이상치 제거
q1,q3 = train_df["전기요금(원)"].quantile([.25,.75])
iqr = q3-q1
train_df = train_df[(train_df["전기요금(원)"]>=q1 - 1.5*iqr) & (train_df["전기요금(원)"]<=q3 + 1.5*iqr)].copy()

FEATURES = ["작업유형_encoded","월","일","시간","요일","주말여부",
            "sin_시간","cos_시간","요금단가",
            "작업유형_te","시간_te","요일_te","시간대_te"]
TARGET = "전기요금(원)"

X = train_df[FEATURES].values
y = train_df[TARGET].values
X_test = test_df[FEATURES].values

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# --- 2. Optuna + K-Fold Tree model training ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
tree_models = {}
oof = {}
test_pred = {}

for name, cls in [("xgb",XGBRegressor),("lgb",LGBMRegressor),("rf",RandomForestRegressor)]:
    def objective(trial):
        if name=="xgb":
            params = {"n_estimators":trial.suggest_int("n_estimators",200,800),
                      "max_depth":trial.suggest_int("max_depth",3,8),
                      "learning_rate":trial.suggest_float("lr",0.01,0.1),
                      "subsample":trial.suggest_float("sub",0.5,1.0),
                      "colsample_bytree":trial.suggest_float("cs",0.5,1.0),
                      "random_state":42}
        elif name=="lgb":
            params = {"n_estimators":trial.suggest_int("n_estimators",200,800),
                      "max_depth":trial.suggest_int("max_depth",3,8),
                      "learning_rate":trial.suggest_float("lr",0.01,0.1),
                      "subsample":trial.suggest_float("sub",0.5,1.0),
                      "colsample_bytree":trial.suggest_float("cs",0.5,1.0),
                      "random_state":42}
        else:
            params = {"n_estimators":trial.suggest_int("n_estimators",100,500),
                      "max_depth":trial.suggest_int("max_depth",5,15),
                      "random_state":42}
        model = cls(**params)
        vt = np.zeros(len(X_scaled))
        for i,(ti,vi) in enumerate(kf.split(X_scaled)):
            model.fit(X_scaled[ti], y[ti])
            vt[vi] = model.predict(X_scaled[vi])
        return mean_absolute_error(y, vt)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best = study.best_params
    tree_models[name] = cls(**best, random_state=42)
    print(name, "best:", best)

    vt = np.zeros(len(X_scaled))
    tp = np.zeros(len(X_test_scaled))
    for ti,vi in kf.split(X_scaled):
        tree_models[name].fit(X_scaled[ti], y[ti])
        vt[vi] = tree_models[name].predict(X_scaled[vi])
        tp += tree_models[name].predict(X_test_scaled)/kf.n_splits

    oof[name] = vt
    test_pred[name] = tp

# --- 3. LSTM model (강화된 구조) ---
TIME = 96*7
seq_scaler = MinMaxScaler()
df_seq = pd.DataFrame(np.concatenate([X_scaled, y.reshape(-1,1)],axis=1),
                      columns=FEATURES+[TARGET])
seq_data = seq_scaler.fit_transform(df_seq)

def mk_seq(data,t):
    xs, ys = [], []
    for i in range(len(data)-t):
        xs.append(data[i:i+t, :-1])
        ys.append(data[i+t-1, -1])
    return np.array(xs), np.array(ys)

Xs, ys = mk_seq(seq_data, TIME)
cut = int(len(Xs)*0.8)
Xst,Xsv = Xs[:cut], Xs[cut:]
yst,ysv = ys[:cut], ys[cut:]

lstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(TIME,len(FEATURES))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.1),
    Dense(16, activation="relu"),
    Dense(1)
])
lstm.compile(Adam(1e-3), "mse")
lstm.fit(Xst, yst, validation_data=(Xsv,ysv),
         epochs=30, batch_size=64,
         callbacks=[EarlyStopping("val_loss",patience=5,restore_best_weights=True)],
         verbose=1)

oof["lstm"] = lstm.predict(Xsv).flatten()

comb = np.vstack([seq_data[-TIME:, :-1], X_test_scaled])
Xs2 = np.array([comb[i:i+TIME] for i in range(len(comb)-TIME)])
test_pred["lstm"] = lstm.predict(Xs2).flatten()[-len(X_test_scaled):]

# --- 4. Stacking Meta Learner (XGB) ---
all_names = list(oof.keys())
min_len = min(len(arr) for arr in oof.values())
Xm = np.column_stack([oof[n][:min_len] for n in all_names])
ym = y[:min_len]

meta = XGBRegressor(random_state=42, n_jobs=-1)
meta.fit(Xm, ym)

Tm = np.column_stack([test_pred[n] for n in all_names])
final = meta.predict(Tm)

print("OOF MAE:", mean_absolute_error(ym, meta.predict(Xm)))

sub = pd.DataFrame({"id": test_df["id"], TARGET: final})
sub.to_csv("submission_enhanced.csv", index=False)
print("✅ Done!")
