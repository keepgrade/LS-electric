import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance as xgb_plot_imp
from lightgbm import LGBMRegressor, plot_importance as lgb_plot_imp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Optionally: Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')
# plt.style.use('seaborn')

# ─── 1. 데이터 전처리 ──────────────────────────────────────────────────────────────

def create_features(df, is_train=True):
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["년"] = df["측정일시"].dt.year
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["시간"] = df["측정일시"].dt.hour
    df["요일"] = df["측정일시"].dt.weekday
    df["주말여부"] = (df["요일"] >= 5).astype(int)

    for unit, period in [("시간",24),("요일",7),("월",12)]:
        df[f"sin_{unit}"] = np.sin(2*np.pi*df[unit]/period)
        df[f"cos_{unit}"] = np.cos(2*np.pi*df[unit]/period)

    df["월초여부"] = (df["일"] <=5).astype(int)
    df["월말여부"] = (df["일"] >=25).astype(int)
    df["peak_time"] = ((df["시간"]>=8) & (df["시간"]<=22)).astype(int)
    df["night_time"] = ((df["시간"]>=22)| (df["시간"]<=6)).astype(int)

    df["계절"] = df["월"].apply(lambda m: "여름" if m in [6,7,8] else ("봄가을" if m in [3,4,5,9,10] else "겨울"))
    df["적용시점"] = df["측정일시"].apply(lambda x: "before" if x < datetime(2024,10,24) else "after")
    mapping = {
        "before":{"여름":{"경부하":93.1,"중간부하":146.3,"최대부하":216.6},
                  "봄가을":{"경부하":93.1,"중간부하":115.2,"최대부하":138.9},
                  "겨울":{"경부하":100.4,"중간부하":146.5,"최대부하":193.4}},
        "after":{"여름":{"경부하":110.0,"중간부하":163.2,"최대부하":233.5},
                 "봄가을":{"경부하":110.0,"중간부하":132.1,"최대부하":155.8},
                 "겨울":{"경부하":117.3,"중간부하":163.4,"최대부하":210.3}}
    }
    def tz(hour, season, when):
        if season in ["여름","봄가을"]:
            if hour>=22 or hour<8: return "경부하"
            if 8<=hour<11 or 12<=hour<13 or 18<=hour<22: return "중간부하"
            return "최대부하"
        else:
            if hour>=22 or hour<8: return "경부하"
            if 8<=hour<9 or 12<=hour<16 or 19<=hour<22: return "중간부하"
            return "최대부하"

    df["시간대"] = df.apply(lambda r: tz(r["시간"], r["계절"], r["적용시점"]),axis=1)
    df["요금단가"] = df.apply(lambda r: mapping[r["적용시점"]][r["계절"]][r["시간대"]],axis=1)

    # label encoding
    if is_train: globals()['le'] = LabelEncoder()
    df["작업유형_encoded"] = globals()['le'].fit_transform(df["작업유형"]) if is_train else globals()['le'].transform(df["작업유형"])

    return df

def target_encode(train, test, col, target, n_folds=5, smoothing=10):
    from sklearn.model_selection import KFold
    global_mean = train[target].mean()
    train[f"{col}_te"] = global_mean
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr, val in kf.split(train):
        agg = train.iloc[tr].groupby(col)[target].agg(["mean","count"])
        weight = 1/(1+np.exp(-(agg['count']-smoothing)))
        enc = global_mean*(1-weight)+agg['mean']*weight
        train.loc[val, f"{col}_te"] = train.loc[val, col].map(enc)
    agg = train.groupby(col)[target].agg(["mean","count"])
    weight = 1/(1+np.exp(-(agg['count']-smoothing)))
    enc = global_mean*(1-weight)+agg['mean']*weight
    test[f"{col}_te"] = test[col].map(enc).fillna(global_mean)

def add_lag_roll(train, test, target):
    for lag in [1,2,3,6,12,24]:
        train[f"{target}_lag{lag}"] = train[target].shift(lag)
        test[f"{target}_lag{lag}"] = train[target].iloc[-lag]
    for w in [3,6,12,24]:
        roll = train[target].rolling(w)
        for fn in ["mean","std","max","min"]:
            train[f"{target}_roll_{fn}{w}"] = getattr(roll,fn)()
            test[f"{target}_roll_{fn}{w}"] = train[f"{target}_roll_{fn}{w}"].iloc[-1]

def load_data(base_dir="../data"):
    tr = pd.read_csv(os.path.join(base_dir,"train.csv"))
    te = pd.read_csv(os.path.join(base_dir,"test.csv"))
    tr = create_features(tr, True)
    te = create_features(te, False)
    target="전기요금(원)"
    for c in ["작업유형","시간","요일","시간대","계절"]:
        target_encode(tr, te, c, target)
    add_lag_roll(tr, te, target)
    tr = tr.dropna()
    tr = tr[(tr[target] >= tr[target].quantile(0.025)) & (tr[target] <= tr[target].quantile(0.975))]
    y = np.log1p(tr[target])
    FEATURES = [c for c in tr.columns if c not in ["id","측정일시","작업유형",target]]
    X = tr[FEATURES].values
    X_test = te[FEATURES].fillna(tr[FEATURES].median()).values
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    pickle.dump((scaler,FEATURES), open("scaler_features.pkl","wb"))
    return X, y, X_test, te[["id","측정일시"]], FEATURES
# ─── 2. 모델 훈련 및 CV ─────────────────────────────────────────────────────────────

def tune_lgb(X,y):
    def obj(trial):
        params = {
            "n_estimators":trial.suggest_int("n_estimators",200,800),
            "max_depth":trial.suggest_int("max_depth",4,12),
            "learning_rate":trial.suggest_float("learning_rate",0.01,0.3),
            "subsample":trial.suggest_float("subsample",0.6,1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),
            "reg_alpha":trial.suggest_float("reg_alpha",0,5),
            "reg_lambda":trial.suggest_float("reg_lambda",0,5),
            "random_state":42
        }
        m = LGBMRegressor(**params)
        cv = TimeSeriesSplit(3)
        maes=[]
        for tr_i, va_i in cv.split(X):
            m.fit(X[tr_i],y[tr_i])
            p = m.predict(X[va_i])
            maes.append(mean_absolute_error(np.expm1(y[va_i]), np.expm1(p)))
        return np.mean(maes)
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=30)
    return study.best_params

def train_cv(X,y,models):
    tscv = TimeSeriesSplit(n_splits=5)
    oofs = {name:np.zeros_like(y) for name in models}
    cv_res = {name:[] for name in models}
    for name, m in models.items():
        for tr,va in tscv.split(X):
            m.fit(X[tr],y[tr])
            p = m.predict(X[va])
            oofs[name][va]=p
            cv_res[name].append(mean_absolute_error(np.expm1(y[va]),np.expm1(p)))
    return oofs,cv_res

def plot_cv(cv_res):
    names=list(cv_res.keys())
    means=[np.mean(cv_res[n]) for n in names]
    stds=[np.std(cv_res[n]) for n in names]
    fig,ax=plt.subplots()
    ax.bar(names,means,yerr=stds)
    ax.set_ylabel("CV MAE")
    plt.show()

def plot_importances(models,FEATURES):
    for name,m in models.items():
        if hasattr(m,"feature_importances_"):
            plt.figure()
            if name=="lgb": lgb_plot_imp(m,max_num_features=10)
            elif name=="xgb": xgb_plot_imp(m,max_num_features=10)
            plt.title(f"{name} 중요 피처")
            plt.show()

# ─── 3. LSTM 모델 ─────────────────────────────────────────────────────────────────

def build_lstm(input_shape):
    m=Sequential([
        LSTM(128,return_sequences=True,input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32,activation="relu"),
        Dropout(0.1),
        Dense(1),
    ])
    m.compile("adam","mse",metrics=["mae"])
    return m

def train_lstm_model(tr_df,FEATURES,y,timesteps=96*7):
    seq = tr_df[FEATURES+[ "전기요금(원)" ]].dropna()
    scaler=MinMaxScaler()
    data = scaler.fit_transform(seq)
    Xs,ys=[],[]
    for i in range(len(data)-timesteps):
        Xs.append(data[i:i+timesteps,:len(FEATURES)])
        ys.append(data[i+timesteps,-1])
    Xs,ys=np.array(Xs),np.array(ys)
    trn=int(len(Xs)*0.8)
    Xtr,Xva=Xs[:trn],Xs[trn:]
    ytr,yva=ys[:trn],ys[trn:]
    m=build_lstm((timesteps,len(FEATURES)))
    es=EarlyStopping(patience=5,restore_best_weights=True)
    rl=ReduceLROnPlateau(patience=3,factor=0.5)
    hist=m.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=20,batch_size=64,callbacks=[es,rl],verbose=0)
    return m,scaler, (Xva,yva),hist

# ─── 4. 실행 부분 ─────────────────────────────────────────────────────────────────

def main():
    X,y,X_test,te_meta = load_data()
    # 모델 정의
    models = {
        "lgb": LGBMRegressor(n_estimators=500),
        "xgb": XGBRegressor(n_estimators=600,verbosity=0),
        "rf": RandomForestRegressor(n_estimators=400,random_state=42)
    }
    # Optuna 튜닝
    if OPTUNA_AVAILABLE:
        best = tune_lgb(X,y)
        models["lgb"] = LGBMRegressor(**best)
        print("Optuna tuned params:",best)
    # CV 훈련
    oofs,cv_res=train_cv(X,y,models)
    plot_cv(cv_res)
    plot_importances(models,pickle.load(open("scaler_features.pkl","rb"))[1])

    # Stacking
    meta_X = np.column_stack([oofs[n] for n in models])
    meta_y = np.expm1(y)
    meta = Ridge()
    meta.fit(meta_X,meta_y)

    # Test 예측
    test_stack = np.column_stack([models[n].predict(X_test) for n in models])
    test_pred = np.expm1(meta.predict(test_stack))
    out = te_meta.copy()
    out["전기요금(원)"] = test_pred
    out[["id","전기요금(원)"]].to_csv("submission_full_cv.csv",index=False)

    for name,m in models.items():
        pickle.dump(m,open(f"model_{name}.pkl","wb"))
    pickle.dump(meta,open("meta_model.pkl","wb"))

    # Optional: LSTM 학습/평가
    # lstm_model, lstm_scaler, (Xva,yva), hist = train_lstm_model(pd.concat([pd.DataFrame(X,columns=pickle.load(open("scaler_features.pkl","rb"))[1]), np.log1p(np.expm1(y))],axis=1),pickle.load(open("scaler_features.pkl","rb"))[1],y)

if __name__=="__main__":
    main()
