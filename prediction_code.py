

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
import joblib

warnings.filterwarnings("ignore")
os.makedirs("Recession-Nowcasting/data/results", exist_ok=True)




FEATURES_PATH   = "Recession-Nowcasting/data/processed/master_features.csv"
EVAL_START      = "2000-01-01"
BLACKOUT_MONTHS = 24       # NBER announcement lag buffer
SIGNAL_THRESHOLD = 0.35   # for lead-time and false-alarm reporting

NBER_ANNOUNCEMENTS = {
    "1990-1991": ("1990-07-01", "1991-04-01"),
    "2001":      ("2001-03-01", "2001-11-01"),
    "2008-2009": ("2007-12-01", "2008-12-01"),
    "2020":      ("2020-02-01", "2020-06-01"),
}

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "learning_rate":     0.03,
    "num_leaves":        15,
    "max_depth":         4,
    "min_child_samples": 10,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.5,
    "reg_lambda":        1.0,
    "n_estimators":      300,
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}


N_BOOTSTRAP = 500
BLOCK_SIZE  = 12    # 12-month blocks preserve recession clustering

HORIZONS = [0, 1, 3, 6]   # months ahead



# LOAD DATA


print(f"\nLoading features from {FEATURES_PATH}...")
df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], index_col="date")

target_col   = "USRECD"
feature_cols = [c for c in df.columns if c != target_col]

X_all = df[feature_cols].copy()
y_all = df[target_col].copy()

print(f"  Shape          : {df.shape}")
print(f"  Features       : {len(feature_cols)}")
print(f"  Date range     : {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Recession months: {int(y_all.sum())}")

# Confirm no regime columns leaked in
regime_cols = [c for c in feature_cols if "MSM" in c or "HMM" in c or "regime" in c.lower()]
if regime_cols:
    print(f"  WARNING: Found regime columns — dropping: {regime_cols}")
    X_all = X_all.drop(columns=regime_cols)
    feature_cols = [c for c in feature_cols if c not in regime_cols]
else:
    print("  Regime columns: none (correct)")

equity_cols = [c for c in feature_cols if any(x in c for x in
               ["SP500", "VIX", "NASDAQ", "WILSHIRE", "EQUITY"])]
print(f"  Equity features : {len(equity_cols)}")




def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_tr = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index, columns=X_train.columns
    )
    X_te = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index, columns=X_test.columns
    )
    return X_tr, X_te, scaler


def get_class_weight(y_train):
    """Scale positive weight to handle class imbalance (~9% recession rate)."""
    n_rec = y_train.sum()
    n_exp = len(y_train) - n_rec
    return float(n_exp / n_rec) if n_rec > 0 else 1.0


def train_lgbm(X_train, y_train, scale_pos_weight):
    params = {**LGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    model  = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model



def run_walk_forward(horizon=0):
    y_shifted = y_all.shift(-horizon) if horizon > 0 else y_all.copy()

    eval_dates = df.index[df.index >= EVAL_START]

    results_lgbm   = []
    results_probit = []

    for i, pred_date in enumerate(eval_dates):

        if i % 24 == 0:
            pct = i / len(eval_dates) * 100
            print(f"  h={horizon}  {pred_date.strftime('%Y-%m')}  "
                  f"{i+1}/{len(eval_dates)}  ({pct:.0f}%)")

        # Training window: apply NBER blackout
        train_end  = pred_date - pd.DateOffset(months=BLACKOUT_MONTHS)
        train_mask = (df.index < pred_date) & (df.index <= train_end)
        train_idx  = df.index[train_mask]

        if len(train_idx) < 60:
            continue

        X_train_raw = X_all.loc[train_idx].copy()
        y_train_raw = y_shifted.loc[train_idx].copy()

        valid   = y_train_raw.notna()
        X_train = X_train_raw[valid]
        y_train = y_train_raw[valid].astype(int)

        if y_train.sum() < 3:
            continue

        X_pred  = X_all.loc[[pred_date]].copy()
        actual  = float(y_shifted.loc[pred_date]) if pred_date in y_shifted.index else np.nan

        # Normalize within training window only (no look-ahead)
        X_tr, X_pr, _ = normalize(X_train, X_pred)
        spw = get_class_weight(y_train)

        # LightGBM
        try:
            lgbm_model = train_lgbm(X_tr, y_train, spw)
            prob_lgbm  = float(lgbm_model.predict_proba(X_pr)[0, 1])
        except Exception:
            prob_lgbm = np.nan

        # Probit benchmark 
        # Uses only T10Y3M and NFCI — the academic standard
        bench_cols = [c for c in ["T10Y3M_level", "NFCI_level"] if c in X_tr.columns]
        if len(bench_cols) == 2:
            try:
                probit = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                probit.fit(X_tr[bench_cols].fillna(0), y_train)
                prob_probit = float(probit.predict_proba(X_pr[bench_cols].fillna(0))[0, 1])
            except Exception:
                prob_probit = np.nan
        else:
            prob_probit = np.nan

        base = {"date": pred_date, "actual": actual}
        results_lgbm.append({**base,   "prob": prob_lgbm})
        results_probit.append({**base, "prob": prob_probit})

    def to_df(results):
        r = pd.DataFrame(results).set_index("date")
        return r.dropna(subset=["prob", "actual"])

    return to_df(results_lgbm), to_df(results_probit)



# PROBABILITY CALIBRATION

def calibrate_isotonic(df_preds, col="prob"):
    n   = len(df_preds)
    mid = n // 2

    cal_train = df_preds.iloc[:mid]
    cal_test  = df_preds.iloc[mid:]
    out       = df_preds[col].copy()

    try:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(cal_train[col].values, cal_train["actual"].astype(int).values)
        out.iloc[mid:] = ir.predict(cal_test[col].values)
    except Exception as e:
        print(f"  Isotonic calibration failed: {e} — using raw probs")

    return out


# EVALUATION METRICS

def brier_decomp(probs, actuals):
    p  = np.array(probs,   dtype=float)
    a  = np.array(actuals, dtype=float)
    bs = float(np.mean((p - a) ** 2))
    rec = a == 1
    exp = a == 0
    bs_rec = float(np.mean((p[rec] - a[rec]) ** 2)) if rec.sum() > 0 else np.nan
    bs_exp = float(np.mean((p[exp] - a[exp]) ** 2)) if exp.sum() > 0 else np.nan
    return {"BS_overall": bs, "BS_recession": bs_rec, "BS_expansion": bs_exp}


def evaluate(df_preds, label, col="prob_cal"):
    p     = df_preds[col].values
    a     = df_preds["actual"].astype(int).values
    brier = brier_decomp(p, a)
    prauc = average_precision_score(a, p)
    rocauc = roc_auc_score(a, p)
    far   = false_alarm_rate(p, a)
    print(f"\n  {label}")
    print(f"    ROC-AUC                : {rocauc:.4f}")
    print(f"    PR-AUC                 : {prauc:.4f}")
    print(f"    Brier Score overall    : {brier['BS_overall']:.4f}")
    print(f"    Brier Score recession  : {brier['BS_recession']:.4f}")
    print(f"    Brier Score expansion  : {brier['BS_expansion']:.4f}")
    print(f"    False alarm rate (>{SIGNAL_THRESHOLD:.0%}): {far:.1f}%")
    return brier, prauc, rocauc, far


def false_alarm_rate(probs, actuals, threshold=SIGNAL_THRESHOLD):
    p, a     = np.array(probs), np.array(actuals)
    exp_mask = a == 0
    if exp_mask.sum() == 0:
        return np.nan
    return float((p[exp_mask] > threshold).mean() * 100)



# BLOCK BOOTSTRAP

def block_bootstrap(probs, actuals, metric_fn,
                    n_boot=N_BOOTSTRAP, block=BLOCK_SIZE):
    
    p   = np.array(probs,   dtype=float)
    a   = np.array(actuals, dtype=float)
    n   = len(p)
    scores = []

    for _ in range(n_boot):
        n_blocks = int(np.ceil(n / block))
        starts   = np.random.randint(0, n - block + 1, size=n_blocks)
        idx      = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
        idx      = idx[:n]
        p_b, a_b = p[idx], a[idx]
        if a_b.sum() < 1:
            continue
        try:
            scores.append(metric_fn(a_b, p_b))
        except Exception:
            continue

    if len(scores) < 10:
        return np.nan, np.nan
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))



# STACKED ENSEMBLE
# Combines LightGBM and probit probabilities via a logistic
def build_ensemble(df_lgbm, df_probit):
    
    common = df_lgbm.index.intersection(df_probit.index)
    X_meta = pd.DataFrame({
        "lgbm":   df_lgbm.loc[common, "prob_cal"],
        "probit": df_probit.loc[common, "prob_cal"],
    })
    y_meta = df_lgbm.loc[common, "actual"].astype(int)

    n   = len(X_meta)
    cut = int(n * 2 / 3)

    X_train_m = X_meta.iloc[:cut]
    y_train_m = y_meta.iloc[:cut]
    X_eval_m  = X_meta.iloc[cut:]

    out = df_lgbm.loc[common, ["actual"]].copy()
    out["prob"] = np.nan

    
    out.iloc[:cut, out.columns.get_loc("prob")] = X_meta.iloc[:cut]["lgbm"].values

    if y_train_m.sum() >= 3:
        try:
            meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            meta.fit(X_train_m.fillna(0.5), y_train_m)
            preds = meta.predict_proba(X_eval_m.fillna(0.5))[:, 1]
            out.iloc[cut:, out.columns.get_loc("prob")] = preds
            print(f"  Stacked ensemble: meta trained on {cut} obs, "
                  f"evaluated on {n - cut} obs")
        except Exception as e:
            print(f"  Stacked ensemble failed: {e} — using LGBM only")
            out.iloc[cut:, out.columns.get_loc("prob")] = X_meta.iloc[cut:]["lgbm"].values
    else:
        out.iloc[cut:, out.columns.get_loc("prob")] = X_meta.iloc[cut:]["lgbm"].values

    out["prob_cal"] = calibrate_isotonic(out)
    return out.dropna(subset=["prob", "actual"])



# LEAD TIME ANALYSIS


def lead_time_analysis(df_preds, threshold=SIGNAL_THRESHOLD, col="prob_cal"):
    probs = df_preds[col]
    rows  = []
    for name, (rec_start, announced) in NBER_ANNOUNCEMENTS.items():
        ts_start  = pd.Timestamp(rec_start)
        ts_announ = pd.Timestamp(announced)

        if ts_start not in probs.index:
            rows.append({"recession": name, "first_signal": "not in sample"})
            continue

        window = probs[(probs.index >= ts_start) & (probs.index <= ts_announ)]
        first_cross = next((d for d, v in window.items() if v >= threshold), None)

        if first_cross:
            months_before = (
                (ts_announ.year  - first_cross.year)  * 12 +
                (ts_announ.month - first_cross.month)
            )
            rows.append({
                "recession":      name,
                "rec_start":      rec_start,
                "announced":      announced,
                "first_signal":   first_cross.strftime("%Y-%m"),
                "months_before":  months_before,
                "prob_at_signal": round(float(window.loc[first_cross]), 3),
            })
        else:
            rows.append({
                "recession":      name,
                "rec_start":      rec_start,
                "announced":      announced,
                "first_signal":   "never crossed",
                "months_before":  None,
                "prob_at_signal": None,
            })
    return pd.DataFrame(rows)



def threshold_sweep(probs, actuals, thresholds=None):
    
    if thresholds is None:
        thresholds = np.arange(0.10, 0.75, 0.05)

    p = np.array(probs,   dtype=float)
    a = np.array(actuals, dtype=float)
    rows = []

    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (a == 1)).sum())
        fp = int(((pred == 1) & (a == 0)).sum())
        fn = int(((pred == 0) & (a == 1)).sum())
        tn = int(((pred == 0) & (a == 0)).sum())

        prec   = tp / (tp + fp)  if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn)  if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        fpr    = fp / (fp + tn)  if (fp + tn) > 0 else 0.0

        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(prec,   3),
            "recall":    round(recall, 3),
            "f1":        round(f1,     3),
            "fpr":       round(fpr,    3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(rows)





def reliability_stats(probs, actuals, n_bins=10):
    
    p, a   = np.array(probs), np.array(actuals, dtype=float)
    bins   = np.linspace(0, 1, n_bins + 1)
    rows   = []
    for i in range(n_bins):
        lo_b, hi_b = bins[i], bins[i+1]
        mask = (p >= lo_b) & (p < hi_b) if i < n_bins - 1 else (p >= lo_b) & (p <= hi_b)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin":       f"{lo_b:.1f}–{hi_b:.1f}",
            "n":         int(mask.sum()),
            "mean_pred": round(float(p[mask].mean()), 4),
            "mean_obs":  round(float(a[mask].mean()), 4),
            "cal_error": round(abs(float(p[mask].mean()) - float(a[mask].mean())), 4),
        })
    return pd.DataFrame(rows)





SHAP_WINDOWS = {
    "2001": {"train_end": "2000-12-01", "pred_start": "2001-01-01", "pred_end": "2001-11-01"},
    "2008": {"train_end": "2007-12-01", "pred_start": "2008-01-01", "pred_end": "2008-12-01"},
    "2020": {"train_end": "2019-12-01", "pred_start": "2020-01-01", "pred_end": "2020-12-01"},
}


def run_shap_analysis(feature_cols_clean):
    all_top_features = {}

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS — 2001, 2008, 2020")
    print("=" * 60)

    for recession, window in SHAP_WINDOWS.items():
        print(f"\n  Recession: {recession}")
        train_end_ts = pd.Timestamp(window["train_end"])

        train_mask = df.index <= train_end_ts
        X_train_s  = X_all.loc[train_mask, feature_cols_clean].copy()
        y_train_s  = y_all.loc[train_mask].dropna().astype(int)
        X_train_s  = X_train_s.loc[y_train_s.index]

        X_pred_s = X_all.loc[
            (df.index >= window["pred_start"]) &
            (df.index <= window["pred_end"]),
            feature_cols_clean
        ].copy()

        if len(X_pred_s) == 0:
            print(f"    Skipped — prediction window not in data")
            continue

        X_tr_s, X_pr_s, _ = normalize(X_train_s, X_pred_s)
        spw_s     = get_class_weight(y_train_s)
        lgbm_shap = train_lgbm(X_tr_s, y_train_s, spw_s)

        try:
            explainer = shap.TreeExplainer(lgbm_shap)
            shap_vals = explainer.shap_values(X_pr_s)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            shap_df = pd.DataFrame(shap_vals, index=X_pr_s.index, columns=X_pr_s.columns)
            shap_df.to_csv(f"Recession-Nowcasting/data/results/shap_values_{recession}.csv")

            mean_abs = shap_df.abs().mean().sort_values(ascending=False)
            all_top_features[recession] = mean_abs

            print(f"    Top 10 drivers:")
            for feat_name, val in mean_abs.head(10).items():
                equity_tag = " [equity]" if any(x in feat_name for x in
                             ["SP500", "VIX", "NASDAQ", "WILSHIRE", "EQUITY"]) else ""
                print(f"      {feat_name:<40} {val:.4f}{equity_tag}")

        except Exception as e:
            print(f"    SHAP failed: {e}")

    # Cross-recession comparison
    if len(all_top_features) >= 2:
        print("\n--- Cross-Recession SHAP: features in top-10 of multiple recessions ---")
        top_sets = {r: set(imp.head(10).index) for r, imp in all_top_features.items()}
        universal = set.intersection(*top_sets.values())
        print(f"  Universal (all 3): {sorted(universal)}")
        for r1, r2 in [("2001", "2008"), ("2001", "2020"), ("2008", "2020")]:
            shared = top_sets.get(r1, set()) & top_sets.get(r2, set()) - universal
            if shared:
                print(f"  Shared {r1}+{r2}: {sorted(shared)}")

    return all_top_features



# MAIN EXECUTION

if __name__ == "__main__":

    np.random.seed(42)

    
    print("\n" + "=" * 60)
    print("WALK-FORWARD: h=0 (NOWCAST)")
    print("=" * 60)

    df_lgbm_h0, df_probit_h0 = run_walk_forward(horizon=0)
    print(f"\nPredictions generated: {len(df_lgbm_h0)}")

    
    print("\nCalibrating probabilities (isotonic regression)...")
    df_lgbm_h0["prob_cal"]   = calibrate_isotonic(df_lgbm_h0)
    df_probit_h0["prob_cal"] = calibrate_isotonic(df_probit_h0)
    print("  Done.")

    
    print("\nBuilding stacked ensemble...")
    df_ensemble = build_ensemble(df_lgbm_h0, df_probit_h0)

    
    print("\n" + "=" * 60)
    print("EVALUATION — h=0 (NOWCAST)")
    print("=" * 60)

    brier_lgbm,     prauc_lgbm,     rocauc_lgbm,     far_lgbm     = evaluate(df_lgbm_h0,   "LightGBM (main)")
    brier_ens,      prauc_ens,      rocauc_ens,       far_ens      = evaluate(df_ensemble,  "Stacked Ensemble")
    brier_probit,   prauc_probit,   rocauc_probit,    far_probit   = evaluate(df_probit_h0, "Probit benchmark")

    
    print("\n--- Bootstrap 95% CI (block bootstrap, block=12) ---")
    for label, df_m, prauc, rocauc in [
        ("LightGBM",          df_lgbm_h0,  prauc_lgbm,   rocauc_lgbm),
        ("Stacked Ensemble",  df_ensemble, prauc_ens,    rocauc_ens),
        ("Probit benchmark",  df_probit_h0, prauc_probit, rocauc_probit),
    ]:
        p = df_m["prob_cal"].values
        a = df_m["actual"].astype(int).values

        pr_lo, pr_hi  = block_bootstrap(p, a, average_precision_score)
        roc_lo, roc_hi = block_bootstrap(p, a, roc_auc_score)

        print(f"  {label:<22}  "
              f"PR-AUC={prauc:.4f} [{pr_lo:.3f}, {pr_hi:.3f}]  "
              f"ROC-AUC={rocauc:.4f} [{roc_lo:.3f}, {roc_hi:.3f}]")

    
    
    print("\n--- Threshold Sweep (LightGBM) ---")
    p_ts = df_lgbm_h0["prob_cal"].values
    a_ts = df_lgbm_h0["actual"].astype(int).values
    thresh_df = threshold_sweep(p_ts, a_ts)
    print(thresh_df.to_string(index=False))
    thresh_df.to_csv("Recession-Nowcasting/data/results/threshold_sweep.csv", index=False)

   
    print(f"\n--- Lead Time vs NBER Announcement (threshold={SIGNAL_THRESHOLD}) ---")
    lead_df  = lead_time_analysis(df_lgbm_h0)
    avg_lead = lead_df["months_before"].dropna().mean()
    print(lead_df.to_string(index=False))
    print(f"\n  Average lead: {avg_lead:.1f} months before NBER announcement")
    lead_df.to_csv("Recession-Nowcasting/data/results/lead_time.csv", index=False)

    
    print("\n--- Reliability Diagram (LightGBM) ---")
    rel_df = reliability_stats(p_ts, a_ts, n_bins=10)
    print(rel_df.to_string(index=False))
    print("  (mean_pred ≈ mean_obs = well calibrated)")
    rel_df.to_csv("Recession-Nowcasting/data/results/reliability.csv", index=False)

    
    print("\n" + "=" * 60)
    print("MULTI-HORIZON EVALUATION  (h = 1, 3, 6)")
    print("=" * 60)

    horizon_rows = [{
        "horizon": 0,
        "prauc":   round(prauc_lgbm, 4),
        "rocauc":  round(rocauc_lgbm, 4),
        "brier":   round(brier_lgbm["BS_overall"], 4),
        "brier_rec": round(brier_lgbm["BS_recession"], 4),
    }]

    for h in [1, 3, 6]:
        print(f"\n  Running h={h}...")
        df_h, _ = run_walk_forward(horizon=h)
        df_h["prob_cal"] = calibrate_isotonic(df_h)
        p_h = df_h["prob_cal"].values
        a_h = df_h["actual"].astype(int).values

        prauc_h  = average_precision_score(a_h, p_h)
        rocauc_h = roc_auc_score(a_h, p_h)
        brier_h  = brier_decomp(p_h, a_h)

        print(f"    PR-AUC={prauc_h:.4f}  ROC-AUC={rocauc_h:.4f}  "
              f"Brier={brier_h['BS_overall']:.4f}")

        horizon_rows.append({
            "horizon":   h,
            "prauc":     round(prauc_h,  4),
            "rocauc":    round(rocauc_h, 4),
            "brier":     round(brier_h["BS_overall"],  4),
            "brier_rec": round(brier_h["BS_recession"], 4),
        })

    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv("Recession-Nowcasting/data/results/metrics_by_horizon.csv", index=False)
    print("\n--- Horizon Decay Table ---")
    print(horizon_df.to_string(index=False))

    
    feature_cols_clean = [c for c in feature_cols
                          if "MSM" not in c and "HMM" not in c]
    all_top_features = run_shap_analysis(feature_cols_clean)

   
    df_lgbm_h0["model"]   = "lgbm"
    df_probit_h0["model"] = "probit"
    df_ensemble["model"]  = "ensemble"

    pd.concat([df_lgbm_h0, df_probit_h0, df_ensemble]).to_csv(
        "Recession-Nowcasting/data/results/predictions_all.csv")
    df_lgbm_h0.to_csv("Recession-Nowcasting/data/results/predictions_full.csv")
    df_ensemble.to_csv("Recession-Nowcasting/data/results/predictions_ensemble.csv")

    metrics = pd.DataFrame({
        "model":        ["lgbm", "ensemble", "probit"],
        "ROC_AUC":      [rocauc_lgbm, rocauc_ens, rocauc_probit],
        "PR_AUC":       [prauc_lgbm,  prauc_ens,  prauc_probit],
        "BS_overall":   [brier_lgbm["BS_overall"],  brier_ens["BS_overall"],  brier_probit["BS_overall"]],
        "BS_recession": [brier_lgbm["BS_recession"], brier_ens["BS_recession"], brier_probit["BS_recession"]],
        "BS_expansion": [brier_lgbm["BS_expansion"], brier_ens["BS_expansion"], brier_probit["BS_expansion"]],
        "false_alarm":  [far_lgbm,    far_ens,    far_probit],
    })
    metrics.to_csv("Recession-Nowcasting/data/results/metrics_summary.csv", index=False)

    
    print("\n" + "=" * 60)
    print("CURRENT RECESSION PROBABILITY")
    print("=" * 60)

    latest      = df_lgbm_h0["prob_cal"].dropna()
    latest_date = latest.index[-1]
    latest_prob = latest.iloc[-1]
    prev        = latest.iloc[-2] if len(latest) > 1 else np.nan
    chg_1m      = latest_prob - prev if not np.isnan(prev) else np.nan

    print(f"\n  Date        : {latest_date.strftime('%Y-%m')}")
    print(f"  Probability : {latest_prob:.1%}")
    if not np.isnan(chg_1m):
        print(f"  1-month chg : {chg_1m:+.1%}")

    print("\n  6-month trend:")
    for date, prob in latest.tail(6).items():
        bar = "" * int(prob * 20)
        print(f"    {date.strftime('%Y-%m')}  {prob:.1%}  {bar}")

    if latest_prob >= 0.50:
        risk = "HIGH — model signals elevated recession risk"
    elif latest_prob >= 0.35:
        risk = "ELEVATED — approaching signal threshold"
    elif latest_prob >= 0.15:
        risk = "MODERATE — worth monitoring"
    else:
        risk = "LOW"
    print(f"\n  Risk level  : {risk}")

   

# Train final model on ALL available data (for live deployment)
    print("\nTraining final model on full dataset for deployment...")
    X_final = X_all.copy()
    y_final = y_all.copy()

    valid_mask = y_final.notna()
    X_final = X_final[valid_mask]
    y_final = y_final[valid_mask].astype(int)

    X_final_scaled, _, final_scaler = normalize(X_final, X_final)
    spw_final = get_class_weight(y_final)
    final_model = train_lgbm(X_final_scaled, y_final, spw_final)

    # Save model + scaler + feature list (all 3 needed for inference)
    os.makedirs("Recession-Nowcasting/model", exist_ok=True)
    joblib.dump(final_model,   "Recession-Nowcasting/model/lgbm_final.joblib")
    joblib.dump(final_scaler,  "Recession-Nowcasting/model/scaler_final.joblib")
    joblib.dump(feature_cols,  "Recession-Nowcasting/model/feature_cols.joblib")

    print("  Saved Recession-Nowcasting/model/lgbm_final.joblib")
    print("  Saved Recession-Nowcasting/model/scaler_final.joblib")
    print("  Saved Recession-Nowcasting/model/feature_cols.joblib")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — FINAL RESULTS")
    print("=" * 60)

    print(f"\n{'Model':<22} {'ROC-AUC':>8} {'PR-AUC':>8} "
          f"{'BS_all':>8} {'BS_rec':>8} {'FalseAlm':>9}")
    print("-" * 67)
    for _, row in metrics.iterrows():
        print(f"  {row['model']:<20} {row['ROC_AUC']:>8.4f} {row['PR_AUC']:>8.4f} "
              f"{row['BS_overall']:>8.4f} {row['BS_recession']:>8.4f} "
              f"{row['false_alarm']:>8.1f}%")

    print(f"\nLead time vs NBER  : {avg_lead:.1f} months avg  (threshold={SIGNAL_THRESHOLD})")
    print(f"Current probability: {latest_prob:.1%}  ({latest_date.strftime('%Y-%m')})")

    print("\nOutput files:")
    outputs = [
        "Recession-Nowcasting/data/results/predictions_full.csv",
        "Recession-Nowcasting/data/results/predictions_ensemble.csv",
        "Recession-Nowcasting/data/results/predictions_all.csv",
        "Recession-Nowcasting/data/results/metrics_summary.csv",
        "Recession-Nowcasting/data/results/metrics_by_horizon.csv",
        "Recession-Nowcasting/data/results/threshold_sweep.csv",
        "Recession-Nowcasting/data/results/lead_time.csv",
        "Recession-Nowcasting/data/results/reliability.csv",
        "Recession-Nowcasting/data/results/shap_values_2001.csv",
        "Recession-Nowcasting/data/results/shap_values_2008.csv",
        "Recession-Nowcasting/data/results/shap_values_2020.csv",
    ]
    for f in outputs:
        print(f"  {f}")
    print("=" * 60)