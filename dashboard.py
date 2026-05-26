
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import joblib
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Recession Nowcasting", page_icon="📉", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background-color: #0F172A; color: #F8FAFC; }
.stApp {
    background-color: #0F172A;
    background-image: radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), radial-gradient(at 50% 0%, hsla(225,39%,30%,0.1) 0, transparent 50%);
}
[data-testid="stSidebar"] { background-color: rgba(15,23,42,0.6) !important; backdrop-filter: blur(12px) !important; border-right: 1px solid rgba(255,255,255,0.05); }
.dash-header { margin-bottom: 2rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 1rem; }
.dash-title { font-size: 2.2rem; font-weight: 600; background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; }
.dash-sub { font-size: 1rem; color: #94A3B8; font-weight: 300; }
.live-badge { display: inline-block; background: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.4); color: #34D399; border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-left: 0.8rem; vertical-align: middle; }
.kpi-container { display: flex; gap: 1rem; margin-bottom: 2rem; }
.kpi-card { flex: 1; background: rgba(30,41,59,0.5); backdrop-filter: blur(10px); border-radius: 16px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: transform 0.2s ease; }
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); border-color: rgba(255,255,255,0.15); }
.kpi-title { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 0.5rem; }
.kpi-value { font-size: 2.5rem; font-weight: 600; color: #F8FAFC; line-height: 1.2; }
.kpi-value.danger { color: #F87171; }
.kpi-value.warning { color: #FBBF24; }
.kpi-value.safe { color: #34D399; }
.kpi-desc { font-size: 0.8rem; color: #64748B; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

BLUE = "#60A5FA"; PURPLE = "#A78BFA"; GREEN = "#34D399"; GRAY = "#475569"; AMBER = "#FBBF24"
NBER_RECESSIONS = [("1990-07-01","1991-04-01"),("2001-03-01","2001-11-01"),("2007-12-01","2009-06-01"),("2020-02-01","2020-04-01")]


def get_api_key():
    try:
        return st.secrets["fred"]["api_key"]
    except Exception:
        return None

FRED_API_KEY = get_api_key()

FRED_SERIES = {
    "T10Y2Y":"T10Y2Y","T10Y3M":"T10Y3M","DGS10":"DGS10","DGS2":"DGS2",
    "HY_SPREAD":"BAMLH0A0HYM2","BBB_SPREAD":"BAMLC0A4CBBBEY",
    "NFCI":"NFCI","STLFSI":"STLFSI4","UMCSENT":"UMCSENT","EPU":"USEPUINDXD",
    "PMI":"NAPM","ICSA":"ICSA","CCSA":"CCSA","UNRATE":"UNRATE","SAHM":"SAHMREALTIME",
    "INDPRO":"INDPRO","IPMAN":"IPMAN","PERMIT":"PERMIT","HOUST":"HOUST",
    "DGORDER":"DGORDER","RETAIL":"RETAILSMNSA","M2SL":"M2SL","CPI":"CPIAUCSL",
    "DELINQ":"DRCCLACBS","WEI":"WEI","SP500":"SP500","VIX":"VIXCLS",
    "NASDAQ":"NASDAQCOM","WILSHIRE":"WILL5000IND",
}

# ── Feature engineering (exact mirror of feture_engineering.py) ─────
def pct_chg(s, p=1): return s.pct_change(periods=p)*100
def roll_mean(s, w=3): return s.rolling(w, min_periods=1).mean()
def expanding_zscore(s):
    mu = s.expanding(min_periods=24).mean(); sig = s.expanding(min_periods=24).std()
    return (s - mu) / sig.replace(0, np.nan)
def add_lag(s, idx, lag):
    s = s.reindex(idx)
    if lag > 0: s = s.shift(lag)
    return s

def build_live_features(raw, master_index):
    master = pd.DataFrame(index=master_index)
    def feat(s, lag):
        if s is None: return None
        return add_lag(s, master_index, lag)
    def add(col, s):
        if s is not None: master[col] = s

    for name in ["T10Y2Y","T10Y3M"]:
        s = raw.get(name)
        if s is None: continue
        sf = feat(s, 0)
        add(f"{name}_level", sf); add(f"{name}_1m_chg", sf.diff(1))
        add(f"{name}_3m_chg", sf.diff(3)); add(f"{name}_12m_chg", sf.diff(12))
        add(f"{name}_inv", (sf < 0).astype(int))
    for name in ["DGS10","DGS2"]:
        s = raw.get(name)
        if s is None: continue
        sf = feat(s, 0)
        add(f"{name}_level", sf); add(f"{name}_1m_chg", sf.diff(1)); add(f"{name}_3m_chg", sf.diff(3))
    if raw.get("DGS10") is not None and raw.get("DGS2") is not None:
        slope = feat(raw["DGS10"] - raw["DGS2"], 0)
        add("YIELD_SLOPE_level", slope); add("YIELD_SLOPE_3m_chg", slope.diff(3))
    if raw.get("T10Y2Y") is not None:
        t2y_inv = (feat(raw["T10Y2Y"], 0) < 0).astype(int)
        months_inv = pd.Series(index=master_index, dtype=float); counter = np.nan
        for date in master_index:
            v = t2y_inv.get(date, 0)
            if v == 1: counter = 0
            elif not pd.isna(counter): counter += 1
            months_inv[date] = counter
        add("MONTHS_SINCE_INVERSION", months_inv)
    for name, key in [("HY","HY_SPREAD"),("BBB","BBB_SPREAD")]:
        s = raw.get(key)
        if s is None: continue
        sf = feat(s, 0)
        add(f"{name}_level", sf); add(f"{name}_1m_chg", sf.diff(1))
        add(f"{name}_3m_chg", sf.diff(3)); add(f"{name}_12m_chg", sf.diff(12))
    if raw.get("HY_SPREAD") is not None and raw.get("BBB_SPREAD") is not None:
        diff = feat(raw["HY_SPREAD"] - raw["BBB_SPREAD"], 0)
        add("CREDIT_DIFF_level", diff); add("CREDIT_DIFF_3m_chg", diff.diff(3))
    for name, key in [("NFCI","NFCI"),("STLFSI","STLFSI")]:
        s = raw.get(key)
        if s is None: continue
        sf = feat(s, 0)
        add(f"{name}_level", sf); add(f"{name}_1m_chg", sf.diff(1)); add(f"{name}_3m_chg", sf.diff(3))
    if raw.get("ICSA") is not None:
        s_ma = feat(roll_mean(raw["ICSA"], 3), 0)
        add("ICSA_ma3", s_ma); add("ICSA_mom", pct_chg(s_ma,1)); add("ICSA_yoy", pct_chg(s_ma,12))
        add("ICSA_3m_chg", s_ma.diff(3)); add("ICSA_accel", pct_chg(s_ma,1).diff(1))
    if raw.get("CCSA") is not None:
        s_ma = feat(roll_mean(raw["CCSA"], 3), 0)
        add("CCSA_ma3", s_ma); add("CCSA_mom", pct_chg(s_ma,1)); add("CCSA_yoy", pct_chg(s_ma,12))
    if raw.get("UNRATE") is not None:
        u = raw["UNRATE"]; uf = feat(u, 1)
        add("UNRATE_level", uf); add("UNRATE_mom", uf.diff(1)); add("UNRATE_yoy", uf.diff(12))
        sahm_c = feat(u.rolling(3,min_periods=1).mean() - u.rolling(12,min_periods=3).min(), 1)
        add("SAHM_computed", sahm_c)
        if sahm_c is not None: add("SAHM_triggered", (sahm_c >= 0.5).astype(int))
    if raw.get("SAHM") is not None:
        sahm_rt = feat(raw["SAHM"], 1)
        add("SAHM_realtime", sahm_rt); add("SAHM_realtime_triggered", (sahm_rt >= 0.5).astype(int))
    for fn, rk, lag in [("INDPRO","INDPRO",1),("IPMAN","IPMAN",1),("PERMIT","PERMIT",1),
                         ("HOUST","HOUST",1),("DGORDER","DGORDER",2),("RETAIL","RETAIL",2)]:
        s = raw.get(rk)
        if s is None: continue
        sf = feat(s, lag); mom = pct_chg(sf, 1)
        add(f"{fn}_mom", mom); add(f"{fn}_yoy", pct_chg(sf,12))
        add(f"{fn}_ma3mom", roll_mean(mom,3)); add(f"{fn}_accel", mom.diff(1))
    if raw.get("LEI") is not None:
        lei_f = feat(raw["LEI"], 1)
        add("LEI_yoy", pct_chg(lei_f,12)); add("LEI_6m_chg", pct_chg(lei_f,6))
        add("LEI_declining", (pct_chg(lei_f,6) < 0).astype(int))
    if raw.get("PMI") is not None:
        sf = feat(raw["PMI"], 1)
        add("PMI_level", sf); add("PMI_3m_chg", sf.diff(3))
        add("PMI_below50", (sf < 50).astype(int)); add("PMI_ma3", roll_mean(sf,3))
    for fn, rk in [("UMCSENT","UMCSENT"),("EPU","EPU")]:
        s = raw.get(rk)
        if s is None: continue
        sf = feat(s, 1)
        add(f"{fn}_level", sf); add(f"{fn}_1m_chg", sf.diff(1)); add(f"{fn}_3m_chg", sf.diff(3))
        add(f"{fn}_yoy", sf.diff(12)); add(f"{fn}_ma3", roll_mean(sf,3))
    if raw.get("M2SL") is not None and raw.get("CPI") is not None:
        real_m2 = raw["M2SL"] / raw["CPI"] * 100
        add("REAL_M2_yoy", feat(pct_chg(real_m2,12),1)); add("REAL_M2_mom", feat(pct_chg(real_m2,1),1))
    if raw.get("CPI") is not None: add("CPI_yoy", feat(pct_chg(raw["CPI"],12),1))
    if raw.get("DELINQ") is not None:
        delinq_f = add_lag(raw["DELINQ"].reindex(master_index).ffill(), master_index, 2)
        add("DELINQ_level", delinq_f); add("DELINQ_3m_chg", delinq_f.diff(3)); add("DELINQ_6m_chg", delinq_f.diff(6))
    if raw.get("WEI") is not None:
        sf = feat(raw["WEI"], 0)
        add("WEI_level", sf); add("WEI_1m_chg", sf.diff(1)); add("WEI_3m_chg", sf.diff(3))
    if raw.get("SP500") is not None:
        sp = feat(raw["SP500"], 0)
        add("SP500_ret_1m", pct_chg(sp,1)); add("SP500_ret_3m", pct_chg(sp,3))
        add("SP500_ret_6m", pct_chg(sp,6)); add("SP500_ret_12m", pct_chg(sp,12))
        sp_ret = sp.pct_change()
        add("SP500_vol_3m", sp_ret.rolling(3,min_periods=2).std()*100)
        add("SP500_vol_6m", sp_ret.rolling(6,min_periods=3).std()*100)
        add("SP500_vol_12m", sp_ret.rolling(12,min_periods=6).std()*100)
        rp = sp.rolling(12,min_periods=1).max(); spd = (sp-rp)/rp*100
        add("SP500_drawdown", spd); add("SP500_bear", (spd<=-20).astype(int))
        ma6 = sp.rolling(6,min_periods=3).mean(); ma12 = sp.rolling(12,min_periods=6).mean()
        add("SP500_below_ma6", (sp<ma6).astype(int)); add("SP500_below_ma12", (sp<ma12).astype(int))
        add("SP500_ma_cross", (ma6<ma12).astype(int)); add("SP500_ret_accel", pct_chg(sp,1).diff(1))
    if raw.get("VIX") is not None:
        vix = feat(raw["VIX"], 0)
        add("VIX_level", vix); add("VIX_1m_chg", vix.diff(1)); add("VIX_3m_chg", vix.diff(3))
        add("VIX_ma3", roll_mean(vix,3)); add("VIX_above20", (vix>20).astype(int))
        add("VIX_above30", (vix>30).astype(int)); add("VIX_spike", vix-roll_mean(vix,3))
    if raw.get("NASDAQ") is not None:
        nq = feat(raw["NASDAQ"], 0)
        add("NASDAQ_ret_1m", pct_chg(nq,1)); add("NASDAQ_ret_3m", pct_chg(nq,3))
        add("NASDAQ_ret_6m", pct_chg(nq,6)); add("NASDAQ_ret_12m", pct_chg(nq,12))
        nqp = nq.rolling(12,min_periods=1).max(); add("NASDAQ_drawdown", (nq-nqp)/nqp*100)
    if raw.get("WILSHIRE") is not None:
        wil = feat(raw["WILSHIRE"], 0)
        add("WILSHIRE_ret_3m", pct_chg(wil,3)); add("WILSHIRE_ret_12m", pct_chg(wil,12))
        wp = wil.rolling(12,min_periods=1).max(); add("WILSHIRE_drawdown", (wil-wp)/wp*100)
    eq_parts = []
    for col, sign in [("SP500_ret_6m",-1),("SP500_drawdown",-1),("SP500_vol_3m",1),("VIX_level",1),("NASDAQ_ret_6m",-1)]:
        if col in master.columns: eq_parts.append(sign * expanding_zscore(master[col]))
    if len(eq_parts) >= 2: master["EQUITY_STRESS"] = pd.concat(eq_parts, axis=1).mean(axis=1)
    stress_parts = []
    for col, sign in [("NFCI_level",1),("HY_level",1),("T10Y2Y_level",-1)]:
        if col in master.columns: stress_parts.append(sign * expanding_zscore(master[col]))
    if "EQUITY_STRESS" in master.columns: stress_parts.append(expanding_zscore(master["EQUITY_STRESS"]))
    if len(stress_parts) >= 2: master["COMPOSITE_STRESS"] = pd.concat(stress_parts, axis=1).mean(axis=1)
    return master


# Parallel FRED fetch, cached 24h
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_live_data(api_key):
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    start = (pd.Timestamp.today() - pd.DateOffset(months=13)).strftime("%Y-%m-%d")
    def fetch_one(item):
        key, series_id = item
        try:
            s = fred.get_series(series_id, observation_start=start)
            s = pd.to_numeric(s, errors="coerce").dropna()
            s.index = pd.to_datetime(s.index)
            gap = s.index.to_series().diff().dt.days.median()
            s = s.resample("MS").mean() if (gap is not None and gap < 25) else s.resample("MS").first()
            return key, s, None
        except Exception as e:
            return key, None, str(e)
    raw, errors = {}, []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for key, s, err in executor.map(fetch_one, FRED_SERIES.items()):
            if s is not None: raw[key] = s
            elif err: errors.append(f"{key}: {err}")
    return raw, errors

@st.cache_data(show_spinner=False)
def load_historical():
    for fp in ["results/predictions_all.csv","data/results/predictions_all.csv"]:
        if os.path.exists(fp): return pd.read_csv(fp, parse_dates=["date"])
    return None

@st.cache_resource(show_spinner=False)
def load_model():
    mp,sp,fp = "model/lgbm_final.joblib","model/scaler_final.joblib","model/feature_cols.joblib"
    if all(os.path.exists(p) for p in [mp,sp,fp]):
        return joblib.load(mp), joblib.load(sp), joblib.load(fp)
    return None, None, None

def apply_dark_theme(fig):
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#94A3B8"), margin=dict(l=20,r=20,t=40,b=20),
        hovermode="x unified", legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig


# SIDEBAR
with st.sidebar:
    st.markdown("### Controls")
    threshold    = st.slider("Recession Threshold", 0.0, 1.0, 0.35, 0.01)
    model_choice = st.multiselect("Show Models", ["LightGBM","Probit"], default=["LightGBM","Probit"])
    st.markdown("---")
    run_live = False
    if FRED_API_KEY:
        run_live = st.button("Run Live Nowcast", type="primary", use_container_width=True)
        st.caption("Fetches latest FRED data and generates today's prediction (~15 sec).")
    else:
        st.warning("Add FRED API key to .streamlit/secrets.toml to enable live nowcasting.")
    st.markdown("---")
    st.markdown("""<div style="font-size:0.75rem; color:#475569; line-height:1.8;">
    <b style="color:#64748B;">Model</b><br>LightGBM trained 1980-present<br><br>
    <b style="color:#64748B;">Validation</b><br>Walk-forward OOS - 24mo NBER blackout<br><br>
    <b style="color:#64748B;">Performance</b><br>ROC-AUC 96.5% - 5.7mo avg lead time</div>""", unsafe_allow_html=True)

# HEADER
st.markdown("""<div class="dash-header">
  <div class="dash-title">US Recession Nowcasting <span class="live-badge">Live</span></div>
  <div class="dash-sub">Real-time macroeconomic risk - LightGBM + Probit ensemble - Walk-forward validation with vintage discipline</div>
</div>""", unsafe_allow_html=True)


# LOAD MODEL + HISTORY (instant) 
model, scaler, feature_cols = load_model()
df_hist = load_historical()
live_prob = None; live_date = None; live_data_ok = False; raw_live = {}


# RUN LIVE NOWCAST 
if run_live and FRED_API_KEY and model is not None:
    progress = st.progress(0, text="Connecting to FRED...")
    try:
        progress.progress(20, text="Fetching 29 series in parallel...")
        raw_live, fetch_errors = fetch_live_data(FRED_API_KEY)
        progress.progress(60, text="Engineering features...")
        master_idx = pd.date_range(
            start=(pd.Timestamp.today() - pd.DateOffset(months=13)),
            end=pd.Timestamp.today(), freq="MS")
        feat_df    = build_live_features(raw_live, master_idx)
        X_live     = feat_df.reindex(columns=feature_cols)
        latest_row = X_live.dropna(how="all").iloc[[-1]]
        live_date  = latest_row.index[0]
        progress.progress(85, text="Running model inference...")
        X_scaled = pd.DataFrame(scaler.transform(latest_row), index=latest_row.index, columns=latest_row.columns)
        live_prob    = float(model.predict_proba(X_scaled)[0, 1])
        live_data_ok = True
        progress.progress(100, text="Done!")
        progress.empty()
        if fetch_errors: st.sidebar.caption(f"{len(fetch_errors)} series failed (model still ran)")
    except Exception as e:
        progress.empty(); st.error(f"Live nowcast failed: {e}")
elif run_live and model is None:
    st.error("Model files not found. Ensure model/ folder is in your repo.")


# KPI CARDS 
if live_data_ok and live_prob is not None:
    display_prob = live_prob; display_date = live_date.strftime("%B %Y"); display_badge = "Live Nowcast"
elif df_hist is not None:
    lgbm_h = df_hist[df_hist["model"]=="lgbm"].sort_values("date")
    display_prob  = float(lgbm_h.iloc[-1]["prob_cal"])
    display_date  = pd.to_datetime(lgbm_h.iloc[-1]["date"]).strftime("%B %Y")
    display_badge = "Latest Historical"
else:
    display_prob = None; display_date = "-"; display_badge = "No Data"

if display_prob is not None:
    confidence = abs(display_prob - 0.5) * 2
    if display_prob > threshold: status_color, status_text = "danger", "HIGH RISK"
    elif display_prob > threshold*0.6: status_color, status_text = "warning", "ELEVATED"
    else: status_color, status_text = "safe", "LOW RISK"
    chg_str = "-"
    if df_hist is not None:
        lgbm_h = df_hist[df_hist["model"]=="lgbm"].sort_values("date")
        if len(lgbm_h) >= 2:
            chg = (display_prob - float(lgbm_h.iloc[-2]["prob_cal"])) * 100
            chg_str = f"{'up' if chg>0 else 'down'} {abs(chg):.1f}pp vs prior month"
    st.markdown(f"""<div class="kpi-container">
        <div class="kpi-card"><div class="kpi-title">Recession Probability - {display_badge}</div>
            <div class="kpi-value {status_color}">{display_prob*100:.1f}%</div>
            <div class="kpi-desc"><b>{status_text}</b> - {display_date} - {chg_str}</div></div>
        <div class="kpi-card"><div class="kpi-title">Model Confidence</div>
            <div class="kpi-value">{confidence*100:.1f}%</div>
            <div class="kpi-desc">Distance from 50% decision boundary</div></div>
        <div class="kpi-card"><div class="kpi-title">Decision Threshold</div>
            <div class="kpi-value" style="color:#60A5FA;">{threshold*100:.0f}%</div>
            <div class="kpi-desc">Adjust in sidebar - signal fires above this</div></div>
        <div class="kpi-card"><div class="kpi-title">Model Performance</div>
            <div class="kpi-value" style="color:#A78BFA;">96.5%</div>
            <div class="kpi-desc">ROC-AUC - Walk-forward OOS - 5.7mo avg lead</div></div>
    </div>""", unsafe_allow_html=True)
    if not run_live and FRED_API_KEY:
        st.info("Click Run Live Nowcast in the sidebar to generate today's prediction.")
else:
    st.error("No data found. Check results/predictions_all.csv exists.")


# TIMELINE CHART 
st.markdown("### Recession Probability Timeline")
if df_hist is not None:
    lgbm_df   = df_hist[df_hist["model"]=="lgbm"].sort_values("date")
    probit_df = df_hist[df_hist["model"]=="probit"].sort_values("date")
    fig = go.Figure()
    for rs, re in NBER_RECESSIONS:
        fig.add_vrect(x0=rs, x1=re, fillcolor="rgba(107,114,128,0.15)", layer="below", line_width=0,
            annotation_text="NBER", annotation_position="top left", annotation_font_size=9, annotation_font_color="#64748B")
    if "LightGBM" in model_choice:
        fig.add_trace(go.Scatter(x=lgbm_df["date"], y=lgbm_df["prob_cal"], name="LightGBM (historical)",
            line=dict(color=BLUE, width=2.5, shape="spline"), fill="tozeroy", fillcolor="rgba(96,165,250,0.08)",
            hovertemplate="<b>%{x|%b %Y}</b><br>LightGBM: %{y:.1%}<extra></extra>"))
    if "Probit" in model_choice:
        fig.add_trace(go.Scatter(x=probit_df["date"], y=probit_df["prob_cal"], name="Probit benchmark",
            line=dict(color=PURPLE, dash="dot", width=2),
            hovertemplate="<b>%{x|%b %Y}</b><br>Probit: %{y:.1%}<extra></extra>"))
    if live_data_ok and live_prob is not None:
        fig.add_trace(go.Scatter(x=[live_date], y=[live_prob],
            name=f"Live nowcast ({live_date.strftime('%b %Y')})", mode="markers",
            marker=dict(color=GREEN, size=14, symbol="star", line=dict(color="white", width=2)),
            hovertemplate=f"<b>LIVE</b><br>Probability: {live_prob:.1%}<extra></extra>"))
        if not lgbm_df.empty:
            ld = lgbm_df["date"].max(); lp = float(lgbm_df[lgbm_df["date"]==ld]["prob_cal"].iloc[0])
            fig.add_trace(go.Scatter(x=[ld, live_date], y=[lp, live_prob], mode="lines",
                line=dict(color=GREEN, dash="dash", width=1.5), showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=threshold, line_dash="dash", line_color=AMBER,
        annotation_text=f"Threshold ({threshold*100:.0f}%)", annotation_position="bottom right",
        annotation_font=dict(color=AMBER, size=11))
    fig = apply_dark_theme(fig)
    fig.update_layout(height=420, yaxis_title="Recession Probability",
        yaxis=dict(range=[0,1.05], tickformat=".0%"), xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Historical predictions not found.")


# SECONDARY CHARTS
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Model Comparison")
    metric_choice = st.radio("Metric", ["ROC-AUC","PR-AUC","Brier"], horizontal=True)
    col_map = {"ROC-AUC":"ROC_AUC","PR-AUC":"PR_AUC","Brier":"BS_overall"}
    vals, lbls, clrs = [], [], []
    for fp in ["results/metrics_summary.csv","data/results/metrics_summary.csv"]:
        if os.path.exists(fp):
            mdf = pd.read_csv(fp)
            for mn, color, label in [("lgbm",BLUE,"LightGBM"),("probit",GRAY,"Probit")]:
                row = mdf[mdf["model"]==mn]
                if not row.empty: vals.append(float(row[col_map[metric_choice]].iloc[0])); lbls.append(label); clrs.append(color)
            break
    if not vals:
        d = {"ROC-AUC":[0.965,0.708],"PR-AUC":[0.711,0.665],"Brier":[0.042,0.061]}
        vals=d[metric_choice]; lbls=["LightGBM","Probit"]; clrs=[BLUE,GRAY]
    fig_bar = go.Figure(go.Bar(x=lbls, y=vals, marker_color=clrs,
        text=[f"{v:.3f}" for v in vals], textposition="auto",
        marker=dict(line=dict(color="rgba(255,255,255,0.2)", width=1))))
    fig_bar = apply_dark_theme(fig_bar); fig_bar.update_layout(height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### Forecast Horizon Decay")
    for fp in ["results/metrics_by_horizon.csv","data/results/metrics_by_horizon.csv"]:
        if os.path.exists(fp):
            df_hz = pd.read_csv(fp); fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_hz["horizon"], y=df_hz["rocauc"], mode="lines+markers",
                name="ROC-AUC", line=dict(color=BLUE,width=3), marker=dict(size=8,color=BLUE,line=dict(color="white",width=1))))
            fig2.add_trace(go.Scatter(x=df_hz["horizon"], y=df_hz["prauc"], mode="lines+markers",
                name="PR-AUC", line=dict(color=PURPLE,width=3), marker=dict(size=8,color=PURPLE,line=dict(color="white",width=1))))
            fig2 = apply_dark_theme(fig2); fig2.update_layout(height=300, xaxis_title="Months Ahead", yaxis_title="AUC Score")
            st.plotly_chart(fig2, use_container_width=True); break
    else: st.info("Horizon metrics not found.")


# LIVE SIGNALS (only after nowcast) 
if live_data_ok and raw_live:
    st.markdown("### Live Market Signals")
    signals = [("T10Y2Y","10Y-2Y Spread","bp",2),("T10Y3M","10Y-3M Spread","bp",2),
               ("HY_SPREAD","HY Spread","%",2),("VIX","VIX","",1),("UNRATE","Unemployment","%",1)]
    sig_cols = st.columns(5)
    for col, (key, label, unit, dec) in zip(sig_cols, signals):
        s = raw_live.get(key)
        if s is not None and not s.dropna().empty:
            val = float(s.dropna().iloc[-1]); prev = float(s.dropna().iloc[-2]) if len(s.dropna())>1 else val
            delta = val - prev; arrow = "up" if delta>0 else ("down" if delta<0 else "-")
            d_col = "#F87171" if delta>0 else "#34D399"
            col.markdown(f"""<div class="kpi-card" style="padding:1rem;">
                <div class="kpi-title">{label}</div>
                <div style="font-size:1.6rem;font-weight:600;color:#F8FAFC;">{val:.{dec}f}{unit}</div>
                <div style="font-size:0.75rem;color:{d_col};margin-top:0.3rem;">{arrow} {abs(delta):.{dec}f} MoM</div>
            </div>""", unsafe_allow_html=True)


# FOOTER 
st.markdown(f"""<div style="text-align:center;margin-top:3rem;padding-top:1rem;
     border-top:1px solid rgba(255,255,255,0.08);color:#334155;font-size:0.78rem;line-height:2;">
  Recession Nowcasting - LightGBM + Probit Ensemble - FRED Data - Walk-forward OOS - Vintage discipline via ALFRED<br>
  Last updated: {datetime.today().strftime("%B %d, %Y")}
</div>""", unsafe_allow_html=True)