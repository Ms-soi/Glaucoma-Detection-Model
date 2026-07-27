"""
ICU Mortality Prediction System — Streamlit Demo App
Final Year Project · Computer Science
Author: [Your Name]

Run with:
    pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn shap plotly
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Mortality Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — dark clinical theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root palette ── */
:root {
    --teal-900: #04342C;
    --teal-800: #085041;
    --teal-600: #0F6E56;
    --teal-400: #1D9E75;
    --teal-200: #5DCAA5;
    --teal-100: #9FE1CB;
    --teal-050: #E1F5EE;
    --amber:    #EF9F27;
    --coral:    #D85A30;
    --sand:     #F1EFE8;
    --ink:      #2C2C2A;
    --muted:    #888780;
}

/* ── App background ── */
.stApp { background: var(--teal-900); }
[data-testid="stSidebar"] { background: #04342C !important; border-right: 1px solid var(--teal-600); }
[data-testid="stSidebar"] * { color: var(--teal-050) !important; }

/* ── Main content area ── */
.block-container { padding: 1.5rem 2rem 3rem 2rem; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #085041;
    border: 1px solid var(--teal-600);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="metric-container"] label { color: var(--teal-100) !important; font-size: 0.75rem; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--teal-050) !important; font-size: 1.8rem; font-weight: 700; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem; }

/* ── Input widgets in sidebar ── */
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stNumberInput > label { color: var(--teal-100) !important; font-size: 0.78rem; letter-spacing: 0.05em; }
[data-testid="stSidebar"] [data-baseweb="select"] { background: var(--teal-800); border-color: var(--teal-600); }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--teal-800);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    color: var(--teal-100);
    border-radius: 8px;
    font-size: 0.83rem;
    letter-spacing: 0.04em;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--teal-400) !important;
    color: white !important;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--teal-400);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    transition: background 0.2s;
}
.stButton > button:hover { background: var(--teal-600); color: white; }

/* ── Divider ── */
hr { border-color: var(--teal-600) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--teal-900); }
::-webkit-scrollbar-thumb { background: var(--teal-600); border-radius: 4px; }

/* ── Custom card ── */
.icu-card {
    background: #085041;
    border: 1px solid #0F6E56;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.icu-card h4 { color: #9FE1CB; font-size: 0.75rem; letter-spacing: 0.1em; margin: 0 0 0.3rem 0; text-transform: uppercase; }
.icu-card h2 { color: #E1F5EE; font-size: 2rem; font-weight: 700; margin: 0; font-family: 'Courier New', monospace; }
.icu-card p  { color: #5DCAA5; font-size: 0.78rem; margin: 0.2rem 0 0 0; }

/* ── Risk badge ── */
.badge-low    { background:#1D9E75; color:white; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
.badge-mod    { background:#EF9F27; color:white; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
.badge-high   { background:#D85A30; color:white; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }

/* ── Hero banner text ── */
.hero-title { font-size: 2.2rem; font-weight: 800; color: #E1F5EE; letter-spacing: -0.02em; margin: 0; font-family: Georgia, serif; }
.hero-sub   { font-size: 0.9rem; color: #5DCAA5; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FAKE MODEL & DATA  (replace with real trained model later)
# ─────────────────────────────────────────────────────────────
MODELS = {
    "XGBoost (Best)":          {"auroc": 0.893, "f1": 0.742, "acc": 0.871, "prec": 0.718, "rec": 0.768},
    "Random Forest":           {"auroc": 0.864, "f1": 0.711, "acc": 0.854, "prec": 0.698, "rec": 0.724},
    "Logistic Regression":     {"auroc": 0.814, "f1": 0.672, "acc": 0.829, "prec": 0.641, "rec": 0.706},
    "Support Vector Machine":  {"auroc": 0.831, "f1": 0.688, "acc": 0.841, "prec": 0.659, "rec": 0.720},
    "Decision Tree":           {"auroc": 0.744, "f1": 0.614, "acc": 0.798, "prec": 0.588, "rec": 0.643},
}

FEATURES = [
    "GCS Motor Score", "Serum Lactate", "Age",
    "Mean Arterial Pressure", "BUN", "Creatinine",
    "Respiratory Rate Max", "SpO2 Min", "Urine Output",
    "PaO2/FiO2 Ratio", "Platelet Count", "Heart Rate Max",
]

SHAP_VALS = [0.38, 0.29, 0.21, 0.18, 0.15, 0.13, 0.11, 0.09, 0.08, 0.06, 0.05, 0.04]

def fake_predict(inputs: dict) -> float:
    """Deterministic fake probability from input features."""
    score = 0.0
    score += max(0, (15 - inputs["gcs"]) * 0.04)
    score += min(inputs["lactate"] / 15, 0.3)
    score += min((inputs["age"] - 18) / 200, 0.2)
    score += max(0, (70 - inputs["map"]) / 200)
    score += min(inputs["bun"] / 120, 0.15)
    score += max(0, (12 - inputs["rr_min"]) / 80) + max(0, (inputs["rr_max"] - 30) / 80)
    score += max(0, (94 - inputs["spo2"]) / 60)
    score += max(0, (1.5 - inputs["creatinine_min"]) / -20) if inputs["creatinine_min"] > 1.5 else 0
    noise = (sum(inputs.values()) % 17) / 340
    return float(np.clip(score + noise, 0.03, 0.97))


def roc_curve_data(auroc: float):
    """Generate plausible fake ROC curve."""
    fpr = np.linspace(0, 1, 100)
    tpr = np.clip(fpr ** (1 / (2 * auroc - 0.6)), 0, 1)
    tpr = (tpr + np.random.default_rng(42).normal(0, 0.01, 100)).clip(0, 1)
    tpr[0], tpr[-1] = 0.0, 1.0
    return fpr, np.sort(tpr)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style='background:#085041;border-radius:12px;padding:16px 20px;margin-bottom:1.5rem;border:1px solid #0F6E56'>
        <div style='font-family:Georgia,serif;font-size:1.15rem;font-weight:700;color:#E1F5EE'>🫀 ICU Mortality</div>
        <div style='font-family:Georgia,serif;font-size:1.15rem;font-weight:700;color:#9FE1CB'>Predictor</div>
        <div style='font-size:0.68rem;letter-spacing:0.1em;color:#5DCAA5;margin-top:6px'>ML-POWERED · MIMIC-IV</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🧍 Patient Demographics")
    age     = st.slider("Age (years)", 18, 95, 65)
    sex     = st.selectbox("Sex", ["Male", "Female"])
    icu_type = st.selectbox("ICU Type", ["Medical ICU", "Surgical ICU", "Cardiac ICU", "Neuro ICU"])

    st.markdown("---")
    st.markdown("##### 🧠 Neurological")
    gcs = st.slider("GCS Total Score", 3, 15, 10)

    st.markdown("---")
    st.markdown("##### 💓 Haemodynamics (first 24h)")
    hr_max  = st.slider("Heart Rate Max (bpm)", 40, 200, 105)
    map_val = st.slider("Mean Arterial Pressure (mmHg)", 30, 130, 68)

    st.markdown("---")
    st.markdown("##### 🫁 Respiratory")
    rr_min  = st.slider("Resp Rate Min (breaths/min)", 5, 40, 14)
    rr_max  = st.slider("Resp Rate Max (breaths/min)", 5, 60, 26)
    spo2    = st.slider("SpO₂ Min (%)", 60, 100, 93)
    pf_ratio = st.slider("PaO₂/FiO₂ Ratio", 50, 500, 220)

    st.markdown("---")
    st.markdown("##### 🧪 Laboratory Values")
    lactate = st.slider("Serum Lactate (mmol/L)", 0.5, 15.0, 2.8, step=0.1)
    bun     = st.slider("BUN (mg/dL)", 5, 120, 28)
    creat   = st.slider("Creatinine Min (mg/dL)", 0.3, 10.0, 1.2, step=0.1)
    urine   = st.slider("Urine Output (mL/24h)", 0, 5000, 1200, step=50)
    platelets = st.slider("Platelet Count (×10³/µL)", 10, 600, 180)

    st.markdown("---")
    selected_model = st.selectbox("🤖 Active Model", list(MODELS.keys()))
    run_btn = st.button("▶  Run Prediction", use_container_width=True)

# Collect inputs
inputs = dict(age=age, gcs=gcs, hr_max=hr_max, map=map_val,
              rr_min=rr_min, rr_max=rr_max, spo2=spo2, pf_ratio=pf_ratio,
              lactate=lactate, bun=bun, creatinine_min=creat,
              urine=urine, platelets=platelets)

risk_prob = fake_predict(inputs)
risk_pct  = risk_prob * 100
model_stats = MODELS[selected_model]

if risk_pct < 25:
    risk_label, risk_class, risk_color = "LOW RISK", "badge-low", "#1D9E75"
elif risk_pct < 55:
    risk_label, risk_class, risk_color = "MODERATE RISK", "badge-mod", "#EF9F27"
else:
    risk_label, risk_class, risk_color = "HIGH RISK", "badge-high", "#D85A30"


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown("""
    <p class='hero-title'>ICU In-Hospital Mortality<br>Prediction System</p>
    <p class='hero-sub'>Classical Machine Learning · MIMIC-IV Database · Final Year Project</p>
    """, unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Prediction",
    "📊  Model Evaluation",
    "🔬  Feature Importance",
    "📋  About & Methods",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    if run_btn:
        with st.spinner("Running inference..."):
            time.sleep(0.6)

    # ── Top row: risk gauge + stats ──
    col_gauge, col_stats = st.columns([2, 3], gap="large")

    with col_gauge:
        # Plotly gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 46, "color": "#E1F5EE", "family": "Courier New"}},
            title={"text": "Mortality Risk", "font": {"size": 14, "color": "#9FE1CB"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#0F6E56",
                         "tickfont": {"color": "#5DCAA5", "size": 10}},
                "bar": {"color": risk_color, "thickness": 0.28},
                "bgcolor": "#085041",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],  "color": "#04342C"},
                    {"range": [25, 55], "color": "#063d31"},
                    {"range": [55, 100],"color": "#07483a"},
                ],
                "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.75, "value": risk_pct},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=10), height=260,
            font={"color": "#E1F5EE"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style='text-align:center;margin-top:-8px'>
            <span class='{risk_class}'>{risk_label}</span>
        </div>
        <div style='text-align:center;margin-top:10px;color:#5DCAA5;font-size:0.78rem'>
            Model: <b style='color:#9FE1CB'>{selected_model}</b>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.markdown("#### Patient Risk Summary")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Mortality Risk", f"{risk_pct:.1f}%",
                   delta=f"{risk_pct - 18.3:+.1f}% vs avg",
                   delta_color="inverse")
        mc2.metric("AUROC (model)", f"{model_stats['auroc']:.3f}")
        mc3.metric("F1-Score", f"{model_stats['f1']:.3f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Key clinical flags
        flags = []
        if gcs <= 8:       flags.append(("🔴", "Severe neurological impairment (GCS ≤ 8)"))
        if lactate > 4.0:  flags.append(("🔴", f"Hyperlactatemia: {lactate:.1f} mmol/L"))
        if map_val < 65:   flags.append(("🟠", f"Hypotension: MAP {map_val} mmHg"))
        if spo2 < 90:      flags.append(("🔴", f"Severe hypoxia: SpO₂ {spo2}%"))
        if pf_ratio < 200: flags.append(("🟠", f"Impaired oxygenation: P/F {pf_ratio}"))
        if bun > 50:       flags.append(("🟡", f"Elevated BUN: {bun} mg/dL"))
        if not flags:      flags.append(("🟢", "No critical flags detected"))

        st.markdown("**Clinical Flags**")
        for icon, msg in flags[:5]:
            st.markdown(f"{icon} <span style='color:#E1F5EE;font-size:0.85rem'>{msg}</span>",
                        unsafe_allow_html=True)

    st.markdown("---")

    # ── SHAP waterfall (fake) ──
    st.markdown("#### 🔍 Prediction Explanation (SHAP Values)")

    # Compute fake SHAP contributions
    feat_names = FEATURES[:8]
    base_val   = 0.18
    shap_contributions = [
        ("GCS Motor Score",       -round(max(0, (15 - gcs) * 0.05), 3)),
        ("Serum Lactate",          round(min(lactate / 12, 0.25), 3)),
        ("Age",                    round((age - 50) / 350, 3)),
        ("Mean Arterial Pressure", round((70 - map_val) / 280, 3)),
        ("BUN",                    round(min(bun / 180, 0.12), 3)),
        ("SpO₂ Min",               round((93 - spo2) / 120, 3)),
        ("PaO₂/FiO₂ Ratio",       round((200 - pf_ratio) / 1400, 3)),
        ("Creatinine",             round(max(0, (creat - 1.0) / 14), 3)),
    ]
    shap_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    names  = [s[0] for s in shap_contributions]
    values = [s[1] for s in shap_contributions]
    colors = ["#D85A30" if v > 0 else "#1D9E75" for v in values]

    fig_shap = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#E1F5EE", "size": 11},
    ))
    fig_shap.add_vline(x=0, line_color="#5DCAA5", line_width=1)
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#085041",
        margin=dict(l=10, r=60, t=20, b=10), height=320,
        xaxis=dict(title="SHAP value (impact on prediction)", color="#9FE1CB",
                   gridcolor="#0F6E56", zerolinecolor="#5DCAA5"),
        yaxis=dict(color="#E1F5EE", autorange="reversed"),
        font={"color": "#E1F5EE"},
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.caption("🔴 Red bars increase mortality risk · 🟢 Green bars decrease mortality risk · Values are SHAP contributions")


# ══════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Model Performance Comparison")

    # ── Metric table ──
    df_perf = pd.DataFrame(MODELS).T.reset_index()
    df_perf.columns = ["Model", "AUROC", "F1-Score", "Accuracy", "Precision", "Recall"]
    df_perf = df_perf.sort_values("AUROC", ascending=False).reset_index(drop=True)

    # Highlight best row
    def highlight_best(row):
        if row["Model"] == "XGBoost (Best)":
            return ["background-color:#1D9E75;color:white"] * len(row)
        return ["color:#E1F5EE;background-color:#085041"] * len(row)

    st.dataframe(
        df_perf.style
            .apply(highlight_best, axis=1)
            .format({"AUROC": "{:.3f}", "F1-Score": "{:.3f}",
                     "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    col_roc, col_bar = st.columns(2, gap="large")

    with col_roc:
        st.markdown("##### ROC Curves")
        fig_roc = go.Figure()
        colors_roc = ["#EF9F27", "#1D9E75", "#5DCAA5", "#9FE1CB", "#888780"]
        for (mname, mstats), col in zip(MODELS.items(), colors_roc):
            fpr, tpr = roc_curve_data(mstats["auroc"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{mname} ({mstats['auroc']:.3f})",
                line=dict(color=col, width=2)
            ))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(color="#5F5E5A", dash="dot", width=1),
                                     showlegend=False, name="Random"))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#085041",
            xaxis=dict(title="False Positive Rate", color="#9FE1CB", gridcolor="#0F6E56"),
            yaxis=dict(title="True Positive Rate", color="#9FE1CB", gridcolor="#0F6E56"),
            legend=dict(bgcolor="#04342C", bordercolor="#0F6E56", font=dict(color="#E1F5EE", size=10)),
            margin=dict(l=10, r=10, t=10, b=10), height=340,
            font={"color": "#E1F5EE"},
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_bar:
        st.markdown("##### AUROC Comparison")
        model_names = [m.replace(" (Best)", " ★") for m in MODELS]
        auroc_vals  = [v["auroc"] for v in MODELS.values()]
        bar_colors  = ["#EF9F27" if "Best" in m else "#1D9E75" for m in MODELS]

        fig_bar = go.Figure(go.Bar(
            x=model_names, y=auroc_vals,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in auroc_vals],
            textposition="outside",
            textfont={"color": "#E1F5EE", "size": 12},
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#085041",
            yaxis=dict(range=[0.6, 0.95], gridcolor="#0F6E56", color="#9FE1CB", title="AUROC"),
            xaxis=dict(color="#9FE1CB"),
            margin=dict(l=10, r=10, t=30, b=10), height=340,
            font={"color": "#E1F5EE"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Confusion matrix for selected model ──
    st.markdown(f"##### Confusion Matrix — {selected_model}")
    prec = model_stats["prec"]
    rec  = model_stats["rec"]
    tp = int(1000 * rec * 0.22)
    fn = int(1000 * 0.22) - tp
    fp = int(tp * (1 - prec) / prec)
    tn = 1000 - tp - fn - fp

    cm = np.array([[tn, fp], [fn, tp]])
    fig_cm = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Survived", "Died"], y=["Survived", "Died"],
        color_continuous_scale=[[0, "#04342C"], [0.5, "#0F6E56"], [1, "#1D9E75"]],
    )
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#085041",
        font={"color": "#E1F5EE"}, height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
    )
    fig_cm.update_traces(textfont_size=18)
    st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Global Feature Importance")

    col_fi, col_dist = st.columns(2, gap="large")

    with col_fi:
        st.markdown("##### Random Forest Feature Importance")
        importance_vals = [0.142, 0.118, 0.097, 0.088, 0.082, 0.071, 0.065, 0.058, 0.051, 0.044, 0.038, 0.030]
        fi_colors = [
            "#EF9F27" if v > 0.10 else "#1D9E75" if v > 0.06 else "#5DCAA5"
            for v in importance_vals
        ]
        fig_fi = go.Figure(go.Bar(
            x=importance_vals, y=FEATURES,
            orientation="h",
            marker_color=fi_colors,
            text=[f"{v:.3f}" for v in importance_vals],
            textposition="outside",
            textfont={"color": "#E1F5EE", "size": 10},
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#085041",
            xaxis=dict(title="Importance", color="#9FE1CB", gridcolor="#0F6E56"),
            yaxis=dict(color="#E1F5EE", autorange="reversed"),
            margin=dict(l=10, r=60, t=10, b=10), height=400,
            font={"color": "#E1F5EE"},
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_dist:
        st.markdown("##### SHAP Summary — Top 6 Features")
        np.random.seed(42)
        n = 300
        fig_bee = go.Figure()
        top_feats = FEATURES[:6]
        shap_palette = ["#EF9F27", "#1D9E75", "#5DCAA5", "#9FE1CB", "#D85A30", "#9FE1CB"]
        for i, (feat, col) in enumerate(zip(top_feats, shap_palette)):
            sv = np.random.normal(SHAP_VALS[i] * 0.5, SHAP_VALS[i] * 0.4, n)
            jitter = np.random.uniform(-0.25, 0.25, n)
            fig_bee.add_trace(go.Scatter(
                x=sv, y=np.full(n, i) + jitter,
                mode="markers",
                marker=dict(color=col, size=3, opacity=0.55),
                name=feat, showlegend=True,
            ))
        fig_bee.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#085041",
            xaxis=dict(title="SHAP value", color="#9FE1CB", gridcolor="#0F6E56", zeroline=True, zerolinecolor="#5DCAA5"),
            yaxis=dict(tickvals=list(range(6)), ticktext=top_feats, color="#E1F5EE"),
            legend=dict(bgcolor="#04342C", font=dict(color="#E1F5EE", size=9)),
            margin=dict(l=10, r=10, t=10, b=10), height=400,
            font={"color": "#E1F5EE"},
        )
        st.plotly_chart(fig_bee, use_container_width=True)

    # Feature correlation heatmap
    st.markdown("##### Feature Correlation Matrix")
    np.random.seed(7)
    n_feat = 8
    rand_corr = np.random.uniform(-0.6, 0.9, (n_feat, n_feat))
    corr_matrix = (rand_corr + rand_corr.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    short_names = ["GCS", "Lactate", "Age", "MAP", "BUN", "RR Max", "SpO₂", "PaO₂/FiO₂"]

    fig_heat = px.imshow(
        corr_matrix, x=short_names, y=short_names,
        color_continuous_scale=[[0, "#D85A30"], [0.5, "#085041"], [1, "#1D9E75"]],
        zmin=-1, zmax=1, text_auto=".2f",
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#085041",
        font={"color": "#E1F5EE", "size": 10}, height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab4:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### 📌 Project Overview")
        st.markdown("""
        <div class='icu-card'>
            <h4>Research Aim</h4>
            <p style='color:#E1F5EE;font-size:0.9rem'>
            Develop and compare classical machine learning models for predicting
            in-hospital mortality among ICU patients using the MIMIC-IV database.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='icu-card'>
            <h4>Dataset</h4>
            <p style='color:#E1F5EE;font-size:0.9rem'>
            <b style='color:#9FE1CB'>MIMIC-IV</b> · Beth Israel Deaconess Medical Center<br>
            50,000+ ICU admissions · 2008–2019 · De-identified EHR data<br>
            Accessed via PhysioNet (MIT Laboratory for Computational Physiology)
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='icu-card'>
            <h4>Models Evaluated</h4>
            <p style='color:#E1F5EE;font-size:0.9rem'>
            Logistic Regression · Decision Tree · Random Forest<br>
            XGBoost (Gradient Boosting) · Support Vector Machine
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### 🔬 Methodology Pipeline")

        pipeline_steps = [
            ("01", "Data Extraction", "SQL from MIMIC-IV PostgreSQL via psycopg2"),
            ("02", "Preprocessing", "Missing value imputation · Outlier removal · Encoding"),
            ("03", "Feature Engineering", "24h summary stats · Lab values · Composite scores"),
            ("04", "Class Balancing", "SMOTE applied to training set only"),
            ("05", "Model Training", "5-fold stratified CV · GridSearchCV tuning"),
            ("06", "Evaluation", "AUROC · F1 · Precision · Recall · Calibration"),
            ("07", "Interpretability", "SHAP values · Feature importance · Odds ratios"),
        ]
        for num, title, desc in pipeline_steps:
            st.markdown(f"""
            <div style='display:flex;gap:16px;align-items:flex-start;margin-bottom:14px'>
                <div style='background:#1D9E75;color:white;border-radius:8px;padding:4px 10px;
                            font-family:monospace;font-size:0.7rem;font-weight:700;white-space:nowrap;
                            min-width:30px;text-align:center'>{num}</div>
                <div>
                    <div style='color:#E1F5EE;font-weight:600;font-size:0.87rem'>{title}</div>
                    <div style='color:#5DCAA5;font-size:0.78rem'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Tech stack
    st.markdown("#### 🛠 Technology Stack")
    tech_cols = st.columns(5)
    techs = [
        ("Python 3.10", "Core language", "🐍"),
        ("scikit-learn", "ML algorithms", "⚙️"),
        ("XGBoost", "Gradient boosting", "🚀"),
        ("SHAP", "Explainability", "🔍"),
        ("Streamlit", "Web interface", "🌐"),
    ]
    for col, (name, desc, icon) in zip(tech_cols, techs):
        col.markdown(f"""
        <div class='icu-card' style='text-align:center;padding:1rem'>
            <div style='font-size:1.6rem'>{icon}</div>
            <div style='color:#E1F5EE;font-weight:700;font-size:0.85rem;margin-top:6px'>{name}</div>
            <div style='color:#5DCAA5;font-size:0.72rem'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#5DCAA5;font-size:0.75rem;padding:1rem 0'>
        ICU Mortality Prediction System · Final Year Project · Computer Science<br>
        Data: MIMIC-IV (PhysioNet) · Framework: Streamlit · All predictions for <b>research purposes only</b>
    </div>
    """, unsafe_allow_html=True)