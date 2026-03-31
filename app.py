"""
Turbofan Engine Predictive Maintenance
Streamlit application using pre-trained Random Forest models
on the NASA C-MAPSS dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turbofan Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
RUL_MAX = 350          # practical max predicted RUL
RUL_CRITICAL = 50
RUL_WARNING = 150

FAILURE_DESCRIPTIONS = {
    "Failure_1": "High Bypass Fan degradation",
    "Failure_2": "Low Pressure Compressor fault",
    "Failure_3": "High Pressure Compressor degradation",
    "Failure_4": "Turbine degradation",
}

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTINGS = [f"op_setting_{i}" for i in range(1, 4)]
ALL_COLS = OP_SETTINGS + SENSORS

# 144 engineered feature names (must match training order)
FEATURE_COLS = []
for col in ALL_COLS:
    for window in [30, 60, 90]:
        FEATURE_COLS.append(f"{col}_avg_{window}")
    FEATURE_COLS.append(f"{col}_min")
    FEATURE_COLS.append(f"{col}_max")
    FEATURE_COLS.append(f"{col}_var")

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background ── */
[data-testid="stAppViewContainer"] { background: #f5f7fa; }
[data-testid="stHeader"]           { background: transparent; }

/* ── Sidebar: white bg + dark text everywhere ── */
[data-testid="stSidebar"] {
    background: #1e3a5f !important;
    border-right: 1px solid #2d5080;
}
[data-testid="stSidebar"] * {
    color: #e8f0fb !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #c8d8f0 !important; }
[data-testid="stSidebar"] code          { background: #2d5080 !important; color: #a8d0ff !important; }
[data-testid="stSidebar"] [data-testid="stInfo"] {
    background: #2d5080 !important;
    border-color: #4a7ab5 !important;
    color: #e8f0fb !important;
}

/* ── Main area text ── */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] .stMarkdown { color: #1e3a5f; }

/* ── Native Streamlit headings in main area ── */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 { color: #1e3a5f !important; }

/* ── Header banner – override everything ── */
.header-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a9e 100%);
    border: 1px solid #3b6ea5;
    padding: 1.8rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
.header-banner h1,
.header-banner h1 * { margin: 0; font-size: 1.9rem; font-weight: 700; color: #ffffff !important; }
.header-banner p,
.header-banner p *  { margin: 0.4rem 0 0; color: #a8c8f0 !important; font-size: 0.97rem; }

/* ── Failure type card ── */
.failure-card {
    background: #ffffff;
    border: 2px solid #bfdbfe;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    height: 100%;
    box-shadow: 0 2px 8px rgba(59,110,165,0.08);
}
.failure-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
.failure-card .type  { font-size: 1.7rem; font-weight: 700; color: #1e3a5f; margin: 0.5rem 0; }
.failure-card .desc  { font-size: 0.93rem; color: #4b5563; }

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-card .metric-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card .metric-value { font-size: 1.55rem; font-weight: 700; color: #1e3a5f; }

/* ── Step boxes ── */
.step-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.step-box .step-num  { font-size: 0.72rem; color: #3b82f6; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.step-box .step-text { font-size: 0.93rem; color: #374151; margin-top: 0.2rem; }

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tab"]                       { color: #4b5563 !important; font-weight: 500; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #1e3a5f !important; border-bottom-color: #3b82f6; }

/* ── File uploader & widgets text ── */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p { color: #1e3a5f !important; }

/* ── Markdown bold/strong in main area ── */
[data-testid="stAppViewContainer"] strong,
[data-testid="stAppViewContainer"] b { color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ── Header banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>✈️ Turbofan Engine Predictive Maintenance</h1>
    <p>NASA C-MAPSS Dataset &nbsp;·&nbsp; Random Forest Models &nbsp;·&nbsp; Real-time RUL &amp; Failure Type Prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About")
    st.info(
        "This tool predicts the **Remaining Useful Life (RUL)** and **failure type** "
        "of turbofan engines using machine learning models trained on the NASA C-MAPSS dataset."
    )
    st.markdown("**Models used**")
    st.markdown("- Random Forest Regressor (RUL prediction)")
    st.markdown("- Random Forest Classifier (Failure type)")
    st.markdown("**Expected CSV formats**")
    st.markdown("*Raw sensor data (multi-cycle):*")
    st.code(
        "engine_id, cycle,\n"
        "op_setting_1, op_setting_2, op_setting_3,\n"
        "sensor_1, sensor_2, ..., sensor_21",
        language="text",
    )
    st.markdown("*Pre-engineered features (one row per engine):*")
    st.code(
        "engine_id,\n"
        "op_setting_1_avg_30, ..., sensor_21_var\n"
        "(144 feature columns)",
        language="text",
    )

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load pre-trained models from disk. Returns None tuple if files missing."""
    try:
        with open("rf_regressor.pkl",  "rb") as f: regressor  = pickle.load(f)
        with open("rf_classifier.pkl", "rb") as f: classifier = pickle.load(f)
        with open("imputer.pkl",       "rb") as f: imputer    = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f: le         = pickle.load(f)
        return regressor, classifier, imputer, le
    except FileNotFoundError:
        return None, None, None, None

regressor, classifier, imputer, le = load_models()
DEMO_MODE = regressor is None

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling window statistics per engine and return one row per engine
    with 144 features matching the training format.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    required_raw = ["engine_id", "cycle"] + OP_SETTINGS + SENSORS
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    records = []
    for eng_id, grp in df.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        row = {"engine_id": eng_id}
        for col in ALL_COLS:
            series = grp[col].astype(float)
            for window in [30, 60, 90]:
                row[f"{col}_avg_{window}"] = series.rolling(window, min_periods=1).mean().iloc[-1]
            row[f"{col}_min"] = series.min()
            row[f"{col}_max"] = series.max()
            row[f"{col}_var"] = series.var(ddof=0) if len(series) > 1 else 0.0
        records.append(row)

    return pd.DataFrame(records)

# ── Helper: status label & colour ─────────────────────────────────────────────
def rul_status(rul: float):
    if rul < RUL_CRITICAL:
        return "CRITICAL", "#c0392b"
    elif rul < RUL_WARNING:
        return "WARNING", "#e67e22"
    return "HEALTHY", "#27ae60"

# ── Helper: RUL gauge chart ───────────────────────────────────────────────────
def rul_gauge(rul: float) -> go.Figure:
    pct = min(rul / RUL_MAX, 1.0)
    if pct > 0.6:
        bar_color = "#70ad47"
    elif pct > 0.3:
        bar_color = "#ffc000"
    else:
        bar_color = "#c0392b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul,
        number={"suffix": " cycles", "font": {"size": 28, "color": "#1a2a4a"}},
        title={"text": "Remaining Useful Life", "font": {"size": 16, "color": "#555"}},
        gauge={
            "axis": {"range": [0, RUL_MAX], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#f5f7fa",
            "borderwidth": 2,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0, RUL_CRITICAL],  "color": "#fde8e8"},
                {"range": [RUL_CRITICAL, RUL_WARNING], "color": "#fef3e2"},
                {"range": [RUL_WARNING, RUL_MAX],      "color": "#eaf5e3"},
            ],
            "threshold": {
                "line": {"color": "#c0392b", "width": 3},
                "thickness": 0.75,
                "value": RUL_CRITICAL,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor="#f5f7fa", font=dict(color="#1e3a5f"))
    return fig

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📂 Upload & Predict", "📊 Model Information", "ℹ️ How It Works"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – UPLOAD & PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if DEMO_MODE:
        st.warning(
            "⚠️ **Demo Mode** – Model files (`rf_regressor.pkl`, `rf_classifier.pkl`, "
            "`imputer.pkl`, `label_encoder.pkl`) were not found. "
            "Run `python train_models.py` to generate them, then restart the app."
        )

    uploaded = st.file_uploader("Upload engine sensor CSV", type=["csv"])

    if uploaded is None:
        # ── No file: show sample format + downloadable demo CSV ──────────────
        st.info("👆 Upload a CSV file with raw engine sensor readings to get predictions.")

        # Build the demo sample from real dataset engine feature values
        @st.cache_data
        def build_sample_csv() -> bytes:
            """
            Build a demo CSV using pre-engineered features directly from the
            training dataset. Rows are selected by dataset index to cover all
            3 RUL zones and all 4 predicted failure types.

            Row index → (demo_id, expected_pred_rul, pred_type):
              75 → (1, ~26,  Failure_3)  CRITICAL
              61 → (2, ~59,  Failure_3)  WARNING
              69 → (3, ~61,  Failure_3)  WARNING
               0 → (4, ~295, Failure_1)  HEALTHY
              27 → (5, ~251, Failure_2)  HEALTHY
              67 → (6, ~180, Failure_3)  HEALTHY
              86 → (7, ~181, Failure_4)  HEALTHY
            """
            df_src = pd.read_csv("final_dataset.csv")
            df_src.columns = df_src.columns.str.strip()

            # (demo_id, dataset_row_index)
            row_map = [(1,75),(2,61),(3,69),(4,0),(5,27),(6,67),(7,86)]
            rows = []
            for demo_id, row_idx in row_map:
                orig = df_src.iloc[row_idx]
                row = {"engine_id": demo_id}
                for col in FEATURE_COLS:
                    row[col] = orig[col]
                rows.append(row)
            return pd.DataFrame(rows).to_csv(index=False).encode()

        sample_bytes = build_sample_csv()
        sample_df = pd.read_csv(pd.io.common.BytesIO(sample_bytes))

        # Status legend
        st.markdown("**Demo dataset — 7 engines covering all statuses & failure types:**")
        legend = pd.DataFrame({
            "Demo Engine": list(range(1, 8)),
            "Expected Status":  ["🔴 CRITICAL","🟠 WARNING","🟠 WARNING",
                                  "🟢 HEALTHY","🟢 HEALTHY","🟢 HEALTHY","🟢 HEALTHY"],
            "Expected RUL":     ["~26 cycles","~59 cycles","~61 cycles",
                                  "~295 cycles","~251 cycles","~180 cycles","~181 cycles"],
            "Failure Type":     ["Failure_3","Failure_3","Failure_3",
                                  "Failure_1","Failure_2","Failure_3","Failure_4"],
        })
        st.dataframe(legend, use_container_width=True, hide_index=True)

        col_prev, col_dl = st.columns([3, 1])
        with col_prev:
            st.markdown("**Pre-engineered feature CSV preview** (one row per engine):")
            st.dataframe(sample_df.iloc[:, :8], use_container_width=True, hide_index=True)
        with col_dl:
            st.markdown("**Try it out**")
            st.markdown("Download and upload this CSV to see predictions with all statuses and failure types.")
            st.download_button(
                label="⬇️ Download sample CSV",
                data=sample_bytes,
                file_name="sample_engines.csv",
                mime="text/csv",
            )

    else:
        # ── File uploaded ─────────────────────────────────────────────────────
        raw_df = pd.read_csv(uploaded)
        raw_df.columns = raw_df.columns.str.strip()

        st.markdown("**Raw data preview**")
        st.dataframe(raw_df.head(10), use_container_width=True)

        try:
            with st.spinner("Computing features and running predictions…"):
                # Detect pre-engineered CSV (has feature columns, no 'cycle' column)
                is_preengineered = (
                    "cycle" not in raw_df.columns
                    and all(c in raw_df.columns for c in FEATURE_COLS[:6])
                )

                if is_preengineered:
                    feat_df = raw_df.copy()
                    if "engine_id" not in feat_df.columns:
                        feat_df.insert(0, "engine_id", range(1, len(feat_df) + 1))
                else:
                    feat_df = engineer_features(raw_df)

                engine_ids = feat_df["engine_id"].values
                X = feat_df[FEATURE_COLS].values

                if not DEMO_MODE:
                    X_imp = imputer.transform(X)
                    rul_preds   = regressor.predict(X_imp).round().astype(int)
                    cls_preds   = le.inverse_transform(classifier.predict(X_imp))
                else:
                    # Demo: random plausible values
                    rng = np.random.default_rng(42)
                    rul_preds = rng.integers(30, 494, size=len(engine_ids))
                    cls_preds = rng.choice(list(FAILURE_DESCRIPTIONS.keys()), size=len(engine_ids))

            n_engines = len(engine_ids)

            # ── Single engine ─────────────────────────────────────────────────
            if n_engines == 1:
                rul  = int(rul_preds[0])
                ftype = cls_preds[0]
                status, _ = rul_status(rul)

                col_gauge, col_card = st.columns(2)

                with col_gauge:
                    st.plotly_chart(rul_gauge(rul), use_container_width=True)

                with col_card:
                    st.markdown(f"""
                    <div class="failure-card">
                        <div class="label">Predicted Failure Type</div>
                        <div class="type">{ftype}</div>
                        <div class="desc">{FAILURE_DESCRIPTIONS.get(ftype, '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")
                if rul < RUL_CRITICAL:
                    st.error(f"🚨 CRITICAL – RUL is only {rul} cycles. Immediate maintenance required.")
                elif rul < RUL_WARNING:
                    st.warning(f"⚠️ WARNING – RUL is {rul} cycles. Schedule maintenance soon.")
                else:
                    st.success(f"✅ Engine is healthy – RUL is {rul} cycles.")

            # ── Multiple engines ──────────────────────────────────────────────
            else:
                statuses = [rul_status(r)[0] for r in rul_preds]
                results_df = pd.DataFrame({
                    "engine_id":            engine_ids,
                    "Predicted RUL":        rul_preds,
                    "Predicted Failure Type": cls_preds,
                    "Status":               statuses,
                })

                st.markdown("### Fleet Prediction Results")

                col_table, col_summary = st.columns([3, 1])

                with col_table:
                    # Colour-coded status column via styler
                    def style_status(val):
                        colours = {
                            "CRITICAL": "background-color: #fde8e8; color: #7f1d1d",
                            "WARNING":  "background-color: #fef3e2; color: #78350f",
                            "HEALTHY":  "background-color: #eaf5e3; color: #14532d",
                        }
                        return colours.get(val, "")

                    styled = results_df.style.map(style_status, subset=["Status"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                with col_summary:
                    n_crit = sum(r < RUL_CRITICAL  for r in rul_preds)
                    n_warn = sum(RUL_CRITICAL <= r < RUL_WARNING for r in rul_preds)
                    n_ok   = sum(r >= RUL_WARNING   for r in rul_preds)
                    avg_rul = int(np.mean(rul_preds))

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🔴 Critical</div>
                        <div class="metric-value">{n_crit}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">🟠 Warning</div>
                        <div class="metric-value">{n_warn}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">🟢 Healthy</div>
                        <div class="metric-value">{n_ok}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">📊 Avg RUL</div>
                        <div class="metric-value">{avg_rul} cy</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── RUL bar chart ─────────────────────────────────────────────
                bar_colours = [rul_status(r)[1] for r in rul_preds]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[f"Eng {e}" for e in engine_ids],
                    y=rul_preds,
                    marker_color=bar_colours,
                    name="Predicted RUL",
                ))
                fig_bar.add_hline(y=RUL_CRITICAL, line_dash="dash", line_color="#c0392b",
                                  annotation_text="Critical (50)", annotation_position="top right")
                fig_bar.add_hline(y=RUL_WARNING,  line_dash="dash", line_color="#e67e22",
                                  annotation_text="Warning (150)", annotation_position="top right")
                fig_bar.update_layout(
                    title="Predicted RUL per Engine",
                    xaxis_title="Engine", yaxis_title="RUL (cycles)",
                    plot_bgcolor="#f5f7fa", paper_bgcolor="#f5f7fa",
                    font=dict(color="#1e3a5f", size=13),
                    title_font=dict(color="#1e3a5f"),
                    xaxis=dict(tickfont=dict(color="#1e3a5f"), title_font=dict(color="#1e3a5f")),
                    yaxis=dict(tickfont=dict(color="#1e3a5f"), title_font=dict(color="#1e3a5f"), gridcolor="#d1dce8"),
                    height=380,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── Failure type pie chart ────────────────────────────────────
                col_pie, col_dl = st.columns([2, 1])
                with col_pie:
                    type_counts = pd.Series(cls_preds).value_counts().reset_index()
                    type_counts.columns = ["Failure Type", "Count"]
                    fig_pie = px.pie(
                        type_counts, names="Failure Type", values="Count",
                        title="Failure Type Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_pie.update_layout(
                        height=350,
                        paper_bgcolor="#f5f7fa",
                        font=dict(color="#1e3a5f", size=13),
                        title_font=dict(color="#1e3a5f"),
                        legend=dict(font=dict(color="#1e3a5f")),
                    )
                    fig_pie.update_traces(textfont=dict(color="#ffffff"))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_dl:
                    st.markdown("**Download predictions**")
                    csv_bytes = results_df.to_csv(index=False).encode()
                    st.download_button(
                        label="⬇️ Export CSV",
                        data=csv_bytes,
                        file_name="turbofan_predictions.csv",
                        mime="text/csv",
                    )

        except ValueError as e:
            st.error(f"**Column error:** {e}")
            st.info(f"Expected columns: `engine_id`, `cycle`, {', '.join(OP_SETTINGS + SENSORS)}")
        except Exception as e:
            st.error(f"**Prediction failed:** {e}")
            st.info("Please check that your CSV matches the expected format shown in the sidebar.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MODEL INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance Comparison")

    # ── Styled HTML table matching the reference design ──────────────────────
    st.markdown("""
    <style>
    .perf-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(30,58,95,0.12);
        margin-bottom: 1.5rem;
    }
    .perf-table thead tr {
        background: #1e3a5f;
        color: #ffffff;
    }
    .perf-table thead th {
        padding: 0.75rem 1rem;
        text-align: center;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid #2d5080;
    }
    .perf-table thead th:first-child { text-align: left; }
    .perf-table tbody tr:nth-child(odd)  { background: #dbeafe; }
    .perf-table tbody tr:nth-child(even) { background: #bfdbfe; }
    .perf-table tbody tr.best-row { background: #16a34a !important; color: #ffffff !important; }
    .perf-table tbody tr.best-row td { color: #ffffff !important; }
    .perf-table tbody td {
        padding: 0.65rem 1rem;
        text-align: center;
        border: 1px solid #93c5fd;
        color: #1e3a5f;
    }
    .perf-table tbody td:first-child { text-align: left; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("**Classification Model – Failure Type**")
    st.markdown("""
    <table class="perf-table">
      <thead>
        <tr>
          <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>CV Mean</th>
        </tr>
      </thead>
      <tbody>
        <tr class="best-row">
          <td>Random Forest</td><td>80.95%</td><td>81.20%</td><td>80.95%</td><td>80.85%</td><td>80.62%</td>
        </tr>
        <tr>
          <td>Logistic Regression</td><td>80.95%</td><td>80.10%</td><td>80.95%</td><td>80.30%</td><td>83.67%</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("**Regression Model – RUL Prediction**")
    st.markdown("""
    <table class="perf-table">
      <thead>
        <tr>
          <th>Model</th><th>MAE (cycles)</th><th>RMSE (cycles)</th><th>R²</th><th>Improvement</th>
        </tr>
      </thead>
      <tbody>
        <tr class="best-row">
          <td>Random Forest</td><td>34.87</td><td>40.87</td><td>0.83</td><td>22.8% better MAE</td>
        </tr>
        <tr>
          <td>Linear Regression</td><td>45.17</td><td>60.42</td><td>0.71</td><td>—</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Top 10 Features for RUL Prediction")

    features = [
        "sensor_15_avg_90", "sensor_15_min", "sensor_15_avg_60",
        "sensor_15_avg_30", "sensor_21_max", "sensor_4_min",
        "sensor_12_max", "sensor_3_min", "sensor_17_avg_60", "sensor_7_max",
    ]
    importances = [0.213, 0.136, 0.083, 0.077, 0.041, 0.034, 0.026, 0.023, 0.020, 0.016]

    fig_feat = go.Figure(go.Bar(
        x=importances[::-1],
        y=features[::-1],
        orientation="h",
        marker_color="#4472c4",
        text=[f"{v:.3f}" for v in importances[::-1]],
        textposition="outside",
    ))
    fig_feat.update_layout(
        title="Feature Importances (Random Forest Regressor)",
        xaxis_title="Importance Score",
        plot_bgcolor="#f5f7fa", paper_bgcolor="#f5f7fa",
        font=dict(color="#1e3a5f", size=13),
        title_font=dict(color="#1e3a5f"),
        xaxis=dict(tickfont=dict(color="#1e3a5f"), title_font=dict(color="#1e3a5f"), gridcolor="#d1dce8"),
        yaxis=dict(tickfont=dict(color="#1e3a5f"), title_font=dict(color="#1e3a5f")),
        height=400,
        margin=dict(l=160),
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.info("💡 Sensor 15 features account for **51.3%** of total predictive power for RUL prediction.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Pipeline Overview")

    steps = [
        ("Step 1", "Upload raw CSV sensor data with engine_id, cycle, operational settings, and 21 sensor readings."),
        ("Step 2", "App automatically computes 144 rolling window features (avg_30, avg_60, avg_90, min, max, var) per sensor and setting."),
        ("Step 3", "Engineered features are fed to the Random Forest Regressor to predict Remaining Useful Life (RUL) in cycles."),
        ("Step 4", "The same features are fed to the Random Forest Classifier to predict the failure type (Failure_1 – Failure_4)."),
        ("Step 5", "Results are displayed with visual gauges, colour-coded alerts, and fleet-level charts."),
    ]
    for num, text in steps:
        st.markdown(f"""
        <div class="step-box">
            <div class="step-num">{num}</div>
            <div class="step-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset Information")

    info_col, _ = st.columns([2, 1])
    with info_col:
        dataset_info = pd.DataFrame({
            "Property": [
                "Source", "Training engines", "Sensors",
                "Operational settings", "Failure types",
                "RUL range", "Mean RUL", "Engineered features",
            ],
            "Value": [
                "NASA C-MAPSS Turbofan Engine Dataset",
                "104 engines (26 unique × 4 failure types)",
                "21 sensors (sensor_1 – sensor_21)",
                "3 (op_setting_1 – op_setting_3)",
                "4 balanced classes (26 engines each)",
                "0 – 347 cycles (true), 26 – 321 (predicted)",
                "260 cycles",
                "144 (rolling statistics per signal)",
            ],
        })
        st.dataframe(dataset_info, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Feature Engineering Detail")
    st.markdown(
        "For each of the 24 signals (21 sensors + 3 operational settings), "
        "the following 6 statistics are computed over the engine's full history, "
        "producing **24 × 6 = 144 features**:"
    )
    fe_df = pd.DataFrame({
        "Feature":     ["col_avg_30", "col_avg_60", "col_avg_90", "col_min", "col_max", "col_var"],
        "Description": [
            "Rolling mean over last 30 cycles",
            "Rolling mean over last 60 cycles",
            "Rolling mean over last 90 cycles",
            "Minimum value across all cycles",
            "Maximum value across all cycles",
            "Variance across all cycles",
        ],
    })
    st.dataframe(fe_df, use_container_width=True, hide_index=True)
