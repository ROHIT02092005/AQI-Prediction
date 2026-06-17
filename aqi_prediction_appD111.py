"""
╔══════════════════════════════════════════════════════════════╗
║       AQI PREDICTION DASHBOARD — Dark Mode Edition          ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
-----------
1. pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly
2. streamlit run aqi_prediction_app_dark.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io, warnings
from twilio.rest import Client

warnings.filterwarnings("ignore")

# TWILIO SETTINGS
account_sid = "AC550f8962374e0245929310b2dcb575df"
auth_token = "c4459af654c0dd953cc015a6431108c1"

twilio_number = "+18777804236"
your_number = "+919335120267"
# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  DARK MODE CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── GLOBAL DARK BASE ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background-color: #0a0f1e !important;
    color: #e2e8f0 !important;
}
.stApp { background-color: #0a0f1e !important; }
.main .block-container { background-color: #0a0f1e !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0f2235 60%, #0a1929 100%) !important;
    border-right: 1px solid rgba(99,140,255,0.15) !important;
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
[data-testid="stSidebar"] input { color: #e2e8f0 !important; background: #1a2840 !important; }
[data-testid="stSidebar"] .stFileUploader { background: #1a2840 !important; border-radius: 10px; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #1e2f48 !important;
    border: 2px dashed rgba(99,140,255,0.4) !important;
    border-radius: 10px !important;
}

/* ── HEADER BANNER ── */
.header-banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #0d3b66 50%, #1a5276 100%);
    padding: 28px 36px;
    border-radius: 18px;
    margin-bottom: 24px;
    border: 1px solid rgba(79,140,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,217,169,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.header-banner h1 {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #7ec8e3;
    font-size: 0.95rem;
    margin: 0;
    opacity: 0.9;
}

/* ── TABS — attractive dark style ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border-radius: 14px !important;
    padding: 6px 8px !important;
    gap: 4px !important;
    border: 1px solid rgba(99,140,255,0.15) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: #8fa3c8 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 18px !important;
    border: none !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(79,140,255,0.12) !important;
    color: #c8d8f0 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a6e, #1a5276) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 12px rgba(79,140,255,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── METRIC CARDS ── */
.metric-card {
    background: linear-gradient(145deg, #111827, #141e30);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    border: 1px solid rgba(99,140,255,0.18);
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #4f8cff, #38d9a9);
    border-radius: 3px 3px 0 0;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(79,140,255,0.2); }
.metric-card h3 { color: #7ec8e3; font-size: 0.78rem; margin: 0 0 10px 0; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; }
.metric-card h2 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0 0 4px 0; }
.metric-card p  { color: #5a7898; font-size: 0.75rem; margin: 0; }
.metric-card .card-icon { font-size: 1.8rem; margin-bottom: 10px; display: block; }

/* ── SECTION TITLES ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #4f8cff;
    padding-left: 14px;
    margin: 28px 0 18px 0;
    letter-spacing: -0.2px;
}

/* ── INFO BOX ── */
.info-box {
    background: rgba(79,140,255,0.07);
    border-left: 4px solid #4f8cff;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 12px 0;
    color: #a8c4e0;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── AQI RESULT ── */
.aqi-result {
    border-radius: 18px;
    padding: 30px 36px;
    margin: 20px 0;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    color: white;
    font-weight: 600;
    position: relative;
    overflow: hidden;
}
.aqi-result::before {
    content: '';
    position: absolute; inset: 0;
    background: rgba(0,0,0,0.2);
    pointer-events: none;
}
.aqi-result h2 { font-size: 3.2rem; margin: 0 0 6px 0; position: relative; }
.aqi-result h3 { font-size: 1.4rem; margin: 0 0 10px 0; position: relative; }
.aqi-result p  { font-size: 0.95rem; margin: 0; opacity: 0.9; position: relative; }

/* ── STEP BADGE ── */
.step-badge {
    display: inline-flex;
    align-items: center; justify-content: center;
    background: linear-gradient(135deg, #4f8cff, #38d9a9);
    color: white;
    border-radius: 50%;
    width: 32px; height: 32px;
    font-weight: 700; font-size: 0.85rem;
    margin-right: 12px;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(79,140,255,0.4);
}
.step-row {
    display: flex; align-items: flex-start;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,140,255,0.12);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.step-row:hover { border-color: rgba(79,140,255,0.35); }

/* ── BUTTONS ── */
div.stButton > button {
    background: linear-gradient(135deg, #1e3a6e, #1a5276) !important;
    color: #ffffff !important;
    border: 1px solid rgba(79,140,255,0.4) !important;
    border-radius: 10px !important;
    padding: 11px 28px !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    letter-spacing: 0.2px !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #254d96, #1f6b91) !important;
    box-shadow: 0 6px 20px rgba(79,140,255,0.3) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active { transform: scale(0.98) !important; }

/* ── INPUTS (number) ── */
.stNumberInput input {
    background: #111827 !important;
    border: 1px solid rgba(99,140,255,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
}
.stNumberInput input:focus {
    border-color: #4f8cff !important;
    box-shadow: 0 0 0 3px rgba(79,140,255,0.15) !important;
}

/* ── POLLUTANT LABEL STRIP ── */
.pollutant-label {
    background: linear-gradient(90deg, #1e3a6e, #1a5276);
    color: white;
    padding: 8px 13px;
    border-radius: 8px 8px 0 0;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: -8px;
    letter-spacing: 0.3px;
    border: 1px solid rgba(79,140,255,0.3);
    border-bottom: none;
}

/* ── DATAFRAME (dark tables) ── */
.stDataFrame { background: #111827 !important; }
[data-testid="stDataFrameResizable"] { background: #111827 !important; }
.dataframe thead tr th { background: #1e3a6e !important; color: white !important; }
.dataframe tbody tr:nth-child(even) { background: #111827 !important; }
.dataframe tbody tr:nth-child(odd)  { background: #0d1829 !important; }

/* ── HR ── */
hr { border: none; border-top: 1px solid rgba(99,140,255,0.15) !important; margin: 24px 0; }

/* ── METRICS (st.metric) ── */
[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid rgba(99,140,255,0.18) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #7ec8e3 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploaderDropzone"] {
    background: #111827 !important;
    border: 2px dashed rgba(79,140,255,0.35) !important;
    border-radius: 12px !important;
}

/* ── WELCOME STEP CARDS ── */
.welcome-card {
    background: linear-gradient(145deg, #111827, #141e30);
    border: 1px solid rgba(99,140,255,0.18);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
}
.welcome-card:hover { transform: translateY(-4px); box-shadow: 0 10px 30px rgba(79,140,255,0.18); }
.welcome-card .wc-icon { font-size: 2.8rem; margin-bottom: 14px; display: block; }
.welcome-card h3 { color: #7ec8e3; font-size: 0.82rem; letter-spacing: 1.5px; text-transform: uppercase; margin: 0 0 8px 0; font-weight: 600; }
.welcome-card p  { color: #8fa3c8; font-size: 0.86rem; margin: 0; line-height: 1.5; }

/* ── FOOTER ── */
.dark-footer {
    text-align: center;
    color: #3d5a80;
    font-size: 0.8rem;
    padding: 20px 0 10px 0;
    border-top: 1px solid rgba(99,140,255,0.1);
    margin-top: 40px;
}

/* ── SELECTBOX / DROPDOWN ── */
[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border: 1px solid rgba(99,140,255,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #4f8cff !important; }

/* ── MARKDOWN TEXT ── */
.stMarkdown p, .stMarkdown li { color: #c8d8f0 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e2e8f0 !important; }
.stMarkdown code { background: #1e2f48 !important; color: #7ec8e3 !important; border-radius: 4px; padding: 2px 6px; }
.stMarkdown pre { background: #111827 !important; border: 1px solid rgba(99,140,255,0.2) !important; border-radius: 10px; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e3a6e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f8cff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
FEATURES = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]

AQI_LIMITS = [
    (0,   50,  "Good",         "#27ae60", "Minimal health impact."),
    (51,  100, "Satisfactory", "#2ecc71", "Minor breathing discomfort to sensitive people."),
    (101, 200, "Moderate",     "#f39c12", "Breathing discomfort to people with lung/heart disease, children & elderly."),
    (201, 300, "Poor",         "#e67e22", "Breathing discomfort to people on prolonged exposure."),
    (301, 400, "Very Poor",    "#e74c3c", "Respiratory illness on prolonged exposure."),
    (401, 500, "Severe",       "#8e44ad", "Affects even healthy people. Serious risk!"),
]

# dark theme colors for charts (keep bar/line colors as is)
DARK_BG    = "#0d1421"
DARK_PAPER = "#0a0f1e"
DARK_GRID  = "rgba(99,140,255,0.1)"   # plotly only — CSS rgba is fine here
DARK_FONT  = "#c8d8f0"
DARK_TICK  = "#5a7898"
# Matplotlib-safe equivalents (hex only — no CSS rgba)
MPL_BG     = "#0d1421"
MPL_SPINE  = "#1e2f48"
MPL_FONT   = "#c8d8f0"
MPL_TICK   = "#5a7898"

def aqi_analysis(aqi_value):
    for lo, hi, cat, color, impact in AQI_LIMITS:
        if lo <= aqi_value <= hi:
            return cat, color, impact
    return "Severe", "#8e44ad", "Affects even healthy people. Serious risk!"

def dark_layout(fig, height=420, title_color=DARK_FONT):
    fig.update_layout(
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_PAPER,
        font_color=DARK_FONT,
        title_font_color=title_color,
        title_font_size=14,
        height=height,
        xaxis=dict(gridcolor=DARK_GRID, tickcolor=DARK_TICK, color=DARK_FONT, linecolor=DARK_GRID),
        yaxis=dict(gridcolor=DARK_GRID, tickcolor=DARK_TICK, color=DARK_FONT, linecolor=DARK_GRID),
    )
    return fig

@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_bytes):
    df = pd.read_csv(io.BytesIO(uploaded_bytes))
    df.drop_duplicates(inplace=True)
    if "Date" in df.columns:
        df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"]  = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"]   = df["Date"].dt.day
    return df

@st.cache_resource(show_spinner=False)
def train_model(uploaded_bytes):
    df = load_and_clean(uploaded_bytes)
    df = df.dropna(subset=["AQI"])
    available = [f for f in FEATURES if f in df.columns]
    X = df[available]; y = df["AQI"]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=available)
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    feat_imp = pd.DataFrame({"Feature": available, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    return dict(model=rf, imputer=imputer, features=available, df=df,
                X_test=X_test, y_test=y_test, y_pred=y_pred,
                mae=mae, mse=mse, rmse=rmse, r2=r2, accuracy=r2*100, feat_imp=feat_imp)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.2rem; margin-bottom:6px;'>🌿</div>
        <div style='font-size:1.15rem; font-weight:700; color:#7ec8e3; letter-spacing:-0.3px;'>AQI Dashboard</div>
        <div style='font-size:0.75rem; color:#3d5a80; margin-top:3px;'>Machine Learning Edition</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.85rem; line-height:1.9; color:#8fa3c8;'>
        <div style='color:#7ec8e3; font-weight:600; margin-bottom:8px; font-size:0.78rem; letter-spacing:1px; text-transform:uppercase;'>How it works</div>
        <div>① Upload your CSV dataset</div>
        <div>② Auto-trains Random Forest</div>
        <div>③ Explore charts & metrics</div>
        <div>④ Predict AQI live</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:0.8rem; color:#7ec8e3; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">Upload Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("city_day.csv", type=["csv"], help="Upload city_day.csv", label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#3d5a80; line-height:1.8;'>
        <div style='color:#4d6a8a; font-weight:600; margin-bottom:4px;'>Model Config</div>
        Algorithm: Random Forest<br>
        Estimators: 200 trees<br>
        Test Split: 20%<br>
        Imputation: Median
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='header-banner'>
    <h1>🌿 Air Quality Index Prediction</h1>
    <p>Machine Learning Dashboard &nbsp;·&nbsp; Random Forest Regressor</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  WELCOME SCREEN (no file)
# ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown('<div class="info-box">👈 <b>Start here:</b> Upload your <code>city_day.csv</code> from the sidebar on the left.</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    steps = [
        ("📁", "Upload CSV", "Upload city_day.csv from the sidebar to begin"),
        ("🤖", "Auto-Train", "Model trains instantly — no code needed"),
        ("🎯", "Predict AQI", "Enter pollutant values for live predictions"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], steps):
        with col:
            st.markdown(f"""
            <div class='welcome-card'>
                <span class='wc-icon'>{icon}</span>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-title">📋 AQI Category Reference</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame([(lo,hi,cat,imp) for lo,hi,cat,_,imp in AQI_LIMITS], columns=["From","To","Category","Health Impact"]),
        use_container_width=True, hide_index=True
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
#  TRAIN MODEL
# ─────────────────────────────────────────────────────────────
with st.spinner("⚙️  Training Random Forest model… please wait"):
    result = train_model(uploaded_file.read())

model    = result["model"]
imputer  = result["imputer"]
features = result["features"]
df       = result["df"]

# ─────────────────────────────────────────────────────────────
#  TABS — attractive dark icons
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠  Dashboard",
    "🔭  Data Explorer",
    "📡  Model Performance",
    "⚡  Predict AQI",
    "💡  About",
])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">⚡ Model Performance Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📉", "MAE",       f"{result['mae']:.2f}",       "Mean Absolute Error"),
        ("📊", "RMSE",      f"{result['rmse']:.2f}",      "Root Mean Squared Error"),
        ("🎯", "R² Score",  f"{result['r2']:.4f}",        "Coefficient of Determination"),
        ("✅", "Accuracy",  f"{result['accuracy']:.1f}%", "Model Accuracy (R² × 100)"),
    ]
    for col, (icon, label, value, tip) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <span class='card-icon'>{icon}</span>
                <h3>{label}</h3>
                <h2>{value}</h2>
                <p>{tip}</p>
            </div>""", unsafe_allow_html=True)

    # Feature Importance
    st.markdown('<div class="section-title">🔬 Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-insight">💡 <b>Key Insight:</b> The longer the bar, the more that pollutant influences AQI predictions. <b>PM2.5</b> is typically the #1 driver — fine particles penetrate deep into lungs. If PM10 or NO₂ rank high, it signals vehicle/industrial pollution. Use this to prioritise which pollutants to monitor most closely.</div>', unsafe_allow_html=True)
    feat_imp = result["feat_imp"]
    fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#1e3a6e","#4f8cff","#38d9a9"],
                     title="Which pollutants drive AQI the most?")
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    dark_layout(fig_imp, height=430)
    st.plotly_chart(fig_imp, use_container_width=True)

    # AQI Distribution
    st.markdown('<div class="section-title">📈 AQI Distribution in Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-insight">💡 <b>Key Insight:</b> A peak in the <b>100–200 range</b> (Moderate) is common for Indian cities. A right-skewed distribution (long tail toward 400–500) means occasional severe pollution events — likely winter months or crop burning seasons. A healthy dataset should have variation across all AQI categories for the model to learn well.</div>', unsafe_allow_html=True)
    aqi_clean = df["AQI"].dropna()
    fig_hist = px.histogram(aqi_clean, nbins=40, color_discrete_sequence=["#4f8cff"],
                            title="How AQI values are spread in your dataset")
    fig_hist.update_layout(xaxis_title="AQI", yaxis_title="Count")
    dark_layout(fig_hist, height=360)
    st.plotly_chart(fig_hist, use_container_width=True)

    # AQI by City (if City column exists)
    if "City" in df.columns:
        st.markdown('<div class="section-title">🏙️ Average AQI by City</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-insight">💡 <b>Key Insight:</b> Cities in <span style="color:#e74c3c; font-weight:600;">red/purple</span> consistently exceed safe AQI levels (>200) and need urgent policy action. Cities in <span style="color:#27ae60; font-weight:600;">green</span> maintain healthy air. Use this comparison to benchmark your city against national averages and identify pollution hotspots.</div>', unsafe_allow_html=True)
        city_aqi = df.groupby("City")["AQI"].mean().dropna().sort_values(ascending=False).head(20).reset_index()
        fig_city = px.bar(city_aqi, x="City", y="AQI",
                          color="AQI", color_continuous_scale=["#27ae60","#f39c12","#e74c3c","#8e44ad"],
                          title="Top 20 cities by average AQI")
        fig_city.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
        dark_layout(fig_city, height=380)
        st.plotly_chart(fig_city, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",    f"{len(df):,}")
    c2.metric("Total Columns", f"{len(df.columns)}")
    c3.metric("AQI Records",   f"{df['AQI'].notna().sum():,}")

    # Interactive filter
    st.markdown('<div class="section-title">🔍 Interactive Data Preview</div>', unsafe_allow_html=True)
    rows_to_show = st.slider("Rows to display", 10, min(500, len(df)), 50, step=10)
    if "City" in df.columns:
        cities = ["All Cities"] + sorted(df["City"].dropna().unique().tolist())
        sel_city = st.selectbox("Filter by City", cities)
        show_df = df if sel_city == "All Cities" else df[df["City"] == sel_city]
    else:
        show_df = df
    st.dataframe(show_df.head(rows_to_show), use_container_width=True)

    # Missing values heatmap
    st.markdown('<div class="section-title">🕳️ Missing Values Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-insight">💡 <b>How to read this:</b> Each column is a feature. A <span style="color:#38d9a9; font-weight:600;">teal cell</span> means that data point is <b>missing</b>. Columns with many teal cells need imputation before training — our model handles this automatically using <b>median values</b>.</div>', unsafe_allow_html=True)
    fig_mv, ax_mv = plt.subplots(figsize=(14, 4))
    fig_mv.patch.set_facecolor(MPL_BG)
    ax_mv.set_facecolor(MPL_BG)
    sns.heatmap(df.isnull(), cbar=False, ax=ax_mv, cmap=["#1e3a6e","#38d9a9"], yticklabels=False)
    ax_mv.set_title("Missing Value Map  (teal = missing data)", color=MPL_FONT, fontsize=13, pad=10)
    ax_mv.tick_params(axis="x", colors=MPL_TICK, labelsize=9)
    ax_mv.tick_params(axis="y", colors=MPL_TICK)
    for spine in ax_mv.spines.values():
        spine.set_edgecolor(MPL_SPINE)
    st.pyplot(fig_mv)
    plt.close(fig_mv)

    # Descriptive stats
    st.markdown('<div class="section-title">📐 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-title">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-insight">💡 <b>How to read this:</b> Values close to <span style="color:#e74c3c; font-weight:600;">+1.0 (red)</span> mean two pollutants rise together — e.g. NO & NOx. Values near <span style="color:#4f8cff; font-weight:600;">−1.0 (blue)</span> mean they move oppositely. <b>High correlation with AQI</b> = that pollutant is a strong AQI predictor. Look at the AQI row/column for key insights.</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
    fig_corr.patch.set_facecolor(MPL_BG)
    ax_corr.set_facecolor(MPL_BG)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr,
                linewidths=0.4, linecolor="#1a2840", annot_kws={"size": 8, "color": "white"})
    ax_corr.set_title("Correlation Between All Numeric Features", color=MPL_FONT, fontsize=13, pad=12)
    ax_corr.tick_params(axis="x", colors=MPL_TICK, labelsize=9, rotation=45)
    ax_corr.tick_params(axis="y", colors=MPL_TICK, labelsize=9)
    for spine in ax_corr.spines.values():
        spine.set_edgecolor(MPL_SPINE)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

# ══════════════════════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    y_test = result["y_test"]
    y_pred = result["y_pred"]

    # Actual vs Predicted
    st.markdown('<div class="section-title">🎯 Actual vs Predicted AQI</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-insight">💡 <b>How to read this:</b> Each dot is one test record. Points hugging the <span style="color:#e74c3c; font-weight:600;">red dashed line</span> = accurate predictions. Dots scattered far from the line = prediction error. A tight cluster along the diagonal means our Random Forest model is performing well across all AQI ranges.</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(x=y_test, y=y_pred,
                             labels={"x": "Actual AQI", "y": "Predicted AQI"},
                             title="How closely does the model predict real AQI values?",
                             color_discrete_sequence=["#4f8cff"], opacity=0.55)
    mn = float(min(y_test.min(), y_pred.min()))
    mx = float(max(y_test.max(), y_pred.max()))
    fig_scatter.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                          line=dict(color="#e74c3c", dash="dash", width=2))
    fig_scatter.add_annotation(x=mx*0.78, y=mx, text="Perfect Fit ✦",
                                showarrow=False, font=dict(color="#e74c3c", size=12))
    dark_layout(fig_scatter, height=460)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Residuals
    st.markdown('<div class="section-title">📉 Residuals Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-insight">💡 <b>How to read this:</b> Residual = Actual AQI minus Predicted AQI. A bell-curve shape <b>centred at 0</b> means the model makes balanced errors — not consistently over- or under-predicting. A skewed or off-centre distribution signals a systematic bias that could be fixed with more features or better tuning.</div>', unsafe_allow_html=True)
    residuals = y_test.values - y_pred
    fig_res = px.histogram(residuals, nbins=50, color_discrete_sequence=["#38d9a9"],
                           title="Residuals centred near zero = good model",
                           labels={"value": "Residual (Actual − Predicted)", "count": "Frequency"})
    dark_layout(fig_res, height=360)
    st.plotly_chart(fig_res, use_container_width=True)

    # Interactive: Top N feature importance
    st.markdown('<div class="section-title">🔬 Feature Importance Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-insight">💡 <b>Interactive:</b> Drag the slider below to zoom into the most impactful features. Importance scores are normalised — they all sum to 1.0. A single feature scoring above <b>0.30</b> dominates predictions. If CO or SO₂ ranks unexpectedly high, it may indicate a specific industrial corridor in the dataset.</div>', unsafe_allow_html=True)
    top_n = st.slider("Show top N features", 3, len(result["feat_imp"]), len(result["feat_imp"]))
    feat_top = result["feat_imp"].head(top_n)
    fig_feat = px.bar(feat_top, x="Feature", y="Importance",
                      color="Importance", color_continuous_scale=["#1e3a6e","#4f8cff","#38d9a9"],
                      title=f"Top {top_n} Features by Importance")
    fig_feat.update_layout(coloraxis_showscale=False)
    dark_layout(fig_feat, height=380)
    st.plotly_chart(fig_feat, use_container_width=True)

    # Metrics table
    st.markdown('<div class="section-title">📋 Full Metrics Report</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        "Metric":   ["MAE","MSE","RMSE","R² Score","Model Accuracy"],
        "Value":    [f"{result['mae']:.4f}", f"{result['mse']:.4f}", f"{result['rmse']:.4f}",
                     f"{result['r2']:.6f}", f"{result['accuracy']:.2f}%"],
        "Meaning":  ["Average absolute error between actual & predicted AQI",
                     "Average squared error — penalises large mistakes",
                     "Square root of MSE — in AQI units",
                     "Proportion of AQI variance explained by model",
                     "R² expressed as percentage"],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  TAB 4 — PREDICT AQI
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🧪 Enter Pollutant Concentrations</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Fill in the pollutant values below and click <b>⚡ Predict AQI</b> to get an instant prediction from the trained model.</div>', unsafe_allow_html=True)

    df_num      = df[features].copy()
    df_num_clean= pd.DataFrame(imputer.transform(df_num), columns=features)
    medians     = df_num_clean.median()

    POLLUTANT_INFO = {
        "PM2.5":   ("PM2.5 — Fine Particulate Matter",    "µg/m³", 0.0,  999.0, "Particles < 2.5 µm. Major AQI driver."),
        "PM10":    ("PM10 — Coarse Particulate Matter",   "µg/m³", 0.0, 1000.0, "Particles < 10 µm."),
        "NO":      ("NO — Nitric Oxide",                  "µg/m³", 0.0,  500.0, "Primary pollutant from combustion."),
        "NO2":     ("NO2 — Nitrogen Dioxide",             "µg/m³", 0.0,  500.0, "Causes respiratory problems."),
        "NOx":     ("NOx — Nitrogen Oxides (Combined)",   "µg/m³", 0.0,  600.0, "NO + NO2 combined."),
        "NH3":     ("NH3 — Ammonia",                      "µg/m³", 0.0,  200.0, "From agriculture & industry."),
        "CO":      ("CO — Carbon Monoxide",               "mg/m³", 0.0,  100.0, "Toxic gas from fuel combustion."),
        "SO2":     ("SO2 — Sulphur Dioxide",              "µg/m³", 0.0,  500.0, "From burning fossil fuels."),
        "O3":      ("O3 — Ozone",                         "µg/m³", 0.0,  500.0, "Ground-level ozone."),
        "Benzene": ("Benzene — Carcinogenic VOC",         "µg/m³", 0.0,  100.0, "Known carcinogen."),
        "Toluene": ("Toluene — Volatile Organic Compound","µg/m³", 0.0,  200.0, "Solvent in paints & glues."),
        "Xylene":  ("Xylene — Aromatic Hydrocarbon",      "µg/m³", 0.0,  100.0, "Used in printing & leather."),
    }

    input_vals = {}
    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    for i, feat in enumerate(features):
        full_name, unit, lo, hi, desc = POLLUTANT_INFO.get(feat, (feat, "", 0.0, 1000.0, ""))
        default_val = max(lo, min(hi, float(medians[feat]) if feat in medians else 0.0))
        with cols[i % 3]:
            st.markdown(f"<div class='pollutant-label'>{full_name}</div>", unsafe_allow_html=True)
            input_vals[feat] = st.number_input(
                f"{feat} ({unit})", min_value=lo, max_value=hi,
                value=round(default_val, 2), step=0.1, help=desc,
                label_visibility="collapsed", key=f"inp_{feat}",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡  Predict AQI Now")

    if predict_clicked:
        input_df  = pd.DataFrame([input_vals])
        input_imp = pd.DataFrame(imputer.transform(input_df), columns=features)

        predicted_aqi = model.predict(input_imp)[0]

        category, color, impact = aqi_analysis(predicted_aqi)

        # =====================================
        # AQI ALERT SMS
        # =====================================
        if predicted_aqi > 300:
            try:
                client = Client(account_sid, auth_token)
                client.messages.create(
                    body=f"AQI ALERT! Current AQI: {predicted_aqi:.1f}",
                    from_="+14783162164",
                    to="+919335120267"
                )
            except Exception as e:
                st.error(f"SMS Error: {e}")

        # Result card
        st.markdown(f"""
        <div class='aqi-result' style='background: linear-gradient(135deg, {color}bb, {color}88);
             border: 1px solid {color}55;'>
            <h2>{predicted_aqi:.1f}</h2>
            <h3>AQI Category: {category}</h3>
            <p>{impact}</p>
        </div>""", unsafe_allow_html=True)

        # Gauge
        st.markdown('<div class="section-title">📊 AQI Scale</div>', unsafe_allow_html=True)
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_aqi,
            number={"font": {"size": 44, "color": color}},
            gauge={
                "axis": {"range": [0, 500], "tickcolor": DARK_FONT, "tickfont": {"color": DARK_FONT}},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor": DARK_BG,
                "bordercolor": "rgba(99,140,255,0.2)",
                "steps": [
                    {"range": [0,   50],  "color": "rgba(39,174,96,0.25)"},
                    {"range": [51,  100], "color": "rgba(46,204,113,0.2)"},
                    {"range": [101, 200], "color": "rgba(243,156,18,0.2)"},
                    {"range": [201, 300], "color": "rgba(230,126,34,0.2)"},
                    {"range": [301, 400], "color": "rgba(231,76,60,0.2)"},
                    {"range": [401, 500], "color": "rgba(142,68,173,0.2)"},
                ],
                "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": predicted_aqi},
            },
            title={"text": f"<b>Predicted AQI: {predicted_aqi:.1f}  —  {category}</b>",
                   "font": {"color": DARK_FONT, "size": 15}},
        ))
        gauge_fig.update_layout(
            paper_bgcolor=DARK_PAPER,
            plot_bgcolor=DARK_BG,
            font_color=DARK_FONT,
            height=360,
            margin=dict(t=70, b=20, l=40, r=40),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Reference table
        st.markdown('<div class="section-title">📋 AQI Category Reference</div>', unsafe_allow_html=True)
        ref_rows = []
        for lo, hi, cat, clr, imp in AQI_LIMITS:
            marker = "◀ YOU ARE HERE" if cat == category else ""
            ref_rows.append({"Range": f"{lo}–{hi}", "Category": cat, "Health Impact": imp, "": marker})
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">💡 About This Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">This professional Streamlit dashboard replicates every step of your <b>AQI_Pred1.ipynb</b> notebook as a fully interactive dark-mode web app.</div>', unsafe_allow_html=True)

    steps_info = [
        ("📁", "Data Loading",          "Reads city_day.csv and removes duplicate rows."),
        ("🗓️", "Date Parsing",          "Converts Date column and extracts Year, Month, Day features."),
        ("🔍", "Exploratory Analysis",   "Missing value heatmap, AQI histogram, correlation heatmap."),
        ("🧹", "Missing Value Handling", "Applies Median Imputation (SimpleImputer) to all numeric columns."),
        ("✂️",  "Train / Test Split",    "80% training data, 20% test data (random_state=42)."),
        ("🤖", "Random Forest Model",   "200 estimators, random_state=42 — same as your notebook."),
        ("📊", "Model Evaluation",      "Reports MAE, MSE, RMSE, R² Score, and Model Accuracy."),
        ("⚡", "Live Prediction",       "Enter custom pollutant values and get an instant AQI prediction."),
    ]
    for i, (icon, title, desc) in enumerate(steps_info, 1):
        st.markdown(f"""
        <div class='step-row'>
            <span class='step-badge'>{i}</span>
            <div>
                <div style='font-weight:600; color:#c8d8f0; margin-bottom:3px;'>{icon} {title}</div>
                <div style='color:#6a87a8; font-size:0.87rem;'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌿 AQI Categories (India Standard)")
    st.dataframe(
        pd.DataFrame([(lo,hi,cat,imp) for lo,hi,cat,_,imp in AQI_LIMITS],
                     columns=["AQI From","AQI To","Category","Health Impact"]),
        use_container_width=True, hide_index=True
    )
    st.markdown("---")
    st.markdown("""
**📦 Required Libraries:**
```
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly
```

**▶️ How to Run:**
```bash
streamlit run aqi_prediction_app_dark.py
```
""")

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='dark-footer'>
    🌿 AQI Prediction Dashboard &nbsp;·&nbsp; Random Forest Regressor &nbsp;·&nbsp; Dark Mode
</div>
""", unsafe_allow_html=True)
