"""
ChemOptix — Streamlit Frontend
Tab 1: Single prediction via form
Tab 2: Batch prediction via CSV upload
"""

import os
import io
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChemOptix — Gas Turbine AI",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

.chemoptix-banner {
    background: linear-gradient(135deg, #051528 0%, #0a2040 100%);
    border: 1px solid rgba(0,210,200,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.chemoptix-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,210,200,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.banner-tag {
    display: inline-block;
    background: rgba(0,210,200,0.1);
    border: 1px solid rgba(0,210,200,0.3);
    color: #00d2c8;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 100px;
    margin-bottom: 0.75rem;
}
.banner-title {
    font-size: 2.2rem; font-weight: 800;
    letter-spacing: -0.03em; margin: 0.25rem 0; color: #e8f4f8;
}
.banner-title span {
    background: linear-gradient(135deg, #00d2c8, #00e87a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.banner-sub { color: #8ba5bf; font-size: 0.95rem; max-width: 560px; margin-top: 0.4rem; }

.section-head {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #00d2c8; margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,210,200,0.15);
}

.metric-row { display: flex; gap: 1rem; margin: 1.25rem 0; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 180px;
    background: rgba(8,18,32,0.8);
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 16px; padding: 1.5rem;
    text-align: center; backdrop-filter: blur(10px);
}
.metric-box.yield  { border-color: rgba(0,232,122,0.25); }
.metric-box.co     { border-color: rgba(0,210,200,0.25); }
.metric-box.nox    { border-color: rgba(255,181,71,0.25); }
.metric-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.metric-lbl  { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #4a6380; margin-bottom: 0.3rem; }
.metric-val  { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.04em; font-family: monospace; line-height: 1; }
.metric-unit { font-size: 0.72rem; color: #4a6380; font-family: monospace; margin-top: 0.15rem; }
.metric-box.yield .metric-val { color: #00e87a; }
.metric-box.co    .metric-val { color: #00d2c8; }
.metric-box.nox   .metric-val { color: #ffb547; }

.pill {
    display: inline-block;
    font-size: 0.65rem; font-weight: 700;
    padding: 3px 10px; border-radius: 100px;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.5rem;
}
.pill-good   { background: rgba(0,232,122,0.15); color: #00e87a; border: 1px solid rgba(0,232,122,0.3); }
.pill-warn   { background: rgba(255,181,71,0.15); color: #ffb547; border: 1px solid rgba(255,181,71,0.3); }
.pill-danger { background: rgba(255,82,82,0.15);  color: #ff5252; border: 1px solid rgba(255,82,82,0.3); }

.risk-banner {
    display: flex; align-items: center; gap: 0.75rem;
    background: rgba(8,18,32,0.8);
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
}
.risk-text { font-size: 0.95rem; color: #8ba5bf; flex: 1; }
.risk-pill {
    font-size: 0.7rem; font-weight: 700;
    padding: 4px 14px; border-radius: 100px;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.risk-low    { background: rgba(0,232,122,0.15); color: #00e87a; border: 1px solid rgba(0,232,122,0.3); }
.risk-medium { background: rgba(255,181,71,0.15); color: #ffb547; border: 1px solid rgba(255,181,71,0.3); }
.risk-high   { background: rgba(255,82,82,0.15);  color: #ff5252; border: 1px solid rgba(255,82,82,0.3); }

.suggestion {
    display: flex; align-items: flex-start; gap: 0.75rem;
    background: rgba(0,210,200,0.04);
    border: 1px solid rgba(0,210,200,0.12);
    border-radius: 10px; padding: 0.85rem 1rem;
    margin-bottom: 0.6rem; color: #8ba5bf; font-size: 0.88rem;
}
.sug-num {
    width: 24px; height: 24px; flex-shrink: 0;
    background: rgba(0,210,200,0.12); color: #00d2c8;
    border: 1px solid rgba(0,210,200,0.3); border-radius: 50%;
    font-size: 0.68rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center; line-height: 1;
}

.api-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}
.api-dot.green { background: #00e87a; box-shadow: 0 0 8px rgba(0,232,122,0.6); }
.api-dot.red   { background: #ff5252; box-shadow: 0 0 8px rgba(255,82,82,0.6); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Batch table styling */
.batch-info {
    background: rgba(0,210,200,0.05);
    border: 1px solid rgba(0,210,200,0.2);
    border-radius: 12px; padding: 1rem 1.25rem;
    font-size: 0.85rem; color: #8ba5bf; margin-bottom: 1rem;
}
.batch-info strong { color: #e8f4f8; }
.template-box {
    background: rgba(8,18,32,0.8);
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 10px; padding: 1rem;
    font-family: monospace; font-size: 0.78rem;
    color: #00d2c8; margin: 0.5rem 0;
    white-space: pre;
}
</style>
""", unsafe_allow_html=True)


# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chemoptix-banner">
  <div class="banner-tag">⚗ Gas Turbine Intelligence Platform</div>
  <div class="banner-title">ChemOptix <span>AI</span></div>
  <div class="banner-sub">
    Predict turbine energy yield, CO and NOX emissions in real-time.
    Single prediction with Gemini AI · Bulk CSV batch prediction.
  </div>
</div>
""", unsafe_allow_html=True)


# ── API health check ───────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

api_ok = check_api_health()
dot_cls = "green" if api_ok else "red"
api_label = "API Connected" if api_ok else "API Offline — start FastAPI backend"
st.markdown(
    f'<p style="font-size:0.78rem;color:#4a6380;margin-bottom:1rem;">'
    f'<span class="api-dot {dot_cls}"></span>{api_label} · {API_URL}</p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🔮  Single Prediction", "📂  Batch Prediction (CSV Upload)"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left_col, right_col = st.columns([1, 1.4], gap="large")

    with left_col:
        st.markdown('<div class="section-head">🌡 Ambient Conditions</div>', unsafe_allow_html=True)
        AT   = st.number_input("Ambient Temperature (°C)",       value=17.71,   min_value=-10.0, max_value=50.0,   step=0.01,  format="%.2f",  help="Outside air temperature")
        AP   = st.number_input("Ambient Pressure (mbar)",        value=1013.07, min_value=980.0, max_value=1050.0, step=0.01,  format="%.2f",  help="Atmospheric pressure")
        AH   = st.number_input("Ambient Humidity (%)",           value=77.87,   min_value=0.0,   max_value=100.0,  step=0.01,  format="%.2f",  help="Outside air humidity")

        st.markdown('<div class="section-head" style="margin-top:1.25rem;">⚙ Turbine Core</div>', unsafe_allow_html=True)
        TIT  = st.number_input("Turbine Inlet Temperature (°C)", value=1081.43, min_value=1000.0,max_value=1110.0, step=0.01,  format="%.2f",  help="Main control variable")
        TAT  = st.number_input("Turbine After Temperature (°C)", value=546.16,  min_value=500.0, max_value=560.0,  step=0.01,  format="%.2f",  help="Temperature after energy extraction")
        CDP  = st.number_input("Compressor Discharge Pressure (bar)", value=12.06, min_value=9.0, max_value=16.0, step=0.001, format="%.3f", help="Pressure before combustion")

        st.markdown('<div class="section-head" style="margin-top:1.25rem;">💨 Flow & Pressure</div>', unsafe_allow_html=True)
        AFDP = st.number_input("Air Filter Diff. Pressure (mbar)", value=3.93,  min_value=2.0,   max_value=8.0,    step=0.001, format="%.3f", help="Filter blockage indicator")
        GTEP = st.number_input("Gas Turbine Exhaust Pressure (mbar)", value=25.56, min_value=15.0, max_value=45.0, step=0.001, format="%.3f", help="Pressure at turbine exhaust")
        year = st.selectbox("Measurement Year", options=[2011,2012,2013,2014,2015], index=4)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict & Optimize", type="primary", use_container_width=True, disabled=not api_ok, key="single_predict")
        if st.button("↺ Reset", use_container_width=True, key="single_reset"):
            st.session_state.pop("single_result", None)
            st.rerun()

    with right_col:
        # Placeholder
        if "single_result" not in st.session_state:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:420px;text-align:center;
                background:rgba(8,18,32,0.5);
                border:1px dashed rgba(0,210,200,0.2);border-radius:20px;color:#4a6380;">
                <div style="font-size:3rem;margin-bottom:1rem;">⚡</div>
                <div style="font-size:1rem;font-weight:600;color:#8ba5bf;">No prediction yet</div>
                <div style="font-size:0.82rem;margin-top:0.5rem;">
                    Fill in the parameters and click<br><strong>Predict & Optimize</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        if predict_btn:
            payload = dict(AT=AT, AP=AP, AH=AH, AFDP=AFDP, GTEP=GTEP, TIT=TIT, TAT=TAT, CDP=CDP, year=year)
            with st.spinner("Running XGBoost + Gemini AI…"):
                try:
                    res = requests.post(f"{API_URL}/api/predict", json=payload, timeout=30)
                    res.raise_for_status()
                    st.session_state["single_result"] = res.json()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach FastAPI. Is it running?")
                    st.stop()
                except requests.exceptions.HTTPError:
                    st.error(f"❌ API Error: {res.json().get('detail', 'Unknown error')}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    st.stop()

        if "single_result" in st.session_state:
            data  = st.session_state["single_result"]
            preds = data["predictions"]
            opt   = data["optimization"]
            TEY, CO, NOX = preds["TEY"], preds["CO"], preds["NOX"]

            def tey_pill(v):
                if v >= 140: return '<span class="pill pill-good">✓ Above Target</span>'
                if v >= 120: return '<span class="pill pill-warn">⚠ Near Target</span>'
                return '<span class="pill pill-danger">✗ Below Target</span>'
            def co_pill(v):
                if v <= 2.0: return '<span class="pill pill-good">✓ Acceptable</span>'
                if v <= 5.0: return '<span class="pill pill-warn">⚠ Elevated</span>'
                return '<span class="pill pill-danger">✗ High CO</span>'
            def nox_pill(v):
                if v <= 65: return '<span class="pill pill-good">✓ Acceptable</span>'
                if v <= 90: return '<span class="pill pill-warn">⚠ Elevated</span>'
                return '<span class="pill pill-danger">✗ High NOX</span>'

            st.markdown(f"""
            <div class="section-head">📊 Prediction Results</div>
            <div class="metric-row">
              <div class="metric-box yield">
                <div class="metric-icon">⚡</div>
                <div class="metric-lbl">Energy Yield</div>
                <div class="metric-val">{TEY:.2f}</div>
                <div class="metric-unit">MWH</div>
                {tey_pill(TEY)}
              </div>
              <div class="metric-box co">
                <div class="metric-icon">💨</div>
                <div class="metric-lbl">CO Emission</div>
                <div class="metric-val">{CO:.4f}</div>
                <div class="metric-unit">mg/m³</div>
                {co_pill(CO)}
              </div>
              <div class="metric-box nox">
                <div class="metric-icon">🌫</div>
                <div class="metric-lbl">NOX Emission</div>
                <div class="metric-val">{NOX:.2f}</div>
                <div class="metric-unit">mg/m³</div>
                {nox_pill(NOX)}
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-head">📈 vs Benchmarks</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("TEY Target", "140 MWH", f"{TEY-140:+.2f} MWH")
                st.progress(min(TEY / 180, 1.0))
            with c2:
                st.metric("CO Limit", "2.0 mg/m³", f"{CO-2.0:+.4f}")
                st.progress(min(CO / 10.0, 1.0))
            with c3:
                st.metric("NOX Limit", "65 mg/m³", f"{NOX-65:+.2f}")
                st.progress(min(NOX / 120.0, 1.0))

            risk = opt.get("risk_level", "medium")
            summary = opt.get("summary", "")
            suggestions = opt.get("suggestions", [])

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-head">✨ Gemini AI Optimization</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="risk-banner">
              <div style="font-size:1.5rem;">🤖</div>
              <div class="risk-text">{summary}</div>
              <span class="risk-pill risk-{risk}">Risk: {risk.upper()}</span>
            </div>""", unsafe_allow_html=True)

            sugg_html = "".join(
                f'<div class="suggestion"><span class="sug-num">{i}</span><span>{s}</span></div>'
                for i, s in enumerate(suggestions, 1)
            )
            st.markdown(sugg_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇ Download Result (JSON)",
                data=json.dumps(data, indent=2),
                file_name="chemoptix_single_result.json",
                mime="application/json",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Info & format guide ────────────────────────────────────────────────────
    st.markdown("""
    <div class="batch-info">
      <strong>How it works:</strong> Upload a CSV file with your turbine readings.
      The model predicts <strong>TEY, CO and NOX</strong> for every row instantly.
      Download the full results table as CSV when done.<br><br>
      <strong>Limits:</strong> Maximum 500 rows per upload · No Gemini suggestions in batch mode
      (use the Single Prediction tab for AI recommendations on specific readings)
    </div>
    """, unsafe_allow_html=True)

    # ── Sample CSV download ────────────────────────────────────────────────────
    st.markdown('<div class="section-head">📋 Required CSV Format</div>', unsafe_allow_html=True)

    sample_data = pd.DataFrame([
        {"AT": 17.71, "AP": 1013.07, "AH": 77.87, "AFDP": 3.93, "GTEP": 25.56, "TIT": 1081.43, "TAT": 546.16, "CDP": 12.06, "year": 2015},
        {"AT": 22.50, "AP": 1010.20, "AH": 65.30, "AFDP": 4.10, "GTEP": 27.80, "TIT": 1090.00, "TAT": 548.00, "CDP": 12.50, "year": 2014},
        {"AT": 10.20, "AP": 1018.50, "AH": 85.00, "AFDP": 3.50, "GTEP": 23.10, "TIT": 1070.00, "TAT": 542.00, "CDP": 11.50, "year": 2013},
    ])

    st.dataframe(sample_data, use_container_width=True, hide_index=True)

    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        "⬇ Download Sample CSV Template",
        data=sample_csv,
        file_name="chemoptix_sample_input.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── File uploader ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">📂 Upload Your Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV file (max 500 rows)",
        type=["csv"],
        help="Columns required: AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP, year",
    )

    REQUIRED_COLS = {"AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "year"}

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            st.stop()

        # Validate columns
        missing_cols = REQUIRED_COLS - set(df_input.columns)
        if missing_cols:
            st.error(f"❌ Missing columns in your CSV: **{', '.join(sorted(missing_cols))}**")
            st.info("Download the sample template above to see the required format.")
            st.stop()

        if len(df_input) > 500:
            st.error("❌ File has more than 500 rows. Please split it into smaller batches.")
            st.stop()

        st.success(f"✅ File loaded — **{len(df_input)} rows** · {len(df_input.columns)} columns detected")

        # Preview
        with st.expander("👁 Preview uploaded data", expanded=False):
            st.dataframe(df_input.head(10), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_batch = st.button(
            f"🚀 Run Batch Prediction ({len(df_input)} rows)",
            type="primary",
            use_container_width=True,
            disabled=not api_ok,
            key="batch_predict",
        )

        if run_batch:
            # Build payload
            rows = df_input[list(REQUIRED_COLS)].copy()
            rows["year"] = rows["year"].astype(int)
            payload = rows.to_dict(orient="records")

            with st.spinner(f"Predicting {len(payload)} rows…"):
                try:
                    res = requests.post(
                        f"{API_URL}/api/predict/batch",
                        json=payload,
                        timeout=60,
                    )
                    res.raise_for_status()
                    batch_results = res.json()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach FastAPI. Is it running?")
                    st.stop()
                except requests.exceptions.HTTPError:
                    st.error(f"❌ API Error: {res.json().get('detail', 'Unknown error')}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    st.stop()

            # Build results DataFrame
            result_rows = []
            for r in batch_results:
                inp = r["inputs"]
                result_rows.append({
                    "Row":   r["row_index"],
                    "AT":    inp["AT"],
                    "AP":    inp["AP"],
                    "AH":    inp["AH"],
                    "AFDP":  inp["AFDP"],
                    "GTEP":  inp["GTEP"],
                    "TIT":   inp["TIT"],
                    "TAT":   inp["TAT"],
                    "CDP":   inp["CDP"],
                    "year":  inp["year"],
                    "TEY (MWH)":   r["TEY"],
                    "CO (mg/m³)":  r["CO"],
                    "NOX (mg/m³)": r["NOX"],
                    "Status":      r["status"],
                })
            df_results = pd.DataFrame(result_rows)

            # Summary stats
            success_df = df_results[df_results["Status"] == "success"]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-head">📊 Batch Summary</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows Processed", len(df_results))
            m2.metric("Avg TEY (MWH)",  f"{success_df['TEY (MWH)'].mean():.2f}"  if not success_df.empty else "—")
            m3.metric("Avg CO (mg/m³)", f"{success_df['CO (mg/m³)'].mean():.4f}" if not success_df.empty else "—")
            m4.metric("Avg NOX (mg/m³)",f"{success_df['NOX (mg/m³)'].mean():.2f}"if not success_df.empty else "—")

            # Color-coded table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-head">📋 Full Results Table</div>', unsafe_allow_html=True)

            def color_tey(val):
                if val >= 140:  return "color: #00e87a"
                if val >= 120:  return "color: #ffb547"
                return "color: #ff5252"
            def color_co(val):
                if val <= 2.0:  return "color: #00e87a"
                if val <= 5.0:  return "color: #ffb547"
                return "color: #ff5252"
            def color_nox(val):
                if val <= 65:   return "color: #00e87a"
                if val <= 90:   return "color: #ffb547"
                return "color: #ff5252"

            styled = (
                df_results.style
                .applymap(color_tey,  subset=["TEY (MWH)"])
                .applymap(color_co,   subset=["CO (mg/m³)"])
                .applymap(color_nox,  subset=["NOX (mg/m³)"])
                .format({"TEY (MWH)": "{:.2f}", "CO (mg/m³)": "{:.4f}", "NOX (mg/m³)": "{:.2f}"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Download results
            st.markdown("<br>", unsafe_allow_html=True)
            csv_out = df_results.to_csv(index=False)
            st.download_button(
                "⬇ Download Full Results (CSV)",
                data=csv_out,
                file_name="chemoptix_batch_results.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(0,210,200,0.1);margin:3rem 0 1rem;" />
<p style="text-align:center;color:#4a6380;font-size:0.78rem;">
  ChemOptix · FastAPI + XGBoost + Google Gemini ·
  <a href="https://github.com/Anshikasharma37/ChemOptix" target="_blank" style="color:#00d2c8;">GitHub</a>
</p>
""", unsafe_allow_html=True)
