"""
app.py
------
Polished Streamlit analytics dashboard for the Smart Student Performance Predictor.

Trained on a fully synthetic dataset of 300 simulated student records.
Predictions are for educational / demonstration purposes only and do not
reflect real-world student performance or establish any real predictive validity.
"""

import datetime
import io
import json
import os
import textwrap

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths (resolved relative to this file's location) ───────────
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "model_artifacts", "student_performance_model.joblib")
META_PATH  = os.path.join(ROOT_DIR, "model_artifacts", "model_metadata.json")

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

# ── Custom CSS — navy / teal professional palette ────────────────
st.markdown(
    """
    <style>
    /* ---- global background ---- */
    .stApp { background-color: #f0f4f8; }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #0f2044;
    }
    [data-testid="stSidebar"] * { color: #dce8f5 !important; }
    [data-testid="stSidebar"] .stSlider > label { color: #a8c8f0 !important; }

    /* ---- top header band ---- */
    .header-band {
        background: linear-gradient(90deg, #0f2044 0%, #1a3a6e 60%, #0e7490 100%);
        padding: 1.4rem 2rem 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.4rem;
    }
    .header-band h1 { color: #ffffff; font-size: 1.85rem; margin: 0; }
    .header-band p  { color: #b0d0ee; font-size: 0.95rem; margin: 0.25rem 0 0 0; }

    /* ---- KPI card ---- */
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        border-left: 5px solid #0e7490;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 0.6rem;
    }
    .kpi-label { font-size: 0.78rem; color: #5a7a99; text-transform: uppercase;
                 letter-spacing: 0.06em; font-weight: 600; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #0f2044; line-height: 1.15; }
    .kpi-sub   { font-size: 0.82rem; color: #5a7a99; margin-top: 0.15rem; }

    /* ---- suggestion card ---- */
    .tip-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        border-left: 4px solid #0e7490;
        margin-bottom: 0.55rem;
        font-size: 0.92rem;
        color: #1e3a5f;
    }

    /* ---- section heading ---- */
    .section-heading {
        font-size: 1.05rem; font-weight: 700;
        color: #0f2044; margin-bottom: 0.5rem;
        border-bottom: 2px solid #d0dff0; padding-bottom: 0.3rem;
    }

    /* ---- tab styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #dce8f5;
        border-radius: 8px;
        padding: 4px 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #1a3a6e;
        font-weight: 600;
        padding: 0.4rem 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f2044 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model artifacts (cached — loaded only once per session) ─
@st.cache_resource
def load_artifacts():
    """Load the trained model and metadata. Raises FileNotFoundError if missing."""
    missing = [p for p in (MODEL_PATH, META_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Required file(s) not found: {missing}\n"
            "Please run  python model/train_model.py  from the project root first."
        )
    mdl = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return mdl, meta["features"]

try:
    model, feature_names = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as e:
    artifacts_loaded = False
    load_error = str(e)

# ── Header band ──────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-band">
      <h1>🎓 Smart Student Performance Predictor</h1>
      <p>Random Forest · ML Internship Demo · Synthetic dataset — for educational purposes only</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Abort early if artifacts are missing ─────────────────────────
if not artifacts_loaded:
    st.error(f"**Model artifacts not found.**\n\n{load_error}")
    st.stop()

# ── Sidebar: user inputs ─────────────────────────────────────────
st.sidebar.markdown("## 📋 Student Details")
st.sidebar.caption("Adjust the sliders and click **Predict** to see the estimated final score.")

study_hours = st.sidebar.slider(
    "Study Hours (per week)",
    min_value=0.0, max_value=12.0, value=6.0, step=0.5,
    help="Average number of hours the student studies per week.",
)
attendance_pct = st.sidebar.slider(
    "Attendance Percentage (%)",
    min_value=0.0, max_value=100.0, value=75.0, step=1.0,
    help="Percentage of classes attended.",
)
prev_exam_score = st.sidebar.slider(
    "Previous Exam Score (out of 100)",
    min_value=0.0, max_value=100.0, value=60.0, step=1.0,
    help="Score obtained in the most recent previous exam.",
)
assignment_score = st.sidebar.slider(
    "Assignment Score (out of 100)",
    min_value=0.0, max_value=100.0, value=65.0, step=1.0,
    help="Average score across all submitted assignments.",
)

st.sidebar.divider()
predict_clicked = st.sidebar.button(
    "🔍 Predict Final Score", use_container_width=True, type="primary"
)

st.sidebar.divider()
st.sidebar.caption(
    "**Note:** This model was trained on 300 artificially generated records. "
    "Results are for educational demonstration only."
)

# ── Prediction logic (run whenever button is clicked) ────────────
prediction_ready = False
safe_prediction  = None

if predict_clicked:
    input_data = pd.DataFrame(
        [[study_hours, attendance_pct, prev_exam_score, assignment_score]],
        columns=feature_names,   # exact order from metadata
    )
    raw_prediction  = model.predict(input_data)[0]
    safe_prediction = float(np.clip(raw_prediction, 0.0, 100.0))
    prediction_ready = True

# ── Tabs ─────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_model, tab_bulk, tab_whatif, tab_report = st.tabs(
    [
        "📊 Predict Performance",
        "💡 Student Insights",
        "🔬 Model Insights",
        "📂 Bulk Predictions",
        "🔀 What-If Simulator",
        "📄 Student Report",
    ]
)

# ════════════════════════════════════════════════════════════════
# TAB 1 — Predict Performance
# ════════════════════════════════════════════════════════════════
with tab_predict:
    col_result, col_inputs = st.columns([1.1, 1], gap="large")

    # ── Result KPI card ─────────────────────────────────────────
    with col_result:
        st.markdown('<p class="section-heading">Predicted Final Score</p>',
                    unsafe_allow_html=True)

        if prediction_ready:
            # Choose accent colour by performance band
            if safe_prediction >= 75:
                band_color = "#16a34a"   # green
                band_label = "Strong performance predicted ✅"
            elif safe_prediction >= 50:
                band_color = "#0e7490"   # teal
                band_label = "Moderate performance predicted ℹ️"
            else:
                band_color = "#b45309"   # amber
                band_label = "Low performance predicted ⚠️"

            st.markdown(
                f"""
                <div class="kpi-card" style="border-left-color:{band_color};">
                  <div class="kpi-label">Estimated Final Exam Score</div>
                  <div class="kpi-value" style="color:{band_color};">
                      {safe_prediction:.2f} <span style="font-size:1.1rem;color:#5a7a99;">/ 100</span>
                  </div>
                  <div class="kpi-sub">{band_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="kpi-card">
                  <div class="kpi-label">Estimated Final Exam Score</div>
                  <div class="kpi-value" style="color:#9ab0cc;">— / 100</div>
                  <div class="kpi-sub">Adjust sliders and click <b>Predict Final Score</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Input summary KPI cards ──────────────────────────────────
    with col_inputs:
        st.markdown('<p class="section-heading">Current Inputs</p>',
                    unsafe_allow_html=True)

        input_items = [
            ("Study Hours / week",    f"{study_hours:.1f} h"),
            ("Attendance",            f"{attendance_pct:.0f} %"),
            ("Previous Exam Score",   f"{prev_exam_score:.0f} / 100"),
            ("Assignment Score",      f"{assignment_score:.0f} / 100"),
        ]
        for label, value in input_items:
            st.markdown(
                f"""
                <div class="kpi-card" style="padding:0.65rem 1.1rem; border-left-color:#1a3a6e;">
                  <div class="kpi-label">{label}</div>
                  <div style="font-size:1.25rem;font-weight:700;color:#0f2044;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ════════════════════════════════════════════════════════════════
# TAB 2 — Student Insights
# ════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<p class="section-heading">General Study Suggestions</p>',
                unsafe_allow_html=True)
    st.caption(
        "These are rule-based general suggestions derived from your current input values. "
        "They are **not** personalised advice and do not guarantee score improvements."
    )

    suggestions = []

    # ── Study hours ─────────────────────────────────────────────
    if study_hours < 2:
        suggestions.append(
            "📚 <strong>Study hours are very low (&lt; 2 h/week).</strong> "
            "Try gradually increasing to at least 2–3 focused sessions per week. "
            "Even short, regular study blocks are more effective than occasional long sessions."
        )
    elif study_hours < 5:
        suggestions.append(
            "📚 <strong>Study hours are building up (2–5 h/week).</strong> "
            "Maintain this consistency and consider using focused study techniques "
            "such as the Pomodoro method or active recall to get more out of each session."
        )
    else:
        suggestions.append(
            "📚 <strong>Study hours are strong (≥ 5 h/week).</strong> "
            "Keep a balanced and consistent routine — steady effort over time "
            "tends to outperform last-minute cramming."
        )

    # ── Attendance ───────────────────────────────────────────────
    if attendance_pct < 75:
        suggestions.append(
            "🏫 <strong>Attendance is below 75%.</strong> "
            "Improving attendance where possible helps you stay current with material "
            "and benefit from in-class explanations that are hard to replicate from notes alone."
        )
    else:
        suggestions.append(
            "🏫 <strong>Attendance is good (≥ 75%).</strong> "
            "Consistent presence in class is a strong foundation — "
            "keep engaging actively during sessions."
        )

    # ── Previous exam score ──────────────────────────────────────
    if prev_exam_score < 50:
        suggestions.append(
            "📝 <strong>Previous exam score was below 50.</strong> "
            "Focus on reviewing fundamental concepts before moving to advanced topics. "
            "Working through practice questions and checking answers carefully can "
            "identify and close specific knowledge gaps."
        )
    else:
        suggestions.append(
            "📝 <strong>Previous exam score was 50 or above.</strong> "
            "Build on this with timed practice under exam conditions to strengthen "
            "both knowledge recall and time management."
        )

    # ── Assignment score ─────────────────────────────────────────
    if assignment_score < 60:
        suggestions.append(
            "📋 <strong>Assignment score is below 60.</strong> "
            "Review any feedback received on past assignments and work through "
            "similar practice exercises. Assignments are an early signal of exam readiness."
        )
    else:
        suggestions.append(
            "📋 <strong>Assignment score is 60 or above.</strong> "
            "Strong assignment performance generally reflects good ongoing engagement "
            "with the material — keep submitting work consistently and on time."
        )

    for tip in suggestions:
        st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

    st.divider()

    # Input radar-style summary table
    st.markdown('<p class="section-heading">Input Snapshot</p>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Feature":      ["Study Hours / week", "Attendance %",
                         "Previous Exam Score", "Assignment Score"],
        "Your Value":   [study_hours, attendance_pct, prev_exam_score, assignment_score],
        "Range":        ["0 – 12 h", "0 – 100 %", "0 – 100", "0 – 100"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<p class="section-heading">Feature Importance</p>',
                unsafe_allow_html=True)
    st.caption(
        "Mean decrease in impurity (MDI) across all 100 decision trees in the Random Forest. "
        "Higher values indicate features the model relied on more heavily when splitting nodes."
    )

    # ── Feature importance chart ─────────────────────────────────
    importances   = model.feature_importances_
    indices       = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    label_map = {
        "study_hours":      "Study Hours",
        "attendance_pct":   "Attendance %",
        "prev_exam_score":  "Prev. Exam Score",
        "assignment_score": "Assignment Score",
    }
    sorted_labels = [label_map.get(n, n) for n in sorted_names]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor("#f0f4f8")
    ax.set_facecolor("#f0f4f8")

    colors = ["#0e7490" if i == 0 else "#1a3a6e" for i in range(len(sorted_labels))]
    bars = ax.barh(
        sorted_labels[::-1], sorted_values[::-1],
        color=colors[::-1], edgecolor="#f0f4f8", height=0.55,
    )
    ax.set_xlabel("Importance (mean decrease in impurity)", fontsize=9, color="#5a7a99")
    ax.set_title("Feature Importance — Random Forest (100 trees)", fontsize=10,
                 color="#0f2044", fontweight="bold", pad=10)
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=9, color="#0f2044")
    ax.set_xlim(0, max(sorted_values) * 1.28)
    ax.tick_params(colors="#0f2044", labelsize=9)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#d0dff0")
    ax.xaxis.set_visible(False)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── Model card ───────────────────────────────────────────────
    st.markdown('<p class="section-heading">Model Card</p>', unsafe_allow_html=True)

    model_card = pd.DataFrame({
        "Property": [
            "Algorithm", "Number of Trees", "Random Seed",
            "Training Split", "Feature Scaling",
            "Training Data", "Target Variable",
        ],
        "Value": [
            "Random Forest Regressor (scikit-learn)",
            "100",
            "42 (fixed — fully reproducible)",
            "80% train / 20% test",
            "None (not required for tree-based models)",
            "300 synthetic student records",
            "Final exam score (0 – 100)",
        ],
    })
    st.dataframe(model_card, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ This model was trained on a fully synthetic dataset. "
        "Evaluation metrics reflect performance on simulated data only and "
        "do not imply real-world predictive validity."
    )

# ════════════════════════════════════════════════════════════════
# TAB 4 — Bulk Predictions
# ════════════════════════════════════════════════════════════════

# Column names and their valid numeric ranges
BULK_FEATURES = ["study_hours", "attendance_pct", "prev_exam_score", "assignment_score"]
BULK_RANGES   = {
    "study_hours":      (0.0, 12.0),
    "attendance_pct":   (0.0, 100.0),
    "prev_exam_score":  (0.0, 100.0),
    "assignment_score": (0.0, 100.0),
}

# Sample template CSV content (two example rows)
SAMPLE_CSV = (
    "study_hours,attendance_pct,prev_exam_score,assignment_score\n"
    "8.0,85.0,72.0,78.0\n"
    "4.5,60.0,45.0,55.0\n"
)

with tab_bulk:
    st.markdown('<p class="section-heading">Bulk Predictions</p>',
                unsafe_allow_html=True)
    st.caption(
        "Upload a CSV file containing one row per student. "
        "Required columns: **study_hours**, **attendance_pct**, "
        "**prev_exam_score**, **assignment_score**. "
        "Extra columns in the file are ignored."
    )

    # ── Sample template download ─────────────────────────────────
    st.download_button(
        label="⬇️ Download sample CSV template",
        data=SAMPLE_CSV,
        file_name="bulk_template.csv",
        mime="text/csv",
        help="Download a template with the correct column headers and two example rows.",
    )

    st.divider()

    # ── File uploader ────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="CSV must contain the four required feature columns listed above.",
    )

    if uploaded_file is not None:
        # ── Parse CSV ────────────────────────────────────────────
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as parse_err:
            st.error(f"**Could not parse the uploaded file.** {parse_err}")
            st.stop()

        if raw_df.empty:
            st.error("**The uploaded file contains no rows.** Please upload a non-empty CSV.")
            st.stop()

        # ── Check required columns ───────────────────────────────
        missing_cols = [c for c in BULK_FEATURES if c not in raw_df.columns]
        if missing_cols:
            st.error(
                f"**Missing required column(s): `{missing_cols}`.**  \n"
                "Download the sample template above for the correct format."
            )
            st.stop()

        # ── Work only on the required columns, in metadata order ─
        work_df = raw_df[BULK_FEATURES].copy()

        # ── Convert to numeric; non-convertible cells → NaN ──────
        for col in BULK_FEATURES:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

        # ── Build per-row validity mask ───────────────────────────
        # Collect a human-readable reason for each invalid row
        invalid_reasons = {}   # row_index → list[str]

        for col, (lo, hi) in BULK_RANGES.items():
            # Missing / non-numeric
            null_mask = work_df[col].isna()
            for idx in work_df.index[null_mask]:
                invalid_reasons.setdefault(idx, []).append(
                    f"`{col}` is missing or non-numeric"
                )
            # Infinite values
            inf_mask = np.isinf(work_df[col].fillna(0))
            for idx in work_df.index[inf_mask]:
                invalid_reasons.setdefault(idx, []).append(
                    f"`{col}` is infinite"
                )
            # Out of range (only for rows that are numeric and finite)
            numeric_finite = ~null_mask & ~inf_mask
            oor_mask = numeric_finite & (
                (work_df[col] < lo) | (work_df[col] > hi)
            )
            for idx in work_df.index[oor_mask]:
                invalid_reasons.setdefault(idx, []).append(
                    f"`{col}` = {work_df.at[idx, col]} is outside valid range [{lo}, {hi}]"
                )

        all_indices   = set(work_df.index)
        invalid_indices = set(invalid_reasons.keys())
        valid_indices   = all_indices - invalid_indices

        n_valid   = len(valid_indices)
        n_invalid = len(invalid_indices)

        # ── Summary counts ────────────────────────────────────────
        count_col1, count_col2 = st.columns(2)
        with count_col1:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left-color:#16a34a;">
                  <div class="kpi-label">Valid Rows</div>
                  <div class="kpi-value" style="color:#16a34a;">{n_valid}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with count_col2:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left-color:#b45309;">
                  <div class="kpi-label">Invalid Rows (excluded)</div>
                  <div class="kpi-value" style="color:#b45309;">{n_invalid}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Show invalid row details ──────────────────────────────
        if n_invalid > 0:
            st.markdown('<p class="section-heading">Invalid Rows</p>',
                        unsafe_allow_html=True)
            invalid_report = []
            for idx in sorted(invalid_indices):
                # Use 1-based display row number (header = row 0 in the file)
                invalid_report.append({
                    "File Row": idx + 2,      # +1 for header, +1 for 1-based
                    "Reason(s)": "; ".join(invalid_reasons[idx]),
                })
            st.dataframe(
                pd.DataFrame(invalid_report),
                use_container_width=True,
                hide_index=True,
            )

        # ── Predict on valid rows ─────────────────────────────────
        if n_valid == 0:
            st.warning("No valid rows to predict. Please fix the issues listed above.")
        else:
            valid_df = work_df.loc[sorted(valid_indices), BULK_FEATURES].reset_index(drop=True)

            # Feature order is guaranteed by the BULK_FEATURES list, which
            # matches feature_names loaded from model_metadata.json
            raw_preds  = model.predict(valid_df[feature_names])
            safe_preds = np.clip(raw_preds, 0.0, 100.0).round(2)

            results_df = valid_df.copy()
            results_df["predicted_final_score"] = safe_preds

            st.markdown('<p class="section-heading">Prediction Results</p>',
                        unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # ── Download results ──────────────────────────────────
            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download results as CSV",
                data=csv_bytes,
                file_name="bulk_predictions.csv",
                mime="text/csv",
            )

# ════════════════════════════════════════════════════════════════
# TAB 5 — What-If Simulator
# ════════════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown('<p class="section-heading">What-If Simulator</p>',
                unsafe_allow_html=True)
    st.caption(
        "Define a baseline student profile, then adjust the three scenario controls "
        "to see how changing a single input affects the model's predicted final score. "
        "Results are model-based estimates on synthetic data — not guaranteed real-world outcomes."
    )

    st.divider()

    # ── Baseline profile ─────────────────────────────────────────
    st.markdown('<p class="section-heading">Baseline Student Profile</p>',
                unsafe_allow_html=True)

    wi_col1, wi_col2 = st.columns(2, gap="large")
    with wi_col1:
        wi_base_study = st.slider(
            "Baseline Study Hours (per week)",
            min_value=0.0, max_value=12.0, value=5.0, step=0.5,
            key="wi_base_study",
            help="Baseline weekly study hours for this hypothetical student.",
        )
        wi_base_attendance = st.slider(
            "Baseline Attendance (%)",
            min_value=0.0, max_value=100.0, value=70.0, step=1.0,
            key="wi_base_attendance",
            help="Baseline attendance percentage.",
        )
    with wi_col2:
        wi_base_prev = st.slider(
            "Baseline Previous Exam Score (out of 100)",
            min_value=0.0, max_value=100.0, value=55.0, step=1.0,
            key="wi_base_prev",
            help="Baseline score from the most recent previous exam.",
        )
        wi_base_assign = st.slider(
            "Baseline Assignment Score (out of 100)",
            min_value=0.0, max_value=100.0, value=60.0, step=1.0,
            key="wi_base_assign",
            help="Baseline average assignment score.",
        )

    st.divider()

    # ── Scenario controls ────────────────────────────────────────
    st.markdown('<p class="section-heading">Scenario Adjustments</p>',
                unsafe_allow_html=True)
    st.caption(
        "Each scenario changes only its labelled feature. "
        "The remaining three features stay at their baseline values."
    )

    sc_col1, sc_col2, sc_col3 = st.columns(3, gap="medium")
    with sc_col1:
        wi_sc_a_study = st.slider(
            "Scenario A — Study Hours",
            min_value=0.0, max_value=12.0,
            value=min(wi_base_study + 2.0, 12.0),
            step=0.5,
            key="wi_sc_a_study",
            help="Adjusted study hours for Scenario A. Other features remain at baseline.",
        )
    with sc_col2:
        wi_sc_b_attend = st.slider(
            "Scenario B — Attendance (%)",
            min_value=0.0, max_value=100.0,
            value=min(wi_base_attendance + 10.0, 100.0),
            step=1.0,
            key="wi_sc_b_attend",
            help="Adjusted attendance for Scenario B. Other features remain at baseline.",
        )
    with sc_col3:
        wi_sc_c_assign = st.slider(
            "Scenario C — Assignment Score",
            min_value=0.0, max_value=100.0,
            value=min(wi_base_assign + 10.0, 100.0),
            step=1.0,
            key="wi_sc_c_assign",
            help="Adjusted assignment score for Scenario C. Other features remain at baseline.",
        )

    st.divider()

    # ── Build prediction DataFrames (exact feature order from metadata) ──
    def _wi_make_df(study, attend, prev, assign):
        """Return a single-row DataFrame in the exact feature order from metadata."""
        return pd.DataFrame(
            [[study, attend, prev, assign]],
            columns=feature_names,   # loaded from model_metadata.json
        )

    wi_df_base = _wi_make_df(wi_base_study, wi_base_attendance, wi_base_prev, wi_base_assign)
    wi_df_sc_a = _wi_make_df(wi_sc_a_study,  wi_base_attendance, wi_base_prev, wi_base_assign)
    wi_df_sc_b = _wi_make_df(wi_base_study,  wi_sc_b_attend,     wi_base_prev, wi_base_assign)
    wi_df_sc_c = _wi_make_df(wi_base_study,  wi_base_attendance, wi_base_prev, wi_sc_c_assign)

    wi_pred_base = float(np.clip(model.predict(wi_df_base)[0], 0.0, 100.0))
    wi_pred_sc_a = float(np.clip(model.predict(wi_df_sc_a)[0], 0.0, 100.0))
    wi_pred_sc_b = float(np.clip(model.predict(wi_df_sc_b)[0], 0.0, 100.0))
    wi_pred_sc_c = float(np.clip(model.predict(wi_df_sc_c)[0], 0.0, 100.0))

    wi_diff_a = wi_pred_sc_a - wi_pred_base
    wi_diff_b = wi_pred_sc_b - wi_pred_base
    wi_diff_c = wi_pred_sc_c - wi_pred_base

    # ── Comparison table ─────────────────────────────────────────
    st.markdown('<p class="section-heading">Predicted Score Comparison</p>',
                unsafe_allow_html=True)

    wi_table_df = pd.DataFrame({
        "Profile": [
            "Baseline",
            "Scenario A (study hours → {:.1f} h)".format(wi_sc_a_study),
            "Scenario B (attendance → {:.0f} %)".format(wi_sc_b_attend),
            "Scenario C (assignment → {:.0f})".format(wi_sc_c_assign),
        ],
        "Changed Feature Value": [
            f"study {wi_base_study:.1f} h / attend {wi_base_attendance:.0f}% / "
            f"prev {wi_base_prev:.0f} / assign {wi_base_assign:.0f}",
            f"study {wi_sc_a_study:.1f} h (was {wi_base_study:.1f})",
            f"attend {wi_sc_b_attend:.0f}% (was {wi_base_attendance:.0f}%)",
            f"assign {wi_sc_c_assign:.0f} (was {wi_base_assign:.0f})",
        ],
        "Predicted Score": [
            f"{wi_pred_base:.2f}",
            f"{wi_pred_sc_a:.2f}",
            f"{wi_pred_sc_b:.2f}",
            f"{wi_pred_sc_c:.2f}",
        ],
        "Δ vs Baseline": [
            "—",
            f"{wi_diff_a:+.2f}",
            f"{wi_diff_b:+.2f}",
            f"{wi_diff_c:+.2f}",
        ],
    })
    st.dataframe(wi_table_df, use_container_width=True, hide_index=True)

    # ── Bar chart ────────────────────────────────────────────────
    wi_labels = [
        "Baseline",
        f"Scenario A\n(study → {wi_sc_a_study:.1f} h)",
        f"Scenario B\n(attend → {wi_sc_b_attend:.0f}%)",
        f"Scenario C\n(assign → {wi_sc_c_assign:.0f})",
    ]
    wi_scores  = [wi_pred_base, wi_pred_sc_a, wi_pred_sc_b, wi_pred_sc_c]
    wi_colors  = ["#1a3a6e", "#0e7490", "#0e7490", "#0e7490"]

    fig_wi, ax_wi = plt.subplots(figsize=(7, 3.8))
    fig_wi.patch.set_facecolor("#f0f4f8")
    ax_wi.set_facecolor("#f0f4f8")

    bars_wi = ax_wi.bar(wi_labels, wi_scores, color=wi_colors,
                        edgecolor="#f0f4f8", width=0.5)
    ax_wi.bar_label(bars_wi, fmt="%.2f", padding=4, fontsize=9, color="#0f2044")
    ax_wi.set_ylim(0, 110)
    ax_wi.set_ylabel("Predicted Final Score (0–100)", fontsize=9, color="#5a7a99")
    ax_wi.set_title("What-If Scenario Comparison — Model Predictions",
                    fontsize=10, color="#0f2044", fontweight="bold", pad=10)
    ax_wi.tick_params(colors="#0f2044", labelsize=8)
    ax_wi.spines[["top", "right"]].set_visible(False)
    ax_wi.spines[["left", "bottom"]].set_color("#d0dff0")
    ax_wi.axhline(wi_pred_base, color="#b45309", linewidth=1.2,
                  linestyle="--", label=f"Baseline ({wi_pred_base:.2f})")
    ax_wi.legend(fontsize=8, framealpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_wi)
    plt.close(fig_wi)

    # ── Interpretation ───────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-heading">Interpretation</p>',
                unsafe_allow_html=True)

    wi_scenario_diffs = {
        "Scenario A (increased study hours)": wi_diff_a,
        "Scenario B (improved attendance)":   wi_diff_b,
        "Scenario C (improved assignment score)": wi_diff_c,
    }
    wi_best_diff  = max(wi_scenario_diffs.values())
    wi_best_names = [k for k, v in wi_scenario_diffs.items() if v == wi_best_diff]

    if wi_best_diff > 0:
        if len(wi_best_names) == 1:
            wi_interp = (
                f"Among the three scenarios, **{wi_best_names[0]}** produces the highest "
                f"predicted score, with a model-estimated gain of **{wi_best_diff:+.2f} points** "
                f"over the baseline ({wi_pred_base:.2f})."
            )
        else:
            wi_tied = " and ".join(f"**{n}**" for n in wi_best_names)
            wi_interp = (
                f"**{wi_tied}** are tied for the highest predicted score, "
                f"each showing a model-estimated gain of **{wi_best_diff:+.2f} points** "
                f"over the baseline ({wi_pred_base:.2f})."
            )
    elif wi_best_diff == 0:
        wi_interp = (
            "None of the three scenarios produce a predicted score different from the baseline "
            f"({wi_pred_base:.2f}) under the current settings."
        )
    else:
        wi_interp = (
            "All three scenarios produce a lower predicted score than the baseline "
            f"({wi_pred_base:.2f}) under the current settings. "
            "Try increasing the scenario values above the baseline."
        )

    st.markdown(wi_interp)
    st.caption(
        "⚠️ This is a model-based what-if comparison using a Random Forest trained on synthetic data. "
        "A higher predicted score for a given scenario does not prove that changing that single "
        "input will cause the same improvement for a real student. "
        "Many other factors influence academic outcomes."
    )

# ════════════════════════════════════════════════════════════════
# TAB 6 — Student Report
# ════════════════════════════════════════════════════════════════

def _rpt_suggestions(study, attend, prev_exam, assign):
    """Return a list of plain-text rule-based suggestions for the report."""
    tips = []
    if study < 2:
        tips.append(
            "Study hours are very low (< 2 h/week). Try gradually increasing to at "
            "least 2-3 focused sessions per week. Short, regular study blocks are "
            "more effective than occasional long sessions."
        )
    elif study < 5:
        tips.append(
            "Study hours are building up (2-5 h/week). Maintain this consistency "
            "and consider focused techniques such as the Pomodoro method or active "
            "recall to get more from each session."
        )
    else:
        tips.append(
            "Study hours are strong (>= 5 h/week). Keep a balanced and consistent "
            "routine - steady effort tends to outperform last-minute cramming."
        )
    if attend < 75:
        tips.append(
            "Attendance is below 75%. Improving attendance where possible helps "
            "stay current with material and benefit from in-class explanations "
            "that are hard to replicate from notes alone."
        )
    else:
        tips.append(
            "Attendance is good (>= 75%). Consistent class presence is a strong "
            "foundation - keep engaging actively during sessions."
        )
    if prev_exam < 50:
        tips.append(
            "Previous exam score was below 50. Focus on reviewing fundamental "
            "concepts before advancing. Working through practice questions and "
            "checking answers carefully can identify and close specific knowledge gaps."
        )
    else:
        tips.append(
            "Previous exam score was 50 or above. Build on this with timed practice "
            "under exam conditions to strengthen both knowledge recall and time management."
        )
    if assign < 60:
        tips.append(
            "Assignment score is below 60. Review any feedback on past assignments "
            "and work through similar practice exercises. Assignments are an early "
            "signal of exam readiness."
        )
    else:
        tips.append(
            "Assignment score is 60 or above. Strong assignment performance generally "
            "reflects good ongoing engagement - keep submitting work consistently and on time."
        )
    return tips


def _rpt_build_html(label, study, attend, prev_exam, assign, score, tips, gen_dt):
    """Build a self-contained printable HTML report string."""
    label_row = f"<tr><td>Student Label</td><td>{label}</td></tr>" if label.strip() else ""
    tips_html  = "".join(
        f"<li>{textwrap.fill(t, width=110)}</li>" for t in tips
    )
    score_color = (
        "#16a34a" if score >= 75 else
        "#0e7490" if score >= 50 else
        "#b45309"
    )
    band_label = (
        "Strong performance predicted"   if score >= 75 else
        "Moderate performance predicted" if score >= 50 else
        "Low performance predicted"
    )

    # Inline SVG bar chart: four feature values scaled to 200 px wide
    def _bar(value, max_val, y, color):
        w = max(int(value / max_val * 200), 2)
        return (
            f'<rect x="160" y="{y}" width="{w}" height="18" fill="{color}" rx="3"/>'
            f'<text x="370" y="{y+14}" font-size="12" fill="#0f2044">{value}</text>'
        )

    chart_items = [
        ("Study Hours",     study,    12,  0,  "#1a3a6e"),
        ("Attendance %",    attend,  100, 26,  "#0e7490"),
        ("Prev Exam Score", prev_exam,100, 52,  "#1a3a6e"),
        ("Assign Score",    assign,  100, 78,  "#0e7490"),
    ]
    bars_svg = ""
    labels_svg = ""
    for name, val, mx, y, col in chart_items:
        bars_svg   += _bar(val, mx, y, col)
        labels_svg += f'<text x="0" y="{y+14}" font-size="12" fill="#5a7a99">{name}</text>'

    svg_chart = f"""
    <svg width="400" height="110" xmlns="http://www.w3.org/2000/svg">
      {labels_svg}
      {bars_svg}
    </svg>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Student Performance Report</title>
<style>
  body {{
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    font-size: 14px; line-height: 1.6; color: #1f2328;
    background: #ffffff; margin: 0; padding: 2rem;
    max-width: 760px; margin-left: auto; margin-right: auto;
  }}
  .header-band {{
    background: linear-gradient(90deg, #0f2044 0%, #1a3a6e 60%, #0e7490 100%);
    padding: 1.2rem 1.6rem; border-radius: 8px; margin-bottom: 1.4rem;
    color: #ffffff;
  }}
  .header-band h1 {{ margin: 0; font-size: 1.4rem; }}
  .header-band p  {{ margin: 0.2rem 0 0 0; font-size: 0.88rem; color: #b0d0ee; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.2rem; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 0.5rem 0.8rem; text-align: left; }}
  th {{ background: #f0f4f8; font-weight: 600; color: #0f2044; }}
  .score-box {{
    border-left: 6px solid {score_color};
    background: #f0f4f8; padding: 0.9rem 1.2rem;
    border-radius: 8px; margin-bottom: 1.2rem;
  }}
  .score-label {{ font-size: 0.75rem; color: #5a7a99; text-transform: uppercase;
                  letter-spacing: 0.06em; font-weight: 600; }}
  .score-value {{ font-size: 2rem; font-weight: 700; color: {score_color}; }}
  .score-band  {{ font-size: 0.82rem; color: #5a7a99; }}
  ul {{ padding-left: 1.4rem; margin: 0.4rem 0 1rem 0; }}
  li {{ margin-bottom: 0.4rem; }}
  .section-title {{
    font-weight: 700; font-size: 0.95rem; color: #0f2044;
    border-bottom: 2px solid #d0dff0; padding-bottom: 0.25rem;
    margin: 1.2rem 0 0.6rem 0;
  }}
  .disclaimer {{
    background: #fff7ed; border-left: 4px solid #b45309;
    padding: 0.7rem 1rem; border-radius: 6px;
    font-size: 0.85rem; color: #78350f; margin-top: 1.2rem;
  }}
  .footer {{
    font-size: 0.78rem; color: #9ab0cc; text-align: center;
    border-top: 1px solid #e5e7eb; margin-top: 1.6rem; padding-top: 0.7rem;
  }}
  @media print {{
    body {{ padding: 0; }}
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>

<div class="header-band">
  <h1>&#127891; Student Performance Report</h1>
  <p>Smart Student Performance Predictor &middot; Generated {gen_dt}</p>
</div>

<div class="section-title">Report Details</div>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  {label_row}
  <tr><td>Report Generated</td><td>{gen_dt}</td></tr>
</table>

<div class="section-title">Input Values</div>
<table>
  <tr><th>Feature</th><th>Value</th><th>Valid Range</th></tr>
  <tr><td>Study Hours (per week)</td><td>{study:.1f} h</td><td>0 &ndash; 12 h</td></tr>
  <tr><td>Attendance Percentage</td><td>{attend:.0f} %</td><td>0 &ndash; 100 %</td></tr>
  <tr><td>Previous Exam Score</td><td>{prev_exam:.0f} / 100</td><td>0 &ndash; 100</td></tr>
  <tr><td>Assignment Score</td><td>{assign:.0f} / 100</td><td>0 &ndash; 100</td></tr>
</table>

<div class="section-title">Input Values Chart</div>
{svg_chart}

<div class="section-title">Predicted Final Score</div>
<div class="score-box">
  <div class="score-label">Estimated Final Exam Score</div>
  <div class="score-value">{score:.2f} / 100</div>
  <div class="score-band">{band_label}</div>
</div>

<div class="section-title">Personalised Study Suggestions</div>
<ul>
{tips_html}
</ul>

<div class="disclaimer">
  <strong>Important:</strong> This report is generated by a Random Forest model trained on
  300 fully synthetic student records. The predicted score is a model-based estimate and is
  <strong>not</strong> a guaranteed exam result. Many real-world factors influence academic
  performance that this model does not capture. These suggestions are general and rule-based,
  not personalised academic advice.
</div>

<div class="footer">
  Smart Student Performance Predictor &middot; Random Forest &middot;
  Synthetic data &mdash; educational demonstration only.
  <br/>To save as PDF: open in a browser and press Ctrl+P &rarr; Save as PDF.
</div>

</body>
</html>"""
    return html


with tab_report:
    st.markdown('<p class="section-heading">Student Report</p>',
                unsafe_allow_html=True)
    st.caption(
        "Fill in the student details and click **Generate Report** to produce a "
        "downloadable HTML report with the predicted score and personalised suggestions. "
        "Open the downloaded file in any browser and use Print → Save as PDF."
    )

    st.divider()

    # ── Inputs ───────────────────────────────────────────────────
    rpt_col1, rpt_col2 = st.columns(2, gap="large")
    with rpt_col1:
        rpt_label = st.text_input(
            "Student Label (optional)",
            value="",
            max_chars=80,
            placeholder="e.g. Student A, Roll No. 42, …",
            help="An optional identifier included in the report. No personal data required.",
            key="rpt_label",
        )
        rpt_study = st.slider(
            "Study Hours (per week)",
            min_value=0.0, max_value=12.0, value=6.0, step=0.5,
            key="rpt_study",
            help="Average weekly study hours.",
        )
        rpt_attend = st.slider(
            "Attendance Percentage (%)",
            min_value=0.0, max_value=100.0, value=75.0, step=1.0,
            key="rpt_attend",
            help="Percentage of classes attended.",
        )
    with rpt_col2:
        rpt_prev = st.slider(
            "Previous Exam Score (out of 100)",
            min_value=0.0, max_value=100.0, value=60.0, step=1.0,
            key="rpt_prev",
            help="Score from the most recent previous exam.",
        )
        rpt_assign = st.slider(
            "Assignment Score (out of 100)",
            min_value=0.0, max_value=100.0, value=65.0, step=1.0,
            key="rpt_assign",
            help="Average assignment score.",
        )

    generate_clicked = st.button(
        "📄 Generate Report", type="primary", key="rpt_generate"
    )

    st.divider()

    if generate_clicked:
        # ── Predict ──────────────────────────────────────────────
        rpt_input_df = pd.DataFrame(
            [[rpt_study, rpt_attend, rpt_prev, rpt_assign]],
            columns=feature_names,   # exact order from model_metadata.json
        )
        rpt_raw   = model.predict(rpt_input_df)[0]
        rpt_score = float(np.clip(rpt_raw, 0.0, 100.0))

        # ── On-screen preview ────────────────────────────────────
        rpt_color = (
            "#16a34a" if rpt_score >= 75 else
            "#0e7490" if rpt_score >= 50 else
            "#b45309"
        )
        rpt_band = (
            "Strong performance predicted ✅"   if rpt_score >= 75 else
            "Moderate performance predicted ℹ️" if rpt_score >= 50 else
            "Low performance predicted ⚠️"
        )
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color:{rpt_color};">
              <div class="kpi-label">Predicted Final Exam Score</div>
              <div class="kpi-value" style="color:{rpt_color};">
                  {rpt_score:.2f}
                  <span style="font-size:1.1rem;color:#5a7a99;">/ 100</span>
              </div>
              <div class="kpi-sub">{rpt_band}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Build HTML report ────────────────────────────────────
        rpt_tips = _rpt_suggestions(rpt_study, rpt_attend, rpt_prev, rpt_assign)
        rpt_gen_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        try:
            rpt_html = _rpt_build_html(
                label    = rpt_label,
                study    = rpt_study,
                attend   = rpt_attend,
                prev_exam= rpt_prev,
                assign   = rpt_assign,
                score    = rpt_score,
                tips     = rpt_tips,
                gen_dt   = rpt_gen_dt,
            )
            rpt_bytes = rpt_html.encode("utf-8")

            safe_label = (
                rpt_label.strip().replace(" ", "_")[:30] if rpt_label.strip() else "student"
            )
            rpt_filename = f"report_{safe_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"

            st.download_button(
                label="⬇️ Download Report (HTML)",
                data=rpt_bytes,
                file_name=rpt_filename,
                mime="text/html",
                help="Open in any browser and use Print → Save as PDF to create a PDF.",
            )
            st.caption(
                "💡 To convert to PDF: open the downloaded file in Chrome or Edge, "
                "press Ctrl+P (or Cmd+P on Mac), and choose **Save as PDF**."
            )
        except Exception as rpt_err:
            st.error(f"Report generation failed: {rpt_err}")
    else:
        st.info(
            "Adjust the inputs above and click **Generate Report** to create "
            "a downloadable report with the predicted score and study suggestions."
        )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Smart Student Performance Predictor · Random Forest · "
    "Built with scikit-learn & Streamlit · "
    "Synthetic data — educational demonstration only."
)
