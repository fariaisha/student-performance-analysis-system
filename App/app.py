import os
import joblib
import pandas as pd
import altair as alt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODELS = {
    "Decision Tree": "decision_tree.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}

@st.cache_resource
def load_models():
    return {
        name: joblib.load(os.path.join(MODEL_DIR, filename))
        for name, filename in MODELS.items()
    }

models = load_models()

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Theme
#
# Concept: an academic transcript / gradebook rendered as a dashboard —
# letterhead-style header with a rule, mono "record" labels, and a wax-seal
# style stamp for the predicted grade. Ink navy + brass gold on warm paper,
# serif display type (Fraunces) paired with Inter for body copy and
# IBM Plex Mono for data/labels.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
    --paper: #F6F5F0;
    --panel: #FFFFFF;
    --ink: #1B2434;
    --ink-soft: #5B6478;
    --rule: #D9DCE3;
    --accent: #3E5C89;
    --accent-soft: #E8EDF5;
    --gold: #A6842C;
    --gold-soft: #F1E9D3;
    --good: #2F6D4F;
    --good-soft: #E6F1EA;
    --warn: #AE4438;
    --warn-soft: #F6E8E5;
}

html, body, [class*="css"], .stMarkdown, p, span, label {
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--ink);
}

.stApp { background: var(--paper); }

/* ---------- General copy color ---------- */
.stMarkdown p, .stMarkdown li, .stText { color: var(--ink); }
.stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--gold) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stDataFrame"] { color: var(--ink); }
.main { padding-top: 0.4rem; }
.block-container { padding-top: 1.6rem; max-width: 1120px; }

/* ---------- Letterhead hero ---------- */
.hero {
    padding: 1.7rem 2rem 1.5rem;
    background: var(--paper);
    border-top: 3px solid var(--ink);
    border-bottom: 1px solid var(--rule);
    margin-bottom: 1.8rem;
}
.hero .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 0 0 0.45rem 0;
}
.hero h1 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 2.15rem;
    color: var(--ink);
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.01em;
}
.hero p {
    color: var(--ink-soft);
    font-size: 0.98rem;
    margin: 0;
    max-width: 620px;
    line-height: 1.5;
}

/* ---------- Section record labels ---------- */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.45rem;
    margin: 0 0 0.9rem 0;
}

h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] h2 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--ink) !important;
    font-size: 1.15rem !important;
}
section[data-testid="stSidebar"] .stAlert {
    background: var(--accent-soft);
    border: 1px solid var(--rule);
    color: var(--ink);
    border-radius: 4px;
}

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div {
    border-radius: 4px !important;
    border-color: var(--rule) !important;
    background: var(--panel) !important;
}
.stNumberInput input {
    border-radius: 4px !important;
    border-color: var(--rule) !important;
}
label, .stNumberInput label, .stSelectbox label {
    color: var(--ink-soft) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ---------- Divider ---------- */
hr { border-color: var(--rule) !important; }

/* ---------- Button ---------- */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    background: var(--ink);
    color: #FFFFFF !important;
    border: none;
    border-radius: 4px;
    letter-spacing: 0.01em;
    padding: 0.6rem 1rem;
    transition: background 0.15s ease;
}
.stButton > button:hover {
    background: var(--accent);
    color: #FFFFFF !important;
}
.stButton > button:active { background: var(--gold); }

/* ---------- Result / stamp panel ---------- */
.result {
    position: relative;
    padding: 1.5rem 1.8rem;
    background: var(--panel);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--stamp-color, var(--ink));
    display: flex;
    align-items: center;
    gap: 1.7rem;
    margin-bottom: 1.2rem;
}
.stamp {
    flex-shrink: 0;
    width: 84px;
    height: 84px;
    border-radius: 50%;
    border: 3px double var(--stamp-color, var(--gold));
    display: flex;
    align-items: center;
    justify-content: center;
    transform: rotate(-6deg);
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 700;
    font-size: 2.3rem;
    color: var(--stamp-color, var(--ink));
    background: var(--stamp-soft, var(--gold-soft));
}
.result-body { flex: 1; }
.result-body .record-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin: 0 0 0.25rem 0;
}
.result-body h2 {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.35rem;
    margin: 0 0 0.3rem 0;
    color: var(--ink);
}
.result-body p.verdict { margin: 0 0 0.55rem 0; color: var(--ink-soft); font-size: 0.94rem; }
.result-body .model-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    background: var(--accent-soft);
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    border: 1px solid var(--rule);
}

/* ---------- Confidence table caption ---------- */
.confidence-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 1.1rem 0 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<p class="eyebrow">Data Mining &middot; Classification Project</p>
<h1>🎓 Student Performance Analysis</h1>
<p>Enter a student's academic and demographic record below to predict their final grade
using a trained classification model.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Prediction Settings")
    selected_model = st.selectbox(
        "Choose Classification Model",
        list(MODELS.keys())
    )
    st.info("Technique: Classification")

st.subheader("👤 Student Information")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<p class="section-label">Background</p>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["female", "male", "other"])
    school_type = st.selectbox("School Type", ["public", "private"])
    parent_education = st.selectbox(
        "Parent Education",
        ["no formal", "high school", "diploma", "college", "graduate", "post graduate", "phd"]
    )
    internet_access = st.selectbox("Internet Access", ["yes", "no"])

with c2:
    st.markdown('<p class="section-label">Study Habits</p>', unsafe_allow_html=True)
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    travel_time = st.selectbox("Travel Time", ["<15 min", "15-30 min", "30-60 min", ">60 min"])
    extra_activities = st.selectbox("Extra Activities", ["yes", "no"])

with c3:
    st.markdown('<p class="section-label">Assessment</p>', unsafe_allow_html=True)
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation = st.selectbox("Test Preparation", ["none", "completed"])
    math_score = st.number_input("Math Score", 0.0, 100.0, 70.0, 1.0)
    science_score = st.number_input("Science Score", 0.0, 100.0, 70.0, 1.0)
    english_score = st.number_input("English Score", 0.0, 100.0, 70.0, 1.0)

overall_score = st.number_input("Overall Score", 0.0, 100.0, 70.0, 1.0)

input_df = pd.DataFrame([{
    "gender": gender,
    "school_type": school_type,
    "parent_education": parent_education,
    "study_hours": study_hours,
    "attendance": attendance,
    "internet_access": internet_access,
    "travel_time": travel_time,
    "extra_activities": extra_activities,
    "lunch": lunch,
    "test_preparation": test_preparation,
    "math_score": math_score,
    "science_score": science_score,
    "english_score": english_score,
    "overall_score": overall_score
}])

st.divider()

if st.button("🔮 Predict Final Grade", type="primary", use_container_width=True):
    model = models[selected_model]
    prediction = model.predict(input_df)[0]
    grade = str(prediction).upper()

    grade_text = {
        "A": "Excellent Performance",
        "B": "Very Good Performance",
        "C": "Good / Average Performance",
        "D": "Needs Improvement",
        "F": "Needs Significant Improvement"
    }.get(grade, "Prediction generated")

    # Stamp color follows the grade: brass for top marks, ink for mid,
    # warm red for grades that need attention.
    stamp_theme = {
        "A": ("#2F6D4F", "#E6F1EA"),
        "B": ("#A6842C", "#F1E9D3"),
        "C": ("#3E5C89", "#E8EDF5"),
        "D": ("#AE4438", "#F6E8E5"),
        "F": ("#AE4438", "#F6E8E5"),
    }.get(grade, ("#1B2434", "#F1E9D3"))
    stamp_color, stamp_soft = stamp_theme

    st.markdown(f"""
    <div class="result" style="--stamp-color:{stamp_color}; --stamp-soft:{stamp_soft};">
        <div class="stamp">{grade}</div>
        <div class="result-body">
            <p class="record-tag">Predicted Final Grade</p>
            <h2>Grade {grade}</h2>
            <p class="verdict">{grade_text}</p>
            <span class="model-tag">Model &middot; {selected_model}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        prob_df = pd.DataFrame({
            "Grade": classes,
            "Probability": probs
        }).sort_values("Probability", ascending=False)
        st.markdown('<p class="confidence-caption">Prediction Confidence</p>', unsafe_allow_html=True)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

        grade_colors = {
            "A": "#2F6D4F",
            "B": "#A6842C",
            "C": "#3E5C89",
            "D": "#AE4438",
            "F": "#AE4438",
        }
        domain = list(prob_df["Grade"])
        range_ = [grade_colors.get(g, "#1B2434") for g in domain]

        base = alt.Chart(prob_df).encode(
            y=alt.Y("Grade:N", sort="-x", title=None,
                    axis=alt.Axis(labelColor="#1B2434", labelFontSize=15,
                                   labelFont="Fraunces", labelFontWeight=600,
                                   domain=False, ticks=False)),
            x=alt.X("Probability:Q", title=None,
                    axis=alt.Axis(format="%", grid=True, gridColor="#E3E6EC",
                                   gridDash=[2, 3], domain=False, tickCount=5,
                                   labelColor="#5B6478", labelFont="IBM Plex Mono",
                                   labelFontSize=10)),
        )

        bars = base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=22).encode(
            color=alt.Color("Grade:N", scale=alt.Scale(domain=domain, range=range_), legend=None),
            tooltip=[alt.Tooltip("Grade:N", title="Grade"),
                     alt.Tooltip("Probability:Q", title="Probability", format=".1%")],
        )

        labels = base.mark_text(align="left", dx=6, font="IBM Plex Mono",
                                 fontSize=11, color="#1B2434").encode(
            text=alt.Text("Probability:Q", format=".1%"),
        )

        chart = (bars + labels).properties(
            height=alt.Step(34),
            background="transparent",
        ).configure_view(strokeWidth=0)

        st.altair_chart(chart, use_container_width=True)

st.caption("Student Performance Analysis System | Classification-based Data Mining Project")