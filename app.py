import streamlit as st
import joblib
import re

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

# ── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

# ── Text Cleaning ──────────────────────────────────────────────────────────
def clean_tweet(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🐦 Twitter Sentiment Analyzer")
st.markdown("Classify any tweet or text as **Positive**, **Neutral**, or **Negative** using a TF-IDF + LinearSVC model trained on 61,000+ tweets.")

st.divider()

# Example tweets
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😊 Positive example"):
        st.session_state.input_text = "I absolutely love this product! Best purchase I've made all year 🎉"
with col2:
    if st.button("😐 Neutral example"):
        st.session_state.input_text = "The event starts at 5pm. There will be food and drinks available."
with col3:
    if st.button("😠 Negative example"):
        st.session_state.input_text = "Terrible customer service. Waited 2 hours and nobody helped me. Never coming back!"

# Text input
text_input = st.text_area(
    "Enter your tweet or text:",
    value=st.session_state.get("input_text", ""),
    height=120,
    placeholder="Type something here or click an example above..."
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_tweet(text_input)
        prediction = model.predict([cleaned])[0]

        # Get confidence scores via decision function
        scores = model.decision_function([cleaned])[0]
        classes = model.classes_
        score_dict = dict(zip(classes, scores))

        # Normalize scores to 0-1 range for display
        min_s, max_s = min(scores), max(scores)
        normalized = {k: (v - min_s) / (max_s - min_s + 1e-9) for k, v in score_dict.items()}
        total = sum(normalized.values())
        confidence = {k: v / total for k, v in normalized.items()}

        st.divider()

        # Result
        emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}
        color = {"Positive": "green", "Neutral": "orange", "Negative": "red"}

        st.markdown(f"### Result: :{color[prediction]}[{emoji[prediction]} {prediction}]")

        # Confidence bars
        st.markdown("**Confidence scores:**")
        for label in ["Positive", "Neutral", "Negative"]:
            st.progress(
                confidence[label],
                text=f"{label}: {confidence[label]*100:.1f}%"
            )

        # Cleaned text
        with st.expander("See cleaned text"):
            st.code(cleaned)

st.divider()
st.markdown("""
**Model Details:**
- Dataset: Twitter Entity Sentiment Analysis (61,121 tweets)
- Algorithm: LinearSVC with TF-IDF (bigrams, 50k features)
- Accuracy: **98.5%** | F1 Score: **0.9855** on validation set
- Classes: Positive · Neutral · Negative
""")
