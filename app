# =====================================================
# Creative Streamlit App: Phishing URL Detector
# Uses trained model (best_model.pkl)
# =====================================================

import streamlit as st
import joblib
import pandas as pd
import re, string, math
from urllib.parse import urlparse

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# ---------------------------
# Feature extraction (must match training)
# ---------------------------
def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    def count_digits(s): return sum(c.isdigit() for c in s)
    def count_special(s): return sum(c in string.punctuation for c in s)
    def entropy(s):
        if not s: return 0
        p = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
        return -sum([pi * math.log(pi, 2) for pi in p])

    feats = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_digits": count_digits(url),
        "num_specials": count_special(url),
        "num_subdirs": path.count('/'),
        "num_query_params": query.count('='),
        "has_https": int(url.startswith("https")),
        "has_ip": int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', domain))),
        "has_at": int('@' in url),
        "has_dash_in_domain": int('-' in domain),
        "entropy": entropy(url),
        "domain_entropy": entropy(domain),
        "path_entropy": entropy(path),
        "contains_login": int("login" in url.lower()),
        "contains_secure": int("secure" in url.lower()),
        "contains_account": int("account" in url.lower()),
        "contains_update": int("update" in url.lower()),
        "contains_verify": int("verify" in url.lower()),
    }
    return pd.DataFrame([feats])

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🛡️ Phishing URL Detector", page_icon="🛡️", layout="wide")

st.markdown("""
# 🛡️ Phishing URL Detector
Check if a URL is **benign** or **phishing** using a trained ML model.
""")

st.markdown("## 🔗 Single URL Detection")
url_input = st.text_input("Enter a URL here:", placeholder="https://example.com/login")

if st.button("Predict"):
    if not url_input.strip():
        st.warning("⚠️ Please enter a valid URL!")
    else:
        with st.spinner("🔍 Analyzing URL..."):
            X_input = extract_features(url_input.strip())
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][prediction] if hasattr(model, "predict_proba") else None

        # Display creative result card
        if prediction == 1:
            st.markdown(
                f"<div style='padding:20px; background-color:#FFCCCC; border-radius:10px;'>"
                f"🚨 <b>Phishing URL Detected!</b><br>"
                f"URL: {url_input}<br>"
                + (f"Confidence: {proba*100:.2f}%" if proba else "")
                + "</div>",
                unsafe_allow_html=True
            )
            if proba:
                st.progress(proba)
        else:
            st.markdown(
                f"<div style='padding:20px; background-color:#CCFFCC; border-radius:10px;'>"
                f"✅ <b>URL is Benign</b><br>"
                f"URL: {url_input}<br>"
                + (f"Confidence: {proba*100:.2f}%" if proba else "")
                + "</div>",
                unsafe_allow_html=True
            )
            if proba:
                st.progress(proba)

# ---------------------------
# Batch CSV upload
# ---------------------------
st.markdown("---")
st.markdown("## 📁 Batch URL Detection")
uploaded_file = st.file_uploader("Upload a CSV file with a column `url`", type=["csv"])

if uploaded_file:
    df_batch = pd.read_csv(uploaded_file)
    if "url" not in df_batch.columns:
        st.error("CSV must contain a column named `url`!")
    else:
        with st.spinner("🔍 Analyzing batch URLs..."):
            features = pd.DataFrame([extract_features(u).iloc[0] for u in df_batch['url']])
            predictions = model.predict(features)
            df_batch['prediction'] = predictions
            df_batch['prediction_label'] = df_batch['prediction'].apply(lambda x: "Phishing" if x==1 else "Benign")
        st.success("✅ Batch prediction complete!")
        st.dataframe(df_batch)
        st.markdown("📥 [Download results](#) (replace with file export if needed)")

st.markdown("---")
st.markdown("💡 *Model trained using RandomForest/XGBoost/LightGBM ensemble — best model selected automatically.*")
