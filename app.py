import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛍️", layout="centered")

MODEL_PATH = Path("models/purchase_intention_pipeline.joblib")

st.title("🛍️ Will This Visitor Buy?")
st.subheader("E-commerce Purchase Intention Predictor")
st.write("Enter a visitor's browsing-session details to estimate the probability of a purchase.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run all cells in notebook.ipynb first to train and save the model.")
    st.stop()

model = joblib.load("purchase_intention_pipeline.joblib")

with st.form("session_form"):
    st.markdown("### Browsing activity")
    c1, c2 = st.columns(2)
    with c1:
        administrative = st.number_input("Administrative pages", min_value=0, value=1)
        administrative_duration = st.number_input("Administrative time (seconds)", min_value=0.0, value=20.0)
        informational = st.number_input("Informational pages", min_value=0, value=0)
        informational_duration = st.number_input("Informational time (seconds)", min_value=0.0, value=0.0)
        product_related = st.number_input("Product-related pages", min_value=0, value=5)
        product_related_duration = st.number_input("Product-related time (seconds)", min_value=0.0, value=120.0)
    with c2:
        bounce_rates = st.number_input("Bounce rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
        exit_rates = st.number_input("Exit rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        page_values = st.number_input("Page value", min_value=0.0, value=0.0)
        special_day = st.select_slider("Closeness to a special day", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)

    st.markdown("### Visitor and session information")
    c3, c4 = st.columns(2)
    with c3:
        month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        operating_systems = st.selectbox("Operating system code", list(range(1, 9)), index=1)
        browser = st.selectbox("Browser code", list(range(1, 14)), index=1)
        region = st.selectbox("Region code", list(range(1, 10)), index=0)
    with c4:
        traffic_type = st.selectbox("Traffic type code", list(range(1, 21)), index=1)
        visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = st.selectbox("Weekend visit?", [False, True])

    submitted = st.form_submit_button("Predict purchase intention", use_container_width=True)

if submitted:
    session = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend
    }])

    probability = float(model.predict_proba(session)[0, 1])
    prediction = int(probability >= 0.5)

    st.divider()
    st.subheader("Prediction result")
    st.metric("Estimated purchase probability", f"{probability:.1%}")
    if prediction == 1:
        st.success("The model predicts that this session is likely to end with a purchase.")
        st.write("Suggested action: prioritize a smooth checkout experience and avoid unnecessary interruptions.")
    else:
        st.warning("The model predicts that this session is less likely to end with a purchase.")
        st.write("Suggested action: consider helpful product recommendations or customer support.")
    st.caption("This is a machine-learning estimate, not a guarantee. Results depend on the training data and model quality.")
