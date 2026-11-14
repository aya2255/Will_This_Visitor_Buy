import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Purchase Intention Predictor", 
    page_icon="🛍️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling for UI enhancements
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("purchase_intention_pipeline.joblib")

# Header Section
st.markdown("<h1 style='text-align: center; color: #1e293b;'>🛍️ Will This Visitor Buy?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1rem;'>E-commerce Purchase Intention Predictor - Real-time Session Analysis</p>", unsafe_allow_html=True)
st.markdown("---")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run all cells in notebook.ipynb first to train and save the model.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Form for Inputs (Organized in Columns)
with st.form("session_form"):
    st.markdown("### 📊 Browsing Activity & Session Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Navigation")
        administrative = st.number_input("Administrative pages", min_value=0, value=1)
        administrative_duration = st.number_input("Administrative time (s)", min_value=0.0, value=20.0)
        informational = st.number_input("Informational pages", min_value=0, value=0)
        informational_duration = st.number_input("Informational time (s)", min_value=0.0, value=0.0)
        product_related = st.number_input("Product-related pages", min_value=0, value=5)
        product_related_duration = st.number_input("Product-related time (s)", min_value=0.0, value=120.0)

    with col2:
        st.markdown("#### Engagement Rates")
        bounce_rates = st.number_input("Bounce rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
        exit_rates = st.number_input("Exit rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        page_values = st.number_input("Page value", min_value=0.0, value=0.0)
        special_day = st.select_slider("Closeness to a special day", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)
        month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    with col3:
        st.markdown("#### Technical & Visitor Info")
        operating_systems = st.selectbox("Operating system code", list(range(1, 9)), index=1)
        browser = st.selectbox("Browser code", list(range(1, 14)), index=1)
        region = st.selectbox("Region code", list(range(1, 10)), index=0)
        traffic_type = st.selectbox("Traffic type code", list(range(1, 21)), index=1)
        visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = st.selectbox("Weekend visit?", [False, True])

    st.markdown("")
    submitted = st.form_submit_button("✨ Predict Purchase Intention", use_container_width=True)

# Results and Interactive Charts Section layout
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
    
    res_col1, res_col2 = st.columns([1, 1], gap="large")
    
    with res_col1:
        st.subheader("🎯 Prediction Result")
        st.metric("Estimated purchase probability", f"{probability:.1%}")
        
        if prediction == 1:
            st.success("The model predicts that this session is likely to end with a purchase.")
            st.markdown("**Suggested action:** Prioritize a smooth checkout experience and avoid unnecessary interruptions.")
        else:
            st.warning("The model predicts that this session is less likely to end with a purchase.")
            st.markdown("**Suggested action:** Consider helpful product recommendations, targeted coupons, or customer support.")
        
        st.caption("This is a machine-learning estimate, not a guarantee. Results depend on the training data and model quality.")

    with res_col2:
        st.subheader("📊 Session Metrics Breakdown")
        # Interactive Plotly Chart visualizing key metrics
        chart_data = pd.DataFrame({
            'Feature': ['Page Value', 'Product Pages', 'Bounce Rate', 'Exit Rate'],
            'Value': [page_values, product_related, bounce_rates * 50, exit_rates * 50]
        })
        fig = px.bar(chart_data, x='Feature', y='Value', color='Feature',
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     title="Key Behavioral Metrics")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("💡 Fill in the session details above and click **'Predict Purchase Intention'** to instantly view real-time results and interactive charts.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>NextGen Institute Machine Learning Final Project | Built with Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)
