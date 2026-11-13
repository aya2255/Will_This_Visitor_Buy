# 🛒 Will This Visitor Buy? Real-Time E-Commerce Purchase Intent Prediction

Predicting whether a website visitor will complete a purchase based on their browsing session features, enabling real-time engagement strategies to increase conversion rates.

🌐 **Live Web App:** [Link to Deployed Streamlit App](https://willthisvisitorbuygit-yfssx8cczhjexzmgddnbkq.streamlit.app/)

---

## 📌 Project Overview
Most e-commerce website visitors leave without buying anything. This project builds a machine learning classification model using the **Online Shoppers Purchasing Intention Dataset (~12.3K sessions)** from UCI. By predicting purchase intent in real-time, online stores can smartly target potential buyers with discounts, chat assistants, or special offers without giving away margins to everyone.

## ✨ Key Features & Web App
The deployed Streamlit web app allows users to input session metrics (such as page values, duration, bounce rates, exit rates, and visitor type) to instantly get:
* **Purchase Probability (%)**
* **Actionable Recommendations:** (e.g., *No action / Show coupon / Show chat*)

---

## 📊 Model Evaluation & Results

All metrics reported below were evaluated on the untouched test set using the best-performing model (Random Forest / Tuned Ensemble):

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.9213** | **92.1%** |
| **Accuracy** | **0.8686** | **86.9%** |
| **Recall (Sensitivity)** | **0.7749** | **77.5%** |
| **F1-Score** | **0.6463** | **64.6%** |
| **Precision** | **0.5543** | **55.4%** |

> *Note: Model tuning was performed using GridSearchCV exclusively on the training data inside a `scikit-learn` Pipeline to prevent data leakage.*

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 
* **Data Manipulation & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Pipelines, Decision Trees, Random Forest, GridSearchCV)
* **Model Persistence:** Joblib
* **Web App & Deployment:** Streamlit, Streamlit Community Cloud

---

## 🚀 How to Run Locally

If you want to run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Will_This_Visitor_Buy.git
   cd Will_This_Visitor_Buy
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Repository Structure
```text
├── notebook.ipynb             # Complete data understanding, EDA, preprocessing, and model training
├── app.py                     # Streamlit web application code
├── model.pkl                  # Trained and serialized machine learning pipeline
├── requirements.txt           # Required Python libraries
└── README.md                  # Project documentation
```

---

## 💼 Portfolio & CV Summary Bullet
> *"Developed a purchase-intent model for 12,330 e-commerce sessions using Random Forest and Scikit-Learn Pipelines, achieving a **0.92 ROC-AUC** and **77.5% Recall**. Deployed a live Streamlit app recommending real-time conversion actions for high-intent visitors."*

---

## 📄 Dataset Source
* **Dataset:** Online Shoppers Purchasing Intention Dataset
* **Source:** UCI Machine Learning Repository (Dataset ID: 468)
