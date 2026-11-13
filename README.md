# Will This Visitor Buy? — E-commerce Purchase Intention Prediction

A machine-learning project that predicts whether an online shopping session will end with a purchase.

## Dataset
- **Source:** UCI Online Shoppers Purchasing Intention Dataset (ID: 468)
- **Size:** 12,330 sessions, 17 input features
- **Target:** `Revenue` (purchase / no purchase)
- **License:** CC BY 4.0
- **Source link:** https://archive.ics.uci.edu/dataset/468/online

## Project workflow
1. Problem framing
2. Data understanding
3. Exploratory data analysis
4. Train/test split and preprocessing pipeline
5. Baseline + Logistic Regression + Decision Tree + Random Forest
6. 5-fold stratified cross-validation
7. Random Forest hyperparameter tuning with GridSearchCV
8. Test evaluation: precision, recall, F1, ROC-AUC, confusion matrix
9. Streamlit prediction app

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Run every cell in `notebook.ipynb` from top to bottom. The notebook downloads the dataset from UCI and saves:
- `models/purchase_intention_pipeline.joblib`
- `reports/test_metrics.csv`

Then launch the app:
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Make sure `requirements.txt`, `app.py`, and the trained model file are included in the repository (or train the model during setup using a separate script).
3. Open https://share.streamlit.io/
4. Select the repository, branch, and `app.py`.
5. Deploy and test the public URL in a private browser window.

**Note:** The trained model file is generated only after running the notebook. Do not claim a public live app until deployment is completed and tested.

## Results
Fill this table only after running the notebook. Never invent scores.

| Model | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | — | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — |
| Tuned Random Forest | — | — | — | — | — | — |

## Limitations
Predictions are estimates, not guarantees. The dataset reflects a particular collection period and website context. Model performance may change on new traffic and should be monitored. Use predictions as decision support, not as the sole basis for customer decisions.

## Dataset citation
Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q. Licensed under CC BY 4.0.
