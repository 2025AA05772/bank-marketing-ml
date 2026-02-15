# Machine Learning Assignment 2
## **UCI Bank Marketing – Classification Models & Streamlit Deployment**

### **Author:** Gaurav Mishra  
### **BITS ID:** 2025AA05772
---

# 1. Problem Statement  
The goal of this project is to build, evaluate, compare, and deploy multiple machine learning classification models to predict **whether a bank customer will subscribe to a term deposit** (`y = yes/no`), based on the **UCI Bank Marketing Dataset**.

The workflow includes:

- Collecting & preprocessing data  
- Training **six ML models**  
- Evaluating them using **six metrics**  
- Creating a **Streamlit web application**  
- Deploying the app on **Streamlit Community Cloud**  
- Uploading **test_data.csv** for external evaluation  

---

# 2️. Dataset Description  
### **Source:** UCI Machine Learning Repository – Bank Marketing Dataset  
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing  

The dataset contains customer attributes related to telephonic marketing campaigns.

### **Dataset Used:**  
- **File:** `bank-additional-full.csv`  
- **Rows:** 41,188  
- **Columns:** 20 input features + 1 target = **21 columns**  
- **Task:** Binary classification (`y` = “yes” or “no”)  
- **Preprocessing Note:**  
  - The feature **`duration`** (call duration in seconds) is **removed** because as per UCI info, it is known **only after the call ends**, and including it leads to **target leakage**.

### **Final Features Used for Modeling:**  
19 input features (after dropping `duration`) + target `y`.

### **Test Data:**  
To evaluate the models on unseen data, a **20% stratified hold‑out split** was created from the full dataset and saved as: test_data.csv. Since the data is highly imbalanced with approx. 89% and 11% class distribution.
The test file is included in the repository and also used in the Streamlit app for prediction.

---

# 3. Models Used  
The following **six classification models** were trained on the same processed dataset:

1. **Logistic Regression**  
2. **Decision Tree Classifier**  
3. **k‑Nearest Neighbors (kNN)**  
4. **Naive Bayes (Gaussian)**  
5. **Random Forest (Ensemble)**  
6. **XGBoost (Ensemble)**  

Each model was evaluated on the test set using:  
- **Accuracy**  
- **AUC Score**  
- **Precision**  
- **Recall**  
- **F1 Score**  
- **MCC (Matthews Correlation Coefficient)**  

---

# 4. Model Comparison Table  

| **ML Model Name**             | **Accuracy** | **AUC**     | **Precision** | **Recall**  | **F1**      | **MCC**     |
|-------------------------------|--------------|-------------|----------------|-------------|-------------|-------------|
| **Logistic Regression**       | 0.835033     | 0.800941    | 0.367872       | 0.646552    | 0.468933    | 0.401084    |
| **Decision Tree**             | 0.846807     | 0.619170    | 0.321581       | 0.324353    | 0.322961    | 0.236599    |
| **kNN**                       | 0.895120     | 0.755854    | 0.573059       | 0.270474    | 0.367496    | 0.345084    |
| **Naive Bayes (Gaussian)**    | 0.804928     | 0.775499    | 0.317178       | 0.634698    | 0.422980    | 0.348985    |
| **Random Forest (Ensemble)**  | **0.866837** | **0.812212** | **0.435150**   | **0.610991**| **0.508292**| **0.442149** |
| **XGBoost (Ensemble)**        | 0.847172     | 0.808920    | 0.391759       | 0.645474    | 0.487586    | 0.421444    |


---

# 5. Observations on Model Performance  

| **Model**                     | **Observation** |
|------------------------------|------------------|
| **Logistic Regression**      | Performs consistently with strong recall and balanced precision. Interpretable and stable baseline model. |
| **Decision Tree**            | Lowest AUC and MCC; shows signs of overfitting due to unconstrained splits. Performs poorly compared to ensemble methods. |
| **kNN**                      | High accuracy but very low recall. Struggles with high‑dimensional one‑hot encoded data. |
| **Naive Bayes (Gaussian)**   | Good recall but lower precision; simplifies feature interactions but performs reasonably well. |
| **Random Forest (Ensemble)** | **Best‑performing model overall with highest AUC, F1, and MCC.** Balanced performance and strong generalization after tuning (`n_estimators=100`, `max_depth=12`). |
| **XGBoost (Ensemble)**       | Very strong model with competitive AUC and MCC. Good handling of class imbalance and robust performance. |


--- 