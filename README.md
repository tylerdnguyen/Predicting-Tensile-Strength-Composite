# 📘 Composite Material Strength Prediction – Capstone Project

This repository contains a machine learning pipeline designed to predict the **tensile strength of composite materials** based on various experimental input features. It is the final project for a capstone course and showcases regression modeling, feature engineering, model evaluation, and comparison.

---

## 📊 Project Overview

Composite materials are widely used in engineering due to their high strength-to-weight ratio. This project aims to:

- Predict **tensile strength (MPa)** from multiple input features like modulus, density, resin content, etc.
- Compare regression models including Linear, Ridge, Lasso, SVR, Random Forest, Gradient Boosting, and Deep Learning.
- Tune hyperparameters using `GridSearchCV`.
- Evaluate models based on R² score and Mean Squared Error (MSE).
- Perform feature importance analysis using **Permutation Importance**.

---

## 🧠 Models Included

- Linear Regression  
- Ridge Regression  
- Lasso Regression with Polynomial Features  
- Support Vector Regression (SVR)  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Deep Neural Network (TensorFlow/Keras)

---

## 📁 Files

- `Capstone.ipynb` — Main Jupyter notebook containing all code, evaluation, and results
- `X_bp.xlsx` — Dataset used for training and testing the models
- `images/comp2.jpg` — Supporting image used in the notebook (optional)

---

## ⚙️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/composite-strength-prediction.git
   cd composite-strength-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook Capstone.ipynb
   ```

---

## 📝 Requirements

* Python 3.8+
* `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `openpyxl`
* (Optional) `scikeras` for `KerasRegressor`

---

## 🔍 Results Summary

| Model                | R² Score | MSE   |
| -------------------- | -------- | ----- |
| Lasso | 0.9983     | 378.9543 |
| LinearRegression        | 0.9983     | 390.0280 |
| Ridge    | 0.9983     | 390.0285 |
| GradientBoosting  | 0.9973     | 606.5007 |
| RandomForest     | 0.9933     | 1506.8009 |
| SVR     | 0.9850     | 3360.6140 |
| Deep Learning Keras     | 0.9829     | 3818.8561 |

---

## 🚀 Future Work

* Incorporate domain-specific composite features (e.g., curing time, fiber layout)
* Add uncertainty estimation (e.g., prediction intervals)
* Deploy as a web app using Streamlit or Flask
* Explore physics-informed ML techniques

---

## 🙋‍♂️ Author

**Thinh Nguyen**
Capstone Project
LinkedIn: https://www.linkedin.com/in/thinh-nguyen-eit-510102150/
Email: tylerdnguyen94@gmail.com
