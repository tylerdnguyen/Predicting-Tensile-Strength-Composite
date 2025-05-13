# Composite Materials Data Analysis (Capstone Project)

This repository contains a comprehensive analysis of a composite materials dataset, focusing on understanding the relationship between material composition and mechanical properties, particularly tensile strength. The analysis was performed as part of a capstone project using Python and Jupyter Notebook.

## 📁 Project Files

- `Capstone.ipynb`: Jupyter Notebook containing full data analysis and visualizations.
- `X_bp.xlsx`: Dataset including matrix-filler ratios, resin and hardener compositions, and mechanical test results.

## 🧪 Dataset Description

The dataset includes 1,023 composite material samples with the following features:

- **Matrix-filler ratio**
- **Density (kg/m³)**
- **Modulus of Elasticity (GPa)**
- **Amount of Hardener (mass %)**
- **Content of Epoxy Groups (%)**
- **Flash Point (°C)**
- **Surface Density (g/m²)**
- **Tensile Modulus (GPa)**
- **Tensile Strength (MPa)** ← *Target variable*
- **Resin Consumption (g/m²)*

## 🔍 Key Analyses

- **Descriptive Statistics**: Summary of all features including means, ranges, and standard deviations.
- **Correlation Analysis**: Weak linear correlation found between tensile strength and other features.
- **Visualizations**: Histograms and pair plots to explore feature distributions and interdependencies.
- **Data Cleaning**: Renaming Russian column headers to English, checking for nulls, and verifying data types.

## 📊 Tools and Libraries

- Python 3.x
- Pandas
- Seaborn
- Matplotlib
- Jupyter Notebook

## 🚀 Future Work

- Implement machine learning models (e.g., Gradient Boosting) to predict tensile strength.
- Include categorical variables like fiber and matrix type (if available).
- Perform SHAP or permutation importance analysis for feature interpretability.

## 🧾 License

This project is released under the MIT License.

---

> 📌 **Note**: This project uses synthetic or anonymized data from [Kaggle](https://www.kaggle.com/datasets/cnezhmar/omposite-materials) for educational purposes.
