# Project Overview

This repository contains all components necessary to run and evaluate our final machine learning project for "Introduction to Machine Learning - Spring 2025".

## 📁 Main Directory Structure

```
├── README.md              # This file
├── .gitignore             # Git configuration to ignore large or unnecessary files
├── ProjectFiles/          # Contains original project instructions and datasets from Moodle
├── Project/               # Main working directory with the Jupyter notebook and submodels
    └── OtherModels/       # Contains additional model runs and performance comparisons
```

---

## 🔧 How to Use This Project

1. **Download the required files**:

    - Go to Moodle and download `train.csv` and `test.csv`
    - Place them in the `Project/` directory
    - ⚠️ Note: All `.csv` files are excluded from this repository using `.gitignore`. You must manually add them before running anything.

2. **Open the main Jupyter Notebook**:

    - Located inside `Project/`
    - This notebook includes data loading, preprocessing, model training, cross-validation, and result generation

3. **Run the notebook**:

    - Make sure `train.csv` and `test.csv` are present in the same folder
    - The final prediction file (`results_31.csv`) will be generated automatically

4. **Explore other models**:

    - Go to `Project/OtherModels/`
    - You'll find additional model definitions that were trained and validated
    - Copy the code into the main notebook to experiment with or compare their performance

---

## 📊 Model Comparison Table

| Model Name                   | AUC (Average) | Runtime         | Description                       |
| ---------------------------- | ------------- | --------------- | --------------------------------- |
| `model_rf_cv100_full_v1`     | \~0.909       | \~12:15 minutes | 100 trees, default depth          |
| `model_rf_cv100x15_leaf5_v2` | \~0.932       | \~2:55 minutes  | 100 trees, depth=15, leaf=5, fast |

All models used 5-Fold Stratified Cross-Validation and were run on the full feature set after preprocessing.

---

## 📝 Notes

-   `.gitignore` is configured to exclude all `.csv` files, including training and test data.
-   You must download and place the required CSVs manually.
-   All models are valid according to course guidelines.
-   The final submission file `results_31.csv` was generated using `model_rf_cv100x15_leaf5_v2`.

Good luck and have fun!
