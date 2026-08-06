# Diabetes Risk Prediction Using BRFSS 2015

## Project Overview

Diabetes is one of the most common chronic illnesses and has a major impact on public health and the economy. Early identification of individuals who are at risk of diabetes or prediabetes can support timely intervention and help reduce long-term health complications.

This project applies machine learning and explainable artificial intelligence methods to predict diabetes status using health indicators from the **2015 Behavioral Risk Factor Surveillance System (BRFSS)** dataset.

The project compares binary and multiclass diabetes prediction to examine:

- Model performance
- Prediction complexity
- Class imbalance
- Feature importance
- The difficulty of identifying prediabetes
- The possibility of developing a compact diabetes-risk screening model

The project includes:

- Exploratory data analysis
- Binary diabetes classification
- Three-class diabetes classification
- Data preprocessing
- Baseline machine learning models
- Class-weighted models
- SMOTE-based imbalance handling
- Logistic Regression
- Random Forest
- XGBoost
- Hyperparameter tuning
- SHAP explainability analysis
- Permutation importance
- Multiple feature-selection methods
- Reduced-feature modeling
- Comparison of full-feature and reduced-feature models

---

## Problem Statement

Although diabetes screening methods are available, many individuals with prediabetes or undiagnosed diabetes are unaware that they have the condition.

Machine learning models may support diabetes-risk prediction using demographic, behavioral, lifestyle, and health-related indicators. However, relatively few studies directly compare binary and multiclass diabetes prediction using BRFSS health indicators.

Explainability is also important because public-health professionals need clear and practical information about the factors influencing model predictions.

This project therefore investigates both predictive performance and feature-level explanations for diabetes and prediabetes classification.

---

## Research Questions

1. How accurately can machine learning models using BRFSS 2015 health feature indicators predict diabetes status, no diabetes, prediabetes, and diabetes?

2. Based on SHAP explainability analysis, which behavioral, demographic, and lifestyle factors are the best indicators of diabetes and prediabetes?

3. What does this indicate about the challenge of differentiating between prediabetes and diabetes, and how does model performance alter when predicting diabetes using a 3-class formulation versus a binary formulation?

4. Is it possible to create a compact and accurate diabetes risk screening tool that is similar to the full feature set using a reduced set of BRFSS health indicators found through feature selection methods?


---

## Dataset Description

This project uses two datasets derived from the **2015 Behavioral Risk Factor Surveillance System**, an annual health-related telephone survey conducted by the Centers for Disease Control and Prevention.

The original BRFSS survey contains demographic, behavioral, lifestyle, and health-related information from more than 400,000 participants.

The two prepared datasets used in this project each contain:

- **253,680 survey responses**
- **21 predictive health indicators**
- One diabetes-status target variable

### 1. Multiclass Dataset

**File name:**

```text
diabetes_012_health_indicators_BRFSS2015.csv
```

The target variable is:

```text
Diabetes_012
```

It contains three classes:

| Class | Description |
|---|---|
| `0` | No diabetes or diabetes only during pregnancy |
| `1` | Prediabetes |
| `2` | Diabetes |

The multiclass dataset is highly imbalanced, particularly because the prediabetes class contains substantially fewer observations than the other classes.

### 2. Binary Dataset

**File name:**

```text
diabetes_binary_health_indicators_BRFSS2015.csv
```

The target variable is:

```text
Diabetes_binary
```

It contains two classes:

| Class | Description |
|---|---|
| `0` | No diabetes |
| `1` | Prediabetes or diabetes |

The binary dataset is also imbalanced and is used to compare binary prediction performance with the three-class formulation.

Both datasets support:

- Data preprocessing
- Exploratory data analysis
- Class-imbalance analysis
- Machine-learning model training
- Feature selection
- Hyperparameter tuning
- Explainability analysis
- Binary versus multiclass comparison

---

## Dataset Source

The datasets are available from Kaggle:

[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)

---

## Repository Structure

```text
project-root/
│
├── data/
│   └── diabetes-health-indicators-dataset.zip
│
├── eda/
│   ├── MRP_EDA.ipynb
│   └── EDA-generated images and plots
│
├── notebooks/
│   └── model_test_1.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── baseline_models.py
│   ├── baseline_evaluation.py
│   ├── baseline_evaluation_balanced.py
│   ├── random_forest_baseline.py
│   ├── random_forest_balanced.py
│   ├── xgboost_baseline.py
│   ├── xgboost_balanced.py
│   ├── smote_preprocessing.py
│   ├── feature_selection.py
│   ├── reduced_models.py
│   ├── hyperparameter_tuning.py
│   └── shap_analysis.py
└── README.md
```

---

## Folder and File Details

### `data/`

The `data` folder contains the dataset as a ZIP file because the extracted CSV files are too large to store directly in the repository.

Before running the notebooks, the ZIP file must be extracted so that the following files are available:

```text
data/
├── diabetes_012_health_indicators_BRFSS2015.csv
└── diabetes_binary_health_indicators_BRFSS2015.csv
```

### `eda/`

The `eda` folder contains:

- The exploratory data analysis notebook
- Target-distribution plots
- Feature-distribution plots
- Class-comparison plots
- Correlation heatmap
- Preliminary feature-importance plot
- Other images produced by the EDA notebook

### `notebooks/`

The `notebooks` folder contains:

```text
model_test_1.ipynb
```

This is the main notebook used to run the complete machine-learning, explainability, feature-selection, and hyperparameter-tuning workflow.

### `src/`

The `src` folder contains reusable Python scripts imported by the main notebook.

### `README.md`

This file provides the project overview, dataset information, repository structure, and instructions for running the project.

---

# Running the Project in Google Colab

## 1. Download or Clone the Repository

Download the repository as a ZIP file from GitHub or clone it using Git.

Example:

```python
!git clone YOUR_REPOSITORY_URL
```

Replace `YOUR_REPOSITORY_URL` with the repository URL.

---

## 2. Upload the Project to Google Drive

Place the complete project folder in Google Drive using the following structure:

```text
MyDrive/
└── MRP/
    └── codes/
        ├── data/
        ├── eda/
        ├── notebooks/
        ├── src/
        └── README.md
```

The notebooks currently expect the project root to be:

```python
/content/drive/MyDrive/MRP/codes
```

When the project is stored in a different location, update the file paths inside the notebooks.

---

## 3. Extract the Dataset ZIP File

The two CSV files must be extracted into the `data` folder before running the notebooks.

After extraction, the folder should contain:

```text
data/
├── diabetes_012_health_indicators_BRFSS2015.csv
└── diabetes_binary_health_indicators_BRFSS2015.csv
```

The following Colab code can be used to extract the dataset:

```python
from pathlib import Path
import zipfile

project_root = Path("/content/drive/MyDrive/MRP/codes")

zip_path = (
    project_root
    / "data"
    / "diabetes-health-indicators-dataset.zip"
)

extract_path = project_root / "data"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully.")
```

Change the ZIP filename when the uploaded ZIP file has a different name.

---

## 4. Install the Required Packages

Run the following cell before running the notebooks:

```python
!pip install -q \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    imbalanced-learn \
    xgboost \
    shap
```

---

## 5. Mount Google Drive

Both notebooks require access to Google Drive.

Run:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Follow the authorization instructions shown by Colab.

---

# Running the EDA Notebook in Google Colab

## 1. Open the EDA Notebook

Open the notebook located in:

```text
eda/mrp_eda.ipynb
```

The notebook can be opened by:

- Uploading it directly to Google Colab
- Opening it from Google Drive
- Opening it from the GitHub repository

## 2. Set the Project Path

For a consistent repository structure, the EDA notebook should load the datasets from the `data` folder.

Use:

```python
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/MRP/codes")

multiclass_path = (
    PROJECT_ROOT
    / "data"
    / "diabetes_012_health_indicators_BRFSS2015.csv"
)

binary_path = (
    PROJECT_ROOT
    / "data"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)
```

Then load the files:

```python
import pandas as pd

df_multi = pd.read_csv(multiclass_path)
df_binary = pd.read_csv(binary_path)

print("Datasets loaded successfully.")
```

## 3. Set the EDA Output Folder

To save all EDA images inside the repository, create an output folder:

```python
EDA_OUTPUT = PROJECT_ROOT / "eda" / "images"
EDA_OUTPUT.mkdir(parents=True, exist_ok=True)

print("EDA images will be saved to:", EDA_OUTPUT)
```

A figure can then be saved using:

```python
plt.savefig(
    EDA_OUTPUT / "Multiclass_Target_Distribution.png",
    bbox_inches="tight"
)
```

This is preferable to using a separate hardcoded `EDA` path.

## 4. Run the EDA Notebook

Run all cells in order:

```text
Runtime → Run all
```

The EDA notebook will:

1. Load the binary and multiclass datasets.
2. Display dataset dimensions and columns.
3. Inspect data types and summary statistics.
4. Check missing values.
5. Count duplicate records.
6. Calculate target-class counts and percentages.
7. Plot binary and multiclass target distributions.
8. Compare numerical and ordinal features across classes.
9. Compare binary health indicators across classes.
10. Generate BMI, general-health, and age boxplots.
11. Compare high blood pressure percentages across classes.
12. Compare high cholesterol percentages across classes.
13. Generate a correlation heatmap.
14. Calculate target correlations.
15. Calculate class-imbalance ratios.
16. Train a preliminary Random Forest model.
17. Plot the top 10 preliminary feature importances.

---

# Running the Main Modeling Notebook in Google Colab

## 1. Open the Main Notebook

Open:

```text
notebooks/model_test_1.ipynb
```

The notebook may be opened directly from Google Drive, uploaded to Colab, or opened from GitHub.

## 2. Select the Runtime

From the Google Colab menu, select:

```text
Runtime → Change runtime type → Python 3
```

A GPU is not required for the machine-learning models in this project. However, a high-memory runtime may be helpful for:

- SMOTE
- Randomized hyperparameter tuning
- SHAP analysis
- Large multiclass model training

## 3. Confirm the Project Root

The main notebook uses:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path("/content/drive/MyDrive/MRP/codes")

sys.path.append(str(PROJECT_ROOT))
```

Make sure this path matches the location of the project in Google Drive.

## 4. Confirm the Dataset Paths

The notebook expects:

```python
binary_path = (
    PROJECT_ROOT
    / "data"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)

multi_path = (
    PROJECT_ROOT
    / "data"
    / "diabetes_012_health_indicators_BRFSS2015.csv"
)
```

## 5. Confirm the `src` Package

The following file must exist:

```text
src/__init__.py
```

The main notebook also creates it using:

```python
!touch /content/drive/MyDrive/MRP/codes/src/__init__.py
```

This allows imports such as:

```python
from src.preprocessing import prepare_modeling_data
```

## 6. Run the Main Notebook

Run all notebook cells in order:

```text
Runtime → Run all
```

The notebook imports the supporting scripts from the `src` folder, trains the models, evaluates performance, performs feature selection, creates reduced models, tunes selected models, and generates explainability results.

Some sections may require more time than others, particularly:

- SMOTE on the multiclass dataset
- Permutation importance
- SHAP analysis
- Randomized hyperparameter tuning
- Multiclass XGBoost training

---

## Important Notes

- Keep `src/__init__.py` in the repository so that Python recognizes `src` as a package.
- Do not change the CSV filenames unless the paths in both notebooks are updated.
- Extract the dataset ZIP before running either notebook.
- Run notebook cells in order because later sections depend on earlier variables and trained models.
- Apply SMOTE only to training data to prevent data leakage.
- Keep the test set unchanged for final model evaluation.
- Update Google Drive paths when the project is stored in a different folder.
- The `__pycache__` folder is generated automatically by Python and does not need to be included in GitHub.
- Notebook checkpoint folders do not need to be uploaded.
- Saved EDA images can be retained in the `eda` folder to document the analysis.
- Hyperparameter tuning and SHAP analysis may require a longer runtime than baseline model training.

---


## Author

**Ishrat Jaben Bushra**

Master of Data Science and Analytics  
Toronto Metropolitan University

