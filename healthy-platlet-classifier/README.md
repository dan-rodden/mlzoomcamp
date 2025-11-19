# MLZoomcamp Midterm Project: Healthy Control Platlet Binary Classification


This is the dataset I am using. Finally!: <link src="https://www.kaggle.com/datasets/samiraalipour/gene-expression-omnibus-geo-dataset-gse68086?resource=download">

Includes Tumor-Educated Platlets (TEPs) with healthy (wt) blood platelet cells. For binary classification, goal is to distinguish the difference between TEPS and healthy blood cells with the hope that that features can be found in blood providing some biomarkers.

The target variable used is `!Sample_characteristics_ch1.3`. With this encoding. `"cancer type: HC"` stands for healthy controls used in the dataset. Other samples are classified as cancer in the `target` columns of the dataframe.
- "cancer type: HC"                55
- "cancer type: GBM"               40
- "cancer type: Lung"              40
- "cancer type: CRC"               38
- "cancer type: Pancreas"          35
- "cancer type: Breast"            24
- "cancer type: Hepatobiliary"     13
- "mutational subclass: wt"        12
- "mutational subclass: KRAS"      11
- "mutational subclass: EGFR"       9
- "mutational subclass: HER2+"      6
- "mutational subclass: PIK3CA"     2



For binary classification the target column is encoded via:
- 0: Cancerous
- 1: Healthy Control Blood Platlet

# Healthy Cell Classifier: Tumor-Educated Platelets (TEP)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## 📋 Problem Description
Liquid biopsies represent a non-invasive alternative to traditional tissue biopsies. This project leverages **Tumor-Educated Platelets (TEPs)**—blood platelets that have altered their RNA profile in response to the presence of a tumor—to classify samples as either **Healthy** or **Cancerous** using RNA-seq gene expression data.

Early detection of cancer through a simple blood draw could significantly improve patient outcomes by identifying malignancies before they become symptomatic.

## 📊 Data Source & Comparative Analysis
The data for this project originates from the landmark study by **Best et al. (2015)** published in *Cancer Cell*.
* **Paper:** [RNA-Seq of Tumor-Educated Platelets Enables Blood-Based Pan-Cancer, Multiclass, and Molecular Pathway Cancer Diagnostics](https://www.cell.com/cancer-cell/fulltext/S1535-6108(15)00349-9)

### Data Encoding
0: Cancerous
1: Healthy Control Blood Platlet

### Model Performance
My XGBoost classifier achieves a marginal improvement over the baseline metrics reported in the original study for the binary classification task (Pan-Cancer vs. Healthy).

| Metric | Best et al. (2015) Paper | My Model (XGBoost) |
| :--- | :--- | :--- |
| **Performance** | ~96% Accuracy | **0.9802 AUC** (Test Set) |

### ⚠️ Critical Data Note: Batch Effects
While the model performance is high, it is important to acknowledge a potential confounding factor in the original dataset. The **Healthy Controls (HC)** were potentially processed in a batch distinct from the cancer samples. 

Consequently, the classifier may be detecting technical differences (batch effects) alongside biological signals. This is a known limitation of the GSE68086 dataset, as subsequent validation studies have highlighted that procedural differences between centers can account for variance in gene expression.


## 🛠️ Tech Stack
* **Model:** XGBoost (Gradient Boosting)
* **API:** Flask (Python)
* **Containerization:** Docker
* **Dependency Management:** Pipenv
* **Deployment:** Render Cloud Hosting

## 📂 Project Structure
```text
healthy-cell-classifier/
├── data/               # Raw RNA-seq data
├── Dockerfile          # Blueprint for building the container
├── notebook.ipynb      # EDA and Parameter Tuning
├── Pipfile             # Dependency definitions
├── Pipfile.lock        # Exact versions for reproduction
├── predict.py          # Flask application (Entry point)
├── test-predict.py     # Client script to test the service
├── train.py            # Script to train and save the model
├── README.md           # Project documentation
└── *.bin               # Saved XGBoost model










