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
- 0: Cancerous
- 1: Healthy Control Blood Platlet

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

## Cloud Deployment Demo
The service is deployed to Render. You can test it using the following command:

```bash
curl -X POST [https://healthy-platelet-classifier.onrender.com/predict](https://healthy-platelet-classifier.onrender.com/predict) \
  -H "Content-Type: application/json" \
  -d '{"url": "[https://raw.githubusercontent.com/dan-rodden/mlzoomcamp/main/healthy-platlet-classifier/test_image.jpg](https://raw.githubusercontent.com/dan-rodden/mlzoomcamp/main/healthy-platlet-classifier/test_image.jpg)"}'
```



<img width="1085" height="636" alt="live_demo_screenshot" src="https://github.com/user-attachments/assets/58b0bf1e-4c56-4472-a6a5-8c1878b8325a" />






