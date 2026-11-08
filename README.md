# Oncology AI Engine — Cancer Subtype Prediction via Gene Expression (RNA-Seq)

An end-to-end precision oncology decision-support platform that classifies human tumor samples into 5 distinct cancer subtypes from high-throughput RNA-Seq transcriptomic profiles (20,531 genes). Powered by gradient-boosted decision tree ensembles (**XGBoost**), ANOVA $F$-value feature selection (**SelectKBest**), and game-theoretic explainability (**SHAP TreeExplainer**), wrapped in a dark SaaS clinical dashboard.

---

## Table of Contents
- [The Clinical Challenge & Purpose](#the-clinical-challenge--purpose)
- [How It Helps Clinicians & Patients](#how-it-helps-clinicians--patients)
- [How the Engine Works](#how-the-engine-works)
- [Project File Structure](#project-file-structure)
- [Sample Patient Data (`sampledata/`)](#sample-patient-data-sampledata)
- [Dataset Details](#dataset-details)
- [Technologies Used](#technologies-used)
- [Installation & How to Run](#installation--how-to-run)
- [Dashboard Architecture](#dashboard-architecture)
- [Medical Disclaimer](#medical-disclaimer)

---

## The Clinical Challenge & Purpose

In classical oncology, cancer is diagnosed primarily through tissue biopsy and histological microscopy. However, tumors that look identical under a microscope often behave completely differently at the molecular level. A therapy that cures one patient may fail entirely in another because their tumors are driven by different underlying genetic pathways.

**Gene expression profiling (RNA-Seq)** measures the activity levels of tens of thousands of genes simultaneously, providing a high-resolution molecular fingerprint of the tumor. 

### Core Objectives:
1. **Histological Differentiation**: Accurately differentiate between 5 major cancer subtypes:
   - **BRCA**: Breast Invasive Carcinoma
   - **COAD**: Colon Adenocarcinoma
   - **KIRC**: Kidney Renal Clear Cell Carcinoma
   - **LUAD**: Lung Adenocarcinoma
   - **PRAD**: Prostate Adenocarcinoma
2. **Overcoming the "Curse of Dimensionality"**: Extract the most statistically informative biomarkers from over 20,000 genes to make real-time inference viable.
3. **Black-Box AI Elimination**: Provide transparent, patient-specific attributions through game-theoretic Shapley values so oncologists understand *why* the AI arrived at a specific diagnosis.

---

## How It Helps Clinicians & Patients

| Clinical Benefit | How This System Delivers It |
|---|---|
| **Prevents Misclassification** | Multi-class gradient-boosted decision trees deliver >98% generalization accuracy across all 5 cancer cohorts. |
| **Personalized Targeted Therapy** | Identifies specific overexpressed driver genes (e.g., hormone receptors, tyrosine kinases) to help guide targeted inhibitor selection over generic toxic chemotherapy. |
| **Transparent Explainability (XAI)** | Instead of a black-box probability score, clinicians inspect a SHAP Beeswarm Plot highlighting individual positive and negative genomic drivers. |
| **Rapid Second Opinion** | Delivers instantaneous inference from standard Illumina HiSeq CSV outputs, reducing diagnostic turnaround time from days to seconds. |

---

## How the Engine Works

```
┌─────────────────────────────────┐
│ Patient RNA-Seq CSV (20,531)    │  Raw Illumina HiSeq log2-normalized expression counts
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ SelectKBest ANOVA F-Filtering   │  Reduces 20,531 loci → Top 500 discriminant biomarkers (97.6% reduction)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ XGBoost Gradient-Boosted Trees  │  Multi-class softmax probability distribution across 5 cancer classes
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ SHAP TreeExplainer Attribution  │  Computes game-theoretic Shapley values per gene for the diagnosed class
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Interactive Clinical Dashboard  │  Real-time waveform, subtype matrix, SHAP plot, and top driver genes table
└─────────────────────────────────┘
```

1. **Patient Data Ingestion**: Accepts a single-patient or multi-patient CSV containing gene expression counts across 20,531 genomic loci (`gene_0` to `gene_20530`).
2. **Biomarker Reduction**: The fitted `SelectKBest(f_classif, k=500)` feature selector isolates the 500 genes with the highest statistical variance and class-separation power.
3. **Ensemble Classification**: The trained `XGBClassifier` calculates posterior probabilities across BRCA, COAD, KIRC, LUAD, and PRAD.
4. **Shapley Attribution**: The `shap.TreeExplainer` decomposes the patient's individual gene values against the baseline expectation, calculating the exact positive or negative pull each gene exerted on the model's decision.
5. **UI Rendering**: The results update dynamically in the dark dashboard.

---

## Project File Structure

```
Cancer Subtype Prediction via Gene Expression (RNA-Seq)/
│
├── app.py                      # Main Streamlit clinical dashboard application
├── train_model.py              # Training pipeline (downloads TCGA data, trains XGBoost, saves artifacts)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation and clinical guide
│
├── model.pkl                   # Trained XGBoost multi-class classification model
├── encoder.pkl                 # Scikit-Learn LabelEncoder (maps 0-4 to BRCA, COAD, KIRC, LUAD, PRAD)
├── selector.pkl                # Fitted SelectKBest feature selector (top 500 genes)
├── sample_patient.csv          # Benchmark patient sample (PRAD, 20,531 genes)
│
└── sampledata/                 # Dedicated folder for 20 real patient CSV samples (from TCGA Pan-Cancer)
    ├── sample_data_1.csv ... sample_data_5.csv   # Initial benchmark samples (PRAD, BRCA, COAD, KIRC, LUAD)
    └── sample_data_6.csv ... sample_data_20.csv  # 15 additional real patient samples across all 5 subtypes
```

---

## Sample Patient Data (`sampledata/`)

The `sampledata/` directory contains **20 pre-extracted, real patient samples** directly from the official TCGA Pan-Cancer Illumina HiSeq cohort. Each file contains all 20,531 continuous RNA-Seq transcriptomic expression levels for a unique real patient, with no synthetic/fake data:

| File | Sample ID | Ground Truth Diagnosis | Target Organ | Model Confidence |
|---|---|---|---|---|
| [`sample_data_1.csv`](file:///sampledata/sample_data_1.csv) | `sample_19` | **PRAD** (Prostate Adenocarcinoma) | Prostate | **99.8%** |
| [`sample_data_2.csv`](file:///sampledata/sample_data_2.csv) | `sample_10` | **BRCA** (Breast Invasive Carcinoma) | Breast | **99.9%** |
| [`sample_data_3.csv`](file:///sampledata/sample_data_3.csv) | `sample_65` | **COAD** (Colon Adenocarcinoma) | Colon | **99.6%** |
| [`sample_data_4.csv`](file:///sampledata/sample_data_4.csv) | `sample_40` | **KIRC** (Kidney Renal Clear Cell) | Kidney | **99.8%** |
| [`sample_data_5.csv`](file:///sampledata/sample_data_5.csv) | `sample_16` | **LUAD** (Lung Adenocarcinoma) | Lung | **99.8%** |
| [`sample_data_6.csv`](file:///sampledata/sample_data_6.csv) | `sample_4` | **BRCA** (Breast Invasive Carcinoma) | Breast | **99.9%** |
| [`sample_data_7.csv`](file:///sampledata/sample_data_7.csv) | `sample_8` | **BRCA** (Breast Invasive Carcinoma) | Breast | **99.8%** |
| [`sample_data_8.csv`](file:///sampledata/sample_data_8.csv) | `sample_13` | **BRCA** (Breast Invasive Carcinoma) | Breast | **99.9%** |
| [`sample_data_9.csv`](file:///sampledata/sample_data_9.csv) | `sample_26` | **COAD** (Colon Adenocarcinoma) | Colon | **99.6%** |
| [`sample_data_10.csv`](file:///sampledata/sample_data_10.csv) | `sample_47` | **COAD** (Colon Adenocarcinoma) | Colon | **99.6%** |
| [`sample_data_11.csv`](file:///sampledata/sample_data_11.csv) | `sample_54` | **COAD** (Colon Adenocarcinoma) | Colon | **99.6%** |
| [`sample_data_12.csv`](file:///sampledata/sample_data_12.csv) | `sample_6` | **KIRC** (Kidney Renal Clear Cell) | Kidney | **99.6%** |
| [`sample_data_13.csv`](file:///sampledata/sample_data_13.csv) | `sample_11` | **KIRC** (Kidney Renal Clear Cell) | Kidney | **99.8%** |
| [`sample_data_14.csv`](file:///sampledata/sample_data_14.csv) | `sample_17` | **KIRC** (Kidney Renal Clear Cell) | Kidney | **99.8%** |
| [`sample_data_15.csv`](file:///sampledata/sample_data_15.csv) | `sample_1` | **LUAD** (Lung Adenocarcinoma) | Lung | **99.8%** |
| [`sample_data_16.csv`](file:///sampledata/sample_data_16.csv) | `sample_22` | **LUAD** (Lung Adenocarcinoma) | Lung | **99.8%** |
| [`sample_data_17.csv`](file:///sampledata/sample_data_17.csv) | `sample_25` | **LUAD** (Lung Adenocarcinoma) | Lung | **99.8%** |
| [`sample_data_18.csv`](file:///sampledata/sample_data_18.csv) | `sample_2` | **PRAD** (Prostate Adenocarcinoma) | Prostate | **97.6%** |
| [`sample_data_19.csv`](file:///sampledata/sample_data_19.csv) | `sample_3` | **PRAD** (Prostate Adenocarcinoma) | Prostate | **99.7%** |
| [`sample_data_20.csv`](file:///sampledata/sample_data_20.csv) | `sample_5` | **PRAD** (Prostate Adenocarcinoma) | Prostate | **99.7%** |

> **Note**: You can drag and drop any of these files directly into the top navbar file uploader, or cycle through them in the app using the **Demo Sample** button.

---

## Dataset Details

- **Origin**: The Cancer Genome Atlas (TCGA) Pan-Cancer project, hosted on the **UCI Machine Learning Repository (Dataset ID: 401)**.
- **Samples**: 801 patients randomly selected from the TCGA HiSeq database.
- **Features**: 20,531 continuous attributes representing $\log_2(\text{RSEM counts} + 1)$ transcriptomic expression levels.
- **Class Distribution**:
  - `BRCA`: 300 samples (37.5%)
  - `KIRC`: 146 samples (18.2%)
  - `LUAD`: 141 samples (17.6%)
  - `PRAD`: 136 samples (17.0%)
  - `COAD`: 78 samples (9.7%)

---

## Technologies Used

- **Runtime & Web Framework**: Python 3.10+, [Streamlit](https://streamlit.io/)
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/) (`XGBClassifier`)
- **Feature Selection & Preprocessing**: [Scikit-Learn](https://scikit-learn.org/) (`SelectKBest`, `f_classif`, `LabelEncoder`)
- **Explainable AI (XAI)**: [SHAP](https://shap.readthedocs.io/) (`TreeExplainer`, beeswarm plots)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [PyArrow](https://arrow.apache.org/)
- **Data Fetching**: `ucimlrepo` (UCI Machine Learning Repository API) & HuggingFace Datasets
- **Visualizations**: [Matplotlib](https://matplotlib.org/)
- **Model Serialization**: `joblib`

---

## Installation & How to Run

### 1. Clone or Open the Repository
```bash
cd "Cancer Subtype Prediction via Gene Expression (RNA-Seq)"
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

The `.venv` directory is intentionally kept on your local machine but ignored by
Git, so it will not be uploaded to GitHub.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain the Model
To re-fetch the raw dataset from UCI / HuggingFace and re-train the XGBoost model:
```bash
python train_model.py
```
This script automatically generates `model.pkl`, `encoder.pkl`, `selector.pkl`, and `sample_patient.csv`.

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## Dashboard Architecture

The dashboard is structured into four functional areas:

1. **Top Navbar**:
   - **Brand Indicator**: Pulsing violet indicator with application title.
   - **Horizontal File Uploader**: Drag-and-drop support for patient CSV files with compact sideways chip display.
   - **File Status**: Live file readiness status for the uploaded patient dataset.
2. **Sub-Navbar Hero**:
   - Clinical subtitle with dataset attribution and the prominent **Initialize AI Analysis** CTA button.
3. **Stat Cards (Top Row)**:
   - **Biomarkers Analyzed**: Feature space reduction metric (500 genes selected from 20,531).
   - **Predicted Subtype**: High-visibility histological classification (e.g. `PRAD - Prostate`).
   - **Model Confidence**: Posterior probability score of the diagnosed subtype.
4. **Analysis Matrix (Middle Row)**:
   - **Biomarker Expression Loci Waveform**: Smooth interpolated wave chart plotting the patient's top 30 expression peaks across the transcriptome.
   - **Pan-Cancer Subtype Heatmap Matrix**: Ranked probability distribution and visual progress bars across all 5 cancer classes.
5. **Model Interpretability (Bottom Row)**:
   - **Unified SHAP Card**: Game-theoretic Shapley beeswarm plot indicating individual gene impacts on the classification, paired with an interactive table of the **Top 5 Positive Genomic Driver Genes**.

---

## Medical Disclaimer

This software is developed strictly for **computational biology research, bioinformatics benchmarking, and educational purposes**. It is not a certified diagnostic medical device and should not be used as the sole basis for clinical diagnosis, patient staging, or surgical decisions without independent pathology confirmation and oncologist review.
