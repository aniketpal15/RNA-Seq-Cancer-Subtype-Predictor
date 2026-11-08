# Oncology AI Engine — Cancer Subtype Prediction via Gene Expression (RNA-Seq)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oncologyaieng.streamlit.app/)
[![Windows Download](https://img.shields.io/badge/Windows-Download%20.exe-blue?logo=windows)](https://github.com/aniketpal15/RNA-Seq-Cancer-Subtype-Predictor/raw/main/Oncology_AI.exe)
[![Android Download](https://img.shields.io/badge/Android-Download%20.apk-brightgreen?logo=android)](https://github.com/aniketpal15/RNA-Seq-Cancer-Subtype-Predictor/raw/main/Oncology_AI.apk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> 🚀 **Live Production Deployment**: Try the interactive web app online at **[https://oncologyaieng.streamlit.app/](https://oncologyaieng.streamlit.app/)**

The local model now includes a `NORMAL` reference class. When that class is
predicted, the dashboard displays **NO CANCER DETECTED**; otherwise it reports
one of the five supported cancer subtypes. This is a research prototype, not a
clinical diagnostic device.

An end-to-end research decision-support platform that distinguishes a normal reference profile from 5 supported cancer subtypes using high-throughput RNA-Seq transcriptomic profiles (20,531 genes). Powered by gradient-boosted decision tree ensembles (**XGBoost**), ANOVA $F$-value feature selection (**SelectKBest**), and game-theoretic explainability (**SHAP TreeExplainer**), wrapped in a dark SaaS clinical dashboard.

---

## Table of Contents
- [Downloads & Standalone Apps](#-downloads--standalone-apps)
- [Live Demo](#live-demo)
- [The Clinical Challenge & Purpose](#the-clinical-challenge--purpose)
- [How It Helps Clinicians & Patients](#how-it-helps-clinicians--patients)
- [How the Engine Works](#how-the-engine-works)
- [Cancer / Normal Decision Flow](#cancer--normal-decision-flow)
- [Project File Structure](#project-file-structure)
- [Sample Patient Data (`sampledata/`)](#sample-patient-data-sampledata)
- [Testing with Local CSVs](#testing-with-local-csvs)
- [Dataset Details](#dataset-details)
- [Technologies Used](#technologies-used)
- [Installation & How to Run](#installation--how-to-run)
- [Dashboard Architecture](#dashboard-architecture)
- [Medical Disclaimer](#medical-disclaimer)

---

## 📥 Downloads & Standalone Apps

Run the Oncology AI Engine directly on your PC or Android mobile phone:

| Platform | Format | Direct Download Link | Description & Usage |
|---|---|---|---|
| **Windows Desktop** | `.exe` (Standalone) | [📥 **Download `Oncology_AI.exe`**](https://github.com/aniketpal15/RNA-Seq-Cancer-Subtype-Predictor/raw/main/Oncology_AI.exe) | Double-click `Oncology_AI.exe`. Launches directly as a native standalone application window on PC (no browser tabs or address bar). Runs 100% offline with local engine. |
| **Android Mobile** | `.apk` (Production) | [📥 **Download `Oncology_AI.apk`**](https://github.com/aniketpal15/RNA-Seq-Cancer-Subtype-Predictor/raw/main/Oncology_AI.apk) | Install on any Android mobile phone to open full-screen in native app format. |
| **Web Browser** | Cloud Web App | [🌐 **Open Web App**](https://oncologyaieng.streamlit.app/) | Access instantly in any browser on any device without downloading anything. |

---

## Live Demo

The platform is continuously deployed and accessible globally on Streamlit Community Cloud:
- **Web App URL**: [https://oncologyaieng.streamlit.app/](https://oncologyaieng.streamlit.app/)
- **Features Online**: Multi-patient upload, sample cohort cycling, instant SelectKBest + XGBoost classification, and SHAP feature attribution directly in your browser.

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
| **Research classification** | A six-class gradient-boosted model includes a normal reference class and five cancer cohorts. |
| **Personalized Targeted Therapy** | Identifies specific overexpressed driver genes (e.g., hormone receptors, tyrosine kinases) to help guide targeted inhibitor selection over generic toxic chemotherapy. |
| **Transparent Explainability (XAI)** | Instead of a black-box probability score, clinicians inspect a SHAP Beeswarm Plot highlighting individual positive and negative genomic drivers. |
| **Rapid Second Opinion** | Delivers instantaneous inference from standard RNA-Seq CSV outputs, reducing diagnostic turnaround time from days to seconds. |

---

## How the Engine Works

```
┌─────────────────────────────────┐
│ Patient RNA-Seq CSV (20,531)    │  Raw log2-normalized RNA-Seq expression counts
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ SelectKBest ANOVA F-Filtering   │  Reduces 20,531 loci → Top 500 discriminant biomarkers (97.6% reduction)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ XGBoost Gradient-Boosted Trees  │  Multi-class probability distribution across NORMAL and 5 cancer classes
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
3. **Ensemble Classification**: The trained `XGBClassifier` calculates posterior probabilities across NORMAL, BRCA, COAD, KIRC, LUAD, and PRAD.
4. **Shapley Attribution**: The `shap.TreeExplainer` decomposes the patient's individual gene values against the baseline expectation, calculating the exact positive or negative pull each gene exerted on the model's decision.
5. **UI Rendering**: The results update dynamically in the dark dashboard.

---

## Cancer / Normal Decision Flow

The current model is a six-class classifier:

1. The uploaded expression profile is converted to numeric values.
2. `SelectKBest` reduces the 20,531 input loci to the 500 features used by the model.
3. XGBoost produces probabilities for `NORMAL`, `BRCA`, `COAD`, `KIRC`, `LUAD`, and `PRAD`.
4. If the winning class is `NORMAL`, the **Predicted Subtype** card displays
   **NO CANCER DETECTED** and the normal probability row is shown first.
5. Otherwise, the card displays the winning supported cancer subtype and its
   posterior confidence.
6. SHAP explanations and the probability matrix are rendered only after the
   user clicks **Initialize AI Analysis**.

The `NORMAL` class is a normal-tissue reference class, not a universal
certificate that a person has no cancer. Results require independent clinical
validation.

---

## Project File Structure

```
Cancer Subtype Prediction via Gene Expression (RNA-Seq)/
│
├── Oncology_AI.exe             # 1-Click native Windows desktop launcher executable
├── app.py                      # Main Streamlit clinical dashboard application
├── train_model.py              # Training pipeline (downloads TCGA data, trains XGBoost, saves artifacts)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation and clinical guide
│
├── model.pkl                   # Trained XGBoost multi-class classification model
├── encoder.pkl                 # Scikit-Learn LabelEncoder (maps six classes including NORMAL)
├── selector.pkl                # Fitted SelectKBest feature selector (top 500 genes)
├── sample_patient.csv          # Benchmark patient sample (PRAD, 20,531 genes)
│
└── sampledata/                 # Real TCGA tumor and GTEx normal reference CSV samples
    ├── sample_data_normal_one.csv
    ├── sample_data_normal_two.csv
    └── sample_data_1.csv ... sample_data_20.csv  # Cancer rows plus GTEx normal rows
```

---

## Sample Patient Data (`sampledata/`)

The `sampledata/` directory contains real public expression samples from TCGA
primary tumors and GTEx non-diseased tissue. `sample_data_normal_one.csv` and
`sample_data_normal_two.csv` each contain 10 real GTEx normal samples. Each
`sample_data_1.csv` through `sample_data_20.csv` retains its existing cancer
rows and now includes two additional real GTEx normal rows for mixed-result
testing. GTEx and TCGA are de-identified public datasets. The normal rows use synthetic
display names and IDs (`PID-NORMAL-####`) so no public sample identifier is
shown in the UI; the gene-expression values remain unchanged.

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

## Testing with Local CSVs

The repository includes UI-ready public-expression test files:

| File | Purpose |
|---|---|
| `real_normal_patient_test.csv` | One GTEx normal-tissue expression profile with synthetic display metadata |
| `real_cancer_patient_test.csv` | One TCGA primary-tumor expression profile with synthetic display metadata |
| `sampledata/sample_data_normal_one.csv` | Ten real GTEx normal-tissue profiles |
| `sampledata/sample_data_normal_two.csv` | Ten real GTEx normal-tissue profiles |
| `sampledata/sample_data_1.csv` – `sample_data_20.csv` | Existing cancer rows plus two real normal rows per file |

Normal test rows use synthetic names and IDs such as `PID-NORMAL-0001` so
public sample identifiers are not displayed in the dashboard. Their
gene-expression values remain from the public normal-tissue data.

Each uploaded file must contain `gene_0` through `gene_20530` in that order.
Optional `Patient ID` and `Patient Name` columns may appear before the gene
columns. To test the UI:

```powershell
streamlit run app.py
```

Upload a CSV, click **Initialize AI Analysis**, and inspect the
**Predicted Subtype** card.

---

## Dataset Details

The current model is trained from the UCSC Xena Toil TCGA/GTEx compendium,
which applies one RNA-seq processing pipeline to both tumor and non-diseased
samples. It uses 400 matched normal-tissue GTEx samples and 80 primary-tumor
samples for each supported subtype. The six training classes are `NORMAL`,
`BRCA`, `COAD`, `KIRC`, `LUAD`, and `PRAD`. The training script selects the
20,531 highest-variance loci and then reduces them to 500 biomarkers.

The uploaded CSV must use the same `gene_0` through `gene_20530` feature order
produced by the current training pipeline. The older UCI-only CSVs remain
syntactically readable but are not guaranteed to be calibrated for this
retrained model.

- **Cancer source**: TCGA primary tumors from the UCSC Xena Toil compendium.
- **Non-cancer source**: GTEx non-diseased tissue from the same compendium.
- **Expression values**: Uniformly processed log-scale RSEM/TPM expression values.

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

### Quick Start (No Installation Needed)
Access the live production deployment directly in your browser:
👉 **[https://oncologyaieng.streamlit.app/](https://oncologyaieng.streamlit.app/)**

---

### Local Setup

#### Option A: 1-Click Windows Desktop Application (No Terminal Required)
Simply double-click **[`Oncology_AI.exe`](Oncology_AI.exe)**:
- Starts the local inference engine in the background.
- Opens the dashboard in a dedicated, distraction-free desktop application window (no browser tabs or URL bar).
- Runs 100% offline on your PC.

#### Option B: Run via Terminal / PowerShell
1. Clone or Open the Repository:
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
To download the public TCGA/GTEx compendium and retrain the six-class model:
```bash
python train_model.py
```
This script automatically generates `model.pkl`, `encoder.pkl`, `selector.pkl`,
and `sample_patient.csv`. The download is large and requires a stable internet
connection.

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
   - **Cancer / Normal Probability Matrix**: Ranked probability distribution across NORMAL and all five cancer classes.
5. **Model Interpretability (Bottom Row)**:
   - **Unified SHAP Card**: Game-theoretic Shapley beeswarm plot indicating individual gene impacts on the classification, paired with an interactive table of the **Top 5 Positive Genomic Driver Genes**.

---

## Medical Disclaimer

This software is developed strictly for **computational biology research, bioinformatics benchmarking, and educational purposes**. It is not a certified diagnostic medical device and should not be used as the sole basis for clinical diagnosis, patient staging, or surgical decisions without independent pathology confirmation and oncologist review.
