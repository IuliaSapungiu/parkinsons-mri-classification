# Multimodal Neuroimaging Classification for Parkinson's Disease

This repository contains the complete end-to-end Machine Learning (ML) and Deep Learning (DL) neuroimaging pipeline for classifying Parkinson's Disease (PD) versus Healthy Controls (HC). 
The study evaluates a multimodal approach combining T1-weighted structural MRI scans and harmonized clinical features using **3D Convolutional Neural Networks (3D CNNs)**, **Random Forest (RF)**, and **Logistic Regression (LR)** models.

* **Internal Cohort:** Parkinson's Progression Markers Initiative (PPMI)
* **External Validation Cohort:** OpenNeuro Dataset

---

## 📋 Table of Contents

## Quick Start and Installation
1. Clone the repository
  ```bash
  git clone [https://github.com/IuliaSapungiu/parkinsons-mri-classification.git](https://github.com/IuliaSapungiu/parkinsons-mri-classification.git)
  cd parkinsons-mri-classification
  ```

2. Set up Virtual Environment
  ```bash
  # Create environment
  python -m venv myenv

  # Activate on Windows (Git Bash):
  source myenv/Scripts/activate

  # Activate on Linux/macOS:
  source myenv/bin/activate
  ```
3. Install dependencies
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## 📁 Repository Structure

```text
PD/
├── openneuro/                                ← External validation raw dataset scans
├── raw/                                      ← Raw clinical and acquisition metadata
│   ├── Code_List_Harmonized_09June2026.csv
│   ├── Data_Dictionary_Harmonized_09June2026.csv
│   ├── demographics.csv                      ← OpenNeuro clinical metadata
│   ├── MDS-UPDRS_Part_III_09Jun2026.csv      ← PPMI clinical metadata
│   ├── PPMI_Baseline_T1_MRI.csv              ← PPMI scan acquisition metadata
│   └── PPMI_Curated_Data_Cut_Public_20251112.xlsx ← PPMI curated clinical dataset
├── processed_data/                           ← Preprocessed features, splits & logs
│   ├── extracted_features/                   ← Radiomic feature matrices (PPMI internal)
│   ├── openneuro_extracted_features/         ← Radiomic feature matrices (OpenNeuro external)
│   ├── HC/                                   ← Raw PPMI DICOM directories
│   ├── PD/                                   ← Raw PPMI DICOM directories
│   ├── split_data/                           ← Stratified PPMI dataset split definitions
│   │   ├── train_clinical.csv
│   │   ├── val_clinical.csv
│   │   └── test_clinical.csv
│   ├── skull_stripped/                       ← Brain-extracted scans & logs (PPMI)
│   ├── openneuro_skull_stripped/             ← Brain-extracted scans & logs (OpenNeuro)
│   ├── registration/                         ← MNI152 registered scans (PPMI)
│   ├── openneuro_registered/                 ← MNI152 registered scans (OpenNeuro)
│   ├── nifti/                                ← Initial NIfTI conversions
│   ├── openneuro_clinical_variables.csv      ← Cleaned OpenNeuro clinical features
│   ├── ppmi_harmonized_clinical_variables.csv← Processed PPMI clinical features
│   ├── pd_demographics_clean.csv             ← Cleaned demographic dataset
│   ├── selected_subjects.txt                 ← Subject selection ID list
│   └── registration_example.png              ← Preprocessing visual QA sample
├── scripts/                                  ← Sequential pipeline scripts & split metadata
│   ├── 00_extract_openneuro.py
│   ├── 00b_preprocessing.py
│   ├── 00b_verify_openneuro.py
│   ├── 01_PPMI_extraction.py
│   ├── 01b_PPMI_clinical_data_prep.py
│   ├── 02_PPMI_MRI_transform.py
│   ├── 03_skull_strip.py
│   ├── 04_registration.py
│   ├── 05_splitting.py
│   ├── 06a_extract_features.py
│   ├── 06b_preprocessing.py
│   ├── 07_train_ml_ppmi.py
│   ├── 08_train_3d_cnn.py
│   ├── 09_openneuro_skull_strip.py
│   ├── 10_openneuro_registration.py
│   ├── 11_openneuro_extract_features.py
│   ├── 12_openneuro_ml.py
│   ├── 13_openneuro_3dcnn.py
│   ├── selected_subjects_public.csv
│   ├── train_subjects.pkl
│   └── test_subjects.pkl
├── results/                                  ← Model evaluation plots & figures
│   └── figures/
├── .gitignore                                ← Git tracking rules
├── requirements.txt                          ← Python dependencies
└── README.md                                 ← Repository documentation
```

## ⚖️ Data Governance & Compliance
To strictly comply with the Parkinson's Progression Markers Initiative (PPMI) Data Usage Agreement and GitHub file size constraints:
* **Excluded from repository:**
  * Raw DICOM scan directories (`HC/`, `PD/`, `openneuro/`)
  * Intermediate NIfTI 3D image volumes (`nifti/`, `skull_stripped/`, `registration/`)
  * Heavy binary model checkpoints and weights (`models/`)
* **Included in repository:**
  * Complete Python source code and execution scripts (`scripts/`)
  * Preprocessing configurations and environment definition (`requirements.txt`)
  * Anonymized tabular radiomic feature matrices (`processed_data/extracted_features/`, `processed_data/openneuro_extracted_features/`)
  * Subject cross-validation split files (`.pkl`, `.csv`)
  * Quantitative evaluation plots, metric curves, and confusion matrices (`results/figures/`)

> **Data Access Note:** Access to raw MRI scans and primary clinical datasets must be obtained directly through the official portals for [PPMI](https://www.ppmi-info.org/) and [OpenNeuro](https://openneuro.org/).



