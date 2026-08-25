import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, 
    confusion_matrix, recall_score, precision_score
)

# Suppress sklearn deprecation warnings for clean execution
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

print("\n" + "="*85)
print(">>> EXECUTING FULLY ALIGNED EXTERNAL VALIDATION (OPENNEURO V3) <<<")
print("="*85 + "\n")

# 1. Directory Paths
PPMI_DATA_DIR = r"D:\UoS\1DISSERTATION\PD\processed_data\extracted_features"
OPENNEURO_DATA_DIR = r"D:\UoS\1DISSERTATION\PD\processed_data\openneuro_extracted_features"
RESULTS_DIR = r"D:\UoS\1DISSERTATION\PD\results\figures\openneuro_eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2. Load Datasets
print("Loading PPMI train, val, and OpenNeuro test datasets...")
train_df = pd.read_csv(os.path.join(PPMI_DATA_DIR, "train_rf_features.csv"))
val_df = pd.read_csv(os.path.join(PPMI_DATA_DIR, "val_rf_features.csv"))
test_df = pd.read_csv(os.path.join(OPENNEURO_DATA_DIR, "openneuro_rf_features.csv"))

target_col = "PD_label"

# Standardise column naming differences between PPMI and OpenNeuro
rename_dict = {"education_yrs": "education_years"}
test_df = test_df.rename(columns=rename_dict)

# 3. Columns & Feature Triage (EXACT MATCH TO INTERNAL PPMI SCRIPT)
leakage_cols = [
    "IDNUM", "updrs_motor_score", "tremor_score", "rigidity_score",
    "bradykinesia_score", "gait_instability_score", "disease_duration_yrs"
]

non_imaging_cols = [
    "age", "sex", "education_years", "family_history_pd",
    "moca_score", "gds_total", "verbal_fluency"
]

# Extract full internal training feature set
all_ppmi_features = [c for c in train_df.columns if c not in leakage_cols and c != target_col]

# Define Exact 3 Feature Paradigms
feature_sets = {
    "Multimodal Full": all_ppmi_features,
    "Multimodal No MoCA": [c for c in all_ppmi_features if c != "moca_score"],
    "Imaging Only": [c for c in all_ppmi_features if c not in non_imaging_cols]
}

# 4. Cross-Validation & Hyperparameter Grids (Matching Internal PPMI)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_param_grid = {
    'n_estimators': [200],
    'max_depth': [3, 5],
    'min_samples_leaf': [4, 6, 8],
    'max_features': ['sqrt']
}

lr_param_grid = {
    'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    'solver': ['liblinear', 'lbfgs']
}

# Helper Function: Tune Decision Threshold on PPMI Validation Set
def find_optimal_threshold(model, X_val_data, y_val_data):
    val_probs = model.predict_proba(X_val_data)[:, 1]
    thresholds = np.linspace(0.20, 0.80, 61)
    best_thresh = 0.50
    best_score = -1.0
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = balanced_accuracy_score(y_val_data, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh

# Storage Containers
results_summary = {}
confusion_matrices = {}

# 5. Core Model Running Logic
def run_experiment(exp_label, model_type, feature_cols, plot_filename=None):
    print(f"\n{'='*65}")
    print(f"Running OpenNeuro Experiment: {exp_label} ({model_type.upper()})")
    print(f"{'='*65}")
    
    # 1. Isolate Features
    X_tr = train_df[feature_cols].copy()
    y_tr = train_df[target_col].copy()
    
    X_v = val_df[feature_cols].copy()
    y_v = val_df[target_col].copy()
    
    # Safely reindex OpenNeuro test set to match internal feature columns exactly
    X_te = test_df.reindex(columns=feature_cols).copy()
    y_te = test_df[target_col].copy()
    
    print(f"Feature count: {len(feature_cols)}")
    print(f"Train rows: {len(X_tr)} | Val rows: {len(X_v)} | Test (OpenNeuro) rows: {len(X_te)}")
    
    # 2. Imputation (Fitted on PPMI Train)
    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_tr)
    X_v_imp = imputer.transform(X_v)
    X_te_imp = imputer.transform(X_te)
    
    # 3. Within-Cohort Z-Score Harmonisation
    scaler_ppmi = StandardScaler()
    X_tr_proc = scaler_ppmi.fit_transform(X_tr_imp)
    X_v_proc = scaler_ppmi.transform(X_v_imp)
    
    scaler_openneuro = StandardScaler()
    X_te_proc = scaler_openneuro.fit_transform(X_te_imp)
    
    # 4. Fit GridSearchCV on PPMI Training Set
    if model_type == 'lr':
        base_lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        grid_search = GridSearchCV(
            estimator=base_lr, param_grid=lr_param_grid,
            cv=cv_strategy, scoring='accuracy', n_jobs=-1
        )
    else:
        base_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            estimator=base_rf, param_grid=rf_param_grid,
            cv=cv_strategy, scoring='accuracy', n_jobs=-1
        )
        
    grid_search.fit(X_tr_proc, y_tr)
    best_model = grid_search.best_estimator_
    cv_acc = grid_search.best_score_ * 100
    
    # 5. Tune Threshold on PPMI Validation Set
    threshold = find_optimal_threshold(best_model, X_v_proc, y_v)
    val_probs = best_model.predict_proba(X_v_proc)[:, 1]
    y_val_pred = (val_probs >= threshold).astype(int)
    val_acc = accuracy_score(y_v, y_val_pred) * 100
    
    # 6. Evaluate on OpenNeuro External Test Set
    test_probs = best_model.predict_proba(X_te_proc)[:, 1]
    y_test_pred = (test_probs >= threshold).astype(int)
    
    test_acc = accuracy_score(y_te, y_test_pred) * 100
    test_bal_acc = balanced_accuracy_score(y_te, y_test_pred) * 100
    healthy_recall = recall_score(y_te, y_test_pred, pos_label=0) * 100
    pd_recall = recall_score(y_te, y_test_pred, pos_label=1) * 100
    gap = val_acc - test_acc
    
    # Terminal Logging
    print(f"Optimal Parameters        : {grid_search.best_params_}")
    print(f"Applied Decision Threshold : {threshold:.2f}")
    print(f"PPMI 5-Fold CV Accuracy    : {cv_acc:.2f}%")
    print(f"PPMI Validation Accuracy   : {val_acc:.2f}%")
    print(f"OpenNeuro Test Accuracy    : {test_acc:.2f}%")
    print(f"Balanced Test Accuracy     : {test_bal_acc:.2f}%")
    print(f"Healthy Recall (Class 0)   : {healthy_recall:.1f}%")
    print(f"PD Recall (Class 1)        : {pd_recall:.1f}%")
    print(f"Val - Test Gap             : {gap:.2f}pp")
    
    print("\nClassification Report (OpenNeuro External Test Set):")
    print(classification_report(y_te, y_test_pred, target_names=['Healthy (0)', 'PD (1)']))
    
    # Store Summary Table Record
    results_summary[exp_label] = {
        "Model": model_type.upper(),
        "Threshold": f"{threshold:.2f}",
        "CV Acc": f"{cv_acc:.2f}%",
        "Val Acc": f"{val_acc:.2f}%",
        "Test Acc": f"{test_acc:.2f}%",
        "Balanced Acc": f"{test_bal_acc:.2f}%",
        "Healthy Recall": f"{healthy_recall:.1f}%",
        "PD Recall": f"{pd_recall:.1f}%",
        "Gap": f"{gap:.2f}pp"
    }
    
    # Store Confusion Matrix for Grid Figure
    cm = confusion_matrix(y_te, y_test_pred)
    confusion_matrices[exp_label] = (cm, test_acc, test_bal_acc)
    
    # 7. Generate Top-10 Feature Importance Plot (EXCLUDING "No MoCA")
    if plot_filename and "No MoCA" not in exp_label:
        plt.figure(figsize=(9, 5))
        if model_type == 'rf':
            importances = best_model.feature_importances_
            feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(10)
            sns.barplot(x=feat_series.values, y=feat_series.index, palette="Blues_r")
            plt.title(f"Top 10 Biomarkers (RF Gini Importance): {exp_label}", fontsize=12, fontweight='bold')
            plt.xlabel("Gini Importance Score")
        else:
            coefs = best_model.coef_[0]
            feat_series = pd.Series(coefs, index=feature_cols).abs().sort_values(ascending=False).head(10)
            sns.barplot(x=feat_series.values, y=feat_series.index, palette="Blues_r")
            plt.title(f"Top 10 Feature Weights (|Coef|): {exp_label}", fontsize=12, fontweight='bold')
            plt.xlabel("Absolute Standardised Weight")
            
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"{plot_filename}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[SAVED] Feature Plot -> {plot_path}")

# ─────────────────────────────────────────────
# 6. EXECUTE THE 6 EXACT EXPERIMENTS
# ─────────────────────────────────────────────

# 1. Multimodal Full (RF)
run_experiment(
    "Multimodal Full (RF)", "rf",
    feature_sets["Multimodal Full"], "rf_multimodal_full_top10"
)

# 2. Multimodal Full (LR)
run_experiment(
    "Multimodal Full (LR)", "lr",
    feature_sets["Multimodal Full"], "lr_multimodal_full_top10"
)

# 3. Multimodal No MoCA (RF) - Plot Skipped
run_experiment(
    "Multimodal No MoCA (RF)", "rf",
    feature_sets["Multimodal No MoCA"], plot_filename=None
)

# 4. Multimodal No MoCA (LR) - Plot Skipped
run_experiment(
    "Multimodal No MoCA (LR)", "lr",
    feature_sets["Multimodal No MoCA"], plot_filename=None
)

# 5. Imaging Only (RF)
run_experiment(
    "Imaging Only (RF)", "rf",
    feature_sets["Imaging Only"], "rf_imaging_only_top10"
)

# 6. Imaging Only (LR)
run_experiment(
    "Imaging Only (LR)", "lr",
    feature_sets["Imaging Only"], "lr_imaging_only_top10"
)

# ─────────────────────────────────────────────
# 7. CONSOLIDATED 2x3 CONFUSION MATRIX GRID
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

exp_labels = list(confusion_matrices.keys())

for idx, label in enumerate(exp_labels):
    cm, acc, bal_acc = confusion_matrices[label]
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Healthy (0)', 'PD (1)'],
                yticklabels=['Healthy (0)', 'PD (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"{label}\nExternal Test Acc: {acc:.2f}% | Bal Acc: {bal_acc:.2f}%", fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Diagnosis', fontsize=10)
    ax.set_xlabel('Predicted Diagnosis', fontsize=10)

plt.suptitle("Figure 4.x: OpenNeuro External Validation – Confusion Matrices Across Feature Paradigms", 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

grid_fig_path = os.path.join(RESULTS_DIR, "combined_openneuro_confusion_matrices.png")
plt.savefig(grid_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[SAVED] Consolidated Confusion Matrix Grid -> {grid_fig_path}")

# ─────────────────────────────────────────────
# 8. PRINT FINAL BENCHMARK SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 105)
print("FINAL OPENNEURO EXTERNAL VALIDATION BENCHMARK COMPARISON TABLE")
print("=" * 105)
summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
print(summary_df.to_string())