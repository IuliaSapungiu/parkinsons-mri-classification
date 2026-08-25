import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, 
    confusion_matrix, recall_score
)

print("=== STARTING PPMI INTERNAL BENCHMARK: LOGISTIC REGRESSION VS RANDOM FOREST ===")

# 1. Define Local Paths
DATA_DIR = r"D:\UoS\1DISSERTATION\PD\processed_data\extracted_features"
RESULTS_DIR = r"D:\UoS\1DISSERTATION\PD\results\figures\internal_ppmi_eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train_rf_features.csv")
val_path = os.path.join(DATA_DIR, "val_rf_features.csv")
test_path = os.path.join(DATA_DIR, "test_rf_features.csv")

# 2. Load the Datasets
print("Loading PPMI train, val, and test datasets...")
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

target_col = "PD_label"

# Feature Triage
leakage_cols = [
    "IDNUM", "updrs_motor_score", "tremor_score", "rigidity_score",
    "bradykinesia_score", "gait_instability_score", "disease_duration_yrs"
]

non_imaging_cols = [
    "age", "sex", "education_years", "family_history_pd",
    "moca_score", "gds_total", "verbal_fluency"
]

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

results_summary = {}
confusion_matrices = {}
model_objects = {}

def find_optimal_threshold(model, X_v, y_v):
    val_probs = model.predict_proba(X_v)[:, 1]
    thresholds = np.linspace(0.20, 0.80, 61)
    best_thresh = 0.50
    best_score = -1.0
    
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = balanced_accuracy_score(y_v, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
            
    return best_thresh

def run_ppmi_experiment(exp_name, model_type, drop_cols):
    print(f"\n{'=' * 60}")
    print(f"Running PPMI Experiment: {exp_name} ({model_type.upper()})")
    print(f"{'=' * 60}")

    X_train = train_df.drop(columns=[target_col] + leakage_cols + drop_cols, errors='ignore')
    y_train = train_df[target_col]

    X_val = val_df.drop(columns=[target_col] + leakage_cols + drop_cols, errors='ignore')
    y_val = val_df[target_col]

    X_test = test_df.drop(columns=[target_col] + leakage_cols + drop_cols, errors='ignore')
    y_test = test_df[target_col]

    feature_names = X_train.columns.tolist()
    print(f"Feature count: {len(feature_names)}")
    print(f"Train rows: {len(X_train)} | Val rows: {len(X_val)} | Test rows: {len(X_test)}")

    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_train)
    X_v_imp = imputer.transform(X_val)
    X_te_imp = imputer.transform(X_test)

    if model_type == 'lr':
        scaler = StandardScaler()
        X_tr_proc = scaler.fit_transform(X_tr_imp)
        X_v_proc = scaler.transform(X_v_imp)
        X_te_proc = scaler.transform(X_te_imp)

        base_lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        grid_search = GridSearchCV(
            estimator=base_lr, param_grid=lr_param_grid,
            cv=cv_strategy, scoring='accuracy', n_jobs=-1
        )
    else:
        X_tr_proc, X_v_proc, X_te_proc = X_tr_imp, X_v_imp, X_te_imp
        base_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            estimator=base_rf, param_grid=rf_param_grid,
            cv=cv_strategy, scoring='accuracy', n_jobs=-1
        )

    grid_search.fit(X_tr_proc, y_train)
    best_model = grid_search.best_estimator_
    cv_acc = grid_search.best_score_ * 100

    best_thresh = find_optimal_threshold(best_model, X_v_proc, y_val)

    val_probs = best_model.predict_proba(X_v_proc)[:, 1]
    test_probs = best_model.predict_proba(X_te_proc)[:, 1]

    y_val_pred = (val_probs >= best_thresh).astype(int)
    y_test_pred = (test_probs >= best_thresh).astype(int)

    val_acc = accuracy_score(y_val, y_val_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred) * 100
    healthy_recall = recall_score(y_test, y_test_pred, pos_label=0) * 100
    pd_recall = recall_score(y_test, y_test_pred, pos_label=1) * 100
    gap = cv_acc - test_acc

    # Print detailed output to terminal
    print(f"Optimal Parameters         : {grid_search.best_params_}")
    print(f"Applied Decision Threshold : {best_thresh:.2f}")
    print(f"5-Fold CV Accuracy (Train) : {cv_acc:.2f}%")
    print(f"Validation Accuracy        : {val_acc:.2f}%")
    print(f"Internal Test Accuracy     : {test_acc:.2f}%")
    print(f"Balanced Test Accuracy     : {test_bal_acc:.2f}%")
    print(f"Healthy Recall (Class 0)   : {healthy_recall:.1f}%")
    print(f"PD Recall (Class 1)        : {pd_recall:.1f}%")
    print(f"CV - Test Gap              : {gap:.2f}pp")

    print(f"\nClassification Report (Internal PPMI Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Healthy (0)', 'PD (1)']))

    full_name = f"{exp_name} ({model_type.upper()})"
    results_summary[full_name] = {
        "Model": model_type.upper(),
        "Threshold": f"{best_thresh:.2f}",
        "CV Acc": f"{cv_acc:.2f}%",
        "Val Acc": f"{val_acc:.2f}%",
        "Test Acc": f"{test_acc:.2f}%",
        "Balanced Acc": f"{test_bal_acc:.2f}%",
        "Healthy Recall": f"{healthy_recall:.1f}%",
        "PD Recall": f"{pd_recall:.1f}%",
        "Gap": f"{gap:.2f}pp"
    }

    confusion_matrices[full_name] = confusion_matrix(y_test, y_test_pred)
    model_objects[full_name] = (best_model, feature_names)

# Execute Experiments
experiments = [
    ("Multimodal Full", "rf", []),
    ("Multimodal Full", "lr", []),
    ("Multimodal No MoCA", "rf", ["moca_score"]),
    ("Multimodal No MoCA", "lr", ["moca_score"]),
    ("Imaging Only", "rf", non_imaging_cols),
    ("Imaging Only", "lr", non_imaging_cols)
]

for name, mtype, drops in experiments:
    run_ppmi_experiment(name, mtype, drops)

# ─────────────────────────────────────────────
# 1. CONSOLIDATED 2x3 CONFUSION MATRIX GRID
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
exp_keys = list(confusion_matrices.keys())

for idx, key in enumerate(exp_keys):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    cm = confusion_matrices[key]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Healthy (0)', 'PD (1)'],
                yticklabels=['Healthy (0)', 'PD (1)'])
    ax.set_title(key, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Diagnosis')
    ax.set_xlabel('Predicted Diagnosis')

plt.tight_layout()
grid_cm_path = os.path.join(RESULTS_DIR, "combined_ppmi_confusion_matrices.png")
plt.savefig(grid_cm_path, dpi=300)
plt.close()
print(f"\n[SAVED] Consolidated Confusion Matrix Grid -> {grid_cm_path}")

# ─────────────────────────────────────────────
# 2. HELPER FUNCTION TO PLOT FEATURE IMPORTANCE / COEFFICIENTS
# ─────────────────────────────────────────────
def plot_feature_importance(model_key, filename_prefix):
    model, features = model_objects[model_key]
    
    plt.figure(figsize=(10, 6))
    if "(RF)" in model_key:
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(10)
        df_plot = pd.DataFrame({'Feature': importances.index, 'Importance': importances.values})
        sns.barplot(data=df_plot, x='Importance', y='Feature', hue='Feature', palette="viridis", legend=False)
        plt.title(f"Top 10 Biomarkers - {model_key}", fontsize=14, fontweight='bold')
        plt.xlabel("Gini Importance Score", fontsize=12)
    else:
        coefs = pd.Series(model.coef_[0], index=features)
        top_coef_keys = coefs.abs().sort_values(ascending=False).head(10).index
        top_signed = coefs.loc[top_coef_keys].sort_values()
        
        df_plot = pd.DataFrame({'Feature': top_signed.index, 'Coefficient': top_signed.values})
        df_plot['Direction'] = df_plot['Coefficient'].apply(lambda x: 'Risk Factor (+)' if x > 0 else 'Protective (-)')
        
        sns.barplot(data=df_plot, x='Coefficient', y='Feature', hue='Direction', palette={'Risk Factor (+)': 'navy', 'Protective (-)': 'crimson'})
        plt.title(f"Top 10 Feature Coefficients - {model_key}", fontsize=14, fontweight='bold')
        plt.xlabel("Standardized Coefficient Weight (Navy = Risk (+), Red = Protective (-))", fontsize=11)
        
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{filename_prefix}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Feature Plot -> {out_path}")

# Generate Plots for both Multimodal Full & Imaging Only
plot_feature_importance("Multimodal Full (RF)", "rf_multimodal_full_top10")
plot_feature_importance("Multimodal Full (LR)", "lr_multimodal_full_top10")
plot_feature_importance("Imaging Only (RF)", "rf_imaging_only_top10")
plot_feature_importance("Imaging Only (LR)", "lr_imaging_only_top10")

# ─────────────────────────────────────────────
# FINAL BENCHMARK SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 90)
print("FINAL PPMI INTERNAL BENCHMARK COMPARISON TABLE")
print("=" * 90)
summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
print(summary_df.to_string())