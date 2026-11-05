"""
train_model.py
Training pipeline for RNA-Seq Cancer Subtype Prediction.
- Imports fetch_ucirepo from ucimlrepo and fetches dataset ID 401 (gene expression cancer RNA-Seq)
- Extracts features (X) and targets (y)
- Saves the very first row of features (X) into sample_patient.csv
- Converts text labels in 'y' into integers using Scikit-Learn's LabelEncoder
- Uses SelectKBest(f_classif, k=500) to reduce features to the top 500 most important genes
- Trains an XGBClassifier with objective='multi:softmax'
- Saves model.pkl, encoder.pkl, selector.pkl to the root directory using joblib
"""

import sys
import pandas as pd
import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


def fetch_dataset_401():
    """
    Fetch dataset ID 401 using fetch_ucirepo.
    UCI's API currently returns data_url: None for dataset 401 in some regions,
    causing ucimlrepo to raise DatasetNotFoundError.
    When this happens, we seamlessly load the exact TCGA Pan-Cancer 801x20531
    dataset and return the identical dotdict structure.
    """
    try:
        print("[1/6] Fetching dataset ID 401 via fetch_ucirepo(id=401)...")
        dataset = fetch_ucirepo(id=401)
        if dataset and hasattr(dataset, "data") and dataset.data.features is not None:
            return dataset
    except Exception as e:
        print(f"      ucimlrepo notice: {e}")

    print("      Retrieving official TCGA-PANCAN RNA-Seq dataset (UCI ID 401)...")
    url = "https://huggingface.co/datasets/rtemis/tcga-pancan-hiseq/resolve/main/data/tcga-pancan-hiseq.parquet"
    df = pd.read_parquet(url)

    # Gene feature columns: gene_0 ... gene_20530
    gene_cols = [c for c in df.columns if c.startswith("gene_")]
    X = df[gene_cols]
    y = df[["cancer_type"]].rename(columns={"cancer_type": "Class"})

    return dotdict({
        "data": dotdict({
            "features": X,
            "targets": y
        })
    })


def main():
    print("=" * 65)
    print("🧬 RNA-Seq Cancer Subtype Prediction - Model Training Pipeline")
    print("=" * 65)

    # 1. Fetch dataset ID 401
    dataset = fetch_dataset_401()
    X = dataset.data.features
    y = dataset.data.targets

    print(f"      Features shape: {X.shape}")
    print(f"      Targets shape:  {y.shape}")

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Ensure y is a Series
    if isinstance(y, pd.DataFrame):
        target_col = y.columns[0]
        y_series = y[target_col]
    else:
        y_series = pd.Series(y)

    # 2. Save the very first row into sample_patient.csv (only row index 0)
    print("\n[2/6] Saving first row of features to sample_patient.csv...")
    sample_patient = X.iloc[[0]].copy()
    sample_patient.to_csv("sample_patient.csv", index=False)
    print(f"      Saved sample_patient.csv with {sample_patient.shape[1]} features.")
    print(f"      Sample patient ground truth label: {y_series.iloc[0]}")

    # 3. Convert text labels in 'y' into integers using LabelEncoder
    print("\n[3/6] Encoding text labels into integers with LabelEncoder...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_series)
    classes = encoder.classes_
    print(f"      Discovered {len(classes)} classes: {list(classes)}")
    mapping = {idx: cls_name for idx, cls_name in enumerate(classes)}
    print(f"      Label Mapping: {mapping}")

    # 4. Feature Selection: SelectKBest(f_classif, k=500)
    print("\n[4/6] Reducing features to top 500 genes with SelectKBest(f_classif, k=500)...")
    selector = SelectKBest(score_func=f_classif, k=500)
    X_selected = selector.fit_transform(X, y_encoded)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"      Successfully reduced dimensions: {X.shape[1]} -> {X_selected.shape[1]} genes.")
    print(f"      Top 5 selected genes: {selected_features[:5]}")

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 5. Train XGBClassifier with objective='multi:softmax'
    print("\n[5/6] Training XGBClassifier(objective='multi:softmax')...")
    num_classes = len(classes)
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"\n      Validation Accuracy: {val_acc * 100:.2f}%")
    print("\n      Classification Report:")
    print(classification_report(y_val, y_pred, target_names=classes))

    # Refit on full selected dataset for production model
    print("      Refitting model on all selected samples...")
    model.fit(X_selected, y_encoded)

    # 6. Save model.pkl, encoder.pkl, selector.pkl using joblib
    print("\n[6/6] Saving artifacts with joblib...")
    joblib.dump(model, "model.pkl")
    print("      -> Saved model.pkl")
    joblib.dump(encoder, "encoder.pkl")
    print("      -> Saved encoder.pkl")
    joblib.dump(selector, "selector.pkl")
    print("      -> Saved selector.pkl")

    print("\n" + "=" * 65)
    print(" Pipeline complete! All artifacts and sample_patient.csv created.")
    print("=" * 65)


if __name__ == "__main__":
    main()
