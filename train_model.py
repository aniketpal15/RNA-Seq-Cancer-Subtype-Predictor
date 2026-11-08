"""
Train the cancer-versus-normal RNA-seq classifier.

The training data comes from the UCSC Xena Toil TCGA/GTEx compendium, where
tumor and non-diseased samples were processed through the same RNA-seq
pipeline. The resulting six-class model contains NORMAL plus five cancer
subtypes and expects the exported gene_0 ... gene_20530 feature format.
"""

import gzip
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


EXPRESSION_URL = (
    "https://toil-xena-hub.s3.dualstack.us-east-1.amazonaws.com/download/"
    "TcgaTargetGtex_rsem_gene_tpm.gz"
)
PHENOTYPE_URL = (
    "https://toil-xena-hub.s3.dualstack.us-east-1.amazonaws.com/download/"
    "TcgaTargetGTEX_phenotype.txt.gz"
)
EXPRESSION_FILE = Path("xena_expression.gz")
PHENOTYPE_FILE = Path("xena_phenotype.txt.gz")
RANDOM_STATE = 42
SAMPLES_PER_CLASS = 80
FEATURE_COUNT = 20531
SELECTED_FEATURE_COUNT = 500


def download_if_missing(url: str, destination: Path) -> None:
    if destination.exists():
        return
    import urllib.request

    print(f"Downloading {destination}...")
    urllib.request.urlretrieve(url, destination)


def choose_samples(phenotype: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return balanced sample IDs and labels from matched tissue groups."""
    cancer_rules = {
        "BRCA": ("Breast Invasive Carcinoma", "Primary Tumor"),
        "COAD": ("Colon Adenocarcinoma", "Primary Tumor"),
        "KIRC": ("Kidney Clear Cell Carcinoma", "Primary Tumor"),
        "LUAD": ("Lung Adenocarcinoma", "Primary Tumor"),
        "PRAD": ("Prostate Adenocarcinoma", "Primary Tumor"),
    }
    normal_rules = {
        "Breast - Mammary Tissue",
        "Colon - Transverse",
        "Colon - Sigmoid",
        "Kidney - Cortex",
        "Lung",
        "Prostate",
    }

    rng = np.random.default_rng(RANDOM_STATE)
    sample_ids: list[str] = []
    labels: list[str] = []

    for label, (category, sample_type) in cancer_rules.items():
        matches = phenotype[
            (phenotype["detailed_category"] == category)
            & (phenotype["_sample_type"] == sample_type)
            & (phenotype["_study"] == "TCGA")
        ]["sample"].dropna().tolist()
        selected = rng.choice(matches, size=min(SAMPLES_PER_CLASS, len(matches)), replace=False)
        sample_ids.extend(selected.tolist())
        labels.extend([label] * len(selected))

    normal_matches = phenotype[
        phenotype["detailed_category"].isin(normal_rules)
        & (phenotype["_sample_type"] == "Normal Tissue")
        & (phenotype["_study"] == "GTEX")
    ]["sample"].dropna().tolist()
    selected = rng.choice(
        normal_matches,
        size=min(SAMPLES_PER_CLASS * len(cancer_rules), len(normal_matches)),
        replace=False,
    )
    sample_ids.extend(selected.tolist())
    labels.extend(["NORMAL"] * len(selected))

    if len(set(labels)) != 6:
        raise RuntimeError(f"Could not assemble all six classes: {sorted(set(labels))}")
    return sample_ids, labels


def load_expression(sample_ids: list[str]) -> pd.DataFrame:
    """Load selected samples and retain the highest-variance 20,531 genes."""
    wanted = set(sample_ids)
    with gzip.open(EXPRESSION_FILE, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    selected_columns = [column for column in header if column in wanted]
    if len(selected_columns) != len(sample_ids):
        missing = sorted(wanted - set(selected_columns))
        raise RuntimeError(f"Expression matrix is missing {len(missing)} selected samples.")

    matrix = pd.read_csv(
        EXPRESSION_FILE,
        sep="\t",
        compression="gzip",
        usecols=["sample", *selected_columns],
        index_col=0,
    )
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(-9.9658)
    matrix = matrix.loc[~matrix.index.duplicated(keep="first")]
    variances = matrix.var(axis=1)
    if len(matrix) < FEATURE_COUNT:
        raise RuntimeError(f"Expected at least {FEATURE_COUNT} genes, found {len(matrix)}.")
    keep_genes = variances.nlargest(FEATURE_COUNT).index
    samples_first = matrix.loc[keep_genes, sample_ids].T
    samples_first.columns = [f"gene_{i}" for i in range(FEATURE_COUNT)]
    return samples_first


def main() -> None:
    download_if_missing(EXPRESSION_URL, EXPRESSION_FILE)
    download_if_missing(PHENOTYPE_URL, PHENOTYPE_FILE)
    phenotype = pd.read_csv(PHENOTYPE_FILE, sep="\t", encoding="latin1")
    sample_ids, labels = choose_samples(phenotype)
    X = load_expression(sample_ids)
    y = pd.Series(labels, name="Class")

    print(f"Training samples: {X.shape[0]} x {X.shape[1]}")
    print(f"Class counts: {y.value_counts().to_dict()}")
    X.to_csv("training_feature_template.csv", index=False)
    X.iloc[[0]].to_csv("sample_patient.csv", index=False)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    selector = SelectKBest(score_func=f_classif, k=SELECTED_FEATURE_COUNT)
    X_selected = selector.fit_transform(X, y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X_selected,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        n_estimators=180,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Validation accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_))
    model.fit(X_selected, y_encoded)

    joblib.dump(model, "model.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(selector, "selector.pkl")
    print(f"Saved classes: {list(encoder.classes_)}")


if __name__ == "__main__":
    main()
