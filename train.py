import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "test.csv"
ASSETS_PATH = ROOT / "assets"
LABELS = ["pas_interessant", "notifiable", "interessant"]


def load_data(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        rows = [
            row
            for row in csv.DictReader(file, delimiter=";")
            if (row.get("label") or "").strip()
        ]

    y = [row["label"].strip() for row in rows]
    unexpected = set(y) - set(LABELS)
    if unexpected:
        raise ValueError(f"Labels inconnus dans le CSV : {unexpected}")

    X = [
        f"De: {row['de']} Sujet: {row['sujet']} Corps: {row['corps_texte']}"
        for row in rows
    ]
    return X, y


def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50_000),
        LogisticRegression(max_iter=1000),
    )


def save_plots(y_true, predictions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=LABELS,
        display_labels=LABELS,
        normalize="true",
        values_format=".0%",
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Prédictions par rapport aux vrais labels")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=LABELS, zero_division=0
    )
    x = np.arange(len(LABELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label="Précision", color="#0072B2")
    ax.bar(x, recall, width, label="Rappel", color="#E69F00")
    ax.bar(x + width, f1, width, label="F1", color="#009E73")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (0 à 1)")
    ax.set_title("Qualité du modèle par classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_metrics.png", dpi=160)
    plt.close(fig)

    actual_counts = [sum(label == class_name for label in y_true) for class_name in LABELS]
    predicted_counts = [
        sum(label == class_name for label in predictions) for class_name in LABELS
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, actual_counts, 0.4, label="Vrai label", color="#0072B2")
    ax.bar(x + 0.2, predicted_counts, 0.4, label="Prédit", color="#E69F00")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15, ha="right")
    ax.set_ylabel("Nombre de mails")
    ax.set_title("Répartition des vrais labels et des prédictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_counts.png", dpi=160)
    plt.close(fig)


def main():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions, labels=LABELS, zero_division=0))
    print("Matrice de confusion — lignes = vrais labels, colonnes = prédictions :")
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    save_plots(y_test, predictions, ASSETS_PATH)
    print(f"Graphiques enregistrés dans : {ASSETS_PATH}")


if __name__ == "__main__":
    main()
