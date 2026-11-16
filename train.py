
import argparse
import re
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sms_spam_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "sms_spam_model.joblib"


def extract_features(message):
    return {
        "message_length": len(message),
        "word_count": len(message.split()),
        "digit_count": sum(char.isdigit() for char in message),
        "uppercase_count": sum(char.isupper() for char in message),
        "url_count": len(re.findall(r"http[s]?://|www\.", message))
    }


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train SMS Spam Detection models"
    )

    parser.add_argument("--nb_alpha", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=1000)

    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["message", "label"])
    df["message"] = df["message"].astype(str)
    df["label"] = df["label"].astype(int)

    # Separate input and target
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Text preprocessing
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )

    X_train_text = tfidf.fit_transform(X_train)
    X_test_text = tfidf.transform(X_test)

    # Numeric feature extraction
    feature_columns = [
        "message_length",
        "word_count",
        "digit_count",
        "uppercase_count",
        "url_count"
    ]

    train_features = pd.DataFrame(
        [extract_features(message) for message in X_train],
        index=X_train.index,
        columns=feature_columns
    )

    test_features = pd.DataFrame(
        [extract_features(message) for message in X_test],
        index=X_test.index,
        columns=feature_columns
    )

    scaler = MinMaxScaler()
    X_train_numeric = scaler.fit_transform(train_features)
    X_test_numeric = scaler.transform(test_features)

    X_train_final = hstack(
        [X_train_text, X_train_numeric]
    ).tocsr()

    X_test_final = hstack(
        [X_test_text, X_test_numeric]
    ).tocsr()

    # Configure MLflow
    tracking_path = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_path}")
    mlflow.set_experiment("SMS_Spam_Detection")

    experiments = [
        ("Naive_Bayes_Default", MultinomialNB(alpha=1.0)),
        (
            "Logistic_Regression",
            LogisticRegression(
                max_iter=args.max_iter,
                random_state=42
            )
        ),
        (
            "Naive_Bayes_Tuned",
            MultinomialNB(alpha=args.nb_alpha)
        )
    ]

    best_model = None
    best_f1 = -1
    best_name = ""

    # Train and track all three experiments
    for name, model in experiments:

        with mlflow.start_run(run_name=name):

            model.fit(X_train_final, y_train)

            metrics = evaluate_model(
                model, X_test_final, y_test
            )

            mlflow.log_param("algorithm", name)
            mlflow.log_param("nb_alpha", args.nb_alpha)
            mlflow.log_param("max_iter", args.max_iter)

            mlflow.log_metrics(metrics)

            print(f"\nExperiment: {name}")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = model
                best_name = name

    # Save the complete preprocessing and model bundle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "tfidf": tfidf,
        "scaler": scaler,
        "classifier": best_model
    }

    joblib.dump(model_bundle, MODEL_PATH)

    print("\nTraining completed successfully!")
    print("Selected model:", best_name)
    print("Best F1-score:", round(best_f1, 4))
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()
