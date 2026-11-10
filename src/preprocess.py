
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EDA_DIR = ROOT_DIR / "reports" / "eda"


def load_data():
    print("Loading dataset...")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    print("Dataset shape:", df.shape)

    return df


def validate_data(df):
    print("\n========== DATA VALIDATION ==========")

    if "Class" not in df.columns:
        raise ValueError("Target column Class is missing")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Invalid target labels")

    print("Missing values:", df.isnull().sum().sum())
    print("Fraud distribution:")
    print(df["Class"].value_counts())

    print("Exact duplicate rows:", df.duplicated().sum())


def clean_data(df):
    print("\n========== DATA CLEANING ==========")

    feature_columns = df.drop(columns=["Class"]).columns

    duplicate_mask = df.duplicated(
        subset=feature_columns,
        keep=False
    )

    duplicate_records = df[duplicate_mask]

    # Identical features with conflicting labels need investigation
    conflicts = (
        duplicate_records
        .groupby(list(feature_columns))["Class"]
        .nunique()
        .gt(1)
    )

    if conflicts.any():
        raise ValueError(
            "Conflicting labels found for identical feature vectors"
        )

    before = len(df)

    df = df.drop_duplicates(
        subset=feature_columns,
        keep="first"
    ).copy()

    print("Records before cleaning:", before)
    print("Records after cleaning:", len(df))
    print("Duplicates removed:", before - len(df))

    return df


def explore_data(df):
    print("\n========== EXPLORATORY DATA ANALYSIS ==========")

    EDA_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Class distribution
    plt.figure(figsize=(7, 5))

    sns.countplot(data=df, x="Class")

    plt.title("Fraud vs Legitimate Transactions")
    plt.xlabel("Transaction Class")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(EDA_DIR / "class_distribution.png")
    plt.close()

    # 2. Transaction amount distribution
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="Amount",
        hue="Class",
        bins=50,
        log_scale=(False, True)
    )

    plt.title("Transaction Amount Distribution")
    plt.tight_layout()

    plt.savefig(EDA_DIR / "amount_distribution.png")
    plt.close()

    # 3. Transaction time distribution
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        bins=50,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title("Transaction Time Distribution")
    plt.tight_layout()

    plt.savefig(EDA_DIR / "time_distribution.png")
    plt.close()

    # 4. Feature correlation
    plt.figure(figsize=(18, 14))

    correlation = df.corr()

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(EDA_DIR / "correlation_heatmap.png")
    plt.close()

    print("EDA plots saved to:", EDA_DIR)


def split_data(df):
    print("\n========== DATA SPLITTING ==========")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 70% training, 30% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # 10% validation, 20% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=2 / 3,
        random_state=42,
        stratify=y_temp
    )

    train = X_train.copy()
    train["Class"] = y_train

    validation = X_val.copy()
    validation["Class"] = y_val

    test = X_test.copy()
    test["Class"] = y_test

    return train, validation, test


def save_data(train, validation, test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False
    )
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("\n========== DATA SAVED ==========")

    print("Training:", train.shape)
    print("Validation:", validation.shape)
    print("Testing:", test.shape)


def generate_report(train, validation, test):
    print("\n========== FINAL DATA REPORT ==========")

    for name, dataset in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test)
    ]:
        print(f"\n{name}")
        print("Records:", len(dataset))
        print("Class distribution:")
        print(dataset["Class"].value_counts().to_dict())


def main():
    df = load_data()

    validate_data(df)

    df = clean_data(df)

    explore_data(df)

    train, validation, test = split_data(df)

    save_data(train, validation, test)

    generate_report(train, validation, test)

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
