"""
Đánh giá tổng thể mô hình Random Forest dự đoán sách bán chạy.

Bao gồm:
1. Cross-Validation 5-Fold
2. Kiểm tra chênh lệch Train/Test
3. Calibration
4. Stability qua nhiều random seed
5. So sánh với baseline
6. Confusion Matrix + Classification Report
7. Feature Importance

Lưu ý:
- Không hardcode kết quả thực nghiệm.
- Các kết quả đều được tính trực tiếp từ dữ liệu và mô hình.
"""

import os
import sys
import io
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier


# ============================================================
# CẤU HÌNH
# ============================================================
DATA_PATH = "data/clean/tiki_books_cleaned.csv"
MODEL_PATH = "models/bestseller_model.pkl"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_DIR = "outputs/charts"

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_SPLITS = 5
STABILITY_SEEDS = [0, 1, 42, 100, 123]

NUMERIC_FEATURES = [
    "price",
    "discount_rate",
    "rating_average",
    "has_rating",
]

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "random_state": RANDOM_STATE,
}


# ============================================================
# HỖ TRỢ HIỂN THỊ UTF-8 TRÊN WINDOWS
# ============================================================
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def prepare_data(df):
    """Tạo One-Hot Encoding cho category và trả về X, y, feature_cols."""
    category_dummies = pd.get_dummies(
        df["category_name"],
        prefix="cat",
        dtype=int,
    )

    df_encoded = pd.concat([df.copy(), category_dummies], axis=1)

    category_features = [
        col for col in df_encoded.columns
        if col.startswith("cat_")
    ]

    feature_cols = NUMERIC_FEATURES + category_features
    X = df_encoded[feature_cols].copy()
    y = df_encoded["is_bestseller"].astype(int).copy()

    return X, y, feature_cols


def cross_validation_evaluation(X, y):
    """
    Stratified 5-Fold Cross-Validation.
    Scaler chỉ được fit trên từng training fold để tránh data leakage.
    """
    print_section("1. CROSS-VALIDATION (5-FOLD)")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train_scaled, y_train)

        score = model.score(X_test_scaled, y_test)
        cv_scores.append(score)
        print(f"Fold {fold}: {score:.1%}")

    cv_scores = np.array(cv_scores)

    print("\nKết quả Cross-Validation:")
    print(f"- Mean Accuracy: {cv_scores.mean():.1%}")
    print(f"- Std Deviation: {cv_scores.std():.3f}")
    print(f"- Min Accuracy : {cv_scores.min():.1%}")
    print(f"- Max Accuracy : {cv_scores.max():.1%}")

    return cv_scores


def train_test_evaluation(X, y):
    """Train/test split 80/20 và đánh giá chênh lệch train-test."""
    print_section("2. TRAIN/TEST EVALUATION")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    gap = train_acc - test_acc

    print(f"Train size: {len(X_train):,}")
    print(f"Test size : {len(X_test):,}")
    print(f"Train Accuracy: {train_acc:.1%}")
    print(f"Test Accuracy : {test_acc:.1%}")
    print(f"Train-Test Gap: {gap:.1%}")

    print(
        "\nNhận xét: Chênh lệch train-test được báo cáo trực tiếp để đánh giá "
        "dấu hiệu quá khớp; không áp dụng ngưỡng cứng để tự động kết luận."
    )

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap,
    }


def calibration_evaluation(model, X_test_scaled, y_test):
    """
    Đánh giá calibration dựa trên mean predicted probability
    và fraction of positives trong từng bin.
    """
    print_section("3. CALIBRATION CHECK")

    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    prob_true, prob_pred = calibration_curve(
        y_test,
        y_proba,
        n_bins=10,
        strategy="uniform",
    )

    print(f"{'Bin':<6}{'Mean Predicted':>18}{'Actual Rate':>16}{'Difference':>14}")
    print("-" * 54)

    differences = []

    for i, (pred_p, true_p) in enumerate(zip(prob_pred, prob_true), start=1):
        diff = abs(true_p - pred_p)
        differences.append(diff)
        print(f"{i:<6}{pred_p:>17.1%}{true_p:>16.1%}{diff:>14.1%}")

    avg_diff = float(np.mean(differences)) if differences else np.nan
    print(f"\nMean absolute calibration difference: {avg_diff:.1%}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, "o-", label="Random Forest")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve - Bestseller Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "calibration_curve.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Đã lưu calibration plot: {output_path}")
    return avg_diff


def stability_evaluation(X, y, feature_cols):
    """Đánh giá độ ổn định của mô hình qua nhiều random seed."""
    print_section("4. STABILITY CHECK")

    results = []

    for seed in STABILITY_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        params = MODEL_PARAMS.copy()
        params["random_state"] = seed

        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        top3 = importance_df.head(3)["feature"].tolist()

        results.append({
            "seed": seed,
            "accuracy": accuracy,
            "top3_features": top3,
        })

        print(
            f"Seed {seed:>3}: Accuracy = {accuracy:.1%}; "
            f"Top 3 = {', '.join(top3)}"
        )

    accuracies = np.array([r["accuracy"] for r in results])

    print("\nThống kê stability:")
    print(f"- Mean Accuracy: {accuracies.mean():.1%}")
    print(f"- Std Deviation: {accuracies.std():.3f}")
    print(f"- Range        : {(accuracies.max() - accuracies.min()):.1%}")

    return results


def baseline_evaluation(X_train_scaled, X_test_scaled, y_train, y_test, model_acc):
    """
    So sánh Random Forest với hai baseline bằng DummyClassifier:
    - most_frequent: luôn dự đoán lớp phổ biến nhất
    - stratified: dự đoán ngẫu nhiên theo phân bố lớp tập huấn luyện
    """
    print_section("5. BASELINE COMPARISON")

    baseline_majority = DummyClassifier(strategy="most_frequent")
    baseline_majority.fit(X_train_scaled, y_train)
    majority_pred = baseline_majority.predict(X_test_scaled)
    majority_acc = accuracy_score(y_test, majority_pred)

    baseline_random = DummyClassifier(
        strategy="stratified",
        random_state=RANDOM_STATE,
    )
    baseline_random.fit(X_train_scaled, y_train)
    random_pred = baseline_random.predict(X_test_scaled)
    random_acc = accuracy_score(y_test, random_pred)

    print(f"Majority baseline Accuracy : {majority_acc:.1%}")
    print(f"Random baseline Accuracy   : {random_acc:.1%}")
    print(f"Random Forest Test Accuracy: {model_acc:.1%}")
    print(f"Improvement vs majority    : {model_acc - majority_acc:+.1%}")
    print(f"Improvement vs random      : {model_acc - random_acc:+.1%}")

    return {
        "majority_acc": majority_acc,
        "random_acc": random_acc,
    }


def classification_evaluation(model, X_test_scaled, y_test):
    """Confusion Matrix và Classification Report."""
    print_section("6. CONFUSION MATRIX & CLASSIFICATION REPORT")

    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(f"{'':<20}{'Predicted 0':>14}{'Predicted 1':>14}")
    print(f"{'Actual 0':<20}{cm[0, 0]:>14}{cm[0, 1]:>14}")
    print(f"{'Actual 1':<20}{cm[1, 0]:>14}{cm[1, 1]:>14}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Non-Bestseller", "Bestseller"],
            digits=3,
        )
    )

    return cm


def feature_importance_evaluation(model, feature_cols):
    """In Feature Importance của Random Forest."""
    print_section("7. FEATURE IMPORTANCE")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("Top features:")
    for _, row in importance_df.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")

    return importance_df


def save_final_model(model, scaler):
    """Lưu model/scaler để khớp với pipeline đánh giá hiện tại."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nĐã lưu model : {MODEL_PATH}")
    print(f"Đã lưu scaler: {SCALER_PATH}")


def main():
    print_section("ĐÁNH GIÁ TỔNG THỂ MODEL BESTSELLER PREDICTION")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

    print("Đang tải dữ liệu...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    print(f"Số sách: {len(df):,}")
    print(f"Số category: {df['category_name'].nunique()}")

    X, y, feature_cols = prepare_data(df)

    print(f"Số features: {len(feature_cols)}")
    print(f"- Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"- Category one-hot: {len(feature_cols) - len(NUMERIC_FEATURES)}")
    print(f"Tỷ lệ Bestseller: {y.mean():.1%} ({int(y.sum()):,}/{len(y):,})")

    cv_scores = cross_validation_evaluation(X, y)

    evaluation = train_test_evaluation(X, y)

    avg_calibration_diff = calibration_evaluation(
        evaluation["model"],
        evaluation["X_test_scaled"],
        evaluation["y_test"],
    )

    stability_results = stability_evaluation(X, y, feature_cols)

    baseline_results = baseline_evaluation(
        evaluation["X_train_scaled"],
        evaluation["X_test_scaled"],
        evaluation["y_train"],
        evaluation["y_test"],
        evaluation["test_acc"],
    )

    classification_evaluation(
        evaluation["model"],
        evaluation["X_test_scaled"],
        evaluation["y_test"],
    )

    importance_df = feature_importance_evaluation(
        evaluation["model"],
        feature_cols,
    )

    save_final_model(
        evaluation["model"],
        evaluation["scaler"],
    )

    print_section("TỔNG KẾT")

    stability_acc = np.array([r["accuracy"] for r in stability_results])

    print(f"CV Mean Accuracy           : {cv_scores.mean():.1%}")
    print(f"CV Std                     : {cv_scores.std():.3f}")
    print(f"Train Accuracy             : {evaluation['train_acc']:.1%}")
    print(f"Test Accuracy              : {evaluation['test_acc']:.1%}")
    print(f"Train-Test Gap             : {evaluation['gap']:.1%}")
    print(f"Calibration Mean Difference: {avg_calibration_diff:.1%}")
    print(f"Stability Mean Accuracy    : {stability_acc.mean():.1%}")
    print(f"Stability Std              : {stability_acc.std():.3f}")
    print(f"Majority Baseline          : {baseline_results['majority_acc']:.1%}")
    print(f"Random Baseline            : {baseline_results['random_acc']:.1%}")

    print("\nTop 5 Feature Importance:")
    for _, row in importance_df.head(5).iterrows():
        print(f"- {row['feature']}: {row['importance']:.4f}")

    print("\nHoàn thành đánh giá model.")


if __name__ == "__main__":
    main()
