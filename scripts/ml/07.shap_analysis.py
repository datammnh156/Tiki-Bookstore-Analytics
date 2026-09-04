"""
Phan tich SHAP (SHapley Additive exPlanations) de giai thich Random Forest model.
So sanh voi feature importance thong thuong va ve cac bieu do SHAP.
"""

import pandas as pd
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set matplotlib backend
plt.switch_backend('Agg')


def load_model_and_data(model_path='models/bestseller_model.pkl',
                        scaler_path='models/scaler.pkl',
                        data_path='data/clean/tiki_books_cleaned.csv'):
    """
    Load model da train, scaler, va du lieu test
    """
    print("=" * 70)
    print("LOAD MODEL VA DU LIEU")
    print("=" * 70)
    
    # Load model va scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Da load model: {}".format(model_path))
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Da load scaler: {}".format(scaler_path))
    
    # Load va chuan bi du lieu
    df = pd.read_csv(data_path, encoding='utf-8')
    print("Da load {} sach".format(len(df)))
    
    # Tao features (giong model training)
    category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    
    feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    feature_cols.extend(cat_cols)
    
    X = df[feature_cols]
    y = df['is_bestseller']
    
    # Chia train/test giong model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale du lieu
    X_test_scaled = scaler.transform(X_test)
    
    print("Test set size: {}".format(len(X_test)))
    print("Features: {}".format(len(feature_cols)))
    
    return model, X_test_scaled, y_test, feature_cols, X_test


def calculate_shap_values(model, X_test_scaled, feature_cols):
    """
    Tinh SHAP values su dung TreeExplainer
    """
    print("\n" + "=" * 70)
    print("TINH SHAP VALUES")
    print("=" * 70)
    
    print("\nDang khoi tao TreeExplainer (dung cho decision tree-based models)...")
    explainer = shap.TreeExplainer(model)
    
    print(f"Dang tinh SHAP values cho {len(X_test_scaled)} mau...")
    shap_values_raw = explainer.shap_values(X_test_scaled)
    
    # Voi RandomForestClassifier nhi phan, shap_values la array 3D: (n_samples, n_features, 2)
    # [:, :, 0] = SHAP values cho lop 0 (Non-Bestseller)
    # [:, :, 1] = SHAP values cho lop 1 (Bestseller) - ta dung cai nay
    if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[:, :, 1]  # Lay SHAP values cho lop Bestseller (class 1)
    elif isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]  # Backup: neu la list thi lay phan tu 1
    else:
        shap_values = shap_values_raw
    
    print("SHAP values shape: {}".format(shap_values.shape))
    print("Da tinh SHAP values thanh cong!")
    
    return explainer, shap_values


def plot_summary(shap_values, X_test_scaled, feature_cols, X_test):
    """
    Ve summary plot (beeswarm) - tong quan feature anh huong
    """
    print("\n" + "=" * 70)
    print("VE SUMMARY PLOT (BEESWARM)")
    print("=" * 70)
    
    print("Dang tao summary plot...")
    
    # Tao SHAP values object cho plotting
    shap_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    plt.figure(figsize=(10, 6))
    shap_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

    shap.summary_plot(shap_values,shap_df,feature_names=feature_cols, plot_type="dot",
        show=False,max_display=15
)
    
    plt.title("SHAP Summary Plot - Random Forest Bestseller Prediction")
    plt.tight_layout()
    plt.savefig('outputs/charts/shap_summary.png', dpi=300, bbox_inches='tight')
    print("Da luu: outputs/charts/shap_summary.png")
    
    plt.close()


def plot_waterfall_examples(model, explainer, shap_values, X_test_scaled, y_test, 
                            feature_cols):
    """
    Ve waterfall plot cho 2 vi du: 1 bestseller va 1 non-bestseller
    """
    print("\n" + "=" * 70)
    print("VE WATERFALL PLOTS CHO VI DU CU THE")
    print("=" * 70)
    
    # Tim 1 vu du bestseller voi xac suat cao
    predictions = model.predict_proba(X_test_scaled)[:, 1]
    bestseller_idx = np.argmax(predictions)  # Bestseller voi xac suat cao nhat
    
    # Tim 1 vu du non-bestseller voi xac suat cao
    non_bestseller_idx = np.argmin(predictions)  # Non-bestseller voi xac suat thap nhat
    
    print("\nVi du 1: Sach BESTSELLER (du doan xac suat: {:.1%})".format(
        predictions[bestseller_idx]))
    print("Thuc te: {}".format("Dung" if y_test.iloc[bestseller_idx] == 1 else "Sai"))
    
    print("\nVi du 2: Sach NON-BESTSELLER (du doan xac suat: {:.1%})".format(
        predictions[non_bestseller_idx]))
    print("Thuc te: {}".format("Dung" if y_test.iloc[non_bestseller_idx] == 0 else "Sai"))
    
    # Ve waterfall plot cho bestseller
    print("\nDang tao waterfall plot cho bestseller...")
    plt.figure(figsize=(10, 6))
    
    # Xac dinh expected_value cho class 1
    if isinstance(explainer.expected_value, np.ndarray):
        expected_value_1 = explainer.expected_value[1]
    else:
        expected_value_1 = explainer.expected_value
    
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[bestseller_idx],
        base_values=expected_value_1,
        data=X_test_scaled[bestseller_idx],
        feature_names=feature_cols
    ), show=False)
    plt.title("SHAP Waterfall - Bestseller Example (High Probability)")
    plt.tight_layout()
    plt.savefig('outputs/charts/shap_example_1.png', dpi=300, bbox_inches='tight')
    print("Da luu: outputs/charts/shap_example_1.png")
    plt.close()
    
    # Ve waterfall plot cho non-bestseller
    print("\nDang tao waterfall plot cho non-bestseller...")
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[non_bestseller_idx],
        base_values=expected_value_1,
        data=X_test_scaled[non_bestseller_idx],
        feature_names=feature_cols
    ), show=False)
    plt.title("SHAP Waterfall - Non-Bestseller Example (Low Probability)")
    plt.tight_layout()
    plt.savefig('outputs/charts/shap_example_2.png', dpi=300, bbox_inches='tight')
    print("Da luu: outputs/charts/shap_example_2.png")
    plt.close()


def compare_importances(model, shap_values, feature_cols):
    """
    So sanh SHAP importance voi Random Forest feature_importance_
    """
    print("\n" + "=" * 70)
    print("SO SANH SHAP IMPORTANCE VS RF FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Tinh SHAP importance (gia tri trung binh tuyet doi cua SHAP values)
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Lay RF feature importance
    rf_importance = model.feature_importances_
    
    # Tao DataFrame de so sanh
    comparison_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Importance': shap_importance,
        'RF_Importance': rf_importance
    })
    
    # Sap xep theo SHAP importance
    comparison_df = comparison_df.sort_values('SHAP_Importance', ascending=False)
    
    print("\nTop 5 Feature theo SHAP Importance:")
    print("-" * 70)
    for idx, (i, row) in enumerate(comparison_df.head(5).iterrows(), 1):
        print("{:>2}. {:30} SHAP: {:.4f}  RF: {:.4f}".format(
            idx, row['Feature'], row['SHAP_Importance'], row['RF_Importance']
        ))
    
    print("\nTop 5 Feature theo Random Forest Importance:")
    print("-" * 70)
    comparison_sorted_rf = comparison_df.sort_values('RF_Importance', ascending=False)
    for idx, (i, row) in enumerate(comparison_sorted_rf.head(5).iterrows(), 1):
        print("{:>2}. {:30} RF: {:.4f}  SHAP: {:.4f}".format(
            idx, row['Feature'], row['RF_Importance'], row['SHAP_Importance']
        ))
    
    # Kiem tra dung thu tu hay khong
    print("\nSo sanh thu tu top 5:")
    shap_top5 = set(comparison_df.head(5)['Feature'].values)
    rf_top5 = set(comparison_sorted_rf.head(5)['Feature'].values)
    
    if shap_top5 == rf_top5:
        print("✓ TOP 5 FEATURES GIONG NHAU (chi khac thu tu)")
    else:
        print("✗ Top 5 khac nhau")
        print("  - Chi SHAP co: {}".format(shap_top5 - rf_top5))
        print("  - Chi RF co: {}".format(rf_top5 - shap_top5))
    
    return comparison_df


def print_shap_explanation():
    """
    In ra giai thich ve SHAP cho thesis
    """
    print("\n" + "=" * 70)
    print("GIAI THICH SHAP CHO THESIS")
    print("=" * 70)
    



def main():
    """
    Ham chinh: chay toan bo quy trinh SHAP analysis
    """
    print("=" * 70)
    print("SHAP ANALYSIS - RANDOM FOREST EXPLAINABILITY")
    print("=" * 70)
    print("\nMuc tieu: Giai thich model Random Forest bang SHAP values")
    print("Va so sanh voi feature importance thong thuong\n")
    
    # Buoc 1: Load model va du lieu
    model, X_test_scaled, y_test, feature_cols, X_test = load_model_and_data()
    
    # Buoc 2: Tinh SHAP values
    explainer, shap_values = calculate_shap_values(model, X_test_scaled, feature_cols)
    
    # Buoc 3: Ve summary plot
    plot_summary(shap_values, X_test_scaled, feature_cols, X_test)
    
    # Buoc 4: Ve waterfall plots cho vu du
    plot_waterfall_examples(model, explainer, shap_values, X_test_scaled, y_test, feature_cols)
    
    # Buoc 5: So sanh importances
    comparison_df = compare_importances(model, shap_values, feature_cols)
    
    # Buoc 6: In giai thich cho thesis
    explanation = print_shap_explanation()
    
    print("\n" + "=" * 70)
    print("HOAN THANH!")
    print("=" * 70)
    print("\nCac file da tao:")
    print("1. outputs/charts/shap_summary.png - Summary plot")
    print("2. outputs/charts/shap_example_1.png - Bestseller waterfall")
    print("3. outputs/charts/shap_example_2.png - Non-bestseller waterfall")
    print("\nDo thi va phan tich da luu.")


if __name__ == "__main__":
    main()