"""
Train model Logistic Regression de du doan bestseller.
Dung cung config voi Random Forest de co the so sanh cong bang.
"""

import pandas as pd
import pickle
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def train_logistic_model(data_path='data/clean/tiki_books_cleaned.csv',
                         model_path='models/logistic_model.pkl',
                         scaler_path='models/logistic_scaler.pkl'):
    """
    Train model Logistic Regression de du doan mot cuon sach co phai bestseller khong.
    
    Tham so:
        data_path: duong dan den file du lieu da lam sach
        model_path: noi luu model sau khi train
        scaler_path: noi luu scaler (de chuan hoa du lieu)
    
    Tra ve:
        dict chua model, scaler, accuracy va danh sach features
    """
    
    print("=" * 70)
    print("TRAIN MODEL LOGISTIC REGRESSION - DU DOAN BESTSELLER")
    print("=" * 70)
    
    # Buoc 1: Load du lieu tu file CSV
    print("\n[1/9] Dang load du lieu...")
    df = pd.read_csv(data_path, encoding='utf-8')
    print("Da load {} sach".format(len(df)))
    
    # Kiem tra ty le bestseller
    bestseller_rate = df['is_bestseller'].mean()
    print("Ty le bestseller: {:.1%}".format(bestseller_rate))
    
    # Buoc 2: Tao one-hot encoding cho category
    # Phai dung cung cach voi Random Forest de so sanh cong bang
    print("\n[2/9] Dang tao features tu category...")
    category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    print("Da tao {} cot category".format(len(category_dummies.columns)))
    
    # Buoc 3: Chon features
    # Dung DUNG 8 features nhu Random Forest
    feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    feature_cols.extend(cat_cols)
    print("Tong so features: {}".format(len(feature_cols)))
    print("  - Numeric: price, discount_rate, rating_average, has_rating")
    print("  - Category: {} cot one-hot".format(len(cat_cols)))
    
    # Tach X va y
    X = df[feature_cols]
    y = df['is_bestseller']
    
    # Buoc 4: Chia train/test set
    # Dung DUNG config: 80/20, random_state=42, stratify=y
    print("\n[3/9] Chia train/test set (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train size: {}".format(len(X_train)))
    print("Test size: {}".format(len(X_test)))
    
    # Buoc 5: Chuan hoa du lieu
    # Dung StandardScaler nhu Random Forest
    print("\n[4/9] Dang scale features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Da scale xong")
    
    # Buoc 6: Train model Logistic Regression
    # KHONG dung class_weight='balanced' de cong bang voi Random Forest
    # max_iter=1000 de dam bao convergence
    print("\n[5/9] Dang train Logistic Regression...")
    print("Tham so: max_iter=1000, random_state=42")
    print("Khong dung class_weight (giong Random Forest)")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print("Training hoan thanh")
    
    # Buoc 7: Danh gia model
    print("\n[6/9] Danh gia model...")
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print("Train Accuracy: {:.1%}".format(train_acc))
    print("Test Accuracy: {:.1%}".format(test_acc))
    print("Gap (Train-Test): {:.1%}".format(train_acc - test_acc))
    
    # In ra bao cao chi tiet
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Non-Bestseller', 'Bestseller'],
                              digits=3))
    
    # Buoc 8: Confusion Matrix
    # De hieu ro model du doan dung/sai o dau
    print("\n[7/9] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("                      Du doan Non-BS  Du doan BS")
    print("Thuc te Non-BS:       {:>10}      {:>10}".format(cm[0,0], cm[0,1]))
    print("Thuc te BS:           {:>10}      {:>10}".format(cm[1,0], cm[1,1]))
    
    # Buoc 9: Xem coefficients (he so hoi quy)
    # Day la "feature importance" cua Logistic Regression
    # He so duong (+) -> tang feature nay thi tang xac suat bestseller
    # He so am (-) -> tang feature nay thi giam xac suat bestseller
    print("\n[8/9] He So Hoi Quy - Top 10 Features Quan Trong:")
    print("(Gia tri tuyet doi lon = anh huong manh)")
    
    # Tao DataFrame de sap xep theo gia tri tuyet doi
    coef_data = []
    for i, feature in enumerate(feature_cols):
        coef_value = model.coef_[0][i]
        coef_data.append({
            'feature': feature,
            'coefficient': coef_value,
            'abs_coef': abs(coef_value)
        })
    
    coef_df = pd.DataFrame(coef_data)
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    # In ra 10 features quan trong nhat
    for idx, row in coef_df.head(10).iterrows():
        sign = "+" if row['coefficient'] > 0 else "-"
        print("{:<25} {} {:.4f}  (coef: {:.4f})".format(
            row['feature'], 
            sign, 
            row['abs_coef'],
            row['coefficient']
        ))
    
    # Buoc 10: Luu model va scaler
    print("\n[9/9] Dang luu model va scaler...")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Da luu model: {}".format(model_path))
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Da luu scaler: {}".format(scaler_path))
    
    print("\n" + "=" * 70)
    print("TRAINING HOAN THANH")
    print("=" * 70)
    
    # Kiem tra ket qua co khop voi ky vong khong
    # (dam bao cau hinh dong bo voi Random Forest)
    print("\nKiem tra ket qua:")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("Ky vong: Accuracy ~0.787, Precision ~0.699, Recall ~0.522")
    print("Thuc te: Accuracy {:.3f}, Precision {:.3f}, Recall {:.3f}".format(
        test_acc, precision, recall
    ))
    
    # Kiem tra xem co khop khong (sai so < 1%)
    if abs(test_acc - 0.787) < 0.01 and abs(precision - 0.699) < 0.01:
        print("Ket qua khop - cau hinh dong bo chinh xac")
    else:
        print("Canh bao: ket qua khac biet - can kiem tra lai")
    
    # Tra ve ket qua
    return {
        'model': model,
        'scaler': scaler,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'feature_cols': feature_cols
    }


if __name__ == "__main__":
    # Chay ham train khi execute file nay truc tiep
    result = train_logistic_model()
    
    print("\nModel Logistic Regression da duoc train va luu thanh cong.")
    print("Su dung trong cac script khac bang cach:")
    print("  model = pickle.load(open('models/logistic_model.pkl', 'rb'))")
    print("  scaler = pickle.load(open('models/logistic_scaler.pkl', 'rb'))")
    print("\nSo sanh voi Random Forest:")
    print("  Random Forest: models/bestseller_model.pkl")
    print("  Logistic Regression: models/logistic_model.pkl")