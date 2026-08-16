"""
Train model Random Forest để dự đoán sách bestseller.
Sau khi train xong sẽ lưu model + scaler vào folder models/ để dùng lại sau.
"""

import pandas as pd
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def train_bestseller_model(data_path='data/clean/tiki_books_cleaned.csv',
                          model_path='models/bestseller_model.pkl',
                          scaler_path='models/scaler.pkl'):
    """
    Train model Random Forest để dự đoán sách có phải bestseller không.
    
    Input:
        data_path: đường dẫn file CSV đã clean
        model_path: lưu model vào đâu
        scaler_path: lưu scaler vào đâu
    
    Output:
        dict chứa model, scaler, accuracy, list features
    """
    
    print("=" * 70)
    print("TRAIN MODEL DU DOAN BESTSELLER")
    print("=" * 70)
    
    # Load dữ liệu từ CSV
    print("\n[1/8] Dang load du lieu...")
    df = pd.read_csv(data_path, encoding='utf-8')
    print("Da load {} sach".format(len(df)))
    
    bestseller_rate = df['is_bestseller'].mean()
    print("Ty le bestseller: {:.1%}".format(bestseller_rate))
    
    # Chuyển category thành số (one-hot encoding)
    # VD: "Sách tiếng Việt" thành cột cat_Sach_tieng_Viet = 1, còn lại = 0
    print("\n[2/8] Dang tao features tu category...")
    category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    print("Da tao {} cot category".format(len(category_dummies.columns)))
    
    # Chọn features để train: 4 cột chính + các cột category vừa tạo
    feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    feature_cols.extend(cat_cols)
    print("Tong so features: {}".format(len(feature_cols)))
    
    # Tách features (X) và label cần dự đoán (y)
    X = df[feature_cols]
    y = df['is_bestseller']
    
    # Chia train/test 80/20
    # Dùng stratify để giữ tỉ lệ bestseller trong train và test giống nhau
    print("\n[3/8] Chia train/test set (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train size: {}".format(len(X_train)))
    print("Test size: {}".format(len(X_test)))
    
    # Scale dữ liệu về cùng 1 thang đo
    # Vì price (hàng trăm nghìn) và rating (0-5) chênh lệch quá nhiều
    print("\n[4/8] Dang scale features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # học cách scale từ train
    X_test_scaled = scaler.transform(X_test)  # áp dụng cách scale đó cho test
    print("Da scale xong")
    
    # Train model với các tham số đã chọn
    print("\n[5/8] Dang train Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print("Training hoan thanh")
    
    # Đánh giá xem model dự đoán tốt không
    print("\n[6/8] Danh gia model...")
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print("Train Accuracy: {:.1%}".format(train_acc))
    print("Test Accuracy: {:.1%}".format(test_acc))
    print("Gap (Train-Test): {:.1%}".format(train_acc - test_acc))
    
    # In báo cáo chi tiết
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Non-Bestseller', 'Bestseller'],
                              digits=3))
    
    # Xem feature nào ảnh hưởng nhiều nhất đến kết quả
    print("\n[7/8] Feature Importance (Top 5):")
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # In 5 features quan trọng nhất
    for idx, row in importance_df.head(5).iterrows():
        print("{:<25} {:.3f}".format(row['feature'], row['importance']))
    
    # Lưu model và scaler để dùng lại sau
    print("\n[8/8] Dang luu model va scaler...")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Da luu model: {}".format(model_path))
    
    # Lưu scaler (phải có khi dự đoán sau này)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Da luu scaler: {}".format(scaler_path))
    
    print("\n" + "=" * 70)
    print("TRAINING HOAN THANH")
    print("=" * 70)
    
    return {
        'model': model,
        'scaler': scaler,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'feature_cols': feature_cols
    }


if __name__ == "__main__":
    # Chạy train khi chạy file này trực tiếp
    result = train_bestseller_model()
    
    print("\nModel da duoc train va luu thanh cong.")
    print("Su dung trong cac script khac bang cach:")
    print("  model = pickle.load(open('models/bestseller_model.pkl', 'rb'))")
    print("  scaler = pickle.load(open('models/scaler.pkl', 'rb'))")