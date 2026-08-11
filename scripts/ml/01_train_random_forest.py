"""
Train model Random Forest de du doan bestseller.
Luu model va scaler vao thu muc models/ de dung cho cac script khac.
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
    Train model Random Forest de du doan mot cuon sach co phai bestseller hay khong.
    
    Tham so:
        data_path: duong dan den file du lieu da lam sach
        model_path: noi luu model sau khi train
        scaler_path: noi luu scaler (de chuan hoa du lieu)
    
    Tra ve:
        dict chua model, scaler, accuracy va danh sach features
    """
    
    print("=" * 70)
    print("TRAIN MODEL DU DOAN BESTSELLER")
    print("=" * 70)
    
    # Buoc 1: Load du lieu tu file CSV
    print("\n[1/8] Dang load du lieu...")
    df = pd.read_csv(data_path, encoding='utf-8')
    print("Da load {} sach".format(len(df)))
    
    # Kiem tra ty le bestseller trong dataset
    bestseller_rate = df['is_bestseller'].mean()
    print("Ty le bestseller: {:.1%}".format(bestseller_rate))
    
    # Buoc 2: Tao one-hot encoding cho category
    # Vi du: "Sach tieng Viet" -> cat_Sach_tieng_Viet = 1, cac cot khac = 0
    print("\n[2/8] Dang tao features tu category...")
    category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    print("Da tao {} cot category".format(len(category_dummies.columns)))
    
    # Buoc 3: Chon cac features de train model
    # Su dung 4 features chinh + cac cot category vua tao
    feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    feature_cols.extend(cat_cols)
    print("Tong so features: {}".format(len(feature_cols)))
    
    # Tach X (features) va y (label can du doan)
    X = df[feature_cols]
    y = df['is_bestseller']
    
    # Buoc 4: Chia train/test set
    # 80% de train, 20% de test
    # stratify=y de dam bao ty le bestseller trong train va test giong nhau
    print("\n[3/8] Chia train/test set (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train size: {}".format(len(X_train)))
    print("Test size: {}".format(len(X_test)))
    
    # Buoc 5: Chuan hoa du lieu (scaling)
    # Vi cac feature co don vi khac nhau (price: tram nghin, rating: 0-5)
    # Nen can scale ve cung muc do de model hoc tot hon
    print("\n[4/8] Dang scale features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit tren train set
    X_test_scaled = scaler.transform(X_test)  # chi transform test set, khong fit lai
    print("Da scale xong")
    
    # Buoc 6: Train model Random Forest
    # n_estimators: so cay decision tree
    # max_depth: do sau toi da cua moi cay (tranh overfit)
    # min_samples_split: so sample toi thieu de chia node
    print("\n[5/8] Dang train Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print("Training hoan thanh")
    
    # Buoc 7: Danh gia model
    print("\n[6/8] Danh gia model...")
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
    
    # Buoc 8: Xem feature nao quan trong nhat
    print("\n[7/8] Feature Importance (Top 5):")
    
    # Tao DataFrame de sap xep theo importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # In ra 5 features quan trong nhat
    for idx, row in importance_df.head(5).iterrows():
        print("{:<25} {:.3f}".format(row['feature'], row['importance']))
    
    # Buoc 9: Luu model va scaler
    print("\n[8/8] Dang luu model va scaler...")
    
    # Luu model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Da luu model: {}".format(model_path))
    
    # Luu scaler (can thiet khi du doan sau nay)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Da luu scaler: {}".format(scaler_path))
    
    print("\n" + "=" * 70)
    print("TRAINING HOAN THANH")
    print("=" * 70)
    
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
    result = train_bestseller_model()
    
    print("\nModel da duoc train va luu thanh cong.")
    print("Su dung trong cac script khac bang cach:")
    print("  model = pickle.load(open('models/bestseller_model.pkl', 'rb'))")
    print("  scaler = pickle.load(open('models/scaler.pkl', 'rb'))")