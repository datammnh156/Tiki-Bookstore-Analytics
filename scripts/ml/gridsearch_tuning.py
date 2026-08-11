"""
Tim kiem hyperparameter toi uu cho Random Forest bang GridSearchCV.
Chi ghi de model neu co cai thien dang ke (F1 tang > 2%).
"""

import pandas as pd
import pickle
import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_and_prepare_data(data_path='data/clean/tiki_books_cleaned.csv'):
    """
    Load va chuan bi du lieu giong het train_model.py
    De dam bao so sanh cong bang
    """
    print("=" * 70)
    print("LOAD VA CHUAN BI DU LIEU")
    print("=" * 70)
    
    df = pd.read_csv(data_path, encoding='utf-8')
    print("\nDa load {} sach".format(len(df)))
    print("Ty le bestseller: {:.1%}".format(df['is_bestseller'].mean()))
    
    # Tao one-hot encoding cho category (giong train_model.py)
    category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    
    # Chon features (GIONG HET train_model.py)
    feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    feature_cols.extend(cat_cols)
    print("Tong so features: {}".format(len(feature_cols)))
    
    X = df[feature_cols]
    y = df['is_bestseller']
    
    # Chia train/test (GIONG HET train_model.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train size: {}".format(len(X_train)))
    print("Test size: {}".format(len(X_test)))
    
    # Scale du lieu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def perform_grid_search(X_train, y_train):
    """
    Thuc hien GridSearchCV de tim hyperparameter toi uu
    """
    print("\n" + "=" * 70)
    print("GRIDSEARCHCV - TIM HYPERPARAMETER TOI UU")
    print("=" * 70)
    
    # Dinh nghia cac tham so can thu
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [5, 10, 20]
    }
    
    print("\nCac tham so can thu:")
    print("- n_estimators: {}".format(param_grid['n_estimators']))
    print("- max_depth: {}".format(param_grid['max_depth']))
    print("- min_samples_split: {}".format(param_grid['min_samples_split']))
    
    # Tinh tong so to hop
    total_combinations = (len(param_grid['n_estimators']) * 
                         len(param_grid['max_depth']) * 
                         len(param_grid['min_samples_split']))
    print("\nTong so to hop: {}".format(total_combinations))
    print("Voi cv=5 folds, tong so model can train: {}".format(total_combinations * 5))
    print("Thoi gian uoc tinh: 3-5 phut")
    
    # Tao base model
    base_model = RandomForestClassifier(random_state=42)
    
    # Khoi tao GridSearchCV
    # scoring='f1' vi du lieu mat can bang (bestseller chi 30%)
    # cv=5 de validate tot hon (5-fold cross-validation)
    print("\nDang chay GridSearchCV...")
    print("- Scoring: f1 (uu tien cho lop bestseller)")
    print("- Cross-validation: 5 folds")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # F1-score cho lop positive (bestseller)
        n_jobs=-1,  # Su dung tat ca CPU cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    print("\nHoan thanh GridSearchCV trong {:.1f} phut".format(elapsed_time / 60))
    
    return grid_search


def evaluate_best_model(grid_search, X_train, X_test, y_train, y_test):
    """
    Danh gia model voi hyperparameter toi uu tren tap test
    """
    print("\n" + "=" * 70)
    print("KET QUA GRIDSEARCHCV")
    print("=" * 70)
    
    # In ra best parameters
    print("\nBo tham so TOI UU:")
    for param, value in grid_search.best_params_.items():
        print("- {}: {}".format(param, value))
    
    # In ra best F1 score tren cross-validation
    print("\nDiem F1 TOI UU tren Cross-Validation (train set):")
    print("F1-score: {:.3f}".format(grid_search.best_score_))
    
    # Lay model tot nhat
    best_model = grid_search.best_estimator_
    
    # Danh gia tren tap test
    print("\n" + "=" * 70)
    print("DANH GIA TREN TAP TEST (test set)")
    print("=" * 70)
    
    # Du doan
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Tinh cac metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print("\nKet qua model TUNED (sau GridSearch):")
    print("- Train Accuracy: {:.3f}".format(train_acc))
    print("- Test Accuracy: {:.3f}".format(test_acc))
    print("- Test Precision (Bestseller): {:.3f}".format(test_precision))
    print("- Test Recall (Bestseller): {:.3f}".format(test_recall))
    print("- Test F1-score (Bestseller): {:.3f}".format(test_f1))
    print("- Train-Test Gap: {:.3f}".format(train_acc - test_acc))
    
    # In classification report
    print("\nClassification Report chi tiet:")
    print(classification_report(y_test, y_test_pred,
                              target_names=['Non-Bestseller', 'Bestseller'],
                              digits=3))
    
    return {
        'model': best_model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'best_params': grid_search.best_params_
    }


def compare_with_baseline(tuned_results):
    """
    So sanh model tuned voi baseline model hien tai
    """
    print("\n" + "=" * 70)
    print("SO SANH VOI BASELINE MODEL")
    print("=" * 70)
    
    # Ket qua baseline (tu train_model.py - model hien tai)
    baseline = {
        'accuracy': 0.845,
        'precision': 0.796,
        'recall': 0.655,
        'f1': 0.719
    }
    
    print("\nBaseline (model hien tai - chua tuned):")
    print("- Accuracy: {:.3f}".format(baseline['accuracy']))
    print("- Precision: {:.3f}".format(baseline['precision']))
    print("- Recall: {:.3f}".format(baseline['recall']))
    print("- F1-score: {:.3f}".format(baseline['f1']))
    
    print("\nTuned (model sau GridSearchCV):")
    print("- Accuracy: {:.3f}".format(tuned_results['test_acc']))
    print("- Precision: {:.3f}".format(tuned_results['precision']))
    print("- Recall: {:.3f}".format(tuned_results['recall']))
    print("- F1-score: {:.3f}".format(tuned_results['f1']))
    
    # Tinh phan tram thay doi
    print("\nSu thay doi:")
    acc_change = tuned_results['test_acc'] - baseline['accuracy']
    prec_change = tuned_results['precision'] - baseline['precision']
    recall_change = tuned_results['recall'] - baseline['recall']
    f1_change = tuned_results['f1'] - baseline['f1']
    
    print("- Accuracy: {:+.3f} ({:+.1%})".format(acc_change, acc_change / baseline['accuracy']))
    print("- Precision: {:+.3f} ({:+.1%})".format(prec_change, prec_change / baseline['precision']))
    print("- Recall: {:+.3f} ({:+.1%})".format(recall_change, recall_change / baseline['recall']))
    print("- F1-score: {:+.3f} ({:+.1%})".format(f1_change, f1_change / baseline['f1']))
    
    # Ket luan
    print("\n" + "=" * 70)
    print("KET LUAN")
    print("=" * 70)
    
    # Nguong cai thien dang ke: F1 tang > 2%
    f1_improvement_threshold = 0.02
    
    if f1_change > f1_improvement_threshold:
        print("\nCAI THIEN DANG KE! (F1 tang > 2%)")
        print("Nen GHI DE model cu bang model tuned nay.")
        should_save = True
    elif f1_change > 0:
        print("\nCo cai thien NHUNG KHONG DANG KE (F1 tang < 2%)")
        print("GIU NGUYEN model cu de tranh overfitting.")
        print("Ket qua nay van duoc ghi nhan trong thesis.")
        should_save = False
    else:
        print("\nKHONG co cai thien (F1 giam hoac bang nhau)")
        print("GIU NGUYEN model cu.")
        print("Baseline da tot, khong can tuning them.")
        should_save = False
    
    return should_save, f1_change


def save_model_if_better(should_save, model, scaler, tuned_results):
    """
    Chi luu model neu co cai thien dang ke
    """
    if should_save:
        print("\n" + "=" * 70)
        print("LUU MODEL MOI")
        print("=" * 70)
        
        # Backup model cu truoc
        import shutil
        try:
            shutil.copy('models/bestseller_model.pkl', 
                       'models/bestseller_model_backup.pkl')
            print("Da backup model cu thanh: models/bestseller_model_backup.pkl")
        except:
            print("Khong co model cu de backup")
        
        # Luu model moi
        with open('models/bestseller_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Da luu model tuned: models/bestseller_model.pkl")
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Da luu scaler: models/scaler.pkl")
        
        print("\nModel moi co hyperparameter:")
        for param, value in tuned_results['best_params'].items():
            print("- {}: {}".format(param, value))
    else:
        print("\n" + "=" * 70)
        print("GIU NGUYEN MODEL CU")
        print("=" * 70)
        print("Model hien tai da du tot, khong can thay the.")
        print("Ket qua GridSearch duoc ghi nhan de bao cao trong thesis.")


def main():
    """
    Ham chinh: chay toan bo quy trinh GridSearchCV
    """
    print("=" * 70)
    print("GRIDSEARCH HYPERPARAMETER TUNING - RANDOM FOREST")
    print("=" * 70)
    print("\nMuc tieu: Tim hyperparameter toi uu cho model du doan bestseller")
    print("Chi ghi de model neu F1-score cai thien > 2%\n")
    
    # Buoc 1: Load va chuan bi du lieu
    X_train, X_test, y_train, y_test, scaler, feature_cols = load_and_prepare_data()
    
    # Buoc 2: Chay GridSearchCV
    grid_search = perform_grid_search(X_train, y_train)
    
    # Buoc 3: Danh gia model tot nhat
    tuned_results = evaluate_best_model(grid_search, X_train, X_test, y_train, y_test)
    
    # Buoc 4: So sanh voi baseline
    should_save, f1_change = compare_with_baseline(tuned_results)
    
    # Buoc 5: Luu model neu tot hon
    save_model_if_better(should_save, tuned_results['model'], scaler, tuned_results)
    
    print("\n" + "=" * 70)
    print("HOAN THANH!")
    print("=" * 70)
    print("\nTom tat:")
    print("- Best params: {}".format(tuned_results['best_params']))
    print("- F1-score: {:.3f} (baseline: 0.719, thay doi: {:+.3f})".format(
        tuned_results['f1'], f1_change))
    print("- Model duoc luu: {}".format("Co" if should_save else "Khong"))


if __name__ == "__main__":
    main()