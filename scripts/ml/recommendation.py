"""
He thong recommendation content-based dung euclidean distance-based similarity.
Goi y 5 sach tuong tu nhat (cung category) cho moi sach.
Tinh similarity = 1 / (1 + euclidean_distance) để chuyển distance thành similarity score.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_and_prepare_data(filepath='data/clean/tiki_books_cleaned.csv'):
    """
    Load du lieu va chuan bi features cho recommendation
    """
    print("=" * 70)
    print("LOAD VA CHUAN BI DU LIEU")
    print("=" * 70)
    
    df = pd.read_csv(filepath, encoding='utf-8')
    print("\nDa load {} sach".format(len(df)))
    
    # Giu nguyen cac cot can thiet
    df_features = df[['id', 'name', 'category_name', 'price', 'rating_average']].copy()
    
    # Xu ly missing values
    df_features['price'] = df_features['price'].fillna(df_features['price'].median())
    df_features['rating_average'] = df_features['rating_average'].fillna(0)
    
    # Chuan hoa price va rating
    scaler = StandardScaler()
    df_features[['price_scaled', 'rating_scaled']] = scaler.fit_transform(
        df_features[['price', 'rating_average']]
    )
    
    # Chi su dung price va rating cho similarity
    # (category da duoc loc sau, khong can dua vao phep tinh similarity)
    features_for_similarity = df_features[['price_scaled', 'rating_scaled']].copy()
    
    print("\nFeatures cho similarity:")
    print("- price_scaled: Gia tien (chuan hoa)")
    print("- rating_scaled: Diem danh gia (chuan hoa)")
    print("Tong so features: {}".format(features_for_similarity.shape[1]))
    print("\nLuu y: Category da duoc loc sau, khong dua vao phep tinh similarity")
    
    return df_features, features_for_similarity


def compute_similarity_matrix(features):
    """
    Tinh ma tran euclidean distance giua tat ca cac sach.
    Chuyển distance sang similarity: similarity = 1 / (1 + distance)
    """
    print("\n" + "=" * 70)
    print("TINH EUCLIDEAN DISTANCE VA CHUYEN SANG SIMILARITY")
    print("=" * 70)
    
    print("\nDang tinh euclidean distance cho {} sach...".format(len(features)))
    print("(Co the mat vai giay...)")
    
    # Tinh euclidean distance
    distances = euclidean_distances(features)
    
    # Chuyen sang similarity: similarity = 1 / (1 + distance)
    # Distance = 0 -> Similarity = 1.0 (giong het)
    # Distance lon -> Similarity nho (khac nhau)
    similarity_matrix = 1 / (1 + distances)
    
    print("Da tinh xong!")
    print("Similarity matrix shape: {}".format(similarity_matrix.shape))
    print("Min similarity: {:.3f}".format(similarity_matrix.min()))
    print("Max similarity: {:.3f}".format(similarity_matrix.max()))
    print("\nPhuong phap: Euclidean distance, chuyen sang similarity bang cong thuc")
    print("            similarity = 1 / (1 + distance)")
    
    return similarity_matrix


def get_recommendations(df_features, similarity_matrix, top_n=5):
    """
    Lay top N sach tuong tu nhat cho moi sach (cung category)
    """
    print("\n" + "=" * 70)
    print("TAO GOI Y CHO TUNG SACH")
    print("=" * 70)
    
    recommendations = []
    
    for idx in range(len(df_features)):
        source_id = df_features.iloc[idx]['id']
        source_name = df_features.iloc[idx]['name']
        source_category = df_features.iloc[idx]['category_name']
        
        # Lay similarity scores cua sach nay voi tat ca sach khac
        sim_scores = list(enumerate(similarity_matrix[idx]))
        
        # Sap xep theo similarity giam dan
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Lay top N, loai tru chinh no va chi lay sach cung category
        count = 0
        for i, score in sim_scores:
            if i == idx:  # Bo qua chinh no
                continue
            
            target_category = df_features.iloc[i]['category_name']
            if target_category != source_category:  # Chi lay cung category
                continue
            
            recommendations.append({
                'source_id': source_id,
                'source_name': source_name,
                'recommended_id': df_features.iloc[i]['id'],
                'recommended_name': df_features.iloc[i]['name'],
                'similarity_score': score
            })
            
            count += 1
            if count >= top_n:
                break
        
        # In progress
        if (idx + 1) % 500 == 0:
            print("  - Da xu ly {} / {} sach".format(idx + 1, len(df_features)))
    
    print("\nDa tao {} goi y (trung binh {} goi y/sach)".format(
        len(recommendations), len(recommendations) / len(df_features)))
    
    return pd.DataFrame(recommendations)


def show_examples(df_recommendations, df_features, num_examples=3):
    """
    Hien thi vi du mau de xem chat luong goi y
    """
    print("\n" + "=" * 70)
    print("VI DU MAU GOI Y")
    print("=" * 70)
    
    # Lay ngau nhien mot vai sach co du 5 goi y
    sample_sources = df_recommendations.groupby('source_id').size()
    sample_sources = sample_sources[sample_sources >= 5].sample(min(num_examples, len(sample_sources)))
    
    for i, source_id in enumerate(sample_sources.index, 1):
        # Lay thong tin sach goc
        source_info = df_features[df_features['id'] == source_id].iloc[0]
        
        print("\n" + "=" * 70)
        print("VI DU {}:".format(i))
        print("=" * 70)
        print("\nSACH GOC:")
        print("  - ID: {}".format(source_id))
        print("  - Ten: {}".format(source_info['name']))
        print("  - Category: {}".format(source_info['category_name']))
        print("  - Gia: {:,.0f}".format(source_info['price']))
        print("  - Rating: {:.1f}".format(source_info['rating_average']))
        
        # Lay top 5 goi y
        recs = df_recommendations[df_recommendations['source_id'] == source_id].head(5)
        
        print("\nTOP 5 SACH TUONG TU:")
        for j, (_, rec) in enumerate(recs.iterrows(), 1):
            rec_info = df_features[df_features['id'] == rec['recommended_id']].iloc[0]
            print("\n  {}. {} (Similarity: {:.3f})".format(
                j, rec['recommended_name'], rec['similarity_score']))
            print("     - Gia: {:,.0f} | Rating: {:.1f}".format(
                rec_info['price'], rec_info['rating_average']))


def main():
    """
    Ham chinh: chay toan bo quy trinh recommendation
    """
    print("=" * 70)
    print("CONTENT-BASED RECOMMENDATION SYSTEM")
    print("=" * 70)
    print("\nMuc tieu: Goi y 5 sach tuong tu nhat cho moi sach")
    print("Phuong phap: Cosine similarity tren features")
    print("  - Category (one-hot encode)")
    print("  - Price (normalized)")
    print("  - Rating (normalized)")
    
    # Buoc 1: Load va chuan bi du lieu
    df_features, features_matrix = load_and_prepare_data()
    
    # Buoc 2: Tinh cosine similarity
    similarity_matrix = compute_similarity_matrix(features_matrix)
    
    # Buoc 3: Tao goi y cho tung sach
    df_recommendations = get_recommendations(df_features, similarity_matrix, top_n=5)
    
    # Buoc 4: Luu ket qua
    output_path = 'data/clean/book_recommendations.csv'
    print("\n" + "=" * 70)
    print("LUU KET QUA")
    print("=" * 70)
    print("\nDang luu vao {}...".format(output_path))
    
    df_recommendations.to_csv(output_path, index=False, encoding='utf-8')
    print("Da luu thanh cong!")
    
    # Thong ke
    print("\n" + "=" * 70)
    print("THONG KE")
    print("=" * 70)
    print("\nTong so goi y: {}".format(len(df_recommendations)))
    print("Tong so sach: {}".format(len(df_features)))
    print("Trung binh goi y/sach: {:.1f}".format(
        len(df_recommendations) / len(df_features)))
    
    # Phan bo similarity scores
    print("\nPhan bo similarity scores:")
    print("  - Mean: {:.3f}".format(df_recommendations['similarity_score'].mean()))
    print("  - Min: {:.3f}".format(df_recommendations['similarity_score'].min()))
    print("  - Max: {:.3f}".format(df_recommendations['similarity_score'].max()))
    print("  - Std: {:.3f}".format(df_recommendations['similarity_score'].std()))
    
    # Buoc 5: Hien thi vi du
    show_examples(df_recommendations, df_features, num_examples=3)
    
    print("\n" + "=" * 70)
    print("HOAN THANH!")
    print("=" * 70)
    print("\nFile da tao: {}".format(output_path))
    print("Co the su dung file nay de hien thi goi y tren website/app.")


if __name__ == "__main__":
    main()