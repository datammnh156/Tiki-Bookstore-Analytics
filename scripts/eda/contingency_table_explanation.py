"""
Giai thich chi tiet: Bang tan suat cheo (Contingency Table) duoc tao nhu the nao?
"""

import pandas as pd
import sys

# Fix encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def explain_contingency_table():
    """
    Giai thich tung buoc tao bang tan suat cheo bang pd.crosstab()
    """
    
    print("=" * 80)
    print("GIAI THICH: BANG TAN SUAT CHEO (CONTINGENCY TABLE)")
    print("=" * 80)
    
    # Buoc 1: Load du lieu
    print("\n[BUOC 1] Load du lieu")
    print("-" * 80)
    df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
    print("DataFrame co {} dong (sach)".format(len(df)))
    print("\nCac cot lien quan:")
    print("- 'has_rating': gia tri 0 hoac 1 (sach co dung/sai rating)")
    print("- 'is_bestseller': gia tri 0 hoac 1 (sach bestseller dung/sai)")
    
    # Xem du lieu mau
    print("\nDu lieu mau (5 dong dau):")
    print(df[['has_rating', 'is_bestseller']].head())
    
    # Buoc 2: Hieu pd.crosstab() la gi
    print("\n[BUOC 2] Hieu pd.crosstab() la gi?")
    print("-" * 80)
    print("""
pd.crosstab(rows, columns, ...) la ham tao bang tan suat cheo
- rows: bien tren hang (trong truong hop nay: has_rating)
- columns: bien tren cot (trong truong hop nay: is_bestseller)
- margins=True: them cot/hang 'All' hien thi tong cong
    """)
    
    # Buoc 3: Tao bang tan suat cheo
    print("\n[BUOC 3] Tao bang tan suat cheo")
    print("-" * 80)
    
    print("\nLenhh Python:")
    print("""
contingency_table = pd.crosstab(
    df['has_rating'],              # Bien hang
    df['is_bestseller'],           # Bien cot
    rownames=['has_rating'],       # Ten hang
    colnames=['is_bestseller'],    # Ten cot
    margins=True                   # Them dong/cot 'All'
)
    """)
    
    contingency_table = pd.crosstab(
        df['has_rating'],
        df['is_bestseller'],
        rownames=['has_rating'],
        colnames=['is_bestseller'],
        margins=True
    )
    
    print("\nKet qua bang tan suat cheo:")
    print(contingency_table)
    
    # Buoc 4: Giai thich tung o trong bang
    print("\n[BUOC 4] Giai thich tung o trong bang")
    print("-" * 80)
    
    print("""
Bang 2x2 (khong tinh dong/cot 'All'):
                    is_bestseller=0    is_bestseller=1
has_rating=0             ?                  ?
has_rating=1             ?                  ?

Tung o trong bang:
- [0, 0] = so sach KHONG co rating VA KHONG phai bestseller = 1440
- [0, 1] = so sach KHONG co rating VA LA bestseller = 32
- [1, 0] = so sach CO rating VA KHONG phai bestseller = 1232
- [1, 1] = so sach CO rating VA LA bestseller = 1126
    """)
    
    # Lay gia tri tung o
    val_00 = contingency_table.loc[0, 0]
    val_01 = contingency_table.loc[0, 1]
    val_10 = contingency_table.loc[1, 0]
    val_11 = contingency_table.loc[1, 1]
    
    print("Gia tri cac o:")
    print("- contingency_table.loc[0, 0] = {}".format(val_00))
    print("  (Khong rating, Khong bestseller)")
    print("- contingency_table.loc[0, 1] = {}".format(val_01))
    print("  (Khong rating, Bestseller)")
    print("- contingency_table.loc[1, 0] = {}".format(val_10))
    print("  (Co rating, Khong bestseller)")
    print("- contingency_table.loc[1, 1] = {}".format(val_11))
    print("  (Co rating, Bestseller)")
    
    # Buoc 5: Dong va cot 'All'
    print("\n[BUOC 5] Y nghia cua dong/cot 'All'")
    print("-" * 80)
    
    row_all = contingency_table.loc['All', :]
    col_all = contingency_table.loc[:, 'All']
    
    print("Dong 'All' - Tong theo cot (is_bestseller):")
    print("- All[0] = {} = so sach KHONG phai bestseller (tong)".format(row_all[0]))
    print("- All[1] = {} = so sach LA bestseller (tong)".format(row_all[1]))
    
    print("\nCot 'All' - Tong theo hang (has_rating):")
    print("- All[0] = {} = so sach KHONG co rating (tong)".format(col_all[0]))
    print("- All[1] = {} = so sach CO rating (tong)".format(col_all[1]))
    
    print("\nO [All, All]:")
    print("- contingency_table.loc['All', 'All'] = {}".format(
        contingency_table.loc['All', 'All']
    ))
    print("  Day la TONG SO SACH (3830)")
    
    # Buoc 6: Tính toán tỷ lệ
    print("\n[BUOC 6] Tinh toan ty le tu bang tan suat cheo")
    print("-" * 80)
    
    # Ty le bestseller trong nhom co rating
    ty_le_bs_co_rating = contingency_table.loc[1, 1] / contingency_table.loc[1, 'All']
    print("Ty le bestseller trong nhom CO rating:")
    print("= {} / {} = {:.1%}".format(
        contingency_table.loc[1, 1],
        contingency_table.loc[1, 'All'],
        ty_le_bs_co_rating
    ))
    
    # Ty le bestseller trong nhom khong co rating
    ty_le_bs_khong_rating = contingency_table.loc[0, 1] / contingency_table.loc[0, 'All']
    print("\nTy le bestseller trong nhom KHONG co rating:")
    print("= {} / {} = {:.1%}".format(
        contingency_table.loc[0, 1],
        contingency_table.loc[0, 'All'],
        ty_le_bs_khong_rating
    ))
    
    print("\nSu chech lech:")
    print("= {:.1%} - {:.1%} = {:.1%}".format(
        ty_le_bs_co_rating,
        ty_le_bs_khong_rating,
        ty_le_bs_co_rating - ty_le_bs_khong_rating
    ))
    print("=> Sach co rating co kha nang la bestseller cao hon 45.5 diem phan tram!")
    
    # Buoc 7: Chi-square kiem tra cai nay
    print("\n[BUOC 7] Chi-square test kiem tra: Co sai khac thuc su khong?")
    print("-" * 80)
    print("""
Chi-square test se:
1. Tinh tan suat KY VONG (neu 2 bien doc lap)
2. So sanh voi tan suat QUAN SAT (bang nay)
3. Neu khac biet lon => co moi lien he
    """)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    explain_contingency_table()