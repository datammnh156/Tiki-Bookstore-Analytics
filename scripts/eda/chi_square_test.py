"""
Kiem dinh Chi-square de xac dinh moi lien he giua has_rating va is_bestseller.
Chi-square test chi chung minh MOI LIEN HE, KHONG chung minh NHAN QUA.
"""

import pandas as pd
import sys
from scipy.stats import chi2_contingency

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def perform_chi_square_test(data_path='data/clean/tiki_books_cleaned.csv'):
    """
    Thuc hien kiem dinh Chi-square giua has_rating va is_bestseller.
    
    Tham so:
        data_path: duong dan den file du lieu da lam sach
    
    Tra ve:
        dict chua ket qua kiem dinh (chi2, p_value, dof, conclusion)
    """
    
    print("=" * 70)
    print("KIEM DINH CHI-SQUARE: HAS_RATING vs IS_BESTSELLER")
    print("=" * 70)
    
    # Buoc 1: Load du lieu
    print("\n[1/4] Dang load du lieu...")
    df = pd.read_csv(data_path, encoding='utf-8')
    print("Da load {} sach".format(len(df)))
    
    # Kiem tra du lieu co day du khong
    print("\nKiem tra du lieu:")
    print("- So sach co has_rating=1: {}".format(df['has_rating'].sum()))
    print("- So sach co has_rating=0: {}".format(len(df) - df['has_rating'].sum()))
    print("- So sach la bestseller: {}".format(df['is_bestseller'].sum()))
    print("- So sach khong phai bestseller: {}".format(len(df) - df['is_bestseller'].sum()))
    
    # Buoc 2: Tao bang tan suat cheo (contingency table)
    # Hang: has_rating (0 hoac 1)
    # Cot: is_bestseller (0 hoac 1)
    print("\n[2/4] Tao bang tan suat cheo (Contingency Table)...")
    contingency_table = pd.crosstab(
        df['has_rating'], 
        df['is_bestseller'],
        rownames=['has_rating'],
        colnames=['is_bestseller'],
        margins=True  # Them tong hang va cot
    )
    
    print("\nBang tan suat cheo:")
    print(contingency_table)
    
    # Giai thich bang
    print("\nGiai thich:")
    print("- Hang 0: Sach KHONG co rating")
    print("- Hang 1: Sach CO rating")
    print("- Cot 0: Sach KHONG phai bestseller")
    print("- Cot 1: Sach LA bestseller")
    print("- 'All': Tong cong")
    
    # Buoc 3: Chay kiem dinh Chi-square
    # Bo hang/cot 'All' (margins) vi chi2_contingency chi can bang 2x2
    print("\n[3/4] Thuc hien kiem dinh Chi-square...")
    contingency_no_margins = pd.crosstab(df['has_rating'], df['is_bestseller'])
    
    # chi2_contingency tra ve: (chi2, p_value, dof, expected_freq)
    chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_no_margins)
    
    print("\nKet qua kiem dinh:")
    print("- Chi-square statistic: {:.4f}".format(chi2_stat))
    print("- P-value: {:.6f}".format(p_value))
    print("- Degrees of freedom: {}".format(dof))
    
    # In ra tan suat ky vong (expected frequencies)
    print("\nTan suat ky vong (neu 2 bien doc lap):")
    expected_df = pd.DataFrame(
        expected_freq,
        index=['has_rating=0', 'has_rating=1'],
        columns=['is_bestseller=0', 'is_bestseller=1']
    )
    print(expected_df.round(2))
    
    # Buoc 4: Ket luan
    print("\n[4/4] Ket luan...")
    print("=" * 70)
    
    alpha = 0.05
    if p_value < alpha:
        print("KET LUAN: Co MOI LIEN HE co y nghia thong ke")
        print("(p-value = {:.6f} < alpha = {})".format(p_value, alpha))
        print("\nGiai thich:")
        print("- Tu chi so Chi-square cao ({:.4f}), ta bac bo gia thuyet H0".format(chi2_stat))
        print("- H0: has_rating va is_bestseller doc lap voi nhau")
        print("- Ket luan: has_rating va is_bestseller CO lien he voi nhau")
        conclusion = "Co moi lien he co y nghia thong ke"
    else:
        print("KET LUAN: KHONG co moi lien he co y nghia thong ke")
        print("(p-value = {:.6f} >= alpha = {})".format(p_value, alpha))
        print("\nGiai thich:")
        print("- Khong du bang chung de bac bo gia thuyet H0")
        print("- H0: has_rating va is_bestseller doc lap voi nhau")
        conclusion = "Khong co moi lien he co y nghia thong ke"
    
    # LUU Y QUAN TRONG
    print("\n" + "=" * 70)
    print("LUU Y QUAN TRONG VE GIAI HAN CUA KIEM DINH CHI-SQUARE:")
    print("=" * 70)
    print("""
CANH BAO: Chi-square test chi chung minh MOI LIEN HE (association),
KHONG chung minh quan he NHAN QUA (causation).

Ket qua nay CHI co nghia la:
- "has_rating va is_bestseller co lien he voi nhau"

KHONG duoc dien giai thanh:
- "Co rating LAM CHO sach ban chay hon" (SAI!)
- "Sach ban chay VI no co rating" (SAI!)

De chung minh nhan qua, can:
- Thiet ke thi nghiem (A/B testing)
- Phan tich du lieu theo thoi gian (longitudinal study)
- Kiem soat cac bien nhiễu (confounding variables)

Trong nghien cuu nay, chi co the ket luan:
"Sach co rating THUONG xuat hien cung voi trang thai bestseller"
nhung KHONG the ket luan ai la nguyen nhan cua ai.
""")
    
    print("=" * 70)
    
    # Tra ve ket qua
    return {
        'chi2': chi2_stat,
        'p_value': p_value,
        'dof': dof,
        'conclusion': conclusion,
        'contingency_table': contingency_table,
        'expected_freq': expected_df
    }


if __name__ == "__main__":
    # Chay kiem dinh
    result = perform_chi_square_test()
    
    print("\n" + "=" * 70)
    print("TOM TAT KET QUA")
    print("=" * 70)
    print("Chi-square statistic: {:.4f}".format(result['chi2']))
    print("P-value: {:.6f}".format(result['p_value']))
    print("Ket luan: {}".format(result['conclusion']))
    print("\nLuu y: Ket qua chi chung minh moi lien he, khong chung minh nhan qua.")