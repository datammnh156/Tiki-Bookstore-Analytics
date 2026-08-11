import pandas as pd

df = pd.read_csv('data/raw/tiki_book_dataset_crawl.csv')
print( "So luong dong trong file: ", len(df))

df = df.drop_duplicates(subset=['id'], keep='first')
print("So luong dong sau khi loai bo trung lap: ", len(df))

#Nhieu sach chua co ai danh gia nen API Tiki tra rating_average = 0
sosach_rating_0 = (df[df['rating_average'] == 0].shape[0])
print("So luong sach co rating = 0: ", sosach_rating_0)
df['has_rating'] = (df['rating_average'] > 0).astype(int)
print(df['has_rating'].value_counts())

#Tao nhan is_bestseller: top 30% quantity_sold trong tung category
df['threshold'] = df.groupby('category_name')['quantity_sold'].transform(lambda x: x.quantile(0.7))

df['is_bestseller'] = (df['quantity_sold'] >= df['threshold']).astype(int)
df.drop(columns=['threshold'], inplace=True)

#So sanh ti le BestSeller giua nhom sach co va 0 co rating
books_with_rating = df[df['has_rating'] >0]
books_without_rating = df[df['has_rating'] == 0]

bestseller_with_rating = books_with_rating['is_bestseller'].mean()*100
bestseller_without_rating = books_without_rating['is_bestseller'].mean()*100

print("Sach co rating co ti le BestSeller : ", f"{bestseller_with_rating:.1f}%")
print("Sach khong co rating co ti le BestSeller : ", f"{bestseller_without_rating:.1f}%")

#Luu du lieu da xu ly ra file csv
df.to_csv("data/clean/tiki_books_cleaned.csv", index=False, encoding="utf-8-sig")
print("Da luu file sach:", len(df), "dong")