import pandas as pd

df = pd.read_csv('data/clean/book_recommendations.csv')
print(f'Total rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print()

print('Confidence distribution:')
conf_dist = df['confidence_level'].value_counts()
for level in ['High', 'Medium', 'Low']:
    count = conf_dist.get(level, 0)
    pct = count / len(df) * 100
    print(f'  {level}: {count:,} ({pct:.1f}%)')

print()
print('Similarity >= 0.99:')
high_sim = (df['similarity_score'] >= 0.99).sum()
pct = high_sim / len(df) * 100
print(f'  {high_sim:,} ({pct:.1f}%)')

print()
print('Similarity stats:')
print(f'  Min: {df["similarity_score"].min():.4f}')
print(f'  Median: {df["similarity_score"].median():.4f}')
print(f'  Max: {df["similarity_score"].max():.4f}')
