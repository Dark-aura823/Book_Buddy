import pandas as pd

df1 = pd.read_csv("All_Books2.csv")
df2 = pd.read_csv("All_Books1.csv")

combined_df = pd.concat([df1, df2], ignore_index=True)
combined_df.dropna(subset=['Title', 'Genre'], inplace=True)
combined_df.drop_duplicates(subset='Title', inplace=True)
combined_df['Genre'] = combined_df['Genre'].str.strip().str.title()

combined_df.to_csv("cleaned_books.csv", index=False)

print("✅ Done combining books!")
