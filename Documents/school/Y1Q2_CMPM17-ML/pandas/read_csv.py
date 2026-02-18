import pandas as pd

df = pd.read_csv(
    r"C:\Users\thatp\Documents\school\code\CMPM17-ML\pandas\pandas_intro_dataset.csv"
)
pd.set_option("display.max_columns", None)

rows_range = df[21:36][["name", "stars", "price"]]
# print(len(df))
# df.drop_duplicates(ignore_index=True, inplace=True)
# df.dropna(ignore_index=True, inplace=True)
# df.drop(columns="narrator", axis=1)

null_values = df.isna().any(axis=1)


def stars_to_float(star: str):
    star = str(star)
    value = star.split(" ")[0]
    try:
        return float(value)
    except ValueError:
        return 0.0


print(df["stars"].apply(stars_to_float))


# print(df)

# print(rows_range)
