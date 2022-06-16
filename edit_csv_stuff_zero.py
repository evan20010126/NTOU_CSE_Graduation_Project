import pandas as pd

summary_file_version = 5

df = pd.read_csv(f'Summary_{summary_file_version}st.csv', header=None)

df = df.fillna(0.0)

df.to_csv(
    f'Summary_stuff_zero_{summary_file_version}st.csv', index=False, header=False)
