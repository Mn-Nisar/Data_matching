import pandas as pd
import os

def read_dataframe(file_path):
    ext = os.path.splitext(file_path)[-1].lower()    
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def clean_df(df):

    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df

def merge_dataframe(df_input, df_org):

    df_input.iloc[:, -1] = df_input.iloc[:, -1].fillna(df_input.iloc[:, -2])

    df = df_input.merge(df_org, how="left",  left_on="Supplier", right_on="Organization")

    return df


