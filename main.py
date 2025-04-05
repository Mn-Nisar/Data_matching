import pandas as pd
import numpy as np
import os
from utils import read_dataframe, clean_df, merge_dataframe
from llms import get_orgs_using_llm


INPUT_FILE_PATH="input/input.csv"
ORG_FILE_PATH="input/org_data.csv"
FACTOR_FILE_PATH="input/factor_data.csv"
RESULT_PATH="output/output.csv"
 

if __name__ == '__main__':

    df_input = read_dataframe(INPUT_FILE_PATH)
    df_org = read_dataframe(ORG_FILE_PATH)
    df_factor = read_dataframe(FACTOR_FILE_PATH)

    # Clean the dataframes
    df_input = clean_df(df_input)
    df_org = clean_df(df_org)
    df_factor = clean_df(df_factor)

    # Merge the dataframes
    df_m = merge_dataframe(df_input, df_org)

    unmatched_sup = df_m.loc[df_m["Organization"].isna(), "Supplier"].unique()

    unmatched_orgs = df_org[~df_org["Organization"].isin(df_input["Supplier"])]
    unmatched_orgs = unmatched_orgs["Organization"].unique()
    
    # using LLM find the relavent  Organization
    orgs = get_orgs_using_llm(unmatched_sup, unmatched_orgs, confidence=0.8)

    
    # Update the dataframe with the matched organizations
    for sup, org in orgs.items():
        df_m.loc[df_m["Supplier"] == sup, "Organization"] = org
        df_m.loc[df_m["Supplier"] == sup, "Sectors"] = df_org.loc[df_org["Organization"] == org, "Sectors"].values[0]
    
    df_m.to_csv(RESULT_PATH)




