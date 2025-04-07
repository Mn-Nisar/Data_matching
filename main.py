import pandas as pd
import numpy as np
import os
from utils import read_dataframe, clean_df, merge_dataframe
from llms import get_orgs_using_llm


INPUT_FILE_PATH="input/input.csv"
ORG_FILE_PATH="input/org_data.csv"
FACTOR_FILE_PATH="input/factor_data.csv"
RESULT_PATH="output/output.csv"
FINAL_COLUMNS = ["Subcategory","Subcategory 2","Supplier","Organization","Sectors","Factor Id","Factor Name"]


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

    unmatched_sup = df_m.loc[df_m["Organization"].isna(), "Supplier"].dropna().unique()

    unmatched_orgs = df_org[~df_org["Organization"].isin(df_input["Supplier"])]
    unmatched_orgs = unmatched_orgs["Organization"].dropna().unique()
    
    # using LLM find the relavent  Organization
    orgs = get_orgs_using_llm(unmatched_sup, unmatched_orgs, confidence=0.8)

    # Update the dataframe with the matched organizations
    for sup, org in orgs.items():
        df_m.loc[df_m["Supplier"] == sup, "Organization"] = org
        df_m.loc[df_m["Supplier"] == sup, "Sectors"] = df_org.loc[df_org["Organization"] == org, "Sectors"].values[0]


    # ========matching with factors data=====

    df_m = df_m.merge(df_factor, how="left", left_on="Activities", right_on="Factor Name")

    unmatched_activity = df_m.loc[df_m["Factor Name"].isna(), "Activities"].dropna().unique()

    unmatched_factor  = df_factor[~df_factor["Factor Name"].isin(df_m["Activities"])]
    unmatched_factor = unmatched_factor["Factor Name"].dropna().unique()
    

    factors = get_orgs_using_llm(unmatched_activity, unmatched_factor, confidence=0.8)

    for act, fact in factors.items():
        df_m.loc[df_m["Activities"] == act, "Factor Name"] = fact
        df_m.loc[df_m["Activities"] == act, "Factor Id"] = df_factor.loc[df_factor["Factor Name"] == fact, "Factor Id"].values[0]


    df_m[FINAL_COLUMNS].to_csv(RESULT_PATH)




