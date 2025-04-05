import pandas as pd
from utils import  clean_df

ORG_FILE_PATH="input/org_data.csv"
FACTOR_FILE_PATH="input/factor_data.csv"

df_org = clean_df(pd.read_csv(ORG_FILE_PATH))
df_factor =  clean_df(pd.read_csv(FACTOR_FILE_PATH))

df = df_factor.merge(df_org,left_on="Factor Name", right_on="Activities")

df.to_csv("factor_with_Activities.csv")

df = df_factor.merge(df_org,left_on="Factor Name", right_on="Sectors")

df.to_csv("factor_with_Sectors.csv")


df = df_factor.merge(df_org,left_on="Factor Name", right_on="Industries")

df.to_csv("factor_with_Industries.csv")


df = df_factor.merge(df_org,left_on="Factor Name", right_on="Primary activity")

df.to_csv("factor_with_Primary activity.csv")