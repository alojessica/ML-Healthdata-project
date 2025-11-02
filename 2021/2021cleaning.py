#%%

import pandas as pd 
from ydata_profiling import ProfileReport


df= pd.read_sas("LLCP2021.XPT")

# %%

df.head()
df.info()

# %%

core_mh_vars = ['_STATE', 'CADULT1', 'BIRTHSEX', 'MENTHLTH',  'POORHLTH',  'ADDEPEV3', 'DECIDE', 'DIFFALON']

optional_mh_vars = [                                         
    'ACEDEPRS', 'ACEDRINK', 'ACEDRUGS', 'ACEPRISN', 'ACEDIVRC',         
    'ACEPUNCH', 'ACEHURT1', 'ACESWEAR', 'ACETOUCH', 'ACETTHEM', 'ACEHVSEX'
]

all_my_vars = core_mh_vars + optional_mh_vars

available_vars = [var for var in all_my_vars if var in df.columns]

df_subset = df[available_vars]


#%%
for var in available_vars:
    if var in df_subset.columns:
        print(df_subset[var].value_counts().sort_index())
    else:
        pass

    # 1.0 = Yes
    # 2.0 = No 
    # 7.0 = Don't know / Not sure
    # 9.0 = Refused
# %%

profile = ProfileReport(df_subset, title = 'CDC Data Exploratory Dashboard',
                        html = {'style': {'full_width': True}},
                        minimal=False)
profile.to_file("CDC_report.html")
# %%
