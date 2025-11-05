#%%

import pandas as pd 
import numpy as np
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

replace_map = {
    'BIRTHSEX': {  
        1: 'Male',
        2: 'Female',
        7: np.nan,
        9: np.nan
    },
    'MENTHLTH': {
        88: 0,
        77: np.nan,
        99: np.nan
    },
    'POORHLTH': {
        88: 0,
        77: np.nan,
        99: np.nan
    },
    'ADDEPEV3': { 
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    },
    'DECIDE': { 
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    },
    'DIFFALON': {
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    },
        
    'ACEDEPRS': {
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    },
    'ACEDRINK': { 
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    },
    'ACEDRUGS': {
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    },
    'ACEPRISN': { 
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    },
    'ACEDIVRC': { 
        1: 'Yes',
        2: 'No',
        8: 'Parents not married',
        7: np.nan,
        9: np.nan
    },
    'ACEPUNCH': {
        1: 'Never',
        2: 'Once',
        3: 'More than once',
        7: np.nan,
        9: np.nan
    },
    'ACEHURT1': {
        1: 'Never',
        2: 'Once',
        3: 'More than once',
        7: np.nan,
        9: np.nan
    },
    'ACESWEAR': { 
        1: 'Never',
        2: 'Once',
        3: 'More than once',
        7: np.nan,
        9: np.nan
    },
    'ACETOUCH': { 
        1: 'Never',
        2: 'Once',
        3: 'More than once',
        7: np.nan,
        9: np.nan
    },
    'ACETTHEM': {
        1: 'Never',
        2: 'Once',
        3: 'More than once',
        7: np.nan,
        9: np.nan
    },
    'ACEHVSEX': {
        1: 'Never',
        2: 'Once',
        3: 'More than once',
        7: np.nan,
        9: np.nan
    }
}

df_subset = df_subset.replace(replace_map)

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
