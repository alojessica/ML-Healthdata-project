#%%

import pandas as pd 
import numpy as np



df= pd.read_sas('/Users/aniyahmcwilliams/Downloads/state cleaning/LLCP2021.XPT ')

# %%

df.head()
# %% 
df['_STATE'].tolist()
# %%
from ydata_profiling import ProfileReport
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
#%% 
# attempting to clean up the state variable 
state_replace_map = {1.0: 'Alabama',
                        2.0: 'Alaska',
                        4.0: 'Arizona',
                        5.0: 'Arkansa',
                        6.0: 'California',
                        8.0: 'Colorado',
                        9.0: 'Connecticut',
                        10.0: 'Delaware',
                        11.0: 'District of Columbia',        
                        13.0 :'Georgia', 
                        15.0 :'Hawaii',
                        16.0 :'Idaho',
                        17.0 :'Illinois',
                        18.0 :'Indiana',
                        19.0 :'Iowa',
                        20.0 :'Kansas',
                        21.0 :'Kentucky',
                        22.0 :'Louisiana',
                        23.0 :'Maine',
                        24.0 :'Maryland',
                        25.0 :'Massachusetts',
                        26.0 :'Michigan',
                        27.0 :'Minnesota',
                        28.0 :'Mississippi',
                        29.0 :'Missouri',
                        30.0 :'Montana',
                        31.0 :'Nebraska',
                        32.0 :'Nevada',
                        33.0 :'New Hampshire',
                        35.0: 'New Mexico',
                        34.0: 'New Jersey', 
                        36.0: 'New York',
                        37.0: 'North Carolina',
                        38.0: 'North Dakota',
                        39.0: 'Ohio',
                        40.0: 'Oklahoma',
                        41.0: 'Oregon',
                        42.0: 'Pennsylvania',
                        44.0: 'Rhode Island',
                        45.0: 'South Carolina',
                        46.0: 'South Dakota',
                        47.0: 'Tennessee',
                        48.0: 'Texas',
                        49.0: 'Utah',
                        50.0: 'Vermont', 
                        51.0: 'Virginia', 
                        53.0: 'Washington',
                        54.0: 'West Virginia',
                        55.0: 'Wisconsin',
                        56.0: 'Wyoming',
                        66.0: 'Guam',
                        72.0: 'Puerto Rico',
                        78.0: 'Virgin Islands'  
}

df_subset['_STATE'] = df_subset['_STATE'].map(state_replace_map)

#%%
df_subset.head()



#%%
profile = ProfileReport(df_subset, title = 'CDC Data Exploratory Dashboard',
                        html = {'style': {'full_width': True}},
                        minimal=False)
profile.to_file("CDC_report.html")
# %%
