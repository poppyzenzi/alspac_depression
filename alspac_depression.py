import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pyreadr
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import scipy.stats as stats
import statsmodels.api as sm
from pathlib import Path

# reading in smfq data, previously cleaned and scores calculated in R script
df = pyreadr.read_r('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_alspac/alspac_dep_long.rds')
print(df.keys()) #check which objects we have: there is only None
df = df[None]
df['id'].nunique() #check 15,645
# df cols are : ['id', 'sex', 'ethnicity', 'time', 'age', 'dep']

# convert to wide - remember to only use first 4 time points in mplus
df = df[['id','time','dep']]
df_wide = pd.pivot(df, index=['id'], columns='time', values='dep') #should have 15,645 rows
df_wide = df_wide.fillna('-9999') #replace NaNs with -9999 for mplus
filepath = Path('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_alspac/mplus_data/alspac_smfq_wide_python.txt')
filepath.parent.mkdir(parents=True, exist_ok=True)
df_wide.to_csv(filepath, header=False, sep='\t')  #save wide data for mplus in gmm_abcd directory

# =================================================
# make new df with id, unique id, Xvars and yclass
alspac_4k = pd.read_table('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/'
                        'Edinburgh/gmm/gmm_alspac/mplus_data/4_class_alspac_test.txt', delim_whitespace=True, header=None)  # this is just test will need to change
alspac_4k.columns = ['y0','y1','y2','y3','id','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','class']
alspac_4k = alspac_4k[['id','class']] #subsetting
alspac_4k.replace('*', np.nan) #replace missing
# merge class with smfq data -
data = pd.merge(df, alspac_4k, on=["id"])
# n8787 unique ids x 11 time points and classes - exclude after 4 time points as this is what model was built on
data = data[data["time"] < 5]

# ===================== EXTRACTING VARS ====================

alspac = pd.read_stata("/Volumes/cmvm/scs/groups/ALSPAC/data/B3421_Whalley_04Nov2021.dta")
als = alspac

# make alspac have same ids as originally coded in R
als['id'] = als['cidB3421'].astype(str) + als['qlet']
als = als.drop(['cidB3421', 'qlet'], axis=1)
als['id'] = pd.factorize(als['id'])[0]  # makes ids unique and numeric, should be 15,645
als = als.rename(columns={'kz021':'sex', 'c804':'ethnicity'}) # rename some cols

# append class and id data to the whole alspac dataframe
# alspac_4k is a df of ids and classes (8787 x 2)
# merge als with alspac_4k by id [but need to make sure these IDs are the same

als = als.rename(columns={'kz021':'sex', 'c804':'ethnicity'}) # renaming some cols

als['id'].nunique() # again check 15,645 unique id's
column_to_move = als.pop("id") # moving id to first col
als.insert(0, "id", column_to_move) # insert column with insert(location, column_name, column_value)
# make ids same object type - both integers
alspac_4k['id'] = alspac_4k['id'].astype(int) # only id and class
# now merge by id .. should be 8787
df = pd.merge(alspac_4k, als, on=["id"])

# select and clean variables
dem_vars = ['id', 'class']
c_vars = ['ku707b', 'kw6602b', 'f8se126']
b_vars = ['sex', 'kv8618', 'kv8617', 'tb8618', 'tb8619', 'f8fp470', 'AT5_n']
all_vars = dem_vars + c_vars + b_vars
X_vars = b_vars + c_vars

df = df[all_vars] # select only vars we want

# continuous vars
for c_var in df[c_vars]:
    df[c_var] = pd.to_numeric(df[c_var], errors='coerce') # this makes cont vars numeric and replaces string values with NaN
df[c_vars] = (df[c_vars] - df[c_vars].min()) / (df[c_vars].max() - df[c_vars].min()) # normalising [0,1]

# binary vars restrict to [0,1,NaN]
df['sex'] = df['sex'].replace(['Female','Male'], [1,0]) # first recode sex
df[b_vars] = df[b_vars].mask(~df[b_vars].isin([0,1]))

# check if features are between 0 and 1 or NaN
for c_var in c_vars:
    if df[c_var].dropna().between(0, 1).all():
        print('ok')
    else:
        print(f"Some non-NaN values in {c_var} are not between 0 and 1.")

# all variables are now in df[X_vars]

# mulitnom logistic regression
# Separate input and target variables, split
X = df[X_vars].dropna()
x = np.array(df['sex'].dropna()).reshape(-1,1) # test with sex
y = np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(x, y)
y_pred = logreg.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


#first use anthropomorphic data to assign unique numeric ID's
anth = pyreadr.read_r('/Volumes/igmm/GenScotDepression/data/abcd/release4.0/iii.data/Physical_Health/abcd_ant01.rds')
anth = anth[None] # let's check what objects we have: there is only None
anth = anth[['src_subject_id','eventname', 'interview_age', 'sex']]
anth.columns = ['id','time','age','sex']
anth['time'] = anth['time'].replace(['baseline_year_1_arm_1','1_year_follow_up_y_arm_1',
                                 '2_year_follow_up_y_arm_1','3_year_follow_up_y_arm_1'],
                                [0,1,2,3])
anth['sex'] = anth['sex'].replace(['F','M'], [1,0])
anth['age'] = anth['age'].div(12)
anth['id'].nunique() #check 11,876
anth['unique_id'] = anth.id.map(hash)

#=============CBCL data===============
df = pyreadr.read_r('/Volumes/igmm/GenScotDepression/data/abcd/release4.0/iii.data/Mental_Health/abcd_cbcls01.rds')
df = df[None]
df = df[['src_subject_id','eventname','cbcl_scr_dsm5_depress_r']]
df.columns = ['id','time','dep']
df['time'] = df['time'].replace(['baseline_year_1_arm_1', '1_year_follow_up_y_arm_1',
                                     '2_year_follow_up_y_arm_1', '3_year_follow_up_y_arm_1'],
                                    [0, 1, 2, 3])
df['id'].nunique() #check 11,876. Should be same but we merge with anth to be certain


#merging into long format df
data = pd.merge(anth, df, on=["id","time"])

#==========now convert data to wide for mplus=============
#first keep only relevant cols
data = data[['id','unique_id','time','dep']]
data_wide = pd.pivot(data, index=['id','unique_id'], columns='time', values='dep') #should have 11,876 rows
data_wide[[0,1,2,3]] = data_wide[[0,1,2,3]].fillna('-9999') #replace NaNs with -9999 for mplus
filepath = Path('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_abcd/mplus_data/abcd_cbcl_wide_python.txt')
filepath.parent.mkdir(parents=True, exist_ok=True)
data_wide.to_csv(filepath, header=False)  #save wide data for mplus in gmm_abcd directory

#=================================================