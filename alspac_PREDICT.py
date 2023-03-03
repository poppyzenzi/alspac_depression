import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pyreadr
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
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
alspac_4k = pd.read_table('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_alspac/mplus_data/4k_alspac_smfq.txt', delim_whitespace=True, header=None)  # this is just test will need to change
alspac_4k.columns = ['y0','y1','y2','y3','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','class','id']
alspac_4k = alspac_4k[['id','class']] #subsetting
alspac_4k.replace('*', np.nan) #replace missing
# merge class with smfq data -
data = pd.merge(df, alspac_4k, on=["id"])
# n8787 unique ids x 11 time points and classes - exclude after 4 time points as this is what model was built on
data = data[data["time"] < 5]

# ===================== EXTRACTING VARS ====================

alspac = pd.read_stata("/Volumes/cmvm/scs/groups/ALSPAC/data/B3421_Whalley_04Nov2021.dta")
als = alspac # call again if need clean

# make alspac have same ids as originally coded in R
als['cidB3421'] = als['cidB3421'].astype(int)
als['IID'] = als['cidB3421'].astype(str).str.cat(als['qlet'], sep='')
als['id'] = als['cidB3421'].astype(str) + als['qlet']
als = als.drop(['cidB3421', 'qlet'], axis=1)
als['id'] = pd.factorize(als['id'])[0] + 1  # makes ids unique and numeric, should be 15,645
als = als.rename(columns={'kz021':'sex', 'c804':'ethnicity'}) # rename some cols
column_to_move = als.pop("id") # moving id to first col
als.insert(0, "id", column_to_move) # insert column with insert(location, column_name, column_value)


# ================= appending PRS scores ==========================
prs_mdd = pd.read_csv('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/prs/prs_alspac_OUT/abcd_mdd_prs_230302.best', sep='\s+')
prs_mdd = prs_mdd[['IID', 'PRS']]
als2 = pd.merge(als, prs_mdd, on='IID', how='left')


# ===============================================================================
# ================== appending class data =======================================


# make ids same object type - both integers
# alspac_4k is a df of ids and classes (8787 x 2)
# merge als with alspac_4k by id [but need to make sure these IDs are the same
alspac_4k['id'] = alspac_4k['id'].astype(int) # only id and class
df = pd.merge(alspac_4k, als2, on=["id"])

# select and clean variables
dem_vars = ['id', 'class']
c_vars = ['ku707b', 'kw6602b', 'f8se126', 'PRS']
b_vars = ['sex', 'kv8618', 'kv8617', 'tb8618', 'tb8619', 'f8fp470', 'AT5_n']
all_vars = dem_vars + c_vars + b_vars
X_vars = b_vars + c_vars

# ==========================================================================================
# ================================= MAKING DESIGN MATRIX ===================================

df = df[all_vars] # select only vars we want

# continuous vars
for c_var in df[c_vars]:
    df[c_var] = pd.to_numeric(df[c_var], errors='coerce') # this makes cont vars numeric and replaces string values with NaN
df[c_vars] = (df[c_vars] - df[c_vars].min()) / (df[c_vars].max() - df[c_vars].min()) # normalising [0,1]


# binary vars restrict to [0,1,NaN]
df['sex'] = df['sex'].replace(['Female','Male'], [1,0]) # first recode sex
df[b_vars] = df[b_vars].mask(~df[b_vars].isin([0,1]))

# check if features are between 0 and 1 or NaN
for var in X_vars:
    if df[var].notnull().all():
        if (df[var] >= 0) & (df[var] <= 1).all():
            pass
        else:
            print(f"Some non-NaN values in {feature} are not between 0 and 1.")
    else:
        print(f"{var} contains NaN values.")

# all variables are now in df[X_vars]

# testing PRS association
# ========================== using statsmodels =====================================

# Filter out rows with missing values in 'prs_mdd' and 'class' columns
df2 = df.dropna(subset=['PRS', 'class'])
df2.loc[:, 'class'] = df2['class'].replace(3.0, 0.0) # replace with 0 for ref level
x = df2['PRS']
y = df2['class']
X = sm.add_constant(x)
model = sm.MNLogit(y, X)
result = model.fit()
print(result.summary())


# ========================== using scikit =====================================


# Filter dataframe to remove null values in both 'PRS' and 'class' columns
filtered_df = df.dropna(subset=['PRS', 'class'])

# Create x and y arrays from the filtered dataframe
x = np.array(filtered_df['PRS']).reshape(-1, 1)
y = np.array(filtered_df['class'])

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(x, y)

print("Coefficients: ", clf.coef_)
print("Intercepts: ", clf.intercept_)
print("Odds Ratios: ", np.exp(clf.coef_))
print("Classes: ", clf.classes_)


# ======================================================================================
# ======================================================================================

# mulitnom logistic regression
# Separate input and target variables, split
X = df[X_vars].dropna()
y = np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(X, y)
y_pred = logreg.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

