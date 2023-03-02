import os
import pandas as pd
import pyreadr
from pathlib import Path

# prepping alspac for lcga analysis in Mplus

# reading in smfq data, previously cleaned and scores calculated in R script
# store in scratch place when running on EDDIE
df = pyreadr.read_r('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_alspac/alspac_dep_long.rds')
df = df[None]
df['id'].nunique() #check 15,645

# convert to wide - remember to only use first 4 time points in mplus
df = df[['id','time','dep']]
df_wide = pd.pivot(df, index=['id'], columns='time', values='dep') #should have 15,645 rows
df_wide = df_wide.fillna('-9999') #replace NaNs with -9999 for mplus
filepath = Path('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_alspac/mplus_data/alspac_smfq_wide_python.txt')
filepath.parent.mkdir(parents=True, exist_ok=True)
df_wide.to_csv(filepath, header=False, sep='\t')  #save wide data for mplus in gmm_abcd directory