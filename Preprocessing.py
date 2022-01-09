import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('weatherAUS.csv')

# print(df.shape)
# print(df.columns)
# print(df['RainTomorrow'].value_counts())
# print(df.head())
# print(df.dtypes)

colna = {
    "column": [],
    "percent_NA": []

}
num_rows = df.shape[0]
for col in df.columns:
    colna["column"].append(col)
    colna["percent_NA"].append(100 * (df[col].isna().sum() / len(df[col])))

NA_df=pd.DataFrame(colna)
NA_df.sort_values(["percent_NA"],inplace=True,ascending=False)
print(NA_df)


conv = ['Cloud3pm','Cloud9am']
for col in conv:
    # df[col]=df[col].apply(cnv)
    median_val=df[col].median()
    df[col].fillna(value=median_val,inplace=True)


drop_features = ['Sunshine','Evaporation','Cloud3pm','Cloud9am']
df.drop(columns=drop_features,inplace=True)
print(df.shape)
df.dropna(inplace=True)
# df.drop(columns=['Date'],inplace=True)
df['Date'] = pd.to_datetime(df['Date']) # parse as datatime

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df[['Year', 'Month', 'Day']] # preview changes made

df.drop('Date', axis=1, inplace = True)
print(df.shape)
# ======================================================================================================================
# LabelEncoding
# ======================================================================================================================

label_enc = ['Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']

LE = LabelEncoder()
for i in label_enc:
    df[i] = LE.fit_transform(df[i])
# print(df.dtypes)

df.to_csv('clean_weatherAUS.csv')

# ======================================================================================================================
# Oversampling Randomly
# ======================================================================================================================

rmos = RandomOverSampler(sampling_strategy=0.65,random_state=42)

x_res, y_res = rmos.fit_resample(df.drop(columns=['RainTomorrow']),df['RainTomorrow'])
print(y_res.value_counts())

x_res['RainTomorrow'] = y_res
x_res.to_csv('random_oversample.csv')

# ======================================================================================================================
# Oversampling SMOTE
# ======================================================================================================================

smos = SMOTE(sampling_strategy=0.65,random_state=42)

x_ress, y_ress = smos.fit_resample(df.drop(columns=['RainTomorrow']),df['RainTomorrow'])

print(y_ress.value_counts())

x_ress['RainTomorrow'] = y_ress
x_ress.to_csv('smote_oversample.csv')



