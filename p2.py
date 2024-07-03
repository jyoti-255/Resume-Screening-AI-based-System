#import lib
import pandas as pd

data=pd.read_csv("Resume Job Recommendation.csv")
print(data.head())

print(data.columns)

print(data['Work Type'].value_counts())
