import pandas as pd
from pandas.stats.api import ols

cats = pd.read_csv('cats.csv')
slope = (sum((cats['Bwt'] - cats['Bwt'].mean()) *
             (cats['Hwt'] - cats['Hwt'].mean())) /
         sum((cats['Bwt'] - cats['Bwt'].mean()) ** 2))

slope = cats['Bwt'].cov(cats['Hwt']) / cats['Bwt'].var()
print slope

intercept = cats['Hwt'].mean() - slope * cats['Bwt'].mean()
print intercept

print ols(y=cats['Hwt'], x=cats['Bwt'])
