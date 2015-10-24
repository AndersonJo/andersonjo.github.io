import pandas as pd

cats = pd.read_csv('cats.csv')
slope = (sum((cats['Bwt'] - cats['Bwt'].mean()) *
             (cats['Hwt'] - cats['Hwt'].mean())) /
         sum((cats['Bwt'] - cats['Bwt'].mean()) ** 2))

intercept = cats['Hwt'].mean() - slope * cats['Bwt'].mean()
print intercept