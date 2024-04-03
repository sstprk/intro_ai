import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.impute as sk

df = pd.read_csv("/Users/sstprk/Desktop/def.csv")

#Question 1 Missing values
def missingVals(daf):
    imp = sk.SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(daf)
    X = daf
    return imp.transform(X)

numeric = pd.DataFrame([df["Month Salary"], df["Debt"], df["Code"]])
numeric_new = missingVals(numeric)


#Question 4 Relationship between Monthly Salary and City
plt.figure()
plt.scatter(df["Month Salary"], df["City"])
plt.show()

#Question 5 Converting categorical values to continuous values
vilnius = 0
kaunas = 0
klaipeda = 0
for idx in range(len(df["City"])):
    if df["City"][idx] == "Vilnius":
        vilnius = vilnius + 1
    
    elif df["City"][idx] == "Kaunas":
        
        kaunas = kaunas +1

    elif df["City"][idx] == "Klaipeda":
        klaipeda = klaipeda +1

vil_rate = vilnius/len(df["City"])*100
kau_rate = kaunas/len(df["City"])*100
kla_rate = klaipeda/len(df["City"])*100

male = 0
female = 0
for idx in range(len(df["City"])):
    if df["Gender"][idx] == 0:
        female = female + 1
    
    elif df["Gender"][idx] == 1:
        
        male = male +1

male_rate = male/len(df["Gender"])*100
female_rate = female/len(df["Gender"])*100

rates = pd.DataFrame([[vil_rate, kau_rate, kla_rate], [male_rate, female_rate]])
correlation = rates.corr()
