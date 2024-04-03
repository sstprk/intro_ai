#Salih Toprak
#Introduction to AI Lab Work 1

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing

#Importing dataset
df = pd.DataFrame(pd.read_csv("/Users/sstprk/Desktop/KTU/Introduction to Artificial Intelligence/Datasets/Heart Attack.csv"))
df_1 = df.copy()

#Data frame info
print(df)

def printInfoNum(column):
    #Function for printing information of numeric values

    print("Information of ",column.name)
    print("Total number of values: ", len(column), 
        " \nCardinality: ", column.nunique(), 
        " \nMin: ", column.min(), 
        " \nMax: ", column.max(), 
        " \nQ1-Q3: ", np.quantile(column, [0.25, 0.75]),
        " \nAverage: ", np.average(column),
        " \nMedian: ", np.median(column),
        " \nStandart Dev.: ", np.std(column), "\n\n")
        
def printInfoCat(column):
    #Function for printing information of categorical values

    print("Information of ",column.name)
    print("Total number of values: ", len(column),
          "\nCardinality: ", column.nunique(), 
          "\nMode1: ", column.mode(),
          "\nFrequency Mode1: ", column.value_counts()[0],
          "\nPercentage Mode1: ", column.value_counts()[0]/len(column)*100,
          "\nFrequency Mode2: ", column.value_counts()[1],
          "\nPercentage Mode2: ", column.value_counts()[1]/len(column)*100, "\n\n")
    
#Printing the info of numeric attributes
printInfoNum(df["age"])
printInfoNum(df["impluse"])
printInfoNum(df["pressurehight"])
printInfoNum(df["pressurelow"])
printInfoNum(df["glucose"])
printInfoNum(df["kcm"])
printInfoNum(df["troponin"])

#Printing the info of categorical attributes
printInfoCat(df["gender"])
printInfoCat(df["class"])

#Histograms of attributes
plt.figure()
plt.subplot(3,3,1)

plt.hist(df["age"])
plt.title("age")
plt.grid(visible=True)

plt.subplot(3,3,2)

plt.hist(df["gender"])
plt.title("gender")
plt.grid(visible=True)

plt.subplot(3,3,3)

plt.hist(df["impluse"])
plt.title("impluse")
plt.grid(visible=True)

plt.subplot(3,3,4)

plt.hist(df["pressurehight"])
plt.title("pressurehight")
plt.grid(visible=True)

plt.subplot(3,3,5)

plt.hist(df["pressurelow"])
plt.title("pressurelow")
plt.grid(visible=True)

plt.subplot(3,3,6)

plt.hist(df["glucose"])
plt.title("glucose")
plt.grid(visible=True)

plt.subplot(3,3,7)

plt.hist(df["kcm"])
plt.title("kcm")
plt.grid(visible=True)

plt.subplot(3,3,8)

plt.hist(df["troponin"])
plt.title("troponin")
plt.grid(visible=True)

plt.subplot(3,3,9)

plt.hist(df["class"])
plt.title("class")
plt.grid(visible=True)

#Removing outliers
q1_impluse = 64
q3_impluse = 85
iqr = q3_impluse - q1_impluse
top_border = q3_impluse + (iqr*1.5)
bottom_border = q1_impluse - (iqr*1.5)

for i in range(len(df_1["impluse"])):
    if (df_1["impluse"][i] < bottom_border) or (df_1["impluse"][i] > top_border):
        df_1["impluse"] = df_1["impluse"].drop(axis="rows", index=i)

q1_troponin = 0.006
q3_troponin= 0.0855
iqr = q3_troponin - q1_troponin
top_border = q3_troponin + (iqr*1.5)
bottom_border = q1_troponin - (iqr*1.5)

for i in range(len(df_1["troponin"])):
    if (df_1["troponin"][i] < bottom_border) or (df_1["troponin"][i] > top_border):
        df_1["troponin"] = df_1["troponin"].drop(axis="rows", index=i)

q1_kcm = 1.655
q3_kcm = 5.805
iqr = q3_kcm - q1_kcm
top_border = q3_kcm + (iqr*1.5)
bottom_border = q1_kcm - (iqr*1.5)

for i in range(len(df_1["kcm"])):
    if (df_1["kcm"][i] < bottom_border) or (df_1["kcm"][i] > top_border):
        df_1["kcm"] = df_1["kcm"].drop(axis="rows", index=i)

#After removing outliers
plt.figure()
plt.subplot(1,3,1)

plt.hist(df_1["impluse"])
plt.title("After removal impluse")
plt.grid(visible=True)

plt.subplot(1,3,2)

plt.hist(df_1["troponin"])
plt.title("After removal troponin")
plt.grid(visible=True)

plt.subplot(1,3,3)

plt.hist(df_1["kcm"])
plt.title("After removal kcm")
plt.grid(visible=True)

#Scatter plots of some attributes
plt.figure()
plt.subplot(2,2,1)

plt.scatter(df_1["kcm"], df_1["troponin"])
plt.grid(visible=True)
plt.xlabel("kcm")
plt.ylabel("troponin")

plt.subplot(2,2,2)

plt.scatter(df_1["age"], df_1["glucose"])
plt.grid(visible=True)
plt.xlabel("age")
plt.ylabel("troponin")

plt.subplot(2,2,3)

plt.scatter(df["pressurelow"], df["pressurehight"])
plt.grid(visible=True)
plt.xlabel("pressurelow")
plt.ylabel("pressurehight")

plt.subplot(2,2,4)

plt.scatter(df["pressurelow"], df["age"])
plt.grid(visible=True)
plt.xlabel("pressurelow")
plt.ylabel("age")

#SPLOM Diagram
df_numeric = df_1.copy()
df_numeric = df_numeric.drop("gender", axis="columns")
df_numeric = df_numeric.drop("class", axis="columns")

scatter_matrix(df_numeric, figsize=(6,6), diagonal="kde")

#Bar plots of categorical attribute
plt.figure()
plt.subplot(1,2,1)

y = [449, 870]
x = ["0(female)", "1(male)"]

plt.bar(x, height=y)
plt.grid(visible=True)
plt.title("gender")

plt.subplot(1,2,2)

y = [509, 810]
x = ["negative", "positive"]

plt.bar(x, height=y)
plt.grid(visible=True)
plt.title("class")

#Boxplots
female_age = df[df["gender"]==0]["age"]
male_age = df[df["gender"]==1]["age"]
data = [female_age, male_age]

fig, ax = plt.subplots()

plt.boxplot(data)
plt.title("gender - age")
plt.xlabel("gender")
plt.ylabel("age")
ax.set_xticklabels(["female", "male"])

positive_glucose = df[df["class"]=="positive"]["glucose"]
negative_glucose = df[df["class"]=="negative"]["glucose"]
data = [positive_glucose, negative_glucose]

fig, ax = plt.subplots()

plt.boxplot(data)
plt.title("class - glucose")
plt.xlabel("class")
plt.ylabel("glucose")
ax.set_xticklabels(["positive", "negative"])

#Histograms
positive_kcm = df[df["class"]=="positive"]["kcm"]
negative_kcm = df[df["class"]=="negative"]["kcm"]

plt.figure()
plt.subplot(1,2,1)

plt.hist(positive_kcm)
plt.hist(negative_kcm)
plt.title("class - kcm")
plt.legend(["positive", "negative"])

male_pressurelow = df[df["gender"]==1]["pressurelow"]
female_pressurelow = df[df["gender"]==0]["pressurelow"]

plt.subplot(1,2,2)

plt.hist(male_pressurelow)
plt.hist(female_pressurelow)
plt.title("gender - pressurelow")
plt.legend(["male", "female"])

#Covariance matrix
cov_matrix = df_numeric.cov()
cov_matrix = cov_matrix
print("Covariance Matrix :\n", cov_matrix, "\n\n")

#Correlation matrix
corr_matrix = df_numeric.corr()
corr_matrix = corr_matrix
print("Correlation Matrix :\n", corr_matrix, "\n\n")

#Heatmap of correlation matrix
plt.figure()
sb.heatmap(corr_matrix ,annot=False, cmap="pink")
plt.title("Correlation Matrix")

plt.show()

#Converting all attributes to numeric
df_converted = df.copy()
temp=[]

for j in range(len(df_converted["class"])):
    if df_converted["class"][j]=="positive":
        temp.append(1)
    else:
        temp.append(0)
df_converted["class"] = temp

print("Converted Dataframe: \n", df_converted)