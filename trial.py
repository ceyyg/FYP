import pandas as pd



train = pd.read_csv("dataset/train_labels.csv")
val = pd.read_csv("dataset/val_labels.csv")

#Exploratory data analysis
#checking the dimensions of the dataset
print(train.shape)  
print(val.shape)

#checking the data types of the dataset
print(train.dtypes)
print(val.dtypes)

#checking the statistical summaries of the dataset
print(train.describe())
print(val.describe())

#checking if there is any null value
print(train.isnull().sum())
print(val.isnull().sum())

#checking the counts of the dataset for race and gender features
print(train['race']. value_counts())
print(val['race'].value_counts())
print(train['gender'].value_counts())
print(val['gender'].value_counts())

#grouping race and gender and their collective counts
train_intersec = train.groupby(['race','gender']).size()
print(train_intersec)
print(train_intersec/ len(train) * 100)
val_intersec = val.groupby(['race', 'gender']).size()
print(val_intersec)
print(val_intersec/ len(val) * 100)

