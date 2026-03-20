import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

#grouping race and gender and their collective counts and percentage composition
train_intersec = train.groupby(['race','gender']).size()
print(train_intersec)
print(train_intersec/ len(train) * 100)
val_intersec = val.groupby(['race', 'gender']).size()
print(val_intersec)
print(val_intersec/ len(val) * 100)

#visualising the race distribution by gender into bar chart
plot = train_intersec.reset_index(name ='count')
plt.figure(figsize= (12,6))
sns.barplot(data = plot, x ='race', y='count', hue = 'gender')
plt.title('Distribution of Race & Gender')
plt.xlabel('Race')
plt.ylabel('Counts')
plt.show()


