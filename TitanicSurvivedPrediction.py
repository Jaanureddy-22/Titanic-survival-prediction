#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("Titanic-Dataset.csv")
data=pd.DataFrame(data)
data.head()
data.isnull().sum()
data.info()
data.describe()
data=data.drop(columns=['PassengerId','Name','Ticket','Cabin'])
data.info()
data.duplicated().sum()
data['Age']=data['Age'].fillna(int(data['Age'].mean()))
data['Embarked']=data['Embarked'].fillna(data['Embarked'].mode()[0])
data.isnull().sum()
sn.barplot(x=data['Survived'],y=data['Survived'].value_counts(),color="skyblue")
plt.title('survivalcount',fontsize=12,color='red')
plt.xlabel("survived")
plt.ylabel("count")
plt.show()
sn.countplot(x='Survived', hue="Sex",data=data,color="skyblue") #palette={"male":"skyblue","female":"pink"})
plt.title("survival count by gender",color="wheat")
sn.countplot(x=("Survived"),hue="Pclass",data=data,palette="viridis")
plt.title("survival counts by pclass")
sn.set(font="serif")

plt.figure(figsize=(8,6))
sn.histplot(data['Age'],bins=15,kde=True,color='deeppink')
plt.title("Age Distribution",fontsize=14,color="violet")
sn.set(font="serif")

plt.figure(figsize=(8,6))
sn.heatmap(data.drop(columns=['Sex','Embarked']).corr(), annot= True, cmap='PiYG', fmt= '.2f')
plt.title("Correlation Matrix", fontsize = 16 , color = 'red')
plt.tight_layout()
plt.show()

data.columns


data['Sex']=LabelEncoder().fit_transform(data['Sex'])
data['Embarked']=LabelEncoder().fit_transform(data['Embarked'])

x=data.drop(columns=['Survived'])
x.head()

y=data['Survived']
y.head()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train
x_test
y_train
y_test

scaler=StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#import  numpy.random import logistic
logistic_model=LogisticRegression()
logistic_model.fit(x_train,y_train)

logistic_model.score(x_test,y_test)  #testing data

logistic_model.score(x_train,y_train)  #trained data

y_pred=logistic_model.predict(x_test)  #prediction
accuracy=accuracy_score(y_test,y_pred)  #accuracy rate of predicted value
report=classification_report(y_test,y_pred)

print(f"accuaracy score is {accuracy*100:.2f}%")

print("classifiaction report is :\n",report)