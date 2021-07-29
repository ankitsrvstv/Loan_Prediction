import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import pickle

def loan_prediction(credithistory):
    train_df=pd.read_csv("Loan.csv")
    train_df.dtypes
    train_df.isnull().sum()

    train_df['LoanAmount'].fillna(train_df['LoanAmount'].median(),inplace=True)
    train_df['Gender'].fillna(train_df['Gender'].mode()[0],inplace=True)
    train_df['Married'].fillna(train_df['Married'].mode()[0],inplace=True)
    train_df['Dependents'].fillna(train_df['Dependents'].mode()[0],inplace=True)
    train_df['Self_Employed'].fillna(train_df['Self_Employed'].mode()[0],inplace=True)
    train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mode()[0],inplace=True)
    train_df['Credit_History'].fillna(train_df['Credit_History'].mode()[0],inplace=True)
    train_df.isnull().sum()

    train_df=train_df.drop('Loan_ID',axis=1)
    train_df=train_df.drop('Dependents',axis=1)

    train_df['Gender']=np.where(train_df['Gender']=="Male",1,0)
    train_df['Married']=np.where(train_df['Married']=="Yes",1,0)
    train_df['Education']=np.where(train_df['Education']=="Graduate",1,0)
    train_df['Self_Employed']=np.where(train_df['Self_Employed']=="Yes",1,0)

    ordinal_label={k: i for i, k in enumerate(train_df['Property_Area'].unique(),0)}
    train_df['Property_Area']=train_df['Property_Area'].map(ordinal_label)
    train_df['Loan_Status']=np.where(train_df['Loan_Status']=="Y",1,0)

    train_df.columns

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(train_df.drop(labels=['Loan_Status'],axis=1),train_df['Loan_Status'],test_size=0.2,random_state=1)
    x_train.dtypes

    from sklearn.feature_selection import mutual_info_classif
    mutual_info=mutual_info_classif(x_train,y_train)

    mutual_info=pd.Series(mutual_info)
    mutual_info.index=x_train.columns
    mutual_info.sort_values(ascending=False)

    mutual_info.sort_values(ascending=False).plot.bar(figsize=(20,8))

    from sklearn.feature_selection import SelectKBest

    best=SelectKBest(mutual_info_classif,k=2)
    best.fit(x_train,y_train)
    x_train.columns[best.get_support()]

    x_train=x_train.drop(['Gender'],axis=1)
    x_train=x_train.drop(['Married'],axis=1)
    x_train=x_train.drop(['LoanAmount'],axis=1)
    x_train=x_train.drop(['Loan_Amount_Term'],axis=1)
    x_train=x_train.drop(['Property_Area'],axis=1)
    x_train=x_train.drop(['ApplicantIncome'],axis=1)
    x_train=x_train.drop(['CoapplicantIncome'],axis=1)
    x_train=x_train.drop(['Education'],axis=1)
    x_train=x_train.drop(['Self_Employed'],axis=1)

    from sklearn.linear_model import LogisticRegression
    regressor=LogisticRegression()
    regressor.fit(x_train,y_train)

    x_test=np.array(credithistory)
    x_test=x_test.reshape((1,-1))

    return regressor.predict(x_test)[0]

    #pickle.dump(regressor,open('loan.pkl','wb'))



