#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv("/home/chandraatul1022/Capstone_project_2025/Customer_Segmentation_Train.csv")
df_test = pd.read_csv("/home/chandraatul1022/Capstone_project_2025/Test_Customer_Segmentation_.csv")

df=df_train.copy()
df.head()


# In[22]:


print("\n Missing Data Analysis - Training data")
print("-" * 25)
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df_train)) * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percentage
}).sort_values('Missing Count', ascending=False)

print(missing_summary)


# In[23]:


# Handle missing values
from sklearn.impute import SimpleImputer

num_cols = ['Work_Experience', 'Family_Size']
cat_cols = ['Ever_Married', 'Graduated', 'Profession', 'Var_1']

num_imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
cat_imputer = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')

df[num_cols] = num_imputer.fit_transform(df[num_cols])

df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("Missing Data Analysis - Training data\n\n", df.isnull().sum())


# In[24]:


segmentation = df["Segmentation"].value_counts()

for segment, count in segmentation.items():
   percentage = (count / df.shape[0]) * 100
   print(f"Segment {segment}: Count: {count}, Percentage: {percentage:.2f}%")


#ploting graph
fig, ax = plt.subplots()
ax.pie(
    segmentation,
    labels=segmentation.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#4e79a7', '#59a14f', '#f28e2c', '#e15759']  # New color palette
)

ax.axis('equal')

plt.title('Segment Distribution')
plt.show()   


# In[ ]:


#Observations on the Target Variable
#There is no significant difference between the segments — only Segment D has about 4% more instances than the others.


# In[ ]:


# Next step 
#   - Feature selection - Feature coorelation with Segmentation


# In[25]:


# data cleaning

def data_cleaning(df):
 df = df.drop(['ID'], axis=1)

 df['Work_Experience'] = df['Work_Experience'].fillna(0)
 df['Ever_Married'] = df['Ever_Married'].fillna("No")
 df['Graduated'] = df['Graduated'].fillna("No")

 df.drop_duplicates(inplace=True)
 df.dropna(inplace=True)

 df['Var_1'] = df['Var_1'].str.replace('Cat_', '').astype(int)

 df['Work_Experience'] = df['Work_Experience'].astype(int)
 df['Family_Size'] = df['Family_Size'].astype(int)
 return df

df = data_cleaning(df)
df.info()
df.head()


# In[27]:


# Graphs

plt.figure(figsize=(10, 6))
plt.hist(df["Family_Size"], bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Family size")
plt.ylabel("Number of Customers")
plt.title("Histogram of Family size")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["Profession"], bins=20, color='lightblue', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Profession")
plt.ylabel("Number of Customers")
plt.title("Histogram of Profession")
plt.show()

plt.figure(figsize=(6, 6))
plt.hist(df["Spending_Score"], bins=20, color='lightyellow', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Spending_Score")
plt.ylabel("Number of Customers")
plt.title("Spending_Score")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["Work_Experience"], bins=20, color='orange', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Work_Experience")
plt.ylabel("Number of Customers")
plt.title("Work_Experience")
plt.show()


# In[28]:


gender_value = df["Gender"].value_counts()

fig, ax = plt.subplots()
ax.pie(
    gender_value,
    labels=gender_value.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightblue', 'lightgreen']  # New color palette
)

ax.axis('equal')

plt.title('Gender Distribution')
plt.show()  


graduated_value = df["Graduated"].value_counts()

fig, ax = plt.subplots()
ax.pie(
    graduated_value,
    labels=graduated_value.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightblue', 'lightgreen']  # New color palette
)

ax.axis('equal')

plt.title('Graduated Distribution')
plt.show()  


married_value = df["Ever_Married"].value_counts()

fig, ax = plt.subplots()
ax.pie(
    married_value,
    labels=married_value.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightblue', 'lightgreen']  # New color palette
)

ax.axis('equal')

plt.title('Ever_Married')
plt.show() 

plt.figure(figsize=(10, 6))
plt.hist(df["Age"], bins=50, color='orange', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.title("Age")
plt.show()


# In[29]:


# label encoding 

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
        
df.info()
df.head()        


# In[30]:


# correlation of feartures with Segmentation 

corr_matrix = df.corr()
corr_matrix["Segmentation"].sort_values(ascending=False)


# In[31]:


df_plot=df.copy()


profession_labels = {
    1: "Engineer", 2: "Doctor", 3: "Artist", 
    4: "Lawyer", 5: "Executive", 6: "Marketing", 7: "Other"
}
df_plot['Profession'] = df_plot['Profession'].map(profession_labels)



pd.crosstab(df_plot['Profession'], df_plot['Segmentation']).plot(
    kind='bar',
    figsize=(10,6),
    colormap='viridis'
)

plt.title("Customer Segmentation by Profession")
plt.xlabel("Profession")
plt.ylabel("Number of Customers")
plt.legend(title="Segmentation")
plt.show()


# In[32]:


scaler = StandardScaler() # implement after spliting
scaler.fit(df)
data_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(data_scaled, columns=df.columns)
df_scaled.head()


# In[12]:


plt.figure(figsize=(10, 6))
plt.hist(df["Family_Size"], bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Family size")
plt.ylabel("Number of Customers")
plt.title("Histogram of Family size")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["Profession"], bins=20, color='lightblue', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Profession")
plt.ylabel("Number of Customers")
plt.title("Histogram of Profession")
plt.show()

plt.figure(figsize=(6, 6))
plt.hist(df["Spending_Score"], bins=20, color='lightyellow', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Spending_Score")
plt.ylabel("Number of Customers")
plt.title("Spending_Score")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["Work_Experience"], bins=20, color='orange', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Work_Experience")
plt.ylabel("Number of Customers")
plt.title("Work_Experience")
plt.show()


# In[33]:




plt.figure(figsize=(10, 6))
plt.hist(df["Age"], bins=50, color='orange', edgecolor='darkgreen', alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.title("Age")
plt.show()


# In[52]:


# Observations - 

# Spliting the data set 

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


#X = df.drop(['Segmentation'],axis = 1)
y = df['Segmentation']
X = df


# In[54]:






#X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size = 0.2 ,random_state = 42)

X_train , X_test , y_train , y_test = train_test_split(df[X] ,df[y] ,test_size = 0.2 ,random_state = 42)


# In[ ]:


X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(df[features], df[target_rul], test_size=0.2, random_state=42)


# In[51]:


from sklearn.linear_model import LogisticRegression

# Softmax regression model
softmax_reg = LogisticRegression(
    multi_class='multinomial',   # enables softmax instead of one-vs-rest
    solver='lbfgs',              # recommended solver for softmax
    max_iter=100,               # increase if convergence warning appears
    C = 10,
    random_state=42
)


print(X_train)
print(y_train)

# Fit the model
softmax_reg.fit(X_train, y_train)


# Predict class labels
y_pred = softmax_reg.predict(X_test)

# Predict class probabilities (softmax output)
y_prob = softmax_reg.predict_proba(X_test)

softmax_reg.predict([[0,1,38,1,2,2,0,3,4]])
softmax_reg.predict_proba([[0,1,38,1,2,2,0,3,4]])


# In[39]:


from sklearn.model_selection import cross_val_score, KFold

scores = cross_val_score(softmax_reg, X, y, cv=10, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))


# In[ ]:




