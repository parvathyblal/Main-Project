
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("train.csv")
df.head()
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[2]:


df.info()


# In[3]:


columns_to_drop = ['age','result','ID','age_desc']
df = df.drop(columns=columns_to_drop)


# In[4]:


df.shape


# In[5]:


df=df.dropna()


# In[6]:


df.shape


# In[7]:


df = df[df != '?'].dropna()


# In[8]:


# Assuming df is your DataFrame
for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values for column {column}:", unique_values)


# In[9]:


maps=[]
def find_category_mappings(data, variable):
    return {k: i for i, k in enumerate(data[variable].unique())}
def integer_encode(df,variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)
for variable in df.columns:
     if df[variable].dtype=="object":
            #print(variable)
            mappings = find_category_mappings(df,variable)
            maps.append(mappings)
            integer_encode(df, variable, mappings)
df.head()


# In[10]:


print(maps)


# In[11]:


file=open("maps.txt","w") #file handling. writing file to text document
file.write(str(maps))
file.close()


# In[12]:


X=df.iloc[:,0:-1]
X.head()


# In[13]:


y=df['Class/ASD']
y.head()


# In[14]:


from collections import Counter #get nos of outputs ie;0's and 1's
Counter(y)


# In[15]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)


# In[16]:


from sklearn.model_selection import train_test_split          #splitting inputs and outputs for testing and training
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


dt_clf= DecisionTreeClassifier(criterion='entropy', random_state=0)  
dt_clf.fit(X_train, y_train) 


# In[19]:


dt_pred=dt_clf.predict(X_test)
dt_pred


# In[20]:


acc_dt=accuracy_score(y_test,dt_pred)*100
print("Accuracy Score:",acc_dt)


# In[21]:


print("Confusion Matrix:\n",confusion_matrix(y_test,dt_pred))


# In[22]:


print("Classification Report:\n",classification_report(y_test,dt_pred))


# In[23]:


#random forest
from sklearn.ensemble import RandomForestClassifier


# In[24]:


rf_clf=RandomForestClassifier()


# In[25]:


rf_clf.fit(X_train, y_train)


# In[26]:


rf_pred=rf_clf.predict(X_test)
rf_pred


# In[27]:


acc_rf=accuracy_score(y_test,rf_pred)*100
print("Accuracy Score:",acc_rf)


# In[28]:


print("Confusion Matrix:\n",confusion_matrix(y_test,rf_pred))


# In[29]:


print("Classification Report:\n",classification_report(y_test,rf_pred))


# In[30]:


import pickle


# In[31]:


pickle.dump(rf_clf,open("model_rf.sav",'wb'))
pickle.dump(dt_clf,open("model_dt.sav",'wb'))

