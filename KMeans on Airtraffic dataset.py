#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
from pylab import*
import seaborn as sns


# In[2]:


df = pd.read_csv("passenger_stats.csv")
df


# In[3]:


df.head()


# In[6]:


figure(figsize=(15,15),dpi=100)
sns.countplot(x="Operating Airline",palette="Set3",data=df)
xticks(rotation=90)
ylabel("Number of flights held")
show()


# In[7]:


figure(figsize=(15,15),dpi=100)
sns.countplot(x="GEO Region",palette="Set3",data=df)
xticks(rotation=90)
ylabel("Number of flights held")
show()


# In[19]:


airline_count = df["Operating Airline"].value_counts()
airline_count.sort_index(inplace=True)
passenger_count = df.groupby("Operating Airline").sum()["Passenger Count"]
passenger_count.sort_index(inplace=True)

from sklearn.preprocessing import scale

x = airline_count.values
y = passenger_count.values

figure(figsize=(10,10),dpi=100)
scatter(x,y)
xlabel("Number of flights held")
ylabel("Passengers")

for i, txt in enumerate(airline_count.index.values):
    a = gca()
    annotate(txt, (x[i], y[i]))
show()


# In[25]:


df1 = airline_count + passenger_count
df1.sort_values(ascending=False,inplace=True)
outliers = df1.head(2).index.values
df1 = df1.drop(outliers)
x = airline_count.values
y = passenger_count.values


# In[26]:


from sklearn.cluster import KMeans
X = np.array(list(zip(x,y)))

inertia = []
for k in range(2,10):
    km = KMeans(n_clusters=k)
    km.fit(X)
    y_pred = km.predict(X)
    inertia.append(km.inertia_)
plot(range(2,10),inertia, "o-g")
xlabel("Number of flights")
ylabel("Clusters")
show()


# In[29]:


km = KMeans(n_clusters=3)
km.fit(X)
y_pred = km.predict(X)

figure(figsize=(15,15),dpi=100)
xlabel("Flights held")
ylabel("Passengers")
scatter(X[:,0], X[:,1],c=y_pred,s=300,cmap="Set1")

for i, txt in enumerate(airline_count.index.values):
    annotate(txt, (X[i,0],X[i,1]),size=7)
show()


# In[ ]:




