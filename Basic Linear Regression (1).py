#!/usr/bin/env python
# coding: utf-8

# ### Lineer Regresyon Modeli

# *Çalışmada Advertising verisi kullanılacaktır.
# *Veride TV, Radio ,Newspaper bağımsız değişkenleri bulunmaktadır.
# *Bağımlı değişken olarak Satış değeri bulunmaktadır.
# *Amacımız bu harcamaların Satışa etkisini bulmaktır . Basic Model kullanacağımız için sadece TV etkisine bakacağım
# 

# In[2]:


# Kütüphaneleri import edelim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[3]:


# Datayı import edelim dataframe oluşturalım
df=pd.read_csv("Advertising.csv",index_col=0)


# In[4]:


#Dataya göz atalım ilk 5 satır için
df.head()


# In[30]:


# Datayı sondan da kontrol edelim
df.tail()


# In[28]:


#Data hakkında genel bilgiler
df.info()


# In[5]:


# Sayısal değişkenlerin istatistiklerini kontrol edelim
df.describe().T


# In[31]:


data=df[["TV","sales"]]


# In[32]:


#Input/Feature
X=data["TV"]

#output
y=data["sales"]


# In[33]:


type(X)


# In[34]:


type(y)


# In[36]:


#Grafik çizelim

plt.figure(figsize=(10,10))

sns.scatterplot(data=data,x="TV",y="sales",color="orange")
plt.title("SALES-TV")
plt.show()


# In[37]:


lr = LinearRegression()


# In[40]:


print("X'in boyutu:" , X.shape)
print("y'nin boyutu:" , y.shape)

# 2 boyutlu olması gerekir 


# In[41]:


X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)


# In[ ]:


print("X'in boyutu:" , X.shape)
print("y'nin boyutu:" , y.shape)
# İstediğimiz gibi 2 boyutlu yaptık


# In[45]:


#Datamızı test/train olarak ayıralım
from sklearn.model_selection import train_test_split


# In[51]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[53]:


X_train.shape


# In[54]:


y_train.shape


# In[55]:


X_test.shape
y_test.shape


# In[56]:


#Modeli fit edelim(öğrenme),öğrendiği şey betalar(katsayıları)hesaplayacak
lr.fit(X_train,y_train)


# In[57]:


#Katsayıları hesaplayalım
#intercept
print("intercept:" ,lr.intercept_)
#coefficient
print("slope:" ,lr.coef_)


# In[58]:


#Tahmin yapalım
y_pred=lr.predict(X_test)


# In[59]:


y_pred
#her bir değere karşılık tahminleme de yaptık


# In[60]:


y_pred.shape


# In[65]:


#Gerçek data/Grans Truth
fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(X_test,y_test,label="Grand Truth",color="red")
ax.scatter(X_test,y_pred,label="Grand Truth",color="green")
plt.xlabel("TV")
plt.ylabel("Satış")
plt.title("TV-Satış")
plt.legend(loc="upper left")

#doğrusal reg olduğu için grafik doğrusal tahminledi kırmızılar gerçek değerlerimiz


# In[67]:


#Her bir değişken arasındaki farkı görelim
indexler=range(1,61)
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(indexler,y_test,label="Grand Truth",color="red",linewidth=2)
ax.plot(indexler,y_pred,label="Grand Truth",color="green",linewidth=2)


# In[ ]:


#Ani yükselmeleri tahminleyememiş ama genel olarak kötü değil


# In[69]:


import numpy as np
indexler=range(1,61)

#Residuals
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(indexler,y_test-y_pred,label="Residuals",color="red",linewidth=2)

#0 doğrusu çizelim
ax.plot(indexler,np.zeros(60),color="black")

#hatalar 0 a yakın olmalı OLS


# In[70]:


from sklearn.metrics import r2_score,mean_squared_error


# In[71]:


r_2=r2_score(y_test,y_pred)


# In[72]:


r_2


# In[74]:


#MSE
mse=mean_squared_error(y_test,y_pred)


# In[75]:


mse


# In[76]:


import math
rmse=math.sqrt(mse)


# In[77]:


rmse


# #Yorumlar
# * Model %67 açıkladı .R2 sonucuna göre
# * Modelimiz ortalamada 2,99 yanılıyor RMSE ye göre 

# In[ ]:




