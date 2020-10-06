#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Linear_Regression_Gradient_Descent import LinearRegressionUsingGD
from sklearn.linear_model import LinearRegression
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("CardioGoodFitness.csv") #Read saved file from local directory


# In[3]:


print(df)


# In[4]:


# Plot sns scatter graphs to observce quick relationships between variables
sns.pairplot(df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


# Analyse product based on age usage using gender as the key
# See each product usage based on the following age groupings 18-25 (young), 26-35 (independant), 36-45 (middle age)
# (senior) 45+


# In[6]:


df_age = df[['Product', 'Age', 'Gender']]
df_age


# In[7]:


# Create a collumn for age bins


# In[8]:


df_age['age_bins'] = pd.cut(x=df_age['Age'], bins=[17, 26, 36, 46, 50])


# In[9]:


df_age


# In[10]:


df_age_bins = df_age[['Product', 'age_bins', 'Gender']]


# In[11]:


df_age_bins


# In[12]:


# Check length of bins
len(df_age_bins['age_bins'].unique())


# In[13]:


# Determine usage per age_bins with Product type as the categorization
df_age_bins.groupby('Product')


# In[14]:


df_age_bins.groupby('Product')['age_bins']


# In[15]:


df_age_bins.groupby('Product')['age_bins'].value_counts()


# In[16]:


#Unstack data into table below
df_age_bins.groupby('Product')['age_bins'].value_counts().unstack('Product')


# In[17]:


# Plot the unstacked table by age bins and frequency, seperated by product type
bar_agebins = df_age_bins.groupby('Product')['age_bins'].value_counts().unstack('Product').plot.bar(stacked=True)
bar_agebins.set(xlabel='Age Groups', ylabel = 'Usage Freq.')


# In[ ]:


# From this plot we observe that the age group of 17-26 used the fitness products the most and TM195 was the most
# popular


# In[18]:


# Next, for our descrptive analysis exercise, is to find if there is a relationship between age and usability of customers per
# product
df


# In[19]:


# Start with Product, TM195
df_TM195 = df.loc[df['Product'] == 'TM195']


# In[20]:


# Order based on age
df_TM195.sort_values('Age')
df_TM195


# In[21]:


# Remove unneccary collumns
df_age_usage = df_TM195[['Age', 'Usage']]
df_age_usage


# In[22]:


# Re-plot and observe scatter plot between age and usage (see sns call above)
df_age_usage.plot.scatter(x='Age', y='Usage').set_title('Correlation between Age and Usage')


# In[23]:


# Out of interest observe the correlation
df_age_usage.corr()


# In[24]:


# There appears to be no correlation only a very minor/ insignificant negative correlation


# In[25]:


# Let's look at the correlation between age and fitness


# In[26]:


# Remove unnesseccary collumns
df_age_fitness = df_TM195[['Age', 'Fitness']]
df_age_fitness


# In[27]:


df_age_fitness.plot.scatter(x='Age', y='Fitness').set_title('Correlation between Age and Fitness')


# In[28]:


df_age_fitness.corr()


# In[29]:


# There appears to be a very weak positive correlation between age and fitness


# In[30]:


# Let us look at a linear regression model between age (feature) and income (label)


# In[31]:


df_age_income = df[['Age','Income']]


# In[32]:


df_age_income


# In[33]:


df_age_income.plot.scatter(x='Age', y='Income').set_title('Scatter plot of Age and Income')


# In[35]:


# Re-shape Age and Income values so that they can fit into sklearn's fit and predict objects
X = df_age_income.iloc[:,0].values.reshape(-1,1)
Y = df_age_income.iloc[:,1].values.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)
# Create the line of best fit
Y_pred = linear_regressor.predict(X)


# In[36]:


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[37]:


# Use sklearn to LR function to predict a specific age value's income
linear_regressor = LinearRegression() # Call LR object again
linear_regressor.fit(df_age_income[['Age']], df.Income)


# In[38]:


print(df_age_income[['Age']])


# In[39]:


# Predict for a value of 43 (value can be changed here by reader)
value = 43
Age_Predict = np.array(value).reshape(-1,1) # Reshape for input into .predict
linear_regressor.predict(Age_Predict)


# In[ ]:


# Returned income based on age=43 gives income=71064


# In[40]:


# Use LinearRegression with Gradient Descent - compare with sklearn's least square's method


# In[41]:


# Appropriately shape the features and labels X and Y
X = df_age_income.iloc[:,0]


# In[42]:


print(X)


# In[43]:


Y = Y = df_age_income.iloc[:,1]


# In[44]:


print(Y)


# In[45]:


from Linear_Regression_Gradient_Descent import LinearRegressionUsingGD #call created class that will calculate LR  using GD


# In[46]:


linear_regressor = LinearRegressionUsingGD(X, Y, 43) # Initialize instance of class (43 here can be user changed)


# In[47]:


linear_regressor.performGD(X, Y, 43)


# In[ ]:


# Results show m = 1822 (gradient) and c = 261 (intercept at 0)
# For age=43, income=78613 - the GD method is more accurate than sklearn's linear regression model??? Perhaps

