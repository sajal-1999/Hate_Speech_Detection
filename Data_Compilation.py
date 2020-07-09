#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


with open('racism.json', 'r') as file:
    data_racism = pd.read_json(file,lines=True)
with open('neither.json', 'r') as file:
    data_neither = pd.read_json(file,lines=True)
with open('sexism.json', 'r') as file:
    data_sexism = pd.read_json(file,lines=True)


# In[3]:


labels = []
text = []
labels = list(data_racism['Annotation'])
text = list(data_racism['text'])
labels.extend(list(data_neither['Annotation']))
labels.extend(list(data_sexism['Annotation']))
text.extend(list(data_neither['text']))
text.extend(list(data_sexism['text']))


# In[ ]:


dataframe = pd.DataFrame(labels, columns =['Annotation'])
dataframe['Tweets'] = texts


# In[ ]:


dataframe.to_csv('Twitter_Data.csv', index=False, encoding='utf-8')

