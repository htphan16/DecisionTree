#!/usr/bin/env python
# coding: utf-8

# In[12]:


import math
import pandas as pd
import numpy as np


def read_file(file):
	return pd.read_csv(file, sep='\t')

def store_examples(df, num_examples):
	return df.sample(num_examples)

titanic = read_file('titanic2.txt')

print(store_examples(titanic, 12))

'''Function to calculate information gain to determine each attribute's importance'''


# In[ ]:




