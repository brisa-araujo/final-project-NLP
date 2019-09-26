#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os, operator, warnings
import logging
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath
from smart_open import smart_open

#dataset manipulation
import pandas as pd
import numpy as np

#model
from gensim import corpora
from gensim.models import LdaModel

#plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from yellowbrick.features.pca import PCADecomposition
import pyLDAvis
import pyLDAvis.gensim

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', palette='Pastel2')
pyLDAvis.enable_notebook()


# In[2]:


lda = LdaModel.load('../03_Dump/model')


# In[6]:


dictionary = corpora.Dictionary.load('dictionary.dict')
bow_corpus = corpora.MmCorpus('bow_corpus.mm')


# In[9]:


p = pyLDAvis.gensim.prepare(lda, bow_corpus, dictionary, mds='mmds')
pyLDAvis.save_html(p, 'lda.html')


# In[ ]:




