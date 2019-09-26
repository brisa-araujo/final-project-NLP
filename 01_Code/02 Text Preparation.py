#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os, re, operator, warnings
import logging
from gensim.utils import simple_preprocess
from smart_open import smart_open

import pandas as pd
import numpy as np

import nltk

from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel, TfidfModel, ldamulticore


warnings.filterwarnings('ignore')
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


# In[41]:


def reviews_per_prod(path_entrada, path_salida):
    '''
    This function takes a dataframe with one line per review 
    and makes it one line per product
    '''
    #Reading file
    base_df = pd.read_csv(path_entrada)
    print(f'File read: {path_entrada}')
    
    #Reframing df and removing non-text values
    df = base_df[['beer_id', 'text']].groupby('beer_id').sum()
    print('Created new grouped DataFrame')
    df = df[(df['text'].isnull() == False)&(df['text'] != 0.0)]
    print('Filtered non-text results')
    df = df.reset_index()
    
    #Filtering reviews that are too long for the LDA model
    #df['text'] = df.text.apply(lambda x: x if len(x) < 1000000 else np.nan)
    #df = df.dropna()
    #print('Filtered reviews longer than 1M characters')
    
    #Saving new DF to CSV
    df.to_csv(path_salida, index=False)
    print(f'Saved new dataframe to path: {path_salida}')
    return df


# In[42]:


def token_lemma(df, path_txt):
    '''
    This function tokenizes and lemmatizes documents, 
    then saves the result to a text file
    '''
    #Load stopwords and lemmatizer
    stopwords = nltk.corpus.stopwords.words('english')
    lemm = nltk.stem.WordNetLemmatizer()
    
    #Tokenize
    df['text'] = df.text.str.lower()
    df['processed_text'] = df.text.apply(nltk.word_tokenize)
    print('Tokenized text')
    
    #Lemmatize
    lista = ['beer', 'head', 'taste', 'flavor', 'smell', 'malt', 
             'nice', 'good', 'aroma', 'like', 'bit', 'one', 'well', 
             'light', 'hop', 'bottle', 'little', 'note', 'really', 'finish', 
             'medium', 'carbonation', 'body', 'hop', 'style', 'overall', 
             'color', 'glass', 'note', 'much', 'pours', 'would', 'slightly', 
             'mouthfeel', 'hint', 'nose', 'great', 'saison', 'get', 'drink', 
             'brew', 'lot', 'moderate', 'amount', 'thin', 'spelt', 'pretty', 
             'lacing', 'flavor', 'flavour', 'quite']
    df['processed_text'] = df.processed_text.apply(
        lambda doc: [lemm.lemmatize(word) for word in doc if word not in stopwords and len(word)>2 and word not in lista])
    print('Lemmatized text')
    
    #Save txt file
    np.savetxt(path_txt, 
               np.array(df.processed_text), 
               newline='\n', 
               encoding='utf-8', 
               fmt="%s")
    
    return print(f'Text corpus saved to file: {path_txt}')


# In[43]:


def prepare_corpus(path_txt):
    '''
    This function creates a dictionary, bag of words and TF-IDF matrix
    from a text file.
    It also saves dictionary, BOW and TF-IDF objects to disk.
    '''
    #Create gensem dictionary
    dictionary = corpora.Dictionary(simple_preprocess(line, 
                                                      deacc=True) for line in open(path_txt, 
                                                                                   encoding='utf-8'))
    print("Created Dictionary.\nFound {} words.\n".format(len(dictionary.values())))
    
    #Filter dictionary for common words
    #dictionary.filter_extremes(no_above=0.5, no_below=300)
    #dictionary.compactify()
    #print("Filtered Dictionary.\nLeft with {} words.\n".format(len(dictionary.values())))
    
    #Create Bag of Words
    bow = []
    for line in smart_open(path_txt, encoding='utf-8'):
        tokenized_list = simple_preprocess(line, deacc=True)
        bow.append(dictionary.doc2bow(tokenized_list, allow_update=True))
    print("Created Bag of Words.\n".format(len(bow)))
    
    #Create TF-IDF Matrix
    tfidf = TfidfModel(bow, smartirs='ntc')
    tfidf_corpus = tfidf[bow]
    print("Created TF-IDF matrix.\n".format(len(tfidf_corpus)))
     
    #Save files to disk
    dictionary.save('dictionary.dict')
    print('Saved dictionary object to disk.')
    corpora.MmCorpus.serialize('bow_corpus.mm', bow)
    print('Saved bag of words corpus object to disk.')
    corpora.MmCorpus.serialize('tfidf_corpus.mm', tfidf_corpus)
    print('Saved TF-IDF corpus object to disk.')
    
    return print('Processed Text')


# In[44]:


def text_processing(path1, path2, path3):
    df = reviews_per_prod(path1, path2)
    print('Created reviews per product DataFrame')
    token_lemma(df, path3)
    prepare_corpus(path3)
    return 'Done. Go apply some models!'


# In[45]:


path_entrada = '../02_Data/reviews_clean.csv'
path_salida = '../02_Data/reviews_per_product.csv'
path_texto = '../02_Data/clean_reviews.txt'

text_processing(path_entrada, path_salida, path_texto)


# In[ ]:


#    lista_negra = ['EX', 'IN', 'MD', 'PRP', 'PRP$', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']


# In[ ]:





# In[ ]:




