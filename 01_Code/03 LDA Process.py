#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, operator, warnings
import logging
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath
from smart_open import smart_open

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel


warnings.filterwarnings('ignore')
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


# In[ ]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Arguments:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Outputs:
    ----------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, 
                             num_topics=num_topics, 
                             id2word=dictionary, 
                             random_state=100, 
                             alpha=.1, 
                             eta=0.01)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, 
                                        texts=texts, 
                                        dictionary=dictionary, 
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    
    limit=limit; start=start; step=step;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
    for i in range(len(model_list)):
        print(model_list[i], coherence_values[i])

    return model_list, coherence_values


# In[ ]:


def chosen_lda(corpus, dictionary, data, n_topics, alpha=.1, eta=0.01):
    '''
    This function trains a Gensim LDA model on chosen hyperparameters
    
    Arguments:
    ----------
    corpus : matrix-format corpus (BOW or TF-IDF)
    dictionary : corpus-related dictionary
    data : text data for coherence score computation
    n_topics : number of desired topics
    alpha : alpha parameter (from 0 to infinity)
    eta : beta parameter (from 0 to infinity)
    
    Outputs:
    ----------
    lda : trained model
    '''
    
    lda = LdaModel(corpus=corpus, 
                id2word=dictionary, 
                num_topics=35, 
                random_state=100, 
                alpha=alpha, 
                eta=eta)
    
    ldatopics = [[word for word, prob in topic] for topicid, topic in lda.show_topics(formatted=False)]
    lda_coherence = CoherenceModel(topics=ldatopics, texts=data, dictionary=dictionary, window_size=10).get_coherence()
    print(lda_coherence)
    lda.print_topics(num_topics=n_topics)
    
    lda.save('../03_Dump/model')
    return lda


# In[5]:


def compute_results(model, corpus, no_topics):
    '''
    This function computes results from model in the format:
    probability of topics per document, and formats dataframes for further analysis
    
    Arguments:
    ----------
    model : trained model
    corpus : corpus used
    no_topics : number of clusters created
    
    Outputs:
    ----------
    final : consolidated dataframe with beer info, reviews clusters and mean scores
    '''
    #Compute results of the model
    results = []
    for i in range(len(model[corpus])):
        tmp={}
        for tupla in model[corpus[i]]:
            tmp[tupla[0]] = tupla[1]
        results.append(tmp)
        
    results_df = pd.DataFrame(results, 
                              columns=[i for i in range(0,no_topics)])
    results_df = results_df.fillna(0)
    print('Created DataFrame with model results.')
    
    #get beer IDs from reviews_per_product dataframe
    per_prod = pd.read_csv('../02_Data/reviews_per_product.csv', 
                           usecols=['beer_id'])
    print('File read: ../02_Data/reviews_per_product.csv')
    per_prod = pd.concat([per_prod, results_df], axis=1)
    
    #merge clustered beers with clean_beers df
    beers_clean = pd.read_csv('../02_Data/beers_clean.csv')
    print('File read: ../02_Data/beers_clean.csv')  
    
    beers_clean = beers_clean.merge(per_prod, 
                                    how='right', 
                                    left_on='id', 
                                    right_on='beer_id')
    beers_clean = beers_clean.drop(columns=['id', 
                                            'brewery_id'], 
                                   axis=1)
    
    #create consolidated dataframe from different sources
    reviews = pd.read_csv('../02_Data/reviews_clean.csv', 
                          usecols=['beer_id', 'look', 'smell', 
                                   'taste', 'feel', 'overall', 
                                   'score'])
    print('File read: ../02_Data/reviews_clean.csv') 
    
    reviews = reviews.groupby('beer_id').mean()
    reviews = reviews.reset_index()
    final = reviews.merge(beers_clean, 
                          how='right', 
                          on='beer_id')
    
    final.to_csv('../02_Data/beers_final.csv', 
                 index=False)
    print('Saved new dataframe to path: ../02_Data/beers_final.csv')
    
    return final


# In[2]:


data = [simple_preprocess(line, 
                          deacc=True) for line in open('../02_Data/clean_reviews.txt', 
                                                       encoding='utf-8')]
dictionary = corpora.Dictionary.load('dictionary.dict')
bow_corpus = corpora.MmCorpus('bow_corpus.mm')


# In[ ]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, 
                                                        corpus=tfidf_corpus, 
                                                        texts=data, 
                                                        start=20, 
                                                        limit=60, 
                                                        step=8)


# In[ ]:


lda1 = chosen_lda(bow_corpus, dictionary, data, 35)


# In[6]:


beers = compute_results(lda1, bow_corpus, 35)

