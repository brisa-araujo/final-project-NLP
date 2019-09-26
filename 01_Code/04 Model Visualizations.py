#!/usr/bin/env python
# coding: utf-8

# *Proceso Visualizaciones:*
#     9. Describir y dar nombre a los grupos
#     10. Generar visualizaciones de tópicos
#         10.1 LDAtovis - distancia entre tópicos
#         10.2 matriz de similitud de tópicos
#         10.3 ejemplo de texto con colores
#         10.4 PCA con evaluación general por tipo de cerveza y grupos

# In[13]:


#file and system 
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
from sklearn.manifold import TSNE

#plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from yellowbrick.features.pca import PCADecomposition
from wordcloud import WordCloud

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', palette='rainbow', rc={'figure.figsize':(10,6)})

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.io as pio

py.init_notebook_mode()


# In[505]:


reviews = pd.read_csv('../02_Data/reviews_clean.csv', nrows=10)


# In[150]:


beers = pd.read_csv('../02_Data/beers_final.csv')


# In[2]:


#load model
lda = LdaModel.load('../03_Dump/model')


# In[152]:


columnas = {'0':'c_bittersweet_coffee', 
            '1':'c_roasted_coffee',
           '2':'c_spicy',
            '3':'c_citrus',
            '4':'c_acid_berry',
            '5':'c_smooth_hop',
            '6':'c_bitter_malt',
            '7':'c_tropical_citrus',
            '8':'c_summer_wheat',
            '9':'c_maple_syrup',
            '10':'c_roasted_caramel',
            '11':'c_dark_oak',
            '12':'c_ginger_ale',
            '13':'c_apple_cider',
            '14':'c_chocolate_cream',
            '15':'c_sweet_amber',
            '16':'c_dark_caramel',
            '17':'c_dry_yeast',
            '18':'c_sweet_dark',
            '19':'c_thick_black',
            '20':'c_dark_dry',
            '21':'c_herbal_clear',
            '22':'c_spicy_brown',
            '23':'c_belgian_yeast',
            '24':'c_bad_sweet',
            '25':'c_strong_dark',
            '26':'c_orange_hazy',
            '27':'c_smooth_roasted',
            '28':'c_strawberry_wheat',
            '29':'c_fruity_caramel',
            '30':'c_bad_citrus',
            '31':'c_creamy_spicy',
            '32':'c_creamy_banana',
            '33':'c_sweet_hop',
            '34':'c_thick_bourbon'}


# In[153]:


beers = beers.rename(columns=columnas, errors='raise')


# In[149]:


for t in [7, 14]:
    plt.figure()
    plt.imshow(WordCloud(background_color="white").fit_words(dict(lda.show_topic(t, 200))))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.savefig(f'../04_Images/Topic_{t}.png')
    plt.show()


# In[446]:


sns.scatterplot(x="c_tropical_citrus", 
                y="score", 
                hue="style", 
                alpha=0.5, 
                data=beers, 
                palette='husl')

plt.legend(bbox_to_anchor=(1.2, 0.5), loc='center right', labelspacing=0.5)
sns.despine(left=True, bottom=True)
plt.savefig(f'../04_Images/tropical_citrus_scatter.png')
plt.show()


# In[445]:


sns.scatterplot(x="c_chocolate_cream", 
                y="score", 
                hue="style", 
                alpha=0.5, 
                data=beers, 
                palette='husl')

plt.legend(bbox_to_anchor=(1.2, 0.5), loc='center right', labelspacing=0.5)
sns.despine(left=True, bottom=True)
plt.savefig(f'../04_Images/chocolate_cream_scatter.png')
plt.show()


# In[399]:


beers['07_binned'] = np.where(beers['c_tropical_citrus'] >=0.5, 1, 0)
beers['14_binned'] = np.where(beers['c_chocolate_cream'] >=0.5, 1, 0)
print(beers['07_binned'].value_counts())
print(beers['14_binned'].value_counts())


# In[415]:


frequency_style = beers.pivot_table(beers, index='style', aggfunc=np.sum)


# In[416]:


freq_07 = frequency_style[['07_binned']].sort_values('07_binned', ascending=False).reset_index()


# In[422]:


total_style = beers['style'].value_counts(ascending=False).reset_index()
total_style.columns = ['style', 'count']


# In[423]:


freq_07 = freq_07.merge(total_style, on='style')


# In[511]:


f, ax = plt.subplots(figsize=(6, 8))

sns.set_color_codes("pastel")
sns.barplot(x="count", y="style", data=freq_07,
            label="Total", color="g", alpha=0.3)

sns.set_color_codes("muted")
sns.barplot(x="07_binned", y="style", data=freq_07,
            label="Tropical Citrus", color="g")

ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(0.8, -0.15))
ax.set(xlim=(0, 3500), ylabel="", xlabel='')
plt.savefig(f'../04_Images/tropical_citrus_bar.png')
sns.despine(left=True, bottom=True)


# In[447]:


freq_14 = frequency_style[['14_binned']].sort_values('14_binned', ascending=False).reset_index()


# In[448]:


freq_14 = freq_14.merge(total_style, on='style')


# In[512]:


f, ax = plt.subplots(figsize=(6, 8))

sns.set_color_codes("pastel")
sns.barplot(x="count", y="style", data=freq_14,
            label="Total", color="#e74c3c", alpha=0.2)

sns.set_color_codes("muted")
sns.barplot(x="14_binned", y="style", data=freq_14,
            label="Chocolate Cream", color="#e74c3c")

ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(0.8, -0.15))
ax.set(xlim=(0, 3500), ylabel="", xlabel='')
sns.despine(left=True, bottom=True)
plt.savefig(f'../04_Images/chocolate_cream_bar.png')
plt.show()


# In[470]:


bins = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5]


# In[471]:


beers['binned_score'] = pd.cut(beers['score'], bins)


# In[3]:


from gensim.matutils import jensen_shannon
from scipy import spatial as scs
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist, squareform


# get topic distributions
topic_dist = lda.state.get_lambda()

# get topic terms
num_words = 300
topic_terms = [{w for (w, _) in lda.show_topic(topic, topn=num_words)} for topic in range(topic_dist.shape[0])]

# no. of terms to display in annotation
n_ann_terms = 10

# use Jensen-Shannon distance metric in dendrogram
def js_dist(X):
    return pdist(X, lambda u, v: jensen_shannon(u, v))

# define method for distance calculation in clusters
linkagefun=lambda x: sch.linkage(x, 'single')

# calculate text annotations
def text_annotation(topic_dist, topic_terms, n_ann_terms, linkagefun):
    # get dendrogram hierarchy data
    linkagefun = lambda x: sch.linkage(x, 'single')
    d = js_dist(topic_dist)
    Z = linkagefun(d)
    P = sch.dendrogram(Z, orientation="bottom", no_plot=True)

    # store topic no.(leaves) corresponding to the x-ticks in dendrogram
    x_ticks = np.arange(5, len(P['leaves']) * 10 + 5, 10)
    x_topic = dict(zip(P['leaves'], x_ticks))

    # store {topic no.:topic terms}
    topic_vals = dict()
    for key, val in x_topic.items():
        topic_vals[val] = (topic_terms[key], topic_terms[key])

    text_annotations = []
    # loop through every trace (scatter plot) in dendrogram
    for trace in P['icoord']:
        fst_topic = topic_vals[trace[0]]
        scnd_topic = topic_vals[trace[2]]
        
        # annotation for two ends of current trace
        pos_tokens_t1 = list(fst_topic[0])[:min(len(fst_topic[0]), n_ann_terms)]
        neg_tokens_t1 = list(fst_topic[1])[:min(len(fst_topic[1]), n_ann_terms)]

        pos_tokens_t4 = list(scnd_topic[0])[:min(len(scnd_topic[0]), n_ann_terms)]
        neg_tokens_t4 = list(scnd_topic[1])[:min(len(scnd_topic[1]), n_ann_terms)]

        t1 = "<br>".join((": ".join(("+++", str(pos_tokens_t1))), ": ".join(("---", str(neg_tokens_t1)))))
        t2 = t3 = ()
        t4 = "<br>".join((": ".join(("+++", str(pos_tokens_t4))), ": ".join(("---", str(neg_tokens_t4)))))

        # show topic terms in leaves
        if trace[0] in x_ticks:
            t1 = str(list(topic_vals[trace[0]][0])[:n_ann_terms])
        if trace[2] in x_ticks:
            t4 = str(list(topic_vals[trace[2]][0])[:n_ann_terms])

        text_annotations.append([t1, t2, t3, t4])

        # calculate intersecting/diff for upper level
        intersecting = fst_topic[0] & scnd_topic[0]
        different = fst_topic[0].symmetric_difference(scnd_topic[0])

        center = (trace[0] + trace[2]) / 2
        topic_vals[center] = (intersecting, different)

        # remove trace value after it is annotated
        topic_vals.pop(trace[0], None)
        topic_vals.pop(trace[2], None)  
        
    return text_annotations


# In[17]:


annotation = text_annotation(topic_dist, topic_terms, n_ann_terms, linkagefun)

# Plot dendrogram
dendro = ff.create_dendrogram(topic_dist, 
                              distfun=js_dist, 
                              labels=range(1, 36), 
                              linkagefun=linkagefun, 
                              hovertext=annotation)
dendro['layout'].update({'width': 1000, 'height': 600})
pio.write_html(dendro, file='../04_Images/dendro.html', auto_open=True)


# In[22]:


# get text annotations
annotation = text_annotation(topic_dist, topic_terms, n_ann_terms, linkagefun)

# Initialize figure by creating upper dendrogram
figure = ff.create_dendrogram(topic_dist, distfun=js_dist, labels=range(1, 36), linkagefun=linkagefun, hovertext=annotation)
for i in range(len(figure['data'])):
    figure['data'][i]['yaxis'] = 'y2'


# In[23]:


# get distance matrix and it's topic annotations
mdiff, annotation = lda.diff(lda, distance="jensen_shannon", normed=False)

# get reordered topic list
dendro_leaves = figure['layout']['xaxis']['ticktext']
dendro_leaves = [x - 1 for x in dendro_leaves]

# reorder distance matrix
heat_data = mdiff[dendro_leaves, :]
heat_data = heat_data[:, dendro_leaves]


# In[25]:


# heatmap annotation
annotation_html = [["+++ {}<br>--- {}".format(", ".join(int_tokens), ", ".join(diff_tokens))
                    for (int_tokens, diff_tokens) in row] for row in annotation]

# plot heatmap of distance matrix
heatmap = go.Data([
    go.Heatmap(
        z=heat_data,
        colorscale='YlGnBu',
        text=annotation_html,
        hoverinfo='x+y+z+text'
    )
])

heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
heatmap[0]['y'] = figure['layout']['xaxis']['tickvals']

# Add Heatmap Data to Figure
figure.add_traces(heatmap)


dendro_leaves = [x + 1 for x in dendro_leaves]

# Edit Layout
figure['layout'].update({'width': 800, 'height': 800,
                         'showlegend':False, 'hovermode': 'closest',
                         })

# Edit xaxis
figure['layout']['xaxis'].update({'domain': [.25, 1],
                                  'mirror': False,
                                  'showgrid': False,
                                  'showline': False,
                                  "showticklabels": True, 
                                  "tickmode": "array",
                                  "ticktext": dendro_leaves,
                                  "tickvals": figure['layout']['xaxis']['tickvals'],
                                  'zeroline': False,
                                  'ticks': ""})
# Edit yaxis
figure['layout']['yaxis'].update({'domain': [0, 0.75],
                                  'mirror': False,
                                  'showgrid': False,
                                  'showline': False,
                                  "showticklabels": True, 
                                  "tickmode": "array",
                                  "ticktext": dendro_leaves,
                                  "tickvals": figure['layout']['xaxis']['tickvals'],
                                  'zeroline': False,
                                  'ticks': ""})
# Edit yaxis2
figure['layout'].update({'yaxis2':{'domain': [0.75, 1],
                                   'mirror': False,
                                   'showgrid': False,
                                   'showline': False,
                                   'zeroline': False,
                                   'showticklabels': False,
                                   'ticks': ""}})

pio.write_html(figure, file='../04_Images/heatmap.html', auto_open=True)


# In[ ]:




