#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


# In[2]:


#function to label encoding a column based on a dictionary
def classifier(s, dictionary):
    for k, v in dictionary.items():
        if s in v:
            return k


# In[3]:


#function to retrieve and clean beer database
def clean_beers(path, to_drop, class_dict, fill_dict):
    df = pd.read_csv(path)  #read csv
    valid = set(df.id[df.retired == 'f']) #create a subset of valid beer ids to be used later
   
    df = df[df.retired == 'f'] #filter df based on beers that are still in production
    df['availability'] = df.availability.str.strip() #clean availability empty spaces
    
    #call classifier function to re-label categorical columns that had too many values
    for k, v in class_dict.items(): 
        df[k] = df[k].apply(lambda x: classifier(x, v))
    
    #fillna from dictionary of column:criterion
    for k, v in fill_dict.items():
        df[k] = df[k].fillna(v)
        
    #drop columns
    df = df.drop(columns=to_drop)
    
    return df, valid   #return both the clean df and the set of valid beer ids


# In[ ]:





# In[4]:


#function to retrieve reviews dataframe and filter interesting rows
def parse_reviews(path, valid):
    iterator = pd.read_csv(path, chunksize=10000) #read csv by chunks, since it has about 10 million rows
    df = pd.DataFrame() #create empty dataframe to store filtered rows
    while len(df) <= 700000: #set a maximum for iteration
        #for each chunk of the data, filter for valid beer ids, drop rows with na and append the rest to df
        try:
            tmp = iterator.get_chunk()
            tmp = tmp[tmp['beer_id'].isin(valid)]
            tmp['text'] = tmp['text'].replace('\xa0\xa0', np.nan)
            tmp = tmp.dropna()
            df = df.append(tmp, ignore_index=True)
        #if the previous task cannot be performed, this means we are out of values from the csv file
        except:
            print('Out of values')
    return df #return filtered dataframe


# In[5]:


#function to finish and save beer df based on which beers have reviews
def beers_subset(df_ref, df_filter, path):
    #get reference dataframe (in this case, the newly created reviews df)and check what is our universe of beer ids
    universe = df_ref.beer_id.unique()
    #filter beers df so to have only beers for which we have reviews
    df_filter = df_filter[df_filter.id.isin(universe)]
    return df_filter.to_csv(path, index=False) #save csv file with the result


# In[6]:


#function to clean review df columns
def prepare_reviews(data, to_drop, to_encode):
    #lower float datatypes 
    for x in data:
        if type(x) == float:
            data[x] = data[x].astype('float16')
    #drop columns
    data = data.drop(columns=to_drop)
    #encode variables
    le = LabelEncoder()
    for x in to_encode:
        data[x] = le.fit_transform(data[x])
    return data #return semi-clean dataframe


# In[10]:


#function to finish and store review df by cleaning text variable
def finish_reviews(df, col, path):
    #lowercase
    df[col] = df[col].str.lower()
    #no numbers
    df[col] = df[col].str.replace(r'\d+', ' ')
    #no punctuation or special characters
    translator = str.maketrans("","" , string.punctuation)
    df[col] = df[col].apply(lambda x: x.translate(translator))
    #remove spaces in the beginning and end
    df[col] = df[col].str.strip()
    #Save file to csv
    df.to_csv(path, index=False)
    return df


# In[8]:


def kickstart_beers():
    #Paths to get and store files
    path_beers = '../02_Data/beers.csv'
    path_reviews = '../02_Data/reviews.csv'
    path_clean_beers = '../02_Data/beers_clean.csv'
    path_clean_reviews = '../02_Data/reviews_clean.csv'
    
    #Variables to drop in each df
    to_drop_beers = ['state', 'notes', 'retired', 'name']
    to_drop_reviews = ['date']
    
    #New labels to encode in beers
    to_classify_beers = {'country':{'US':['US'], 
                        'Canada':['CA'],
                        'America':['MX', 'CR', 'NI', 'BR', 'AR', 'PY', 'CL', 
                                   'EC', 'PE', 'DO', 'PA', 'VE', 'UY', 'CO', 
                                   'GT', 'HN', 'PR', 'HT', 'EG', 'BS', 'CU', 
                                   'JM', 'GU', 'LC', 'TT', 'AG', 'VG', 'BO', 
                                   'MQ', 'BB', 'GP', 'DM', 'GQ', 'AW', 'BZ', 
                                   'SR', 'VC', 'VI'],
                        'Europe':['NO', 'IT', 'SE', 'PL', 'DE', 'GB', 'RU', 
                                  'BE', 'ES', 'IE', 'DK', 'FI', 'NL', 'AT', 
                                  'FR', 'EE', 'AM', 'PT', 'CZ', 'LV', 'CH', 
                                  'UA', 'RO', 'LT', 'HU', 'RS', 'MK', 'RE', 
                                  'HR', 'GL', 'BY', 'TR', 'BG', 'FO', 'SI', 
                                  'GR', 'LU', 'GF', 'MD', 'IS', 'IR', 'SK', 
                                  'AD', 'SV', 'BA', 'JE', 'GD', 'MT', 'SM', 
                                  'AL', 'MC', 'LI', 'MO'], 
                        'Asia':['IN', 'JP', 'TH', 'CN', 'KZ', 'ID', 'HK', 
                                'PH', 'KG', 'IL', 'MN', 'VN', 'KR', 'GE', 
                                'TN', 'PS', 'PG', 'AZ', 'KP', 'KH', 'UZ', 
                                'TW', 'BM', 'IM', 'LA', 'BD', 'CW', 'MM', 
                                'NP', 'JO', 'MW', 'LK', 'BJ', 'LB', 'AO', 
                                'KY', 'LY', 'AE', 'SY', 'BT', 'TJ', 'IQ', 
                                'TM', 'BF'], 
                        'Africa':['ZW', 'ZA', 'SG', 'BG', 'CY', 'CI', 'PF', 
                                  'ME', 'GH', 'NG', 'KE', 'MG', 'MZ', 'GN', 
                                  'ET', 'UG', 'MA', 'CV', 'LS', 'WS', 'ER', 
                                  'MY', 'PK', 'RW', 'SN', 'LR', 'CG', 'DZ', 
                                  'ST', 'BW', 'TD', 'CM', 'GM', 'SZ', 'ZM', 
                                  'GW', 'GA', 'CF', 'SS', 'NE'], 
                        'Oceania':['AU', 'NZ', 'MU', 'NC', 'FJ', 'TZ', 'TC', 
                                   'SC', 'GG', 'YT', 'FM', 'PW', 'CK', 'VU', 
                                   'TG', 'SB', 'TO']}, 
              'style':{'bock':['German Bock', 
                                   'German Doppelbock', 
                                   'German Eisbock', 
                                   'German Maibock', 
                                   'German Weizenbock'],
                           'brown_ale':['American Brown Ale', 
                                        'English Brown Ale', 
                                        'English Dark Mild Ale', 
                                        'German Altbier'],
                           'dark_ale':['American Black Ale', 
                                       'Belgian Dark Ale', 
                                       'Belgian Dubbel', 
                                       'German Roggenbier',
                                       'Scottish Ale', 
                                       'Winter Warmer'],
                           'dark_lager':['American Amber / Red Lager', 
                                         'European Dark Lager',
                                         'German Märzen / Oktoberfest',
                                         'German Rauchbier',
                                         'German Schwarzbier',
                                         'Munich Dunkel Lager',
                                         'Vienna Lager'],
                           'hybrid':['American Cream Ale',
                                     'Bière de Champagne / Bière Brut',
                                     'Braggot',
                                     'California Common / Steam Beer'],
                           'IPA':['American Brut IPA', 
                                  'American Imperial IPA',
                                  'American IPA',
                                  'Belgian IPA',
                                  'English India Pale Ale (IPA)',
                                  'New England IPA'],
                           'pale_ale':['American Amber / Red Ale',
                                       'American Blonde Ale',
                                       'American Pale Ale (APA)',
                                       'Belgian Blonde Ale',
                                       'Belgian Pale Ale',
                                       'Belgian Saison',
                                       'English Bitter',
                                       'English Extra Special / Strong Bitter (ESB)',
                                       'English Pale Ale',
                                       'English Pale Mild Ale',
                                       'French Bière de Garde',
                                       'German Kölsch',
                                       'Irish Red Ale'],
                           'pilsener_pale_lager':['American Adjunct Lager',
                                                  'American Imperial Pilsner',
                                                  'American Lager',
                                                  'American Light Lager',
                                                  'American Malt Liquor',
                                                  'Bohemian Pilsener',
                                                  'European Export / Dortmunder',
                                                  'European Pale Lager',
                                                  'European Strong Lager',
                                                  'German Helles',
                                                  'German Kellerbier / Zwickelbier',
                                                  'German Pilsner'],
                           'porter':['American Imperial Porter',
                                     'American Porter',
                                     'Baltic Porter',
                                     'English Porter',
                                     'Robust Porter',
                                     'Smoke Porter'],
                           'specialty':['Chile Beer',
                                        'Finnish Sahti',
                                        'Fruit and Field Beer',
                                        'Herb and Spice Beer',
                                        'Japanese Happoshu',
                                        'Japanese Rice Lager',
                                        'Low Alcohol Beer',
                                        'Pumpkin Beer',
                                        'Russian Kvass',
                                        'Rye Beer',
                                        'Scottish Gruit / Ancient Herbed Ale',
                                        'Smoke Beer'],
                           'stout':['American Imperial Stout',
                                    'American Stout',
                                    'English Oatmeal Stout',
                                    'English Stout',
                                    'English Sweet / Milk Stout',
                                    'Foreign / Export Stout',
                                    'Irish Dry Stout',
                                    'Russian Imperial Stout'],
                           'strong_ale':['American Barleywine',
                                         'American Imperial Red Ale',
                                         'American Strong Ale',
                                         'American Wheatwine Ale',
                                         'Belgian Quadrupel (Quad)',
                                         'Belgian Strong Dark Ale',
                                         'Belgian Strong Pale Ale',
                                         'Belgian Tripel',
                                         'British Barleywine',
                                         'English Old Ale',
                                         'English Strong Ale',
                                         'Scotch Ale / Wee Heavy'],
                           'wheat':['American Dark Wheat Ale',
                                    'American Pale Wheat Ale',
                                    'Belgian Witbier',
                                    'Berliner Weisse',
                                    'German Dunkelweizen',
                                    'German Hefeweizen',
                                    'German Kristalweizen'],
                           'wild_sour':['American Brett',
                                        'American Wild Ale',
                                        'Belgian Faro',
                                        'Belgian Fruit Lambic',
                                        'Belgian Gueuze',
                                        'Belgian Lambic',
                                        'Flanders Oud Bruin',
                                        'Flanders Red Ale',
                                        'Leipzig Gose']}}
    #values to fill na in beers
    to_fill_beers = {'country':'Unknown', 
           'style':'other', 
           'abv':0}
    
    #variables to encode in reviews
    to_encode_reviews = ['username']
    
    #call function to get and clean beer df
    beers, valid_ids = clean_beers(path_beers, to_drop_beers, to_classify_beers, to_fill_beers)
    
    #call function to get and filter reviews df - based on first criteria
    reviews_raw = parse_reviews(path_reviews, valid_ids)
    
    #call function to finish up beers dataframe, selecting only beers for which we have reviews
    beers_subset(reviews_raw, beers, path_clean_beers)
    
    #call function to prepare reviews df columns
    reviews_semi = prepare_reviews(reviews_raw, to_drop_reviews, to_encode_reviews)
    
    #call function to clean text column and finish up reviews df
    reviews = finish_reviews(reviews_semi, 'text', path_clean_reviews)
    
    #return clean dataframes to play with
    return beers, reviews
    


# In[11]:


beers, reviews = kickstart_beers()

