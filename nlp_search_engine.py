#!/usr/bin/env python
# coding: utf-8

# # Inverse indexing, index search, and signal page rank¶

# ## PART I: Preparing the documents/webpages

# In[1]:


# Load libraries

import pandas as pd
import numpy as np 
import string
import random
import json
import tqdm

import nltk
from nltk.corpus import brown

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer


# In[2]:


#load arXiv dataset
with open('arxivData.json') as fp:
    data = json.load(fp)
len(data)


# In[3]:


#view text from one document 
#reuters.raw(fileids=['test/14826'])[0:201]
print(data[1]['title'])
print(data[1]['summary'])


# In[4]:


# remove punctuation from all DOCs 
exclude = str.maketrans('', '', string.punctuation)
alldocslist = []

for i in data:
    text = (i['title'] + '\n' + i['summary']).translate(exclude)
    alldocslist.append(text)
    
print(alldocslist[1])


# In[5]:


#tokenize words in all DOCS 
plot_data = []

for doc in tqdm.tqdm(alldocslist):
    tokentext = word_tokenize(doc)
    plot_data.append(tokentext)
    
print(plot_data[1])


# In[6]:


# Navigation: first index gives all documents, second index gives specific document, third index gives words of that doc
plot_data[1][0:10]


# In[7]:


#make all words lower case for all docs 
for index, x in enumerate(plot_data):
    lowers = [word.lower() for word in x]
    plot_data[index] = lowers

plot_data[1][0:10]


# In[8]:


# remove stop words from all docs 
stop_words = set(stopwords.words('english'))

for index, x in enumerate(plot_data):
    filtered_sentence = [w for w in x if not w in stop_words]
    plot_data[index] = filtered_sentence

plot_data[1][0:10]


# In[9]:


#stem words EXAMPLE (could try others/lemmers )

snowball_stemmer = SnowballStemmer("english")
stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
print(stemmed_sentence[0:10])

porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
stemmed_sentence = [ porter_stemmer.stem(w) for w in filtered_sentence]
print(stemmed_sentence[0:10])


# # PART II: CREATING THE INVERSE-INDEX

# In[10]:


# Create inverse index which gives document number for each document and where word appears

#first we need to create a list of all words
flatten = [item for sublist in plot_data for item in sublist]
words = flatten
wordsunique = set(words)
wordsunique = list(wordsunique)


# In[11]:


# create functions for TD-IDF / BM25
import math
#from textblob import TextBlob as tb

def tf(word, doc):
    return doc.count(word) / len(doc)

def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc)

def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)))

def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))


# In[42]:


# Create dictonary of words
# THIS ONE-TIME INDEXING IS THE MOST PROCESSOR-INTENSIVE STEP AND WILL TAKE TIME TO RUN (BUT ONLY NEEDS TO BE RUN ONCE)
import os
import numpy as np

plottest = plot_data[:2000]

file_name = 'worddic_2000.npy'
if os.path.isfile(file_name):
    worddic = np.load(file_name, allow_pickle=True).item()
else:
    worddic = {}

    for index, doc in enumerate(tqdm.tqdm(plottest)):
        for word in wordsunique:
            if word in doc:
                positions = list(np.where(np.array(doc) == word)[0])
                idfs = tfidf(word,doc,plottest)
                try:
                    worddic[word].append([index,positions,idfs])
                except:
                    worddic[word] = []
                    worddic[word].append([index,positions,idfs])
                    
    # pickel (save) the dictonary to avoid re-calculating
    np.save('worddic_2000.npy', worddic)


# In[43]:


# the index creates a dic with each word as a KEY and a list of doc indexs, word positions, and td-idf score as VALUES
worddic['neural'][:30]


# # PART III: The Search Engine

# In[46]:


# create word search which takes multiple words and finds documents that contain both along with metrics for ranking:

    ## (1) Number of occruances of search words 
    ## (2) TD-IDF score for search words 
    ## (3) Percentage of search terms
    ## (4) Word ordering score 
    ## (5) Exact match bonus 


from collections import Counter

def search(searchsentence):
    try:
        # split sentence into individual words 
        searchsentence = searchsentence.lower()
        try:
            words = searchsentence.split(' ')
        except:
            words = list(words)
        enddic = {}
        idfdic = {}
        closedic = {}
        
        # remove words if not in worddic 
        realwords = []
        for word in words:
            if word in list(worddic.keys()):
                realwords.append(word)  
        words = realwords
        numwords = len(words)
        
        # make metric of number of occurances of all words in each doc & largest total IDF 
        for word in words:
            for indpos in worddic[word]:
                index = indpos[0]
                amount = len(indpos[1])
                idfscore = indpos[2]
                enddic[index] = amount
                idfdic[index] = idfscore
                fullcount_order = sorted(enddic.items(), key=lambda x:x[1], reverse=True)
                fullidf_order = sorted(idfdic.items(), key=lambda x:x[1], reverse=True)

                
        # make metric of what percentage of words appear in each doc
        combo = []
        alloptions = {k: worddic.get(k, None) for k in (words)}
        for worddex in list(alloptions.values()):
            for indexpos in worddex:
                for indexz in indexpos:
                    combo.append(indexz)
        comboindex = combo[::3]
        combocount = Counter(comboindex)
        for key in combocount:
            combocount[key] = combocount[key] / numwords
        combocount_order = sorted(combocount.items(), key=lambda x:x[1], reverse=True)
        
        # make metric for if words appear in same order as in search
        if len(words) > 1:
            x = []
            y = []
            for record in [worddic[z] for z in words]:
                for index in record:
                     x.append(index[0])
            for i in x:
                if x.count(i) > 1:
                    y.append(i)
            y = list(set(y))

            closedic = {}
            for wordbig in [worddic[x] for x in words]:
                for record in wordbig:
                    if record[0] in y:
                        index = record[0]
                        positions = record[1]
                        try:
                            closedic[index].append(positions)
                        except:
                            closedic[index] = []
                            closedic[index].append(positions)

            x = 0
            fdic = {}
            for index in y:
                csum = []
                for seqlist in closedic[index]:
                    while x > 0:
                        secondlist = seqlist
                        x = 0
                        sol = [1 for i in firstlist if i + 1 in secondlist]
                        csum.append(sol)
                        fsum = [item for sublist in csum for item in sublist]
                        fsum = sum(fsum)
                        fdic[index] = fsum
                        fdic_order = sorted(fdic.items(), key=lambda x:x[1], reverse=True)
                    while x == 0:
                        firstlist = seqlist
                        x = x + 1
        else:
            fdic_order = 0
                    
        # also the one above should be given a big boost if ALL found together 
           
        
        #could make another metric for if they are not next to each other but still close 
        
        
        return(searchsentence,words,fullcount_order,combocount_order,fullidf_order,fdic_order)
    
    except:
        return("")


search('convolutional neural network')[1]


# In[60]:


# 0 return will give back the search term, the rest will give back metrics (see above)

search('FPGA')


# In[51]:


# save metrics to dataframe for use in ranking and machine learning 
result1 = search('convolutional neural network')
result2 = search('computer vision')
result3 = search('natural language processing')
result4 = search('information retrieval')
result5 = search('speech recognition')
result6 = search('FPGA')
result7 = search('accelerator')
result8 = search('trade')
df = pd.DataFrame([result1,result2,result3,result4,result5,result6,result7,result8])
df.columns = ['search term', 'actual_words_searched','num_occur','percentage_of_terms','td-idf','word_order']
df


# In[54]:


# look to see if the top documents seem to make sense

print(alldocslist[1377])


# # PART IV: Rank and return (rules based)

# In[55]:


# create a simple (non-machine learning) rank and return function

def rank(term):
    results = search(term)

    # get metrics 
    num_score = results[2]
    per_score = results[3]
    tfscore = results[4]
    order_score = results[5]
    
    final_candidates = []

    # rule1: if high word order score & 100% percentage terms then put at top position
    try:
        first_candidates = []

        for candidates in order_score:
            if candidates[1] > 1:
                first_candidates.append(candidates[0])

        second_candidates = []

        for match_candidates in per_score:
            if match_candidates[1] == 1:
                second_candidates.append(match_candidates[0])
            if match_candidates[1] == 1 and match_candidates[0] in first_candidates:
                final_candidates.append(match_candidates[0])

    # rule2: next add other word order score which are greater than 1 

        t3_order = first_candidates[0:3]
        for each in t3_order:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    # rule3: next add top td-idf results
        final_candidates.insert(len(final_candidates),tfscore[0][0])
        final_candidates.insert(len(final_candidates),tfscore[1][0])

    # rule4: next add other high percentage score 
        t3_per = second_candidates[0:3]
        for each in t3_per:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    #rule5: next add any other top results for metrics
        othertops = [num_score[0][0],per_score[0][0],tfscore[0][0],order_score[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)
                
    # unless single term searched, in which case just return 
    except:
        othertops = [num_score[0][0],num_score[1][0],num_score[2][0],per_score[0][0],tfscore[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)

    for index, results in enumerate(final_candidates):
        if index < 5:
            print("RESULT", index + 1, ":", alldocslist[results][0:100],"...")


# In[56]:


# example of output 
rank('convolutional neural network')


# In[58]:


# example of output 
rank('speech recognition')


# # PART V: Rank and return (BM25)

# In[93]:


def rank_bm25(searchsentence):
    # split sentence into individual words 
    searchsentence = searchsentence.lower()
    try:
        words = searchsentence.split(' ')
    except:
        words = list(words)
    enddic = {}
    idfdic = {}
    closedic = {}

    # remove words if not in worddic
    words = [word for word in words if word in worddic]
    numwords = len(words)
    
    docs = {}
    for word in set(words):
        for doc in worddic[word]:
            docs[doc[0]] = 0
    
    num_docs = len(plottest)
    avg_dl = sum(len(x) for x in plottest) / num_docs
    
    k1 = 1.2
    b = 0.75
    k3 = 500
    
    for word in set(words):
        doc_count = len(worddic[word])
        IDF = math.log(1 + (num_docs - doc_count + 0.5) / (doc_count + 0.5))
        query_term_weight = words.count(word)
        QTF = (k3 + 1) * query_term_weight / (k3 + query_term_weight)
        for doc in worddic[word]:
            doc_term_count = len(doc[1])
            doc_size = len(plottest[doc[0]])
            TF = (k1 + 1) * doc_term_count / (k1 * (1 - b + b * doc_size / avg_dl) + doc_term_count)
            docs[doc[0]] += IDF * TF * QTF
    
    result = list(docs.items())
    result.sort(key = lambda x: -x[1])
    index = 0
    for r in result:
        if index >= 5:
            break
        print(f"RESULT {index + 1} (score = {r[1]}):", alldocslist[r[0]][0:100],"...")
        index += 1
        
    return result


# In[94]:


# example of output 
rank_bm25('convolutional neural network')


# In[109]:


def author_recommend(q: str):
    result = rank_bm25(q)
    authors = {}
    for r in result:
        for index, author in enumerate(eval(data[r[0]]['author'])):
            author_name = author['name']
            if author_name not in authors:
                authors[author_name] = (r[1] * (0.8**index), [data[r[0]]])
            else:
                authors[author_name] = (authors[author_name][0] + r[1] * (0.8**index), authors[author_name][1] + [data[r[0]]])
    authors = [(x, y[0], y[1]) for x, y in authors.items()]
    authors.sort(key = lambda x: x[1])
    return authors


# In[110]:


author_recommend('convolutional neural network')


# # TO-DO / Improvements
# 
# Indexer:
# - Improve stem/lemm
# - Add new metrics (e.g. bonus for exact matches / closeness metric)
# - Add BM25 (and variants)
# 
# Search Engine:
# - Add query expansion / synonyms
# - Add spellchecker functions
# - Add confidence level 
# 
# Data sources:
# - Find another source with a proper truth set
# - Download wikipedia and try it with this 
# 
# Machine Learning:
# - Fix ML example compiler (crashes if len(col) is so short it is an int and so no len function)
# - Try different algorithms 
# 
# GUI:
# - Build GUI interface
# - Add feedback mechanism

# In[ ]:




