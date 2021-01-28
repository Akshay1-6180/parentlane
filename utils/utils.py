import pandas as pd
import numpy as np
from textblob import TextBlob
from mlxtend.frequent_patterns import apriori
from sklearn import preprocessing

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
from nltk.corpus import stopwords

df = pd.read_csv("dataset/dataset_cleaned.csv")
df_norm = pd.read_csv("dataset/dis_sym_dataset_norm.csv")
df_comb = pd.read_csv("dataset/dis_sym_dataset_comb.csv")
df_symps = df_norm[df_norm.columns.difference(['label_dis'])]


#########################

def most_frequent_symptoms():
    
    df_aff = df_symps[(df_symps > 0).sum(axis=1) >= 1]
    frequent_itemsets_plus = apriori(df_aff, min_support=0.07, 
                                     use_colnames=True).sort_values('support', ascending=False).reset_index(drop=True)

    frequent_itemsets_plus['length'] = frequent_itemsets_plus['itemsets'].apply(lambda x: len(x))
    a = frequent_itemsets_plus[(frequent_itemsets_plus['length']  == 1) &
                               (frequent_itemsets_plus['support'] >= 0.07)]
    freq_symp = [list(x)[0] for x in a["itemsets"].values]
    
    return freq_symp

#########################
def vocab():
    
    vocabulary = df_norm.columns.unique()
    vocabulary = vocabulary[1:]
    
    return vocabulary

def symptoms():
    symptoms_list = df_norm.columns[1:]
    
    return symptoms_list

def features():
    features_list = df_comb.iloc[:, 1:].columns
    
    return features_list

def sample_x(final_symptoms):
    sample_x = []
    for i in range(len(features())):
        sample_x.append(0)
    for j in range(len(features())):
        if(features()[j] in final_symptoms): 
            sample_x[j] = 1
            
    sample_x = np.array(sample_x).reshape(1,len(sample_x))
    
    return sample_x

def conditions():
    conditions_set = {}
    for i in df.Conditions.unique():
        conditions_set[i] = []

    for i in range(len(df["Conditions"])):
        conditions_set[df["Conditions"][i]].append(df["Symptoms"][i])
        
    return conditions_set
  
#########################
  
def pre_processing(question):
    def lemmatize_with_pos_tag(sentence):
        tokenized_sentence = TextBlob(sentence)
        tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
        words_and_tags = [(word, tag_dict.get(pos[0], 'n')) for word, pos in tokenized_sentence.tags]
        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        return " ".join(lemmatized_list)
    stop_words = set(stopwords.words('english'))  
  
    word_tokens = word_tokenize(question)  
  
    filtered_sentence = [w for w in word_tokens if not w in stop_words]  
  
    filtered_sentence = []  
  
    for w in word_tokens:  
        if w not in stop_words:  
            filtered_sentence.append(" " + w)
    filtered_sentence = lemmatize_with_pos_tag("".join(filtered_sentence))
    return filtered_sentence

#########################

def closest_symptoms():
    closest_symp = {}
    for i in df_symps.columns:
        closest_symp[i] = set()
    for c in df_symps.columns:
        indices = df_symps[df_symps[c] == 1].index.values
        for i in indices:
            for j in df_symps.columns[df_symps.iloc[i] == 1]:
                closest_symp[c].add(j)
        
        closest_symp[c].remove(c)
    
    return closest_symp


def predict(model, sample_, predictions, weight=1, k = 10, final_symptoms=None):
    le = preprocessing.LabelEncoder()
    le.fit(list(df_comb.label_dis.unique()))
    prediction = model.predict_proba(sample_x(final_symptoms))
    diseases = list(set(df_norm['label_dis']))
    diseases.sort()
    topk = prediction[0].argsort()[-k:][::-1]
    for pred in topk:
        try:
            predictions[le.inverse_transform([pred])[0]] += round(prediction[0][pred]/sum(prediction[0])*100, 2)*weight
        except KeyError:
            predictions[le.inverse_transform([pred])[0]] = round(prediction[0][pred]/sum(prediction[0])*100, 2)
    return predictions