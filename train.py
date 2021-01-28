import pandas as pd
import pickle
import sklearn
from sklearn import preprocessing

from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB

def train(path_to_data='dataset/dis_sym_dataset_comb.csv',
          mnb_save_path='weights/mnb_model.sav',
          cnb_save_path='weights/cnb_model.sav'):

    df_comb = pd.read_csv(path_to_data)

    le = preprocessing.LabelEncoder()
    le.fit(list(df_comb.label_dis.unique()))
    X = df_comb.iloc[:, 1:]
    y = le.transform(df_comb.label_dis)
    
    # Multinomal Classifier
    mnb = MultinomialNB()
    mnb = mnb.fit(X, y) #Train
    pickle.dump(mnb, open(mnb_save_path, 'wb')) #Saving the model weights here

    #Complement Naive Bayes
    cnb = ComplementNB()
    cnb = cnb.fit(X, y) #Train
    pickle.dump(cnb, open(cnb_save_path, 'wb')) #Saving the model weights here

    print("Done!")
    
train()