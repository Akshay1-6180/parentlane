import random
from sentence_transformers import SentenceTransformer
from utils.utils import pre_processing, symptoms, most_frequent_symptoms, closest_symptoms, predict, sample_x
from utils.keywords import relevant_symptoms
import itertools
import pickle

model = SentenceTransformer('bert_model')
final_symptoms = set()
unselected_symptoms = set()
closest_words1 = set()
closest_words2 = set()
exit = 0
count = 0
co_symps = set()
random_symps = []

print(
'''

         ========================================
         |                                      |
         |                                      |
         |    WELCOME TO THE PARENTLANE APP     |
         |                                      |
         |                                      |
         ========================================

'''
    )

description = input("Tell me the symptoms you are experiencing: ")
filtered_description = pre_processing(description)
try:
    keyword_symptoms = relevant_symptoms(filtered_description, model)
except ValueError:
    keyword_symptoms = []

for keys in keyword_symptoms:
    qu = keys.split()
    for s in symptoms():
        if(qu[0] in pre_processing(s)):
            closest_words1.add(s)
        if(qu[1] in pre_processing(s)):
            closest_words2.add(s)
            
closest_words =  closest_words1.union(closest_words2)

random_symps = []
if (len(closest_words)>1):
    num = 17
    if (len(closest_words) < 17):
        num = len(closest_words)
    print("\n\nPlease enter the numbers (with spaces) of the appropriate symptoms from this list:\n")
    for idx, symp in enumerate(random.sample(closest_words, num)):
        random_symps.append(symp)
        unselected_symptoms.add(symp)
        print(idx,":",symp)
        
        if (idx>0 and idx%8==0):
            #print("\nDo you have any other symptoms from the following list?\n")
            closest_word = list(closest_words)
            select_list = input("\nPlease select the relevant symptoms (if none, say none): ").split()
            print('\n')
            if (select_list[0]=="none"):
                select_list = []
            for i in select_list:
                final_symptoms.add(random_symps[int(i)])
                unselected_symptoms.difference(final_symptoms)
else:
    print("Couldnt find matching symptoms, please select symptoms from this list")
for idx, symp in enumerate(most_frequent_symptoms()[:10]):
    unselected_symptoms.add(symp)
    print(idx,":",symp)
    
select_list = input("\nPlease select the relevant symptoms(or, say none): ").lower().split()
print("\n")

if select_list[0] != 'none':
    for i in select_list:
        final_symptoms.add(most_frequent_symptoms()[int(i)])
        unselected_symptoms.difference(final_symptoms)
else:
    pass
    
###########################

for subset in itertools.combinations(final_symptoms, 2):
    count+=1
    symps = closest_symptoms()[subset[0]].intersection(closest_symptoms()[subset[1]])
    symps = symps.difference(final_symptoms)
    co_symps = co_symps.union(symps)

if (len(co_symps)>1):
    num = 24
    if (len(co_symps)<25):
        num = len(co_symps)
    for idx, symp in enumerate(random.sample(co_symps, num)):
        random_symps.append(symp)
        unselected_symptoms.add(symp)
        print(idx,":",symp)
        
        if ((idx>0 and idx%8==0 ) or idx==len(co_symps)-1):
            #print("Top matching symptoms from your search!")

            select_list = input("\nPlease select the final relevant symptoms(or, say none): ").split()
            if (select_list[0]=="none"):
                select_list = []
            for i in select_list:
                final_symptoms.add(random_symps[int(i)])
                unselected_symptoms.difference(final_symptoms)
                
print("\nFinal Symptoms: {}".format(final_symptoms))

predictions = {}
mnb = pickle.load(open('weights/mnb_model.sav', 'rb'))
cnb = pickle.load(open('weights/cnb_model.sav', 'rb'))
predictions = predict(mnb,sample_x(final_symptoms), predictions, final_symptoms=final_symptoms)
predictions = predict(cnb,sample_x(final_symptoms), predictions,4, final_symptoms=final_symptoms)
print("Our best prediction: \n")
for cond, acc in sorted(predictions.items(), key=lambda x:x[1],reverse=True):
    print(cond, acc)