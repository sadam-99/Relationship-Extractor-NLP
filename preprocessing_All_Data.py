# CODE AUTHOR
# SHIVAM GUPTA (NET ID: SXG19040)
# PRACHI VATS  (NET ID: PXV180021)
# Entities Relationship Extraction Project


import re, sys, nltk
import os
import numpy as np
from nltk.tokenize.stanford import StanfordTokenizer
path_to_jar = "data/stanford-postagger.jar"
tokenizer = StanfordTokenizer(path_to_jar)
import _pickle as pickle


# For Training Data

# Extracting the Relations 
# Please comment this when preprocessing the sentences.
# for training data open "TRAIN_FILE.TXT" and for test data open "TEST_FILE_FULL.TXT"

input_Sentence = []
for line in open("data/Main_Data/TRAIN_FILE.TXT"):
    input_Sentence.append(line.strip())

rel = []
for i, w in enumerate(input_Sentence):
    if((i+3)%4==0):
        rel.append(w)
        
f = open("data/Main_Data/train_relations.txt", 'w')
for relation in rel:
    f.write(relation+'\n')
print(relation)


test_lines = []
for line in open("data/Main_Data/TEST_FILE.TXT"):
    test_lines.append(line.strip())

test_relations = []
for i, w in enumerate(test_lines):
    
    if((i+3)%4==0):
        print(i, w)
        test_relations.append(w)
        
f = open("data/Main_Data/test_relations.txt", 'w')
c=0
for rel in test_relations:
    c=c+1
    f.write(rel+'\n')
# print(test_relations)



# len(test_relations)
# t_R = np.array(test_relations)
# len(np.unique(t_R))


import numpy as np
R = np.array(rel)
len(np.unique(R))
# np.unique(R)


# ## For training Preprocessing

# For preprocessing Training data open "TRAIN_FILE.TXT and for Test data open "TEST_FILE.txt
lines = []
# for line in open("data/TRAIN_FILE.TXT"): 
for line in open("data/Main_Data/TRAIN_FILE.TXT"):
    m = re.match(r'^([0-9]+)\s"(.+)"$', line.strip())
    if(m is not None):
        lines.append(m.group(2))

# len(rel)

data = []
entity1_POS = []
entity2_POS = []
for j,line in enumerate(lines):
    tokenList = []
    tempList = []
    t = line.split("<e1>")
    tokenList.append(t[0])
    tempList.append(t[0])

    t = t[1].split("</e1>")
    e1_text = tokenList
    e1_text = " ".join(e1_text)
    e1_text = nltk.word_tokenize(e1_text)
    tokenList.append(t[0])
    e11= t[0]
    y = nltk.word_tokenize(t[0])
    y[0] +="E11"
    tempList.append(" ".join(y))
    t = t[1].split("<e2>")
    tokenList.append(t[0])
    tempList.append(t[0])
    t = t[1].split("</e2>")
    e22 = t[0]
    e2_text = tokenList
    e2_text = " ".join(e2_text)
    e2_text = nltk.word_tokenize(e2_text)
    tokenList.append(t[0])
    tokenList.append(t[1])
    y = nltk.word_tokenize(t[0])
    y[0] +="E22"
    tempList.append(" ".join(y))
    tempList.append(t[1])

    tokenList = " ".join(tokenList)
    tokenList = nltk.word_tokenize(tokenList)
    tempList = " ".join(tempList)
    tempList = nltk.word_tokenize(tempList)


    q1 = nltk.word_tokenize(e11)[0]
    q2 = nltk.word_tokenize(e22)[0]
    for i, word in enumerate(tokenList):
        if(word.find(q1)!=-1):
            if(tempList[i].find("E11")!=-1):
                entity1_POS.append(i)            
                break
    for i, word in enumerate(tokenList):
        if(word.find(q2)!=-1):
                if(tempList[i].find("E22")!=-1):
                    entity2_POS.append(i)   
    tokenList = " ".join(tokenList)
    data.append(tokenList)
    print(j, tokenList)


len(data), len(entity1_POS), len(entity2_POS)

# for saving training data open "train_data" and for test data open "test_data"

with open('data/Main_Data/train_data', 'wb') as f:
    pickle.dump((data, entity1_POS, entity2_POS), f)
    f.close()


# k=0
# for tes_rel in test_relations:
#     if tes_rel is " ":
#         k+=1
#         print(tes_rel)
# k


# ## For Testing Preproceesing

# For preprocessing Training data open "TRAIN_FILE.TXT and for Test data open "TEST_FILE.txt
Test_lines = []
# for line in open("data/TRAIN_FILE.TXT"): 
for line in open("data/Main_Data/TEST_FILE.TXT"):
    m = re.match(r'^([0-9]+)\s"(.+)"$', line.strip())
    if(m is not None):
        Test_lines.append(m.group(2))


Test_sentences = []
entity1_POS = []
entity2_POS = []
for j,line in enumerate(Test_lines):
    tokenList = []
    tempList = []
    t = line.split("<e1>")
    tokenList.append(t[0])
    tempList.append(t[0])

    t = t[1].split("</e1>")
    e1_text = tokenList
    e1_text = " ".join(e1_text)
    e1_text = nltk.word_tokenize(e1_text)
    tokenList.append(t[0])
    e11= t[0]
    y = nltk.word_tokenize(t[0])
    y[0] +="E11"
    tempList.append(" ".join(y))
    t = t[1].split("<e2>")
    tokenList.append(t[0])
    tempList.append(t[0])
    t = t[1].split("</e2>")
    e22 = t[0]
    e2_text = tokenList
    e2_text = " ".join(e2_text)
    e2_text = nltk.word_tokenize(e2_text)
    tokenList.append(t[0])
    tokenList.append(t[1])
    y = nltk.word_tokenize(t[0])
    y[0] +="E22"
    tempList.append(" ".join(y))
    tempList.append(t[1])

    tokenList = " ".join(tokenList)
    tokenList = nltk.word_tokenize(tokenList)
    tempList = " ".join(tempList)
    tempList = nltk.word_tokenize(tempList)

    q1 = nltk.word_tokenize(e11)[0]
    q2 = nltk.word_tokenize(e22)[0]
    for i, word in enumerate(tokenList):
        if(word.find(q1)!=-1):
            if(tempList[i].find("E11")!=-1):
                entity1_POS.append(i)            
                break
    for i, word in enumerate(tokenList):
        if(word.find(q2)!=-1):
                if(tempList[i].find("E22")!=-1):
                    entity2_POS.append(i)   
    tokenList = " ".join(tokenList)
    Test_sentences.append(tokenList)
    print(j, tokenList)


len(Test_sentences), len(entity1_POS), len(entity2_POS)

# for saving training data open "train_data" and for test data open "test_data"

with open('data/Main_Data/test_data', 'wb') as f:
    pickle.dump((Test_sentences, entity1_POS, entity2_POS), f)
    f.close()

