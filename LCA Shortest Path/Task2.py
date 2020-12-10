# CODE AUTHOR
# SHIVAM GUPTA (NET ID: SXG19040)
# PRACHI VATS  (NET ID: PXV180021)
# Entities Relationship Extraction Project

import numpy as np
import pandas as pd
import nltk
import spacy

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
TokenNLP = spacy.load("en_core_web_sm")

import re
from nltk.stem import WordNetLemmatizer 
from spacy import displacy
from IPython.core.display import display, HTML
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")
class NLP_Part2:

  def __init__(self, path):
    self.path = path
    

  def clean_str(self, text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

  def preprocess(self):
    sentenceList = []
    relationList = []
    lines = [line.strip() for line in open(self.path)]
    max_sentence_length = 0
    for idx in range(0, len(lines), 4):
      id = lines[idx].split("\t")[0]
      relation = lines[idx + 1]
      sentence = lines[idx].split("\t")[1][1:-1]
      sentence = sentence.replace('<e1>', ' _e11_ ')
      sentence = sentence.replace('</e1>', ' _e12_ ')
      sentence = sentence.replace('<e2>', ' _e21_ ')
      sentence = sentence.replace('</e2>', ' _e22_ ')
      sentence = self.clean_str(sentence)
      relationList.append(relation)
      sentenceList.append(sentence)
    return relationList, sentenceList

  
  def getRelationsOfSentences(self):
    relationMap = {}
    relationCount = 0
    relationList, sentenceList = self.preprocess()
    for sentence in sentenceList:
      e11Index = sentence.index('e11') + 3
      e12Index = sentence.index('e12')
      firstEntity = sentence[e11Index : e12Index]

      e21Index = sentence.index('e21') + 3
      e22Index = sentence.index('e22') 
      secondEntity = sentence[e21Index : e22Index]
      #print(firstEntity, " ", secondEntity, " ", relationList[relationCount])
      relationMap[relationCount] = [firstEntity, secondEntity, relationList[relationCount]]
      relationCount += 1
    print(" Entities and relation between them is -> ", relationMap)  
    return relationMap

  def tokenize(self, sentenceList):
    sentenceCount = 0
    tokenMap = {}
    for sentence in sentenceList:
      #print(sentenceCount, " ", sentence)
      sentence = sentence.replace('e11', '').replace('e12','').replace('e21','').replace('e22','')
      tokens = nltk.word_tokenize(sentence)
      tokenMap[sentenceCount] = tokens
      sentenceCount += 1
    print("Tokens of the sentence are -> " , tokenMap)
    return tokenMap

  def lemmatize(self, tokenMap):
    lemmatizer = WordNetLemmatizer() 
    lemmaMap = {}
    lemmaCount = 0
    for tokenList in tokenMap.values():
      lemmaList = []
      #print(" tokenList ", tokenList , "/n")
      for token in tokenList:
        lemmaList.append(lemmatizer.lemmatize(token))
      lemmaMap[lemmaCount] = lemmaList
      lemmaCount += 1
    print("Lemmas are -> ", lemmaMap)
    return lemmaMap

  def posTags(self, tokenMap):
    posMap = {}
    posCount = 0
    for tokenList in tokenMap.values():
      posTags= nltk.pos_tag(tokenList)
      #print("posTags ", posTags)
      posMap[posCount] = posTags
      posCount += 1
    print("Pos Tags are -> ", posMap)
    return posMap

  def getHyponymsAndHypernyms(self, tokenMap):
    print(" Hypernyms, Hyponyms, Meronyms are  ->   ")
    for sentence in sentenceList:
      sentence = sentence.replace('e11', " ").replace('e12'," ").replace('e21'," ").replace('e22'," ")
      tokendoc = TokenNLP(sentence)
      for token in tokendoc:
        print(" Token -> ", token)
        for ss in wn.synsets(token.lemma_):
          
          print ("hypernyms: "+str(ss.hypernyms())+" hyponyms: " + str(ss.hyponyms())+" holonyms: " + str(ss.member_holonyms())+" meronyms: "+ str(ss.part_meronyms()))
            
  def NERTags(self):
    NERMap = {}
    NERCount = 0
    for firstEntity, secondEntity, relationList in self.getRelationsOfSentences().values():
      docFirst = nlp(firstEntity)
      docSecond = nlp(secondEntity)
      NERList = []
      for ent in docFirst.ents:
        NERList.append((ent.text, ent.label_))
      for ent in docSecond.ents:
        NERList.append((ent.text, ent.label_))
      NERMap[NERCount] = NERList
      NERCount += 1
    
    print("NER Tags of the Entities are -> ", NERMap)
    return NERMap  

  def dependencyTree(self, sentenceList):
  
    dependecyMap = {}
    dependecyCount = 0
    for sentence in sentenceList:
      sentence = sentence.replace('e11', " ").replace('e12'," ").replace('e21'," ").replace('e22'," ")
      depedenlist = []
      tokendoc = TokenNLP(sentence)
      for token in tokendoc:
        depedenlist.append((token,token.dep_))
      dependecyMap[dependecyCount] = depedenlist
      dependecyCount += 1
      html = displacy.render(tokendoc, style="dep")
      # Uncomment below 2 lines
      print("Dependency Tree is ")
      display(HTML(html))
    print(" Dependency Tags are -> ", dependecyMap)

# Function calling Part
nlpPart2 = NLP_Part2("../DATA_FILES/Main_Data/Task2_TEST.TXT")
relationList, sentenceList = nlpPart2.preprocess()
tokenMap = nlpPart2.tokenize(sentenceList)
nlpPart2.lemmatize(tokenMap)
nlpPart2.posTags(tokenMap)
nlpPart2.getHyponymsAndHypernyms(sentenceList)
nlpPart2.NERTags()
nlpPart2.dependencyTree(sentenceList)