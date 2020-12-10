# CODE AUTHOR
# SHIVAM GUPTA (NET ID: SXG19040)
# PRACHI VATS  (NET ID: PXV180021)
# Entities Relationship Extraction Project

import os
import nltk
java_path = "C:/Program Files/Java/jdk-13.0.2/bin/java.exe"
os.environ['JAVAHOME'] = java_path
import os
from nltk.parse import stanford
from nltk.parse.stanford import StanfordDependencyParser
import pickle
os.getcwd()
# nltk.download()


# Dependency Tree
os.environ['CLASSPATH'] = r"E:/The University of Texas at Dallas/Fall 2020\Natural Language Processing- CS 6320.501/NLP Project/stanford-parser-full-2020-11-17"
Mod_path = r"E:/The University of Texas at Dallas/Fall 2020/Natural Language Processing- CS 6320.501/NLP Project/Dep_Model/englishPCFG.caseless.ser.gz"
dependencyParser = StanfordDependencyParser(model_path=Mod_path)
print([parse.tree() for parse in dependencyParser.raw_parse("University of Dallas of texas is located in US.")])

# This function calculates the Lowest Common Ancestor Along the Shortest path to the Root Node
def lowestCommonAncester(tree, index1, index2):
    node = index1
    path_1 = []
    path_2 = []
    path_1.append(index1)
    path_2.append(index2)
    while(node != tree.root):
        node = tree.nodes[node['head']]
        path_1.append(node)
    node = index2
    while(node != tree.root):
        node = tree.nodes[node['head']]
        path_2.append(node)
    for l1, l2 in zip(path_1[::-1],path_2[::-1]):
        if(l1==l2):
            nodeTemp = l1
    return nodeTemp


# In[6]:


# This function Evaluated the Path from E1/E2 to the Lowest Common Ancestor
def path_lowestCommonAncester(tree, node, lca_node):
    all_path = []
    all_path.append(node)
    while(node != lca_node):
        node = tree.nodes[node['head']]
        all_path.append(node)
    return all_path



# Picke for Train Data
import _pickle 
f = open("../data/Main_Data/train_data", 'rb')
sentences, e1, e2 = _pickle.load(f)
f.close()




Train_Sent_Count = len(sentences)



Train_Sent_Count = len(sentences)
word_path1 = []
word_path2 = []
rel_path1 = []
rel_path2 = []
pos_path1 = []
pos_path2 = []
for i in range(Train_Sent_Count):
    word_path1.append(0)
    word_path2.append(0)
    rel_path1.append(0)
    rel_path2.append(0)
    pos_path1.append(0)
    pos_path2.append(0)



for i in range(Train_Sent_Count):
    try:
        parse_tree = dependencyParser.raw_parse(sentences[i])
        for trees in parse_tree:
            tree = trees
        node_1 = tree.nodes[e1[i]+1]
        node_2 = tree.nodes[e2[i]+1]
        if node_1['address']!=None and node_2['address']!=None:
            print(i, "success")
            lca_node = lowestCommonAncester(tree, node_1, node_2)
            path1 = path_lowestCommonAncester(tree, node_1, lca_node)
            path2 = path_lowestCommonAncester(tree, node_2, lca_node)

            word_path1[i] = [p["word"] for p in path1]
            word_path2[i] = [p["word"] for p in path2]
            rel_path1[i] = [p["rel"] for p in path1]
            rel_path2[i] = [p["rel"] for p in path2]
            pos_path1[i] = [p["tag"] for p in path1]
            pos_path2[i] = [p["tag"] for p in path2]
        else:
            print(i, node_1["address"], node_2["address"])
    except AssertionError:
        print(i, "error")
    

with open('../data/Main_Data/trainpaths/train_paths', 'wb') as f:
    pickle.dump([word_path1, word_path2, rel_path1, rel_path2, pos_path1, pos_path2], f)


f = open('../data/Main_Data/trainpaths/train_paths_new', 'rb')
word_p1, word_p2, dep_p1, dep_p2, pos_p1, pos_p2 = pickle.load(f)
