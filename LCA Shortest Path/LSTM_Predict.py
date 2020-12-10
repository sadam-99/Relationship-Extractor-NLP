# CODE AUTHOR
# SHIVAM GUPTA (NET ID: SXG19040)
# PRACHI VATS  (NET ID: PXV180021)
# Entities Relationship Extraction Project

import re, sys, nltk
import _pickle as pickle
import pickle
import os
import nltk
from nltk.parse import stanford
# import warnings

java_path = "C:/Program Files/Java/jdk-13.0.2/bin/java.exe"
os.environ['JAVAHOME'] = java_path
# For preprocessing  Testing data open "TEST_FILE.txt"
lines = []
for line in open("../DATA_FILES/Main_Data/TEST_Sent.TXT"):
    m = re.match(r'^([0-9]+)\s"(.+)"$', line.strip())
    if(m is not None):
        lines.append(m.group(2))

batch_size = 10  
lines = lines*batch_size

import sys, os, _pickle as pickle
import tensorflow as tf
import numpy as np
import nltk
from sklearn.metrics import f1_score
import keras

DATA_DIR = '../DATA_FILES'                        # Directory for Data and Other files
CKPT_DIR = '../Checkpoint_Model'                  # Directory for Checkpoints 
MODEL_CKPT_DIR = r"Checkpoint_Model/Epochs/" # Directory for Checkpoints of Model

pos_embd_dim = 25         # Dimension of embedding layer for POS Tags
dep_embd_dim = 25         # Dimension of embedding layer for Dependency Types
word_embd_dim = 100       # Dimension of embedding layer for words


word_state_size = 100
other_state_size = 100
Relation_Labels = 19     # No. of Relation Classes
state_size = 100          # Dimension of States of LSTM-RNNs
batch_size = 10           # Batch Size for training
Vocab_SIZE = 400001  # Vocab size for Words
POS_Dep_size = 10       # Vocab size for POS Tags
DEP_Parse_size = 21       # Vocab size for Dependency Types


MAX_path_L = 10         # Maximum Sent_Number of sequence
channels = 100      # No. of types of features to feed in LSTM-RNN
lambda_l2 = 0.0001

# Dependency Tree
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordDependencyParser
os.environ['CLASSPATH'] = r"E:/The University of Texas at Dallas/Fall 2020\Natural Language Processing- CS 6320.501/NLP Project/stanford-parser-full-2020-11-17"
CFG_Mod_path = r"E:/The University of Texas at Dallas/Fall 2020/Natural Language Processing- CS 6320.501/NLP Project/Dep_Model/englishPCFG.caseless.ser.gz"
dep_parser=StanfordDependencyParser(model_path=CFG_Mod_path)
# print([parse.tree() for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")])


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
    # print(j, tokenList)
#     print(j, text)
entity1 = tokenList.split(" ")[entity1_POS[0]]
entity2 = tokenList.split(" ")[entity2_POS[0]]
print("entity1:---->", entity1 ,"=======", "entity2:----->",  entity2)



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


# This function Evaluated the Path from E1/E2 to the Lowest Common Ancestor
def path_lowestCommonAncester(tree, node, lca_node):
    all_path = []
    all_path.append(node)
    while(node != lca_node):
        node = tree.nodes[node['head']]
        all_path.append(node)
    return all_path


sentences = data
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
        parse_tree = dep_parser.raw_parse(sentences[i])
        for trees in parse_tree:
            tree = trees
        node1 = tree.nodes[entity1_POS[i]+1]
        node2 = tree.nodes[entity2_POS[i]+1]
        if node1['address']!=None and node2['address']!=None:
            # print(i, "success")
            lca_node = lowestCommonAncester(tree, node1, node2)
            path1 = path_lowestCommonAncester(tree, node1, lca_node)
            path2 = path_lowestCommonAncester(tree, node2, lca_node)

            word_path1[i] = [p["word"] for p in path1]
            word_path2[i] = [p["word"] for p in path2]
            rel_path1[i] = [p["rel"] for p in path1]
            rel_path2[i] = [p["rel"] for p in path2]
            pos_path1[i] = [p["tag"] for p in path1]
            pos_path2[i] = [p["tag"] for p in path2]
        else:
            print(i, node1["address"], node2["address"])
    except AssertionError:
        print(i, "error")


grap = tf.Graph()

with grap.as_default():
    with tf.name_scope("input"):
    
    # Length of the sequence = 2X10
        PATH_LEN = tf.placeholder(tf.int32, shape=[2, batch_size], name="path1_length") 

        # Words in the sequence  = 2X10X10
        TOKEN_ids = tf.placeholder(tf.int32, shape=[2, batch_size, MAX_path_L], name="word_ids") 

         # POS Tags in the sequence = 2X10X10
        POS_IDs = tf.placeholder(tf.int32, [2, batch_size, MAX_path_L], name="pos_ids") 

        # Dependency Types in the sequence = 2X10X10
        dep_ids = tf.placeholder(tf.int32, [2, batch_size, MAX_path_L], name="dep_ids") 

         # True Relation btw the entities = [10]
        True_Label = tf.placeholder(tf.int32, [batch_size], name="y")   
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

# Embedding Layer of Words 
with grap.as_default():
    with tf.name_scope("word_embedding"):
        W = tf.Variable(tf.constant(0.0, shape=[Vocab_SIZE, word_embd_dim]), name="W")
        embedding_placeholder = tf.placeholder(tf.float32,[Vocab_SIZE, word_embd_dim])
        embedding_init = W.assign(embedding_placeholder)
        embedded_word = tf.nn.embedding_lookup(W, TOKEN_ids)
        word_embedding_saver = tf.train.Saver({"word_embedding/W": W})
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

    # Embedding Layer of POS Tags 
    with tf.name_scope("pos_embedding"):
        W = tf.Variable(tf.random_uniform([POS_Dep_size, pos_embd_dim]), name="W")
        embedded_pos = tf.nn.embedding_lookup(W, POS_IDs)
        pos_embedding_saver = tf.train.Saver({"pos_embedding/W": W})
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

    # Embedding Layer of Dependency Types 
    with tf.name_scope("dep_embedding"):
        W = tf.Variable(tf.random_uniform([DEP_Parse_size, dep_embd_dim]), name="W")
        embedded_dep = tf.nn.embedding_lookup(W, dep_ids)
        dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope


n_units = 100
n_layers = 3
channels_word=100
word_sz_sequence = 10

#  For Word Embeddings
with grap.as_default():
    with tf.variable_scope("word_lstm1"):
        inputs = tf.keras.layers.Input(batch_shape=(batch_size, word_sz_sequence, channels_word))
        cells = [tf.keras.layers.GRUCell(n_units) for _ in range(n_layers)]
        state_series_w1 = tf.keras.layers.RNN(cells, stateful=True, return_sequences=True, return_state=False)(embedded_word[0])
        state_series_word1 = tf.reduce_max(state_series_w1, axis=1)
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope
        # print("state_series_w1", state_series_w1)
        # print("state_series_word1", state_series_word1)
    #  For Word Embeddings 2
    with tf.variable_scope("word_lstm2"):
        inputs = tf.keras.layers.Input(batch_shape=(batch_size, word_sz_sequence, channels_word))
        cells = [tf.keras.layers.GRUCell(n_units) for _ in range(n_layers)]
        state_series_w2 = tf.keras.layers.RNN(cells, stateful=True, return_sequences=True, return_state=False)(embedded_word[1])
        state_series_word2 = tf.reduce_max(state_series_w1, axis=1)
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope
        # print("state_series_w2", state_series_w2)
        # print("state_series_word2", state_series_word2)
    

#  For POS Embeddings 1
pos_sz_sequence_POS = 10
channels_POS = 25
n_units_POS = 25
with grap.as_default():
    with tf.variable_scope("pos_lstm1"):
        inputs = tf.keras.layers.Input(batch_shape=(batch_size, pos_sz_sequence_POS, channels_POS)) # 10X10X25
        cells = [tf.keras.layers.GRUCell(n_units_POS) for _ in range(n_layers)]
        state_series_p1 = tf.keras.layers.RNN(cells, stateful=True, return_sequences=True, return_state=False)(embedded_pos[0])
        state_series_pos1 = tf.reduce_max(state_series_p1, axis=1)
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope
        # print("state_series_p1", state_series_p1)
        # print("state_series_pos1", state_series_pos1)

    #  For POS Embeddings 2
    with tf.variable_scope("pos_lstm2"):
        inputs = tf.keras.layers.Input(batch_shape=(batch_size, pos_sz_sequence_POS, channels_POS)) # 10X25X3
        cells = [tf.keras.layers.GRUCell(n_units_POS) for _ in range(n_layers)]
        state_series_p2 = tf.keras.layers.RNN(cells, stateful=True, return_sequences=True, return_state=False)(embedded_pos[1])
        state_series_pos2 = tf.reduce_max(state_series_p2, axis=1)
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope
        # print("state_series_w2", state_series_p2)
        # print("state_series_pos2", state_series_pos2)


# with grap.as_default():
#     my_scope = 'pos_lstm2'
#     scope_variables=  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=my_scope )
#     print(scope_variables)


#  For POS Embeddings 2
dep_sz_sequence = 10
channels_DEP = 25
n_units_DEP = 25
with grap.as_default():
    with tf.variable_scope("dep_lstm1"):
        inputs = tf.keras.layers.Input(batch_shape=(batch_size, dep_sz_sequence, channels_DEP))
        cells = [tf.keras.layers.GRUCell(n_units_DEP) for _ in range(n_layers)]
        state_series_d1 = tf.keras.layers.RNN(cells, stateful=True, return_sequences=True, return_state=False)(embedded_dep[0])
        state_series_dep1 = tf.reduce_max(state_series_d1, axis=1)
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope
        # print("state_series_d1", state_series_d1)
        # print("state_series_dep1", state_series_dep1)

    with tf.variable_scope("dep_lstm2"):
        inputs = tf.keras.layers.Input(batch_shape=(batch_size, dep_sz_sequence, channels_DEP))#10X10X25
        cells = [tf.keras.layers.GRUCell(n_units_DEP) for _ in range(n_layers)]
        state_series_d2 = tf.keras.layers.RNN(cells, stateful=True, return_sequences=True, return_state=False)(embedded_dep[1])
        state_series_dep2 = tf.reduce_max(state_series_d2, axis=1)
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope
        # print("state_series_d2", state_series_d2)
        # print("state_series_dep2", state_series_dep2)


with grap.as_default():
    state_series1 = tf.concat([state_series_word1, state_series_pos1, state_series_dep1], 1)
    state_series2 = tf.concat([state_series_word2, state_series_pos2, state_series_dep2], 1)

    Concat_state_series = tf.concat([state_series1, state_series2], 1)
    init=tf.global_variables_initializer()   # Initializing all Variables within the scope
    print("Concat_state_series", Concat_state_series)


with grap.as_default():
    with tf.name_scope("hidden_layer"):
        W = tf.Variable(tf.truncated_normal([300, 100], -0.1, 0.1), name="W")
#         b = tf.Variable(tf.zeros([100]), name="b")
        b = tf.Variable(tf.zeros([100]), name="b")
        y_hidden_layer = tf.matmul(Concat_state_series, W) + b
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

    with tf.name_scope("softmax_layer"):
        W = tf.Variable(tf.truncated_normal([100, Relation_Labels], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([Relation_Labels]), name="b")
        logits = tf.matmul(y_hidden_layer, W) + b
        predictions = tf.argmax(logits, 1)
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

# with grap.as_default():
#     my_scope = 'softmax_layer'
#     scope_variables=  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= my_scope )
#     for scope_var in scope_variables:
#         print(scope_var.name)

with grap.as_default():
    Train_VAR = tf.trainable_variables()
    Trainable_VAR_R = []
    non_reg = ["word_embedding/W:0","pos_embedding/W:0",'dep_embedding/W:0',"global_step:0",'hidden_layer/b:0','softmax_layer/b:0']
    for t in Train_VAR:
        if t.name not in non_reg:
#             if(t.name.find('biases')==-1):
            if(t.name.find('bias')==-1):
                Trainable_VAR_R.append(t)


with grap.as_default():
    with tf.name_scope("loss"):
        l2_loss = lambda_l2 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in Trainable_VAR_R ])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=True_Label))
        total_loss = loss + l2_loss
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

    global_step = tf.Variable(0, name="global_step")

    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, global_step=global_step)
    init=tf.global_variables_initializer()  # Initializing the optimizer variables


f = open(DATA_DIR + '/vocab.pkl', 'rb')
vocab = pickle.load(f)
f.close()

Token_2_ID = dict((w, i) for i,w in enumerate(vocab))
ID_2_Token = dict((i, w) for i,w in enumerate(vocab))

unknown_token = "UNKNOWN_TOKEN"
Token_2_ID[unknown_token] = Vocab_SIZE -1
ID_2_Token[Vocab_SIZE-1] = unknown_token

pos_tags_vocab = []
for line in open(DATA_DIR + '/pos_tags.txt'):
        pos_tags_vocab.append(line.strip())

dep_vocab = []
for line in open(DATA_DIR + '/dependency_types.txt'):
    dep_vocab.append(line.strip())

relation_vocab = []
for line in open(DATA_DIR + '/relation_types.txt'):
    relation_vocab.append(line.strip())


Label_2_ID = dict((w, i) for i,w in enumerate(relation_vocab))
ID_2_Label = dict((i, w) for i,w in enumerate(relation_vocab))

POS_2_ID = dict((w, i) for i,w in enumerate(pos_tags_vocab))
ID_2_POS = dict((i, w) for i,w in enumerate(pos_tags_vocab))

DEP_parse_2_ID = dict((w, i) for i,w in enumerate(dep_vocab))
ID_2_DEP_parse = dict((i, w) for i,w in enumerate(dep_vocab))

POS_2_ID['OTH'] = 9
ID_2_POS[9] = 'OTH'

DEP_parse_2_ID['OTH'] = 20
ID_2_DEP_parse[20] = 'OTH'

RB_pos_tags = ['RB', 'RBR', 'RBS']
PRP_pos_tags = ['PRP', 'PRP$']
VB_pos_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
_pos_tags = ['CC', 'CD', 'DT', 'IN']
JJ_pos_tags = ['JJ', 'JJR', 'JJS']
NN_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']


def Get_POS_Tagging(x):
    if x in JJ_pos_tags:
        return POS_2_ID['JJ']
    if x in NN_pos_tags:
        return POS_2_ID['NN']
    if x in RB_pos_tags:
        return POS_2_ID['RB']
    if x in PRP_pos_tags:
        return POS_2_ID['PRP']
    if x in VB_pos_tags:
        return POS_2_ID['VB']
    if x in _pos_tags:
        return POS_2_ID[x]
    else:
        return 9



# f = open(data_dir + '/test_pathsv1', 'rb')
# word_p1, word_p2, dep_p1, dep_p2, pos_p1, pos_p2 = pickle.load(f)
# f.close()
word_p1, word_p2, dep_p1, dep_p2, pos_p1, pos_p2  = word_path1, word_path2, rel_path1, rel_path2, pos_path1, pos_path2
# print(word_p1, word_p2, dep_p1, dep_p2, pos_p1, pos_p2)

relations = []
for line in open(DATA_DIR + '/test_relations.txt'):
    relations.append(line.strip().split()[0])



Sent_Number = len(word_p1)
num_batches = int(Sent_Number/batch_size)

for i in range(Sent_Number):
    if type(word_p1[i]) is not list:
        word_p1[i] = [""]
    if type(word_p2[i]) is not list:
        word_p2[i] = [""]
    if type(dep_p1[i]) is not list:
        dep_p1[i] = [""]
    if type(dep_p2[i]) is not list:
        dep_p2[i] = [""]
    if type(pos_p1[i]) is not list:
        pos_p1[i] = [""]
    if type(pos_p2[i]) is not list:
        pos_p2[i] = [""]
    for j, word in enumerate(word_p1[i]):
        word = word.lower()
        word_p1[i][j] = word if word in Token_2_ID else unknown_token 
    for k, word in enumerate(word_p2[i]):
        word = word.lower()
        word_p2[i][k] = word if word in Token_2_ID else unknown_token 
    for l, d in enumerate(dep_p1[i]):
        dep_p1[i][l] = d if d in DEP_parse_2_ID else 'OTH'
    for m, d in enumerate(dep_p2[i]):
        dep_p2[i][m] = d if d in DEP_parse_2_ID else 'OTH'

word_p1_ids = np.ones([Sent_Number, MAX_path_L],dtype=int)
word_p2_ids = np.ones([Sent_Number, MAX_path_L],dtype=int)
pos_p1_ids = np.ones([Sent_Number, MAX_path_L],dtype=int)
pos_p2_ids = np.ones([Sent_Number, MAX_path_L],dtype=int)
dep_p1_ids = np.ones([Sent_Number, MAX_path_L],dtype=int)
dep_p2_ids = np.ones([Sent_Number, MAX_path_L],dtype=int)
rel_ids = np.array([Label_2_ID[rel] for rel in relations])
path1_len = np.array([len(w) for w in word_p1], dtype=int)
path2_len = np.array([len(w) for w in word_p2])

for i in range(Sent_Number):
    for j, w in enumerate(word_p1[i]):
        word_p1_ids[i][j] = Token_2_ID[w]
    for j, w in enumerate(word_p2[i]):
        word_p2_ids[i][j] = Token_2_ID[w]
    for j, w in enumerate(pos_p1[i]):
        pos_p1_ids[i][j] = Get_POS_Tagging(w)
    for j, w in enumerate(pos_p2[i]):
        pos_p2_ids[i][j] = Get_POS_Tagging(w)
    for j, w in enumerate(dep_p1[i]):
        dep_p1_ids[i][j] = DEP_parse_2_ID[w]
    for j, w in enumerate(dep_p2[i]):
        dep_p2_ids[i][j] = DEP_parse_2_ID[w]



# MODEL_CKPT_DIR = r"E:\The University of Texas at Dallas\Fall 2020\Natural Language Processing- CS 6320.501\NLP Project\Relation-Classification-using-Bidirectional-LSTM-Tree\Relation-Classification-using-Bidirectional-LSTM\LCA Shortest Path\checkpoint\modelv1\Epoch19"
# MODEL_CKPT_DIR = r"E:\The University of Texas at Dallas\Fall 2020\Natural Language Processing- CS 6320.501\NLP Project\Relation-Classification-using-Bidirectional-LSTM-Tree\Relation-Classification-using-Bidirectional-LSTM\LCA Shortest Path\checkpoint\NewTraining\Epoch15"
# MODEL_CKPT_DIR = r"E:\The University of Texas at Dallas\Fall 2020\Natural Language Processing- CS 6320.501\NLP Project\Submision\SDP-LSTM-Model\Checkpoint_Model\Latesttraining\Epoch26"
MODEL_CKPT_DIR = r"E:\The University of Texas at Dallas\Fall 2020\Natural Language Processing- CS 6320.501\NLP Project\Submision\SDP-LSTM-Model\Checkpoint_Model\Epochs\Epoch100"
# MODEL_CKPT_DIR
with grap.as_default():
    sess_f1 = tf.Session()
    sess_f1.run(tf.global_variables_initializer())

# Testing Begins
with grap.as_default():
    checkpoint_file = tf.train.latest_checkpoint(MODEL_CKPT_DIR)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess_f1, checkpoint_file)
    all_predictions = []
    for j in range(num_batches):
        path_dict = [path1_len[j*batch_size:(j+1)*batch_size], path2_len[j*batch_size:(j+1)*batch_size]]
        word_dict = [word_p1_ids[j*batch_size:(j+1)*batch_size], word_p2_ids[j*batch_size:(j+1)*batch_size]]
        pos_dict = [pos_p1_ids[j*batch_size:(j+1)*batch_size], pos_p2_ids[j*batch_size:(j+1)*batch_size]]
        dep_dict = [dep_p1_ids[j*batch_size:(j+1)*batch_size], dep_p2_ids[j*batch_size:(j+1)*batch_size]]
        # y_dict = rel_ids[j*batch_size:(j+1)*batch_size]

        Tensor_DICT = {
            PATH_LEN:path_dict,
            TOKEN_ids:word_dict,
            POS_IDs:pos_dict,
            dep_ids:dep_dict,
            }
            # True_Label:y_dict
        batch_predictions = sess_f1.run(predictions, Tensor_DICT)
        all_predictions.append(batch_predictions)

    y_pred = []
    for i in range(num_batches):
        for pred in all_predictions[i]:
            y_pred.append(pred)

    # count = 0
    # for i in range(batch_size*num_batches):
    #     count += y_pred[i]==rel_ids[i]
    # accuracy = count/(batch_size*num_batches) * 100
    # print("test accuracy", accuracy)

Relation_Pred = batch_predictions[0]

print("\n")
print("The Test Sentence is:", lines[0])
print("The Entitity Relationship between", "entity1:", entity1 ,"AND", "entity2:", entity2, ":----> \n", ID_2_Label[Relation_Pred])
print("\n")