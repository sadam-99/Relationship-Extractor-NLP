import sys, os, _pickle as pickle
import tensorflow as tf
import numpy as np
import nltk
from sklearn.metrics import f1_score
import keras


data_dir = '../data'                        # Directory for Data and Other files
ckpt_dir = '../checkpoint'                  # Directory for Checkpoints 
word_embd_dir = '../checkpoint/word_embd'   # Directory for Checkpoints of Word Embedding Layer
# model_dir = '../checkpoint/modelv1'         # Directory for Checkpoints of Model
model_DIR = r"checkpoint/modelv1/" # Directory for Checkpoints of Model

word_embd_dim = 100       # Dimension of embedding layer for words
pos_embd_dim = 25         # Dimension of embedding layer for POS Tags
dep_embd_dim = 25         # Dimension of embedding layer for Dependency Types

word_vocab_size = 400001  # Vocab size for Words
# word_vocab_size = 100001  # Vocab size for Words
pos_vocab_size = 10       # Vocab size for POS Tags
dep_vocab_size = 21       # Vocab size for Dependency Types
word_state_size = 100
other_state_size = 100
relation_classes = 19     # No. of Relation Classes
state_size = 100          # Dimension of States of LSTM-RNNs
batch_size = 10           # Batch Size for training

channels = 100      # No. of types of features to feed in LSTM-RNN
lambda_l2 = 0.0001
max_len_path = 10         # Maximum Sent_Count of sequence


grap = tf.Graph()

with grap.as_default():
    with tf.name_scope("input"):
    
    # Length of the sequence = 2X10
        path_length = tf.placeholder(tf.int32, shape=[2, batch_size], name="path1_length") 

        # Words in the sequence  = 2X10X10
        word_ids = tf.placeholder(tf.int32, shape=[2, batch_size, max_len_path], name="word_ids") 

         # POS Tags in the sequence = 2X10X10
        pos_ids = tf.placeholder(tf.int32, [2, batch_size, max_len_path], name="pos_ids") 

        # Dependency Types in the sequence = 2X10X10
        dep_ids = tf.placeholder(tf.int32, [2, batch_size, max_len_path], name="dep_ids") 

         # True Relation btw the entities = [10]
        y = tf.placeholder(tf.int32, [batch_size], name="y")   
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

# Embedding Layer of Words 
with grap.as_default():
    with tf.name_scope("word_embedding"):
        W = tf.Variable(tf.constant(0.0, shape=[word_vocab_size, word_embd_dim]), name="W")
        embedding_placeholder = tf.placeholder(tf.float32,[word_vocab_size, word_embd_dim])
        embedding_init = W.assign(embedding_placeholder)
        embedded_word = tf.nn.embedding_lookup(W, word_ids)
        word_embedding_saver = tf.train.Saver({"word_embedding/W": W})
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

    # Embedding Layer of POS Tags 
    with tf.name_scope("pos_embedding"):
        W = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embd_dim]), name="W")
        embedded_pos = tf.nn.embedding_lookup(W, pos_ids)
        pos_embedding_saver = tf.train.Saver({"pos_embedding/W": W})
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

    # Embedding Layer of Dependency Types 
    with tf.name_scope("dep_embedding"):
        W = tf.Variable(tf.random_uniform([dep_vocab_size, dep_embd_dim]), name="W")
        embedded_dep = tf.nn.embedding_lookup(W, dep_ids)
        dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope


word_sz_sequence = 10
n_units = 100
n_layers = 3
channels_word=100
# tf.reset_default_graph()
# with grap.as_default():
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

    state_series = tf.concat([state_series1, state_series2], 1)
    init=tf.global_variables_initializer()   # Initializing all Variables within the scope
    print("state_series", state_series)


with grap.as_default():
    with tf.name_scope("hidden_layer"):
        W = tf.Variable(tf.truncated_normal([300, 100], -0.1, 0.1), name="W")
#         b = tf.Variable(tf.zeros([100]), name="b")
        b = tf.Variable(tf.zeros([100]), name="b")
        y_hidden_layer = tf.matmul(state_series, W) + b
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

    with tf.name_scope("softmax_layer"):
        W = tf.Variable(tf.truncated_normal([100, relation_classes], -0.1, 0.1), name="W")
        b = tf.Variable(tf.zeros([relation_classes]), name="b")
        logits = tf.matmul(y_hidden_layer, W) + b
        predictions = tf.argmax(logits, 1)
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

# with grap.as_default():
#     my_scope = 'softmax_layer'
#     scope_variables=  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= my_scope )
#     for scope_var in scope_variables:
#         print(scope_var.name)

with grap.as_default():
    tv_all = tf.trainable_variables()
    tv_regu = []
    non_reg = ["word_embedding/W:0","pos_embedding/W:0",'dep_embedding/W:0',"global_step:0",'hidden_layer/b:0','softmax_layer/b:0']
    for t in tv_all:
        if t.name not in non_reg:
#             if(t.name.find('biases')==-1):
            if(t.name.find('bias')==-1):
                tv_regu.append(t)


with grap.as_default():
    with tf.name_scope("loss"):
        l2_loss = lambda_l2 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv_regu ])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        total_loss = loss + l2_loss
        init=tf.global_variables_initializer()   # Initializing all Variables within the scope

    global_step = tf.Variable(0, name="global_step")

    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, global_step=global_step)
    init=tf.global_variables_initializer()  # Initializing the optimizer variables


f = open(data_dir + '/vocab.pkl', 'rb')
vocab = pickle.load(f)
f.close()

word2id = dict((w, i) for i,w in enumerate(vocab))
id2word = dict((i, w) for i,w in enumerate(vocab))

unknown_token = "UNKNOWN_TOKEN"
word2id[unknown_token] = word_vocab_size -1
id2word[word_vocab_size-1] = unknown_token

pos_tags_vocab = []
for line in open(data_dir + '/pos_tags.txt'):
        pos_tags_vocab.append(line.strip())

dep_vocab = []
for line in open(data_dir + '/dependency_types.txt'):
    dep_vocab.append(line.strip())

relation_vocab = []
for line in open(data_dir + '/relation_types.txt'):
    relation_vocab.append(line.strip())


rel2id = dict((w, i) for i,w in enumerate(relation_vocab))
id2rel = dict((i, w) for i,w in enumerate(relation_vocab))

pos_tag2id = dict((w, i) for i,w in enumerate(pos_tags_vocab))
id2pos_tag = dict((i, w) for i,w in enumerate(pos_tags_vocab))

dep2id = dict((w, i) for i,w in enumerate(dep_vocab))
id2dep = dict((i, w) for i,w in enumerate(dep_vocab))

pos_tag2id['OTH'] = 9
id2pos_tag[9] = 'OTH'

dep2id['OTH'] = 20
id2dep[20] = 'OTH'

JJ_pos_tags = ['JJ', 'JJR', 'JJS']
NN_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
RB_pos_tags = ['RB', 'RBR', 'RBS']
PRP_pos_tags = ['PRP', 'PRP$']
VB_pos_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
_pos_tags = ['CC', 'CD', 'DT', 'IN']

def pos_tag(x):
    if x in JJ_pos_tags:
        return pos_tag2id['JJ']
    if x in NN_pos_tags:
        return pos_tag2id['NN']
    if x in RB_pos_tags:
        return pos_tag2id['RB']
    if x in PRP_pos_tags:
        return pos_tag2id['PRP']
    if x in VB_pos_tags:
        return pos_tag2id['VB']
    if x in _pos_tags:
        return pos_tag2id[x]
    else:
        return 9



f = open(data_dir + '/Main_Data/train_paths', 'rb')
word_p1, word_p2, dep_p1, dep_p2, pos_p1, pos_p2 = pickle.load(f)
f.close()

relations = []
for line in open(data_dir + '/Main_Data/train_relations.txt'):
#     print(line.strip().split())
    relations.append(line.strip().split()[0])
# print(relations)


Sent_Count = len(word_p1)
num_batches = int(Sent_Count/batch_size)

for i in range(Sent_Count):
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
        word_p1[i][j] = word if word in word2id else unknown_token 
    for k, word in enumerate(word_p2[i]):
        word = word.lower()
        word_p2[i][k] = word if word in word2id else unknown_token 
    for l, d in enumerate(dep_p1[i]):
        dep_p1[i][l] = d if d in dep2id else 'OTH'
    for m, d in enumerate(dep_p2[i]):
        dep_p2[i][m] = d if d in dep2id else 'OTH'

word_p1_ids = np.ones([Sent_Count, max_len_path],dtype=int)
word_p2_ids = np.ones([Sent_Count, max_len_path],dtype=int)
pos_p1_ids = np.ones([Sent_Count, max_len_path],dtype=int)
pos_p2_ids = np.ones([Sent_Count, max_len_path],dtype=int)
dep_p1_ids = np.ones([Sent_Count, max_len_path],dtype=int)
dep_p2_ids = np.ones([Sent_Count, max_len_path],dtype=int)
rel_ids = np.array([rel2id[rel] for rel in relations])
path1_len = np.array([len(w) for w in word_p1], dtype=int)
path2_len = np.array([len(w) for w in word_p2])

for i in range(Sent_Count):
    for j, w in enumerate(word_p1[i]):
        word_p1_ids[i][j] = word2id[w]
    for j, w in enumerate(word_p2[i]):
        word_p2_ids[i][j] = word2id[w]
    for j, w in enumerate(pos_p1[i]):
        pos_p1_ids[i][j] = pos_tag(w)
    for j, w in enumerate(pos_p2[i]):
        pos_p2_ids[i][j] = pos_tag(w)
    for j, w in enumerate(dep_p1[i]):
        dep_p1_ids[i][j] = dep2id[w]
    for j, w in enumerate(dep_p2[i]):
        dep_p2_ids[i][j] = dep2id[w]



with grap.as_default():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
#     saver = tf.train.Saver(defer_build=True)

# with grap.as_default():
#     for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dep_embedding'):
#         print(i.name)

# model_dir = r"E:\The University of Texas at Dallas\Fall 2020\Natural Language Processing- CS 6320.501\NLP Project\Relation-Classification-using-Bidirectional-LSTM-Tree\Relation-Classification-using-Bidirectional-LSTM\checkpoint\modelv1"
# model = tf.train.latest_checkpoint(model_DIR)
# saver.restore(sess, model)
tf.train.latest_checkpoint(model_DIR)
# import numpy as np
# from tensorflow.python.layers import base
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# def model_summary():
#     model_vars = tf.trainable_variables()
#     slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# with grap.as_default():
#     model_summary()


with grap.as_default():
        train_vars= tf.trainable_variables()
        all_vars = tf.global_variables()
print("len(all_vars), len(train_vars)", len(all_vars), len(train_vars))
# model_DIR

num_epochs = 20
with grap.as_default():
    for epoch in range(num_epochs):
        for j in range(num_batches):
            path_dict = [path1_len[j*batch_size:(j+1)*batch_size], path2_len[j*batch_size:(j+1)*batch_size]]
            word_dict = [word_p1_ids[j*batch_size:(j+1)*batch_size], word_p2_ids[j*batch_size:(j+1)*batch_size]]
            pos_dict = [pos_p1_ids[j*batch_size:(j+1)*batch_size], pos_p2_ids[j*batch_size:(j+1)*batch_size]]
            dep_dict = [dep_p1_ids[j*batch_size:(j+1)*batch_size], dep_p2_ids[j*batch_size:(j+1)*batch_size]]
            y_dict = rel_ids[j*batch_size:(j+1)*batch_size]

            feed_dict = {
                path_length:path_dict,
                word_ids:word_dict,
                pos_ids:pos_dict,
                dep_ids:dep_dict,
                y:y_dict}
#             with grap.as_default():
    #             with tf.Session(graph=grap) as sess:
            # print("=======Sess Running=========")
            if j % 50 == 0:
                print("===== Trained Batches:", j,"Epoch Number", epoch)
            _, loss, step = sess.run([optimizer, total_loss, global_step], feed_dict)
            if step%10==0:
                print("=============Step:", step, "loss:============",loss)
        if (epoch % 1) == 0:
            os.makedirs(model_DIR + "Epoch" + str(epoch))
            Model_Path = os.path.join(model_DIR, "Epoch" + str(epoch) + "/")
            print(Model_Path)
            NUMBER_OF_CKPT = 200
            saver.save(sess, Model_Path, global_step=NUMBER_OF_CKPT)
            print("============Model has been Saved Model for Epoch Number:=======>", epoch)


# training accuracy
all_predictions = []
for j in range(num_batches):
    path_dict = [path1_len[j*batch_size:(j+1)*batch_size], path2_len[j*batch_size:(j+1)*batch_size]]
    word_dict = [word_p1_ids[j*batch_size:(j+1)*batch_size], word_p2_ids[j*batch_size:(j+1)*batch_size]]
    pos_dict = [pos_p1_ids[j*batch_size:(j+1)*batch_size], pos_p2_ids[j*batch_size:(j+1)*batch_size]]
    dep_dict = [dep_p1_ids[j*batch_size:(j+1)*batch_size], dep_p2_ids[j*batch_size:(j+1)*batch_size]]
    y_dict = rel_ids[j*batch_size:(j+1)*batch_size]

    feed_dict = {
        path_length:path_dict,
        word_ids:word_dict,
        pos_ids:pos_dict,
        dep_ids:dep_dict,
        y:y_dict}
    with grap.as_default():
        batch_predictions = sess.run(predictions, feed_dict)
        print("Sess Running for predictions")
        all_predictions.append(batch_predictions)

y_pred = []
for i in range(num_batches):
    for pred in all_predictions[i]:
        y_pred.append(pred)

count = 0
for i in range(batch_size*num_batches):
    count += y_pred[i]==rel_ids[i]
accuracy = count/(batch_size*num_batches) * 100

print("Final training accuracy", accuracy)