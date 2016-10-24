#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, absolute_import,
                        print_function, unicode_literals)
import math
import random
import zipfile
import argparse
import collections

from six.moves import xrange
import numpy as np
import tensorflow as tf




filename = 'text8.zip'

# Read the data into a list of strings.
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)
print('Data size', len(words))

vocab_size = 50000


# Step 1: Build the dictionary and replace rare words with UNK token.
# sort by frequency, most frequent word is indexed as 0
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    # map from word to id
    word2id = {}
    for word, _ in count:
        word2id[word] = len(word2id)
    data = []
    unk_count = 0
    for word in words:
        if word in word2id:
            index = word2id[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # map from id to word
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return data, count, word2id, id2word



data, count, word2id, id2word = build_dataset(words)
del words

# To store the path and the binary code for every word in the vovab list, the index is the same as the count_list, that is data_index starting from 0
path = []
code = []

def build_huffman(count):
    global vocab_size
    count = [c for c,_ in count] + [1e15]* (vocab_size - 1)
    parent = [0] * (2 * vocab_size -2)
    binary = [0] * (2 * vocab_size -2)
    
    pos1 = vocab_size - 1
    pos2 = vocab_size

    for i in xrange(vocab_size - 1):
        # Find min1
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min1 = pos1
                pos1 -= 1
            else:
                min1 = pos2
                pos2 += 1
        else:
            min1 = pos2
            pos2 += 1

        # Find min2
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min2 = pos1
                pos1 -= 1
            else:
                min2 = pos2
                pos2 += 1
        else:
            min2 = pos2
            pos2 += 1
        count[vocab_size + i] = count[min1] + count[min2]
        parent[min1] = vocab_size + i
        parent[min2] = vocab_size + i
        binary[min2] = 1
        
    root_idx = 2 * vocab_size - 2
    for i in range(0, vocab_size):
        path_i = []
        code_i = []
        node_idx = i
        while node_idx < root_idx:
            if nodex_idx > vocab_size: path_i.append(node_idx)    
            code_i.append(binary[node_idx])
            node_idx = parent[node_idx]
        path_i.append(root_idx)
        path.append([j - vocab_size for j in path[::-1]])
        code.append(code_i[::-1])
    
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [id2word[i] for i in data[:10]])


# Step 2: Generate a training batch for the skip-gram model.
data_index = 0

# change this part to realize the CBOW model
# num_skips is the surrounding size, which should be consistent with the skip_window, also affect the batch size.
def generate_batch(batch_size, window_size, to_predict):
    global data_index
    # assert batch_size % num_skips == 0
    assert  window_size == 2 * to_predict
    batch = np.ndarray(shape=(batch_size, window_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = window_size + 1
    buffer = collections.deque(maxlen=span)
    for _ in xrange(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in xrange(batch_size):
        target = to_predict
        batch[i] = [buffer[id] for id in list(range(0, to_predict) + range(to_predict+1, window_size+1))]
        labels[i, 0] = buffer[to_predict]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, window_size=2, to_predict=1)
print (batch)
    #for i in xrange(8):
    #print(batch[i], id2word[batch[i]], '->',
#     labels[i, 0], id2word[labels[i, 0]])

# Step 3: Build and train the skip-gram model
batch_size = 64
embedding_size = 128  # Dimension of the embedding vector
to_predict = 1       # Length of context to predict
window_size = 2         # How may times to reuse a context

# Pick a random validation set to sample nearest neighbors.
valid_size = 50 
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size, window_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # model parameters
    embeddings = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([vocab_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocab_size]))

    # Look up embeddings for inputs
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    embed = tf.reduce_mean(embed, 1)
    # size should be batch_size * path_length * embedding_size
    huff = tf.nn.embedding_lookup(nce_weights, train_outputs)
    inner_product = tf.matmul(embed, huff, transpose_b = True)
    # should be a multilabel loss function, similar to sampled_softmax.
    loss = tf.nn.sigmoid_cross_entropy_with_logits(inner_product,label)
    # Hierarchical Softmax
    logits = tf.nn.logits

    # get the mean of the input layer
    # Compute the average NCE loss
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                       num_sampled, vocab_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Computer the cosine similarity between examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normed_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normed_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normed_embeddings, transpose_b=True)

    init = tf.initialize_all_variables()

# Step 4: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as sess:
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, window_size, to_predict)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = id2word[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = id2word[nearest[k]]
                    log_str = '%s %s, ' % (log_str, close_word)
                print(log_str)
    final_embeddings = normed_embeddings.eval()
