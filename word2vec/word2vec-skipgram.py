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
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [id2word[i] for i in data[:10]])


# Step 2: Generate a training batch for the skip-gram model.
data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert  num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in xrange(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in xrange(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in xrange(num_skips):
            while target in targets_to_avoid:
                target = np.random.randint(0, span)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in xrange(8):
    print(batch[i], id2word[batch[i]], '->',
          labels[i, 0], id2word[labels[i, 0]])

# Step 3: Build and train the skip-gram model
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector
skip_window = 1       # Length of context to predict
num_skips = 2         # How may times to reuse a context

# Pick a random validation set to sample nearest neighbors.
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
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
            batch_size, num_skips, skip_window)
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
