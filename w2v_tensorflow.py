import tensorflow as tf
import zipfile
import collections
import numpy as np
import random
import math

data_path = "/home/xiexk/machineLearning/data_files/"


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = (tf.compat.as_str(f.read(f.namelist()[0])).split())
    return data

words = read_data(data_path + "text8.zip")
print("Data Size", len(words))
vocabulary_size = 50000


def build_dataSet(words):
    wordCount = [["UNK", -1]]
    wordCount.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = {}
    for i, _ in wordCount:
        dictionary[i] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    wordCount[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, dictionary, wordCount, reverse_dictionary

data, dictionary, wordCount, reverse_dictionary = build_dataSet(words)
del words
print('Most common words:', wordCount[:10])
print('Sample data:', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    span = 2 * skip_window + 1
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(128, 2, 1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

batch_size = 200
skip_window = 5
num_skip = 10
embedding_size = 200
num_sampled = 64
valid_size = 16
valid_window = 100
valid_example = np.random.choice(valid_window, valid_size, replace=False)

# valid_example = np.array([dictionary['boy'], dictionary['girl'], dictionary['son'], dictionary['daughter']])
print(valid_example)

graph = tf.Graph()
with graph.as_default():
    train_input = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_example)
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_input)
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0/math.sqrt(embedding_size)))
    nce_bias = tf.Variable(tf.zeros([vocabulary_size]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, labels=train_labels,
                                         inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # valid_data_1 = tf.constant(np.array([dictionary['boy']]))
    # valid_embed_1 = tf.nn.embedding_lookup(normalized_embeddings, valid_data_1)
    # print(valid_embed_1.shape)
    #
    # valid_data_2 = tf.constant(np.array([dictionary['girl']]))
    # valid_embed_2 = tf.nn.embedding_lookup(normalized_embeddings, valid_data_2)
    # print(valid_embed_2.shape)
    #
    # valid_data_3 = tf.constant(np.array([dictionary['son']]))
    # valid_embed_3 = tf.nn.embedding_lookup(normalized_embeddings, valid_data_3)
    # print(valid_embed_3.shape)
    #
    # valid_embed = valid_embed_1 - valid_embed_2 + valid_embed_3
    # print(valid_embed.shape)
    # find_sim = tf.matmul(valid_embed, normalized_embeddings, transpose_b=True)

num_steps = 100000
checkpoint_dir = '/home/xiexk/machineLearning/'
isTrain = False

with tf.Session(graph=graph) as session:
    if isTrain:
        init.run()
        print("Initialized")
        average_loss = 0.0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skip, skip_window)
            feed_dict = {train_input: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict)
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    print('Average loss at step', step, ':', average_loss)
                    average_loss = 0
        saver.save(session, checkpoint_dir)
    else:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)

            # find_sim_val = find_sim.eval()
            # nearest = (-find_sim_val[0, :]).argsort()[1:10]
            # log_str = 'result is = '
            # for k in range(9):
            #     close_word = reverse_dictionary[nearest[k]]
            #     log_str = "%s %s," % (log_str, close_word)
            # print(log_str)

            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_example[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                # nearest = (find_sim_val[i, :]).argsort()[0:top_k]
                log_str = "Nearest to %s:" % valid_word

                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," %(log_str, close_word)
                print(log_str)
        else:
            pass


