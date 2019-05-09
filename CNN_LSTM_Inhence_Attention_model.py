import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import rnn
import load_data_new
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# --------CNN parameter------------
# the longest length of word
word_limit = 15
# the most number of day news title
day_news_limit = 30
# the most number of week news title
midterm_news_limit = 124
# the most number of month news title
long_news_limit = 212
# the dimension of word2vec
word2vec_dimension = 300
# --------LSTM parameter-----------
# unrolled through 28 time steps
n_steps=30
# hidden LSTM units
n_inputs=14
# rows of 28 pixels
num_units=n_inputs
# learning rate for adam
learning_rate = 0.0001
# mnist is meant to be classified in 10 classes(0-9).
n_classes = 2
# size of batch
batch_size = 128
# lstm
forget = 1
# iters
training_iters = 80800


# self attention
def attention_fun(Q, K, scaled_=True, masked_=False):
    attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

    if scaled_:
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

    if masked_:
        raise NotImplementedError

    attention = tf.nn.softmax(attention, dim=-1)  # [batch_size, sequence_length, sequence_length]
    return attention


def self_attention(data, hidden_dim, n_classes):
    Q = tf.layers.dense(data, hidden_dim)  # [batch_size, sequence_length, hidden_dim]
    K = tf.layers.dense(data, hidden_dim)  # [batch_size, sequence_length, hidden_dim]
    V = tf.layers.dense(data, n_classes)  # [batch_size, sequence_length, n_classes]

    attention = attention_fun(Q, K)  # [batch_size, sequence_length, sequence_length]
    output = tf.matmul(attention, V)  # [batch_size, sequence_length, n_classes]
    output = tf.reduce_mean(output, axis=1)
    return output


# CNN
def weight_variable(shape, ans):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=ans)


def bias_variable(shape, ans):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=ans)


def conv2d1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]
    # day news input


def conv2d2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]
    # day news input


def max_pool_15x1(x):
    return tf.nn.max_pool(x, ksize=[1, 15, 1, 1],
                          strides=[1, 15, 1, 1], padding='SAME')
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize                 : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides               : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.


def max_pool_2x32(x):
    return tf.nn.max_pool(x, ksize=[1, 32, 1, 1],
                          strides=[1, 32, 1, 1], padding='SAME')
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize                 : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides               : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.


# LSTM
def RNN(x_in):
    lstm_layer = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=forget)
    # init
    init_state = lstm_layer.zero_state(batch_size, dtype=tf.float32)

    # (steps , batch_size , 8)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_layer, x_in, dtype="float32", initial_state=init_state)
    outputs = tf.transpose(outputs, [1, 0, 2])[-1]

    return outputs


tf.reset_default_graph()
with tf.device('/device:GPU:0'):
    x_day_news_one_hot = tf.placeholder(tf.int32, [batch_size, word_limit, day_news_limit], name='x_day_news_one_hot')
    x_week_news_one_hot = tf.placeholder(tf.int32, [batch_size, word_limit, midterm_news_limit],
                                         name='x_week_news_one_hot')
    x_month_news_one_hot = tf.placeholder(tf.int32, [batch_size, word_limit, long_news_limit],
                                          name='x_month_news_one_hot')
    x_stocks = tf.placeholder("float", [None, n_steps, n_inputs], name='x_stocks')
    y = tf.placeholder("float", [None, n_classes], name='y')

    out_w = tf.Variable(tf.random_normal([8 + 14, n_classes]), name='out_w')
    out_b = tf.Variable(tf.random_normal([n_classes]), name='out_b')

    W_news_input = weight_variable([900, 8], ans='W_news_input')
    b_news_input = bias_variable([8], ans='b_news_input')

    W_day_conv1 = weight_variable([5, 1, day_news_limit, day_news_limit], ans='W_day_conv1')
    b__day_conv1 = bias_variable([1], ans='b__day_conv1')
    W_week_conv1 = weight_variable([5, 1, midterm_news_limit, midterm_news_limit], ans='W_week_conv1')
    b__week_conv1 = bias_variable([1], ans='b__week_conv1')
    W_month_conv1 = weight_variable([5, 1, long_news_limit, long_news_limit], ans='W_month_conv1')
    b__month_conv1 = bias_variable([1], ans='b__month_conv1')

    W_day_conv2 = weight_variable([1, 30, day_news_limit, 32], ans='W_day_conv2')
    b__day_conv2 = bias_variable([32], ans='b__day_conv2')
    W_week_conv2 = weight_variable([1, 30, midterm_news_limit, 32], ans='W_week_conv2')
    b__week_conv2 = bias_variable([32], ans='b__week_conv2')
    W_month_conv2 = weight_variable([1, 30, long_news_limit, 32], ans='W_month_conv2')
    b__month_conv2 = bias_variable([32], ans='b__month_conv2')

    # RNN
    tmp = np.load('data_news/numpy_w2c.npy')
    embedding_float64_tmp = tf.Variable(tf.ones([tmp.shape[0], tmp.shape[1]]))
    embedding_float64 = embedding_float64_tmp.assign(tmp)
    embedding = tf.cast(embedding_float64, tf.float32)

    x_day_news_one_hot_1d = tf.reshape(x_day_news_one_hot, [-1])
    x_week_news_one_hot_1d = tf.reshape(x_week_news_one_hot, [-1])
    x_month_news_one_hot_1d = tf.reshape(x_month_news_one_hot, [-1])

    x_day_news_w2c_1d = tf.nn.embedding_lookup(embedding, x_day_news_one_hot_1d)
    x_week_news_w2c_1d = tf.nn.embedding_lookup(embedding, x_week_news_one_hot_1d)
    x_month_news_w2c_1d = tf.nn.embedding_lookup(embedding, x_month_news_one_hot_1d)

    x_day_news = tf.transpose(tf.reshape(x_day_news_w2c_1d, [batch_size, word_limit, day_news_limit,
                                                             word2vec_dimension]), (0, 1, 3, 2))
    x_week_news = tf.transpose(tf.reshape(x_week_news_w2c_1d, [batch_size, word_limit, midterm_news_limit,
                                                               word2vec_dimension]), (0, 1, 3, 2))
    x_month_news = tf.transpose(tf.reshape(x_month_news_w2c_1d, [batch_size, word_limit, long_news_limit,
                                                                 word2vec_dimension]), (0, 1, 3, 2))

    h_conv1_day = tf.nn.relu(conv2d1(x_day_news, W_day_conv1) + b__day_conv1)
    h_conv1_week = tf.nn.relu(conv2d1(x_week_news, W_week_conv1) + b__week_conv1)
    h_conv1_month = tf.nn.relu(conv2d1(x_month_news, W_month_conv1) + b__month_conv1)

    h_pool1_day = max_pool_15x1(h_conv1_day)
    h_pool1_week = max_pool_15x1(h_conv1_week)
    h_pool1_month = max_pool_15x1(h_conv1_month)

    # h_pool1(batch, 1, 300, 30) -> h_pool2(batch, 1, 300, 1)
    h_attn_day = tf.transpose(tf.squeeze(h_pool1_day), (0, 2, 1))  # (batch, 30, 300)
    h_attn_week = tf.transpose(tf.squeeze(h_pool1_week), (0, 2, 1))
    h_attn_month = tf.transpose(tf.squeeze(h_pool1_month), (0, 2, 1))

    h_attn_day_v = self_attention(h_attn_day, 300, 300)
    h_attn_week_v = self_attention(h_attn_day, 300, 300)
    h_attn_month_v = self_attention(h_attn_day, 300, 300)

    # h_conv2_day = tf.nn.relu(conv2d2(h_pool1_day, W_day_conv2) + b__day_conv2)
    # h_conv2_week = tf.nn.relu(conv2d2(h_pool1_week, W_week_conv2) + b__week_conv2)
    # h_conv2_month = tf.nn.relu(conv2d2(h_pool1_month, W_month_conv2) + b__month_conv2)
    # #print(h_conv2_month.get_shape())

    # h_conv2_day_tmp = tf.transpose( h_conv2_day , (0, 3, 2, 1))
    # h_conv2_week_tmp = tf.transpose( h_conv2_week , (0, 3, 2, 1))
    # h_conv2_month_tmp = tf.transpose( h_conv2_month , (0, 3, 2, 1))

    # h_pool2_day_tmp = max_pool_2x32(h_conv2_day_tmp)
    # h_pool2_week_tmp = max_pool_2x32(h_conv2_week_tmp)
    # h_pool2_month_tmp = max_pool_2x32(h_conv2_month_tmp)

    # h_pool2_day = tf.transpose( h_pool2_day_tmp , (0, 3, 2, 1))
    # h_pool2_week = tf.transpose( h_pool2_week_tmp , (0, 3, 2, 1))
    # h_pool2_month = tf.transpose( h_pool2_month_tmp , (0, 3, 2, 1))

    h_news_input = tf.concat([h_attn_day_v, h_attn_week_v, h_attn_month_v], axis=1)
    # print(h_news_input.get_shape())
    # h_news_input = tf.squeeze(h_news_input)
    # (?, 1, 900, 1)

    output_cnn = tf.nn.relu(tf.matmul(h_news_input, W_news_input) + b_news_input)
    # print('cnn' , output_cnn.get_shape())
    # print(output_cnn.get_shape())

    # LSTM part
    output_lstm = RNN(x_stocks)

    # we have output_lstm and output_cnn
    # then just concat it and put it into the hidden layer + softmax layer
    # '''
    # weights and biases of appropriate shape to accomplish above task
    outputs = tf.concat([output_lstm, output_cnn], axis=1)

    # softmax
    pred = tf.nn.softmax(tf.matmul(outputs, out_w) + out_b)
    # print(pred.get_shape())

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), name='cost')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

# loading news
# (505,160,3)
x_news_train = np.load('data_news/x_news_train.npy')
x_news_val = np.load('data_news/x_news_val.npy')
x_news_test = np.load('data_news/x_news_test.npy')

# (505, 160, 30, 8)
x_stocks_train = np.load('data_prices/x_train_log_scaled_2013_inhence.npy')
x_stocks_val = np.load('data_prices/x_val_log_scaled_2013_inhence.npy')
x_stocks_test = np.load('data_prices/x_test_log_scaled_2013_inhence.npy')

# (505 , 160)
y_train = np.load('data_prices/y_train_2013.npy' )
y_val = np.load('data_prices/y_val_2013.npy' )
y_test = np.load('data_prices/y_test_2013.npy' )

# (13083 , dict)
dictionary_word = np.load('data_news/dictionary_word.npy').tolist()
# (13084 , 300)
numpy_w2c = np.load('data_news/numpy_w2c.npy')

# stock
Stock_Data_train = load_data_new.Stock_Data(x_stocks_train, y_train)
Stock_Data_val = load_data_new.Stock_Data(x_stocks_val, y_val)
Stock_Data_test = load_data_new.Stock_Data(x_stocks_test, y_test)
# Stock_Data_test_jia = data_loader_v2.Stock_Data(x_stocks_val , y_val )

# news
News_Data_train = load_data_new.News_Data(x_news_train, dictionary_word, word_limit, day_news_limit,
                                          midterm_news_limit, long_news_limit)
News_Data_val = load_data_new.News_Data(x_news_val, dictionary_word, word_limit, day_news_limit,
                                        midterm_news_limit, long_news_limit)
News_Data_test = load_data_new.News_Data(x_news_test, dictionary_word, word_limit, day_news_limit,
                                         midterm_news_limit, long_news_limit)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# val loss
plot_train_x = []
plot_val_x = []
plot_val_acc = []
plot_val_loss = []
plot_train_acc = []
plot_train_loss = []

with tf.device('/device:GPU:0'):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options,
                                          allow_soft_placement=True)) as sess:
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = Stock_Data_train.next_batch(batch_size)
            batch_x_day, batch_x_week, batch_x_month = News_Data_train.next_batch(batch_size)

            batch_xs_val, batch_ys_val = Stock_Data_val.next_batch(batch_size)
            batch_x_day_val, batch_x_week_val , batch_x_month_val = News_Data_val.next_batch(batch_size)

            if batch_xs.shape[0] < 128 or batch_x_day.shape[0] < 128:
                break
            sess.run([train_op], feed_dict={
                x_stocks: batch_xs,
                x_day_news_one_hot: batch_x_day,
                x_week_news_one_hot: batch_x_week,
                x_month_news_one_hot: batch_x_month,
                y: batch_ys})

            if step % 20 == 0:
                plot_train_x.append(step)
                print('---train loss---')
                tmp_train_loss = sess.run(cost, feed_dict={
                x_stocks: batch_xs,
                x_day_news_one_hot: batch_x_day,
                x_week_news_one_hot: batch_x_week,
                x_month_news_one_hot: batch_x_month,
                y: batch_ys})

                plot_train_loss.append(tmp_train_loss)
                print(tmp_train_loss)

                print('---train accuracy---')
                tmp_train_acc = sess.run(accuracy, feed_dict={
                x_stocks: batch_xs,
                x_day_news_one_hot: batch_x_day,
                x_week_news_one_hot: batch_x_week,
                x_month_news_one_hot : batch_x_month,
                y: batch_ys})

                plot_train_acc.append(tmp_train_acc)
                print(tmp_train_acc)
                # val
                if batch_xs_val.shape[0] < 128 or batch_x_day_val.shape[0] < 128:
                    continue
                plot_val_x.append(step)
                print('---dev loss---')
                tmp_val_loss = sess.run(cost, feed_dict={
                x_stocks: batch_xs_val,
                x_day_news_one_hot: batch_x_day_val,
                x_week_news_one_hot: batch_x_week_val,
                x_month_news_one_hot: batch_x_month_val,
                y: batch_ys_val})

                plot_val_loss.append(tmp_val_loss)
                print(tmp_val_loss)

                print('---dev accuracy---')
                tmp_val_acc = sess.run(accuracy, feed_dict={
                x_stocks: batch_xs_val,
                x_day_news_one_hot: batch_x_day_val,
                x_week_news_one_hot: batch_x_week_val,
                x_month_news_one_hot: batch_x_month_val,
                y: batch_ys_val})

                plot_val_acc.append(tmp_val_acc)
                print(tmp_val_acc)
            step += 1

        # save the model
        save_path = saver.save(sess, "save_LSTM_CNN_Inhence_Attention_model/cnn_lstm_attention_model_v1.ckpt")
        print ("Model saved in file: ", save_path)
        print('start testing...')
        acc_test = []
        while True:
            batch_xs_test, batch_ys_test = Stock_Data_test.next_batch(batch_size)
            batch_x_day_test, batch_x_week_test, batch_x_month_test = News_Data_test.next_batch(batch_size)
            if batch_xs_test.shape[0] < batch_size or batch_x_day_test.shape[0] < batch_size:
                break
            tmp_train_acc = sess.run(accuracy, feed_dict={
                x_stocks: batch_xs_test,
                x_day_news_one_hot: batch_x_day_test,
                x_week_news_one_hot: batch_x_week_test,
                x_month_news_one_hot: batch_x_month_test,
                y: batch_ys_test})

            acc_test.append(tmp_train_acc)
        print(np.mean(np.array(acc_test)))


plt.subplot(2, 2, 1)
plt.title('Train Accuracy Analysis')
# plt.plot(plot_x, plot_val_acc, color='green', label='plot_val_acc')
plt.plot(plot_train_x, plot_train_acc, color='red', label='plot_train_acc')
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.subplot(2, 2, 2)
plt.title('Train Loss Analysis')
# plt.plot(plot_x, plot_val_acc, color='green', label='plot_val_acc')
plt.plot(plot_train_x, plot_train_loss, color='blue', label='plot_train_loss')
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('rate')

plt.subplot(2, 2, 3)
# plt.title('Valiadtion Accuracy Analysis')
# plt.plot(plot_x, plot_val_acc, color='green', label='plot_val_acc')
plt.plot(plot_val_x, plot_val_acc, color='red', label='plot_val_acc')
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.subplot(2, 2, 4)
# plt.title('Valiadtion Loss Analysis')
# plt.plot(plot_x, plot_val_acc, color='green', label='plot_val_acc')
plt.plot(plot_val_x, plot_val_loss, color='blue', label='plot_val_loss')
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()


