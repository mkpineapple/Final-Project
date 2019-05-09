import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import rnn
import load_data
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------LSTM parameter-----------
# unrolled through 28 time steps
n_steps = 30
# hidden LSTM units
n_inputs = 8
# rows of 28 pixels
num_units = n_inputs
# learning rate for adam
learning_rate = 0.001
# mnist is meant to be classified in 10 classes(0-9).
n_classes = 2
# size of batch
batch_size = 128
# lstm
forget = 1
# iters
training_iters = 80800


def RNN(x_in):
    lstm_layer = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=forget)
    # init
    init_state = lstm_layer.zero_state(batch_size, dtype=tf.float32)

    # (steps , batch_size , 8)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_layer, x_in, dtype="float32", initial_state=init_state)
    outputs = tf.transpose(outputs, [1, 0, 2])[-1]

    return outputs


tf.reset_default_graph()

x_stocks = tf.placeholder("float", [None, n_steps,n_inputs], name='x_stocks')
y = tf.placeholder("float", [None, n_classes], name='y')
out_w = tf.Variable(tf.random_normal([8, n_classes]), name='out_w')
out_b = tf.Variable(tf.random_normal([n_classes]), name='out_b')

# ---LSTM---
output_lstm = RNN(x_stocks)

# output
# we have output_lstm and output_cnn
# then just concat it and put it into the hidden layer + softmax layer

# weights and biases of appropriate shape to accomplish above task
# softmax
pred = tf.nn.softmax(tf.matmul(output_lstm, out_w) + out_b)
# print(pred.get_shape())

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

saver = tf.train.Saver()

init = tf.global_variables_initializer()

# --------data loading------------
# (505,160,3)
x_news_train = np.load('data_news/x_news_train.npy')
x_news_val = np.load('data_news/x_news_val.npy')
x_news_test = np.load('data_news/x_news_test.npy')

# (505, 160, 30, 8)
x_stocks_train = np.load('data_prices/x_train_log_scaled_2013.npy')
x_stocks_val = np.load('data_prices/x_val_log_scaled_2013.npy')
x_stocks_test = np.load('data_prices/x_test_log_scaled_2013.npy')

# (505 , 160)
y_train = np.load('data_prices/y_train_2013.npy')
y_val = np.load('data_prices/y_val_2013.npy')
y_test = np.load('data_prices/y_test_2013.npy')

# (13083 , dict)
dictionary_word = np.load('data_news/dictionary_word.npy').tolist()

# (13084 , 300)
numpy_w2c = np.load('data_news/numpy_w2c.npy')

Stock_Data_train = load_data.Stock_Data(x_stocks_train, y_train)
Stock_Data_val = load_data.Stock_Data(x_stocks_val, y_val)
Stock_Data_test = load_data.Stock_Data(x_stocks_test, y_test)

# News_Data_train = load_data.News_Data(x_news_train, dictionary_word, word_limit, day_news_limit, midterm_news_limit,
#                                       long_news_limit)
# News_Data_val = load_data.News_Data(x_news_val, dictionary_word, word_limit, day_news_limit, midterm_news_limit,
#                                     long_news_limit)
# News_Data_test = load_data.News_Data(x_news_test, dictionary_word, word_limit, day_news_limit, midterm_news_limit,
#                                      long_news_limit)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# val loss
plot_train_x = []
plot_val_x = []
plot_val_acc = []
plot_val_loss = []
plot_train_acc = []
plot_train_loss = []


with tf.device('/device:GPU:0'):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                          gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = Stock_Data_train.next_batch(batch_size)
            # batch_x_day , batch_x_week , batch_x_month = News_Data_train.next_batch(batch_size)

            batch_xs_val, batch_ys_val = Stock_Data_val.next_batch(batch_size)
            # batch_x_day_val , batch_x_week_val , batch_x_month_val = News_Data_val.next_batch(batch_size)

            if batch_xs.shape[0] < 128:  #or batch_x_day.shape[0] < 128 :
                break
            sess.run([train_op], feed_dict={
                x_stocks: batch_xs,
                # x_day_news_one_hot : batch_x_day,
                # x_week_news_one_hot : batch_x_week,
                # x_month_news_one_hot : batch_x_month,
                y: batch_ys
            })
            if step % 20 == 0:
                plot_train_x.append(step)
                print('---train loss---')
                tmp_train_loss = sess.run(cost, feed_dict={
                x_stocks: batch_xs,
                # x_day_news_one_hot : batch_x_day,
                # x_week_news_one_hot : batch_x_week,
                # x_month_news_one_hot : batch_x_month,
                y: batch_ys})
                plot_train_loss.append(tmp_train_loss)
                print(tmp_train_loss)

                print('---train accuracy---')
                tmp_train_acc = sess.run(accuracy, feed_dict={
                x_stocks: batch_xs,
                # x_day_news_one_hot : batch_x_day,
                # x_week_news_one_hot : batch_x_week,
                # x_month_news_one_hot : batch_x_month,
                y: batch_ys})

                plot_train_acc.append(tmp_train_acc)
                print(tmp_train_acc)
                # val
                if batch_xs_val.shape[0] < 128:  #or batch_x_day_val.shape[0] < 128 :
                    continue
                plot_val_x.append(step)
                print('---dev loss---')
                tmp_val_loss = sess.run(cost, feed_dict={
                x_stocks: batch_xs_val,
                #x_day_news_one_hot : batch_x_day,
                #x_week_news_one_hot : batch_x_week,
                #x_month_news_one_hot : batch_x_month,
                y: batch_ys_val})
                plot_val_loss.append(tmp_val_loss)
                print(tmp_val_loss)

                print('---dev accuracy---')
                tmp_val_acc = sess.run(accuracy, feed_dict={
                x_stocks: batch_xs_val,
                # x_day_news_one_hot : batch_x_day,
                # x_week_news_one_hot : batch_x_week,
                # x_month_news_one_hot : batch_x_month,
                y: batch_ys_val})

                plot_val_acc.append(tmp_val_acc)
                print(tmp_val_acc)
            step += 1

        #save the model
        save_path = saver.save(sess, "save_LSTM_model/lstm_model_v1.ckpt")
        print("Model saved in file: ", save_path)
        print('start testing。。。')
        acc_test = []
        while True:
            batch_xs_test, batch_ys_test = Stock_Data_test.next_batch(batch_size)
            # batch_x_day_test , batch_x_week_test , batch_x_month_test = News_Data_test.next_batch(batch_size)
            if batch_xs_test.shape[0] < batch_size:  #or batch_x_day_test.shape[0] < batch_size :
                break
            tmp_train_acc = sess.run(accuracy, feed_dict={
                x_stocks: batch_xs_test,
                # x_day_news_one_hot : batch_x_day,
                # x_week_news_one_hot : batch_x_week,
                # x_month_news_one_hot : batch_x_month,
                y: batch_ys_test})

            acc_test.append(tmp_train_acc)
        print(np.mean(np.array(acc_test)))

plt.figure()     
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
