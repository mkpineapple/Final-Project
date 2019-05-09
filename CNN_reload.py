
import tensorflow as tf
import numpy as np
import load_data
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
n_steps = 30
# hidden LSTM units
n_inputs = 14
# rows of 28 pixels
num_units = n_inputs
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


# loading stock
Stock_Data_train = load_data.Stock_Data(x_stocks_train, y_train)
Stock_Data_val = load_data.Stock_Data(x_stocks_val, y_val)
Stock_Data_test = load_data.Stock_Data(x_stocks_test, y_test)

# loading news
News_Data_train = load_data.News_Data(x_news_train, dictionary_word, word_limit, day_news_limit, midterm_news_limit,
                                      long_news_limit)
News_Data_val = load_data.News_Data(x_news_val, dictionary_word, word_limit, day_news_limit, midterm_news_limit,
                                    long_news_limit)
News_Data_test = load_data.News_Data(x_news_test, dictionary_word, word_limit, day_news_limit, midterm_news_limit,
                                     long_news_limit)

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

        saver = tf.train.import_meta_graph('save_CNN_model/cnn_model_v1.ckpt.meta')
        saver.restore(sess, "save_CNN_model/cnn_model_v1.ckpt")
        graph = tf.get_default_graph()
        # x_stocks = graph.get_tensor_by_name("x_stocks:0")
        x_day_news_one_hot = graph.get_tensor_by_name("x_day_news_one_hot:0")
        x_week_news_one_hot = graph.get_tensor_by_name("x_week_news_one_hot:0")
        x_month_news_one_hot = graph.get_tensor_by_name("x_month_news_one_hot:0")
        y = graph.get_tensor_by_name("y:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        print('start testing...')
        acc_test = []
        while True:
            batch_xs_test, batch_ys_test = Stock_Data_test.next_batch(batch_size)
            # print(batch_ys_test.shape[0])
            batch_x_day_test, batch_x_week_test, batch_x_month_test = News_Data_test.next_batch(batch_size)
            if batch_xs_test.shape[0] < batch_size:
                break
            tmp_train_acc = sess.run(accuracy, feed_dict={
                # x_stocks: batch_xs_test,
                x_day_news_one_hot: batch_x_day_test,
                x_week_news_one_hot: batch_x_week_test,
                x_month_news_one_hot: batch_x_month_test,
                y: batch_ys_test})
            acc_test.append(tmp_train_acc)
        print('the test accuracy is :')
        print(np.mean(np.array(acc_test)))

