# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:22:38 2017

@author: Charles
"""

'''
在mnist数据集上实现centerloss
复现论文中的结果
param 中所需的参数
lamb 控制类内分散程度的惩罚力度
alpha 中心更新速度
batch_size
lr
ckpt_dir 保存模型的地址
keep_prob
'''
from net_structures import net_1, net_2
import tensorflow as tf
import numpy as np
class mnist_centerloss():
    def __init__(self, param, load_last_model = False):
        '''
        param 包含了模型需要的所有超参数， 类实例，可用.访问其属性
        '''

        self.load_last_model = load_last_model
        self.sess = tf.Session()
        self.param = param
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        self.inputs = tf.placeholder(dtype = tf.float32, shape = (None,28,28,1), name = 'inputs')
        self.labels = tf.placeholder(tf.int32, shape = (None,), name = 'labels')
        self.lr = tf.placeholder(tf.float32, (), 'param_learning_rate')
        self.alpha = tf.placeholder(tf.float32, (), 'center_learning_rate')
        self.lamb = tf.placeholder(tf.float32, (), 'lambda')
        self.labels_onehot = tf.one_hot(self.labels, 10,  name = 'label_onehot')
        self.features, self.logits = self.inference()
        self.center = tf.Variable(initial_value = tf.random_normal(shape = (10,2), 
                                                                   name = 'center_initializer'),
                                                           
                                  trainable = False,
                                  name = 'center',
                                  dtype = tf.float32)
        self.center_of_centers = tf.reduce_mean(self.center, axis = 0)
        self.center_loss, self.softmax_loss, self.delta_center = self.loss()
        self.total_loss = self.softmax_loss + self.lamb * self.center_loss
        self.update_center = tf.scatter_sub(self.center, self.labels, self.alpha*self.delta_center)
        self.accuracy = self.acc()
        self.optimizer = tf.train.AdamOptimizer(self.lr, name = 'optimizer')\
                            .minimize(loss = self.total_loss)
        self.train_x, self.train_y,\
        self.valid_x, self.valid_y,\
        self.test_x,  self.test_y  = self.load_data()
        self.train_size = len(self.train_x)
        self.batch_size = self.param.batch_size
        tf.summary.scalar('train_softmax_loss', self.softmax_loss)
        tf.summary.scalar('train_center_loss', self.center_loss)
        tf.summary.scalar('train_total_loss', self.total_loss)
#        tf.summary.scalar('valid_total_loss', self.total_loss, collections = ['valid'])
#        tf.summary.scalar('valid_softmax_loss', self.softmax_loss, collections = ['valid'])
#        tf.summary.scalar('valid_center_loss', self.center_loss, collections = ['valid'])        
        self.summaries = tf.summary.merge_all()
#        self.summaries_valid = tf.summary.merge_all(key = 'valid')
        self.writer = tf.summary.FileWriter(logdir = './/log//', graph = self.sess.graph)
        self.saver = tf.train.Saver()
        
    def load_data(self, s = 0.2):
        '''
        从训练集中分离出 s 比例的样本作为验证集
        '''
        def scale(x):
            return (x - 127.5)/128.
        with tf.name_scope('load_data'):
            (train_valid_x, train_valid_y), (test_x, test_y) = tf.contrib.keras.datasets.mnist.load_data()
            train_valid_x = train_valid_x.reshape((-1, 28, 28, 1))
            test_x = test_x.reshape((-1, 28, 28, 1))
            train_valid_x = scale(train_valid_x)
            test_x = scale(test_x)
            train_end = int(len(train_valid_y)*(1 - s))
            train_x = train_valid_x[0: train_end]
            train_y = train_valid_y[0: train_end]
            valid_x = train_valid_x[train_end:]
            valid_y = train_valid_y[train_end:]
            return train_x, train_y, valid_x, valid_y, test_x, test_y     
    def inference(self):
        with tf.name_scope('inference'):
            return net_2(self.inputs, self.keep_prob)
    def loss(self):
        with tf.name_scope('Loss'):
            with tf.name_scope('center_loss'):
                feature_center = tf.gather(self.center, self.labels)
                assert self.features.shape.as_list() == feature_center.shape.as_list()
                center_loss = tf.reduce_mean((tf.abs(self.features - feature_center)) **2)
            with tf.name_scope('softmax_loss'):
#                softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_onehot,
#                                                                  logits = self.logits,
#                                                                  name = 'softmax_loss')
                softmax_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.labels_onehot,
                                                               logits = self.logits,
                                                               label_smoothing = 0.05
                                                               )#label_smoothing 
                softmax_loss_mean = tf.reduce_mean(softmax_loss, name = 'softmax_loss_mean')
            with tf.name_scope('center_diff'):
                unique_label, unique_idx, unique_count = tf.unique_with_counts(self.labels)
                appear_times = tf.gather(unique_count, unique_idx)#每个label在这个batch中出现次数，形状为(batch_size,)
                appear_times = tf.reshape(appear_times, (-1,1))
                diff = (feature_center - self.features)/tf.cast(1 + appear_times, tf.float32)
            return center_loss, softmax_loss_mean, diff
    def shuffle(self, data_x, data_y):
        assert len(data_x) == len(data_y)
        seed = np.random.permutation(len(data_y))
        return data_x[seed], data_y[seed]
    def train(self):
        best_valid_loss = float('inf')
        best_train_loss = float('inf')
        patience = 5#验证误差不减少的忍耐次数
        max_iterations = 200
        with tf.Session() as sess:
            p = 0
            if self.load_last_model:#载入最新的模型及其参数，作为网络参数的初始化
                self.saver.restore(sess, self.param.ckpt_dir + 'model.ckpt')
                _, best_valid_loss = self.validation(sess) 
            else:#重新训练网络
                sess.run(tf.global_variables_initializer())
            try:
                for iteration in range(max_iterations):
                    train_x, train_y = self.shuffle(self.train_x, self.train_y)
                    batch_num = 0
                    train_loss = 0#训练集误差
                    for batch_begin, batch_end in zip(range(0,self.train_size, self.batch_size),\
                                                  range(self.batch_size,self.train_size,\
                                                  self.batch_size)):
                        batch_imgs = train_x[batch_begin:batch_end]
                        batch_labels = train_y[batch_begin:batch_end]
                        feed_dict = {self.inputs: batch_imgs, 
                                     self.labels: batch_labels,
                                     self.alpha: self.param.alpha,
                                     self.lr: self.param.lr,
                                     self.lamb: self.param.lamb,
                                     self.keep_prob: self.param.keep_prob}
                        center,_,softmax_loss, center_loss, total_loss, summary = sess.run([self.update_center,
                                                                                self.optimizer,
                                                                                self.softmax_loss, 
                                                                                self.center_loss,
                                                                                self.total_loss,
                                                                                self.summaries
                                                                                ],
                                                                              feed_dict = feed_dict
                                                                              )
                        batch_num += 1
                        train_loss += total_loss * self.batch_size/len(train_y)
                        self.writer.add_summary(summary, iteration*self.train_size/self.batch_size + batch_num)
                    #训练误差不减少，降低学习率
                    if train_loss > best_train_loss:
                        self.param.lr *= self.param.lr_decay_rate
                    else:
                        best_train_loss = train_loss
                    #一个 iteration 后打印训练误差
                    print ('iteration {} on train:\nsoftmax_loss:{}\ncenter_loss:{}\ntotal_loss:{}\n'\
                           .format(iteration,softmax_loss, center_loss, total_loss))  
                    #计算验证误差
                    valid_loss, valid_accuracy = self.validation(sess)                                                        
                    #若验证误差比当前最好验证误差还小则保存模型，更新忍耐次数 p = 0, 否则 p +=1                                                  })
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        self.saver.save(sess, self.param.ckpt_dir + 'model.ckpt')#保存模型
                        print('model successfully saved in {}'.format(self.param.ckpt_dir))
                        p = 0
                    else:
                        p += 1
                    #打印验证误差
                    print ('验证集误差：\n当前 total_loss:{}\n最好 total_loss:{}\n准确率: {}\n'\
                           .format(valid_loss, best_valid_loss, valid_accuracy))
                    #若忍耐次数达上限，扔出异常，终止训练
                    if p > patience:
                        raise KeyboardInterrupt
            except KeyboardInterrupt:
                print ('终止训练。。')
            finally:
                print ('center: {}'.format(sess.run(self.center)))
    def acc(self):
        with tf.name_scope('acc_compute'):
            result = tf.argmax(self.logits, axis = 1)
            result = tf.cast(result, tf.int32)
            correct_or_not = tf.equal(self.labels, result)
            correct_or_not = tf.cast(correct_or_not, tf.float32)
            accuracy = tf.reduce_mean(correct_or_not)
            return accuracy
    def validation(self, sess):
        total = len(self.valid_y)
        batch_size = 1000
        average_loss = 0
        average_acc = 0
#        loss = {l:[] for l in ['softmax','center','total']}
        for start in range(0, total, batch_size):
            imgs = self.valid_x[start: (start + batch_size)]
            labels = self.valid_y[start: (start + batch_size)]
            feed_dict = {self.inputs: imgs, self.labels:labels, self.keep_prob: 1,
                         self.lamb: self.param.lamb}
            loss, acc = sess.run([self.total_loss, self.accuracy],\
                                           feed_dict = feed_dict                                                  
                                           )
#            loss['softmax'].append(softmax_loss)
#            loss['center'].append(center_loss)
            average_loss += len(labels) * loss/float(total)
            average_acc += len(labels) * acc/float(total)
#        s = sum(loss['softmax'])/len(loss['softmax'])
#        c = sum(loss['center'])/len(loss['center'])
        return average_loss, average_acc
    def visualized_on(self, dataset = 'train'):
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        if dataset == 'valid':
            data_x = self.valid_x
            data_y = self.valid_y
        elif dataset == 'test':
            data_x = self.test_x
            data_x = self.test_y
        else:
            data_x = self.train_x
            data_y = self.train_y
        #建立作图区域
        fig = plt.figure(figsize = (10,10))
        axes = fig.add_subplot(1,1,1)
        #按标签把数据分类,得到每一类数据的索引，方便后续作图
        index = [[] for i in range(10)]
        for idx, label in enumerate(data_y):
            index[label].append(idx)
        #运行计算图把所有训练图片的特征计算出来，分批计算，显存不够！
        batch_size = 2000
        imgs_features =[]
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
        types = []
        with tf.Session() as sess:
            self.saver.restore(sess, self.param.ckpt_dir + 'model.ckpt')
            for start in range(0, len(self.train_x), batch_size):
                imgs = data_x[start:(start + batch_size)]
                feed_dict = {self.inputs: imgs, self.keep_prob: 1}
                batch_features = sess.run(self.features, feed_dict = feed_dict)
                imgs_features.append(batch_features)
        imgs_features = np.concatenate(imgs_features)
        #作图
        for label in range(10):
            features_label = imgs_features[index[label]]
            type_label = axes.scatter(features_label[:,0], features_label[:,1], s = 10, c = colors[label])
            types.append(type_label)
        #添加提示
        axes.legend(types, list(range(10)))
        fig
            
if __name__ == '__main__':
    class parameter:
        def __init__(self,
                     lamb = 0,
                     alpha = 0.5, 
                     batch_size = 64,
                     lr = 0.001,
                     ckpt_dir = '.\\ckpt\\',
                     lr_decay_rate = 0.5,
                     keep_prob = 0.8
                     ):
            self.lamb = lamb
            self.alpha = alpha
            self.batch_size = batch_size
            self.lr = lr
            self.ckpt_dir = ckpt_dir
            self.lr_decay_rate = lr_decay_rate
            self.keep_prob = keep_prob
            

    param = parameter()
    test = mnist_centerloss(param)
    test.train()
    test.visualized_on()
    test.visualized_on('valid')
       
        
        
        
        