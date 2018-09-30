# -*- coding:utf-8 -*-
import tensorflow as tf
#import tensorlayer as tl
import os
import sys
import argparse

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/data_set'," ")
tf.app.flags.DEFINE_string('model_dir', '/save_models',"")
tf.app.flags.DEFINE_string('tb_dir', '/logs',"")
tf.app.flags.DEFINE_integer('batch_size',512,"")
tf.app.flags.DEFINE_string('set_name','data.tfrecord',"")
tf.app.flags.DEFINE_string('check_point','model.ckpt',"")

class One_convenlution_net:

    def __init__(self):
        self.input_image = None
        self.input_label = None
        self.keep_prob = None
        self.lamb = None
        self.image_size = [FLAGS.batch_size ,224, 224, 3]
        self.predict = None
        self.batch_size = FLAGS.batch_size

    def weight_variable(self, shape, name):
        with tf.name_scope('init_w'):
            stddev = tf.sqrt(x=2/(shape[0]*shape[1]*shape[2]))
            initial = tf.truncated_normal(shape=shape, stddev=stddev)
            w = tf.Variable(initial_value=initial, name=name)
        return w

    def bias_variable(self, shape, name):
        with tf.name_scope('init_b'):
            initial = tf.random_normal(shape=shape)
            return tf.Variable(initial_value=initial, name=name)

    def conv2d(self, x, W, name):
        with tf.name_scope('conv'):
            conv = tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='VALID', name=name)
            print(conv.shape)
            return conv
    
    def drop_out(self, x, name):
        probs = tf.nn.dropout(x=x, keep_prob=self.keep_prob, name=name)
        return probs

    def net_model(self, loss='dice'):
        INPUT_IMAGE_CHANNEL = self.image_size[3]
        #inpput
        with tf.name_scope('input'):
            batch_size = self.batch_size
            self.input_image = tf.placeholder(dtype=tf.float32, shape=self.image_size, name='input_images')
            self.input_label = tf.placeholder(dtype=tf.int64, shape=[batch_size, 6], name='input_labels')
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            
        #layer 1
        with tf.name_scope('layer_1'):
            W_layer1_conv1 = self.weight_variable(shape=[224, 224, INPUT_IMAGE_CHANNEL,6], name='W_1')
            b_layer1_conv1 = self.bias_variable(shape=[6], name='b_1')
            c_layer1_conv1 = self.conv2d(x=self.input_image, W=W_layer1_conv1, name='conv1') + b_layer1_conv1
        with tf.name_scope('dropout_1'):
            h_layer1_drop1 = self.drop_out(x=c_layer1_conv1, name='dropout_1')

        #reshape
        with tf.name_scope('prediction'):
            predict = tf.reshape(h_layer1_drop1,[-1,6])
            print(predict.shape)

        #loss
        with tf.name_scope('softmax_loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label,logits=predict))
            self.loss_mean = cross_entropy

        #accuracy
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(self.input_label,1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)

        #gradient descent
        with tf.name_scope('gradient_descent'):
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cross_entropy)
        

    def read_image(self, file_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )
        image = features['image']
        label = features['label']
        image = tf.decode_raw(image,tf.float32)
        image = tf.reshape(image,[224,224,3])
        return image,label

    def read_image_batch(self, file_queue, batch_size):
        image, label = self.read_image(file_queue)
        min_after_dequeue = 2000
        capacity = 4000
        image_batch, label_batch = tf.train.shuffle_batch(
            tensors=[image, label], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue
        )

        one_hot_labels = tf.one_hot(label_batch, 6, 1, 0)

        return image_batch, one_hot_labels

    def train(self):
        TRAIN_SET_NAME = FLAGS.set_name
        train_file_path = os.path.join(FLAGS.data_dir, TRAIN_SET_NAME)
        train_image_filename_queue = tf.train.string_input_producer(
            [os.path.abspath('.') + train_file_path], num_epochs=None, shuffle=True
        )

        CHECK_POINT_PATH = os.path.join(FLAGS.model_dir, FLAGS.check_point)

        ckpt_path = os.path.abspath('.') + CHECK_POINT_PATH
        batch_size = self.batch_size
        train_images, train_labels = self.read_image_batch(train_image_filename_queue, batch_size)

        tf.summary.scalar('loss', self.loss_mean)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tb_dir = os.path.abspath('.')
            tb_dir = tb_dir + FLAGS.tb_dir
            summary_writer = tf.summary.FileWriter(tb_dir, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            all_parameters_saver.restore(sess, ckpt_path)
            try:
                epoch = 1
                while not coord.should_stop():
                    image, label = sess.run([train_images, train_labels])
                    loss,acc, summary_str = sess.run(
                        [self.loss_mean, self.accuracy ,merged_summary],
                        feed_dict={
                            self.input_image:image,
                            self.input_label:label,
                            self.keep_prob:1.0, 
                        }
                    )
                    summary_writer.add_summary(summary_str, epoch)
                    if epoch % 10 == 0:
                        print('num %d, loss:%.6f, acc:%.6f' % (epoch, loss, acc))
                    
                    sess.run(
                        [self.train_step],
                        feed_dict={
                            self.input_image:image,self.input_label:label,
                            self.keep_prob:0.6,
                        }
                    )
                    epoch += 1
            except tf.errors.OutOfRangeError:
                print('error')
            finally:
                all_parameters_saver.save(sess=sess, save_path=ckpt_path)
                coord.request_stop()

            coord.join(threads)
        print("Done training")
    
    def eval(self):
        EVAL_SET_NAME = FLAGS.set_name
        eval_file_path = os.path.join(FLAGS.data_dir, EVAL_SET_NAME)
        eval_image_filename_queue = tf.train.string_input_producer(
            [os.path.abspath('.') + eval_file_path], num_epochs=None, shuffle=None
        )

        CHECK_POINT_PATH = os.path.join(FLAGS.model_dir, FLAGS.check_point)

        ckpt_path = os.path.abspath('.') + CHECK_POINT_PATH
        batch_size = self.batch_size
        eval_images, eval_labels = self.read_image_batch(eval_image_filename_queue, batch_size)

        all_parameters_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            all_parameters_saver.restore(sess, ckpt_path)

            image, label = sess.run([eval_images, eval_labels])
            loss,acc = sess.run(
                [self.loss_mean, self.accuracy ],
                feed_dict={
                    self.input_image:image,
                    self.input_label:label,
                    self.keep_prob:1.0, 
                    }
                )

            print('loss:%.6f, acc:%.6f' % (loss, acc))
            coord.request_stop()
            coord.join(threads)

def main():
    pass

if __name__ == '__main__':

    main()
    print('model buiding!!!')
    net = One_convenlution_net()
    net.net_model()
    print('model finished!')
    net.train()