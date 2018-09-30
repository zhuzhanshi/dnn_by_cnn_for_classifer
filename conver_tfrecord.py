# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import glob
import os
import argparse

image_path = 'data/validation/'
data_name = 'test.tfrecord'

def write_label_txt(label, num):
    cwd = os.getcwd()
    txtfile = cwd + '/label.txt'
    f = open(txtfile, 'a') #zhuijiashuju
    line = label + '\t' + str(num) + '\n'
    f.write(line)
    f.close

def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img,(224,224))

    img = img/255
    img = img.astype(np.float32)
    return img

def convert():
    writer = tf.python_io.TFRecordWriter(data_name)
    label = 0
    #创建example
    for class_num in os.listdir(image_path):
        path_class = image_path + class_num
        label = label + 1
        write_label_txt(label=class_num, num=label)
        print(class_num, '\t' + str(label))
        for img in os.listdir(path_class):
            img = path_class + '/' + img
            image = load_image(img)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image':tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image.tostring()])
                        ),
                        'label':tf.train.Feature(
                            int64_list=tf.train.Int64List (value=[label])
                        )
                    }
                )
            )
            #序列化
            serialized = example.SerializeToString()
            #写入文件
            writer.write(serialized)
    writer.close()

#定义一个函数,创建从"文件中读一个样本"的操作
def read_single_sample(filename):
    #创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    #reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
 
    # get feature from serialized example
    #解析符号化的样本
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

#-----main function-----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--data_dir',
      default='data/train/',
      type=str,
    )
    parser.add_argument(
      '--data_name',
      default='train.tfrecord',
      type=str,
    )

    image_path = parser.parse_known_args()[0].data_dir
    data_name = parser.parse_known_args()[0].data_name

    print('tfrecording ... ')
    convert()
    print('finished!')

    # create tensor
    images,labels = read_single_sample(data_name)

    images_batch, labels_batch = tf.train.shuffle_batch([images,labels], batch_size=32, capacity=500, min_after_dequeue=100, num_threads=2)
    print(images_batch.shape)
    print(labels_batch.shape)
    # sess
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
 
    tf.train.start_queue_runners(sess=sess)

    for step in range(50):
        a_val, b_val = sess.run([images_batch, labels_batch])
        image = a_val[0,:,:,:]
        label = b_val[0,]
        image = (image*255).astype(int)
        cv2.imwrite(data_name + '.png',image)
        print(label)
        exit()