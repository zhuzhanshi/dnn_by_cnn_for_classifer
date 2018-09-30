# -*- coding:utf-8 -*-
import tensorflow as tf
#import tensorlayer as tl
import os
import sys
import argparse
from model import One_convenlution_net,FLAGS

#FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--data_dir',
      default='/data_set',
      type=str,
    )
    parser.add_argument(
      '--model_dir',
      default='/save_models',
      type=str,
    )
    parser.add_argument(
      '--tb_dir',
      default='/logs',
      type=str,
    )
    parser.add_argument(
      '--batch_size',
      default=1200,
      type=int,
    )
    parser.add_argument(
      '--set_name',
      default='val.tfrecord',
      type=str,
    )
    parser.add_argument(
      '--check_point',
      default='model.ckpt',
      type=str,
    )

    FLAGS.data_dir = parser.parse_known_args()[0].data_dir
    FLAGS.batch_size  = parser.parse_known_args()[0].batch_size
    FLAGS.model_dir = parser.parse_known_args()[0].model_dir
    FLAGS.tb_dir = parser.parse_known_args()[0].tb_dir
    FLAGS.set_name = parser.parse_known_args()[0].set_name
    FLAGS.check_point = parser.parse_known_args()[0].check_point

    print('model buiding!!!')
    net = One_convenlution_net()
    net.net_model()
    print('model finished!')
    net.eval()
    