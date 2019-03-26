import matplotlib as mpl
import argparse
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale
from common import get_sample_images
from networks.networks import get_network
from loguru import logger
from alfred.utils.log import init_logger

init_logger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet_v2_1.4', help='model name')
    parser.add_argument('--datapath', type=str, default='/data/public/rw/coco/annotations')
    parser.add_argument('--imgpath', type=str, default='/data/public/rw/coco/')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--lr', type=str, default='0.001')
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--checkpoint', type=str, default='./models/train/')

    parser.add_argument('--input-width', type=int, default=432)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--quant-delay', type=int, default=-1)
    args = parser.parse_args()

    # define input placeholder
    w, h =args.input_width, args.input_height
    
    # write graph, you can also write it separately
    input_node = tf.placeholder(tf.float32, shape=(None, h, w, 3), name='image')
    net, pretrain_path, last_layer = get_network(args.model, input_node, None, trainable=False)
    with tf.Session() as sess:
        loader = tf.train.Saver(net.restorable_variables())
        # loader.restore(sess, pretrain_path)
        tf.train.write_graph(sess.graph_def, os.path.dirname(args.checkpoint), 'model.pbtxt', as_text=True)
        logger.info('model graph written.')
