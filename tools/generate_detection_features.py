#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = (
'__background__', 'sunglasses', 'pants', 'jeans', 'shirt', 'tie', 'suit', 'shoes', 'skirt', 'jacket', 'dress', 'coat',
'shorts')
NETS = {'res101': ('res101_faster_rcnn_iter_490000.ckpt',)}
DATASETS = {'visual_genome': ('visual_genome_categories_1_train_490000_0.003',)}


def vis_detections(im_file, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    # dets = dets[inds,:]
    # print(dets)
    # inds = [dets[:,-1].argmax()]
    # print(inds)
    im = im[:, :, (2, 1, 0)]
    flag = False
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        print(bbox)
        score = dets[i, -1]
        print(score)

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(im_file)


def demo(sess, net, frame_name,saveDirectory,det_file):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    frame_file = os.path.join(cfg.DATA_DIR, 'runway1', 'img1',frame_name)
    fra = cv2.imread(frame_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, roi_pooling = im_detect(sess, net, fra)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    fig, ax = plt.subplots(figsize=(12, 12))
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # print(dets.shape)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        object_features = roi_pooling[keep,:]
        # print(dets.shape)
        save_name = frame_file
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        # print(inds)
        if len(inds) == 0:
            continue
        fra = fra[:, :, (2, 1, 0)]
        ax.imshow(fra, aspect='equal')
        with open(saveDirectory + '/' + det_file,'a') as f:
            for i in inds:
                bbox = dets[i, :4]
                # print(bbox)
                score = dets[i, -1]
                # print(score)
                f.write('{:s} '.format(frame_name.split('.')[0]))
                f.write('{:d} '.format(-1))
                for v_bbox in bbox:
                     f.write("{:d} ".format(int(v_bbox)))
                f.write("{:.2f} ".format(score))
                f.write('{:d} {:d} {:d} '.format(-1,-1,-1))
                for v_roi_pooling in roi_pooling[i, :]:
                     f.write("{:.2f} ".format(v_roi_pooling))
                f.write('\n')

                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3.5)
                )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(cls, score),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')
                plt.hold(True)
                plt.axis('off')
                plt.tight_layout()
                plt.draw()
            # plt.savefig(save_name + 'res.jpg')
    temp_dir = save_name.rsplit('/', 1)
    save_dir = temp_dir[0] + '/res/'
    print('temp_dir is : %s' % (temp_dir))
    print('save_dir is : %s' % (save_dir))
    save_name = temp_dir[1].split('.jpg')[0]
    print(save_name)
    if os.path.exists(save_dir):
        plt.savefig(save_dir + save_name + '_res.jpg')
    else:
        os.mkdir(save_dir)
        plt.savefig(save_dir + save_name + '_res.jpg')

        # vis_detections(save_name,im, cls, dets, thresh=CONF_THRESH)
        # im = cv2.imread(save_name)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='visual_genome')
    parser.add_argument('--video', dest='video', help='Path to the video folder')
    parser.add_argument('--saveDirectory', dest='saveDirectory', help='Path to save object embeddings')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('resources','networks','output', demonet, DATASETS[dataset][0], 'default',
                           NETS[demonet][0])
    print(tfmodel)

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 13,
                            tag='default', anchor_scales=[2, 4, 8, 16, 32], anchor_ratios=[0.25, 0.5, 1, 2, 4])

    # saver = tf.train.import_meta_graph(tfmodel + '.meta')
    # saver.restore(sess, tfmodel)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    saveDirectory = args.saveDirectory
    video_frames_directory = args.video
    det_file = 'det.txt'
    frames = [file for file in os.listdir(video_frames_directory)]
    for frame in frames:
        print(frame)
        print(saveDirectory)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(frame))
        try:
           demo(sess, net, frame,saveDirectory,det_file)
        except AttributeError:
           continue
    plt.show()
