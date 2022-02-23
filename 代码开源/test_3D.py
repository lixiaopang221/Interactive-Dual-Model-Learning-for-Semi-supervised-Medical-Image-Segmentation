import argparse
import os
import shutil
from glob import glob

import torch

from networks.unet_3D import unet_3D
from networks.unet_3D_dual import unet_3D_dual
from test_3D_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/xueli/dataset/BraTS2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTS2019/Deep Mutual Learning_75_labeled', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D_dual', help='model_name') 


def Inference(FLAGS):
    snapshot_path = "/home/xueli/semi_supervised/model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "/home/xueli/semi_supervised/model/BraTS2019/Deep Mutual Learning_75_labeled/{}_Prediction".format(
        FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    #net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    net = unet_3D_dual(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
