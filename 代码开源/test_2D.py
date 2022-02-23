import argparse
import os
import shutil
import logging
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

#from networks.efficientunet import UNet
from models.whole_model import MODULE, MODULE_single
from networks.net_factory import net_factory


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                      default='/home/xueli/dataset/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/MobileNet_Interpolation_Consistency_Training', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mobilenet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')

os.environ["CUDA_VISIBLE_DEVICES"]='1'

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt) 
    asd = metric.binary.asd(pred, gt) 
    hd95 = metric.binary.hd95(pred, gt) 
    return dice, hd95, asd
    #return dice


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        #slice = zoom(slice, (512 / x, 512 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urpc":
                out_main, _, _, _ = net(input)
            else:
                #out_main, out2 = net(input)
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            '''out = torch.sigmoid(out_main).squeeze(0).squeeze(0)
            out[out>0.5] = 1
            out[out<=0.5] = 0'''
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            #pred = zoom(out, (x / 512, y / 512), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS, snapshot_path):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    test_save_path = "/home/xueli/semi_supervised/model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    #net = MODULE().cuda()
    #net = MODULE_single().cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    logging.info('init weight from {}'.format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        print(case)
        if case=="patient068_frame02":
           continue
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        #first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    num = len(image_list)-1
    avg_metric = [first_total / num, second_total /
                  num, third_total / num]
    #avg_metric = [first_total / num]
    logging.info(' first_total_avg is {}, second_total is {}, third_total is {}'.format(avg_metric[0], avg_metric[1], avg_metric[2]))
    #logging.info(' first_total_avg is {}'.format(avg_metric[0]))
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    snapshot_path = "/home/xueli/semi_supervised/model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    log_path = snapshot_path+"/{}_test_log.txt".format(FLAGS.model)
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(filename=snapshot_path+"/{}_test_log.txt".format(FLAGS.model), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    metric = Inference(FLAGS, snapshot_path)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    logging.info('avg_metric is {}'.format((metric[0]+metric[1]+metric[2])/3))
    logging.info('avg_metric is {}'.format(metric))