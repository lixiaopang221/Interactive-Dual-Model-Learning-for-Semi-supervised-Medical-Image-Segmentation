import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_dual
#import models.network as models
from models.whole_model import MODULE
from data_transform.lib import transforms_for_rot, transforms_back_rot, transforms_for_noise, transforms_for_scale, transforms_back_scale, postprocess_scale

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/xueli/dataset/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Dual_Students_MobileNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mobilenet', help='model_name, if backbone is unet_dual or denseunet mobilenet')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=14,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency_rampup', type=float,
                    default=181.0, help='consistency_rampup')
parser.add_argument('--consistency-scale', default=1.0, type=float, metavar='WEIGHT',
                    help='use consistency loss with given weight (default: None)')

parser.add_argument('--stabilization-rampup', default=181.0, type=float, metavar='EPOCHS',
                    help='length of the stabilization loss ramp-up')
parser.add_argument('--stable-threshold', default=0.4, type=float, metavar='THRESHOLD',
                    help='threshold for stable sample')
parser.add_argument('--stable-threshold-teacher', default=0.5, type=float, metavar='THRESHOLD',
                    help='threshold for stable sample')
parser.add_argument('--stabilization-scale', default=1.0, type=float, metavar='WEIGHT',
                    help='use stabilization loss with given weight (default: None)')                    

parser.add_argument('--logit-distance-cost', default=0.05, type=float, metavar='WEIGHT',
                    help='let the student model have two outputs and use an MSE loss '
                    'the logits with the given weight (default: only have one output)')
                    
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    if "LITS" in dataset:
        ref_dict = {"6": 714, "11": 1516, "22": 3024,
                    "32": 4607, "40": 6110, "52": 7682, "104": 15227}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, total_epoch):
    lr = args.base_lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.base_lr - args.initial_lr) + args.initial_lr

    # decline lr
    lr *= ramps.zero_cosine_rampdown(epoch, total_epoch)

    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr
    return lr

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    teacher_threshold = args.stable_threshold_teacher
    student_threshold = args.stable_threshold
    def create_model(ema=False):
        # Network definition
        #model = MODULE().cuda()
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    stu1_model = create_model()  # student1 model
    stu2_model = create_model()  # student2 model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    stu1_model.train()
    stu2_model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    stu1_optimizer = optim.SGD(stu1_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    stu2_optimizer = optim.SGD(stu2_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    residual_logit_criterion = losses.symmetric_mse_loss
    eucliden_distance = losses.softmax_mse_loss
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
        stabilization_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
        stabilization_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    val_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    noise_r=0.2
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume1_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
               
            noise = torch.clamp(torch.randn_like(
                volume1_batch) * 0.1, -noise_r, noise_r)
            volume2_batch = volume1_batch + noise

            stu1_out1 = stu1_model(volume1_batch)
            stu1e_out2 = stu1_model(volume2_batch)
            stu2_out2 = stu2_model(volume1_batch)
            stu2e_out1 = stu2_model(volume2_batch)
            
            assert len(stu1_out1) == 2
            stu1_seg_logit, stu1_cons_logit = stu1_out1
            stu2_seg_logit, stu2_cons_logit = stu2_out2
            stu1e_seg_logit, stu1e_cons_logit = stu1e_out2
            stu2e_seg_logit, stu2e_cons_logit = stu2e_out1

            stu1_res_loss = args.logit_distance_cost * residual_logit_criterion(stu1_seg_logit, stu1_cons_logit) / batch_size
            stu2_res_loss = args.logit_distance_cost * residual_logit_criterion(stu2_seg_logit, stu2_cons_logit) / batch_size
            
            # segmentation loss
            stu1_loss_ce = ce_loss(stu1_seg_logit[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) / args.labeled_bs
            stu1_loss_dice = dice_loss(torch.softmax(stu1_seg_logit[:args.labeled_bs], dim=1), label_batch[:args.labeled_bs].unsqueeze(1))
            stu1_seg_loss = 0.5 * (stu1_loss_dice + stu1_loss_ce)
            
            stu2_loss_ce = ce_loss(stu2_seg_logit[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) / args.labeled_bs
            stu2_loss_dice = dice_loss(torch.softmax(stu2_seg_logit[:args.labeled_bs], dim=1), label_batch[:args.labeled_bs].unsqueeze(1))
            stu2_seg_loss = 0.5 * (stu2_loss_dice + stu2_loss_ce)
             
            stu1_loss = stu1_seg_loss+stu1_res_loss
            stu2_loss = stu2_seg_loss+stu2_res_loss
            
            # consistency loss
            consistency_weight = args.consistency_scale * ramps.sigmoid_rampup(epoch_num, args.consistency_rampup)

            stu1_seg_logit = Variable(stu1_seg_logit.detach().data, requires_grad=False)
            stu1_consistency_loss1 = consistency_weight * consistency_criterion(stu1e_cons_logit, stu1_seg_logit)
            stu1_consistency_loss = torch.mean(stu1_consistency_loss1)       
            stu1_loss += stu1_consistency_loss
            
            stu2_seg_logit = Variable(stu2_seg_logit.detach().data, requires_grad=False)
            stu2_consistency_loss1 = consistency_weight * consistency_criterion(stu2e_cons_logit, stu2_seg_logit)
            stu2_consistency_loss = torch.mean(stu2_consistency_loss1)
            stu2_loss += stu2_consistency_loss 

            # stabilization loss
            # value (seg_v) and index (seg_i) of the max probability in the prediction
            stu1_seg_v, stu1_seg_i = torch.max(F.softmax(stu1_seg_logit[args.labeled_bs:], dim=1), dim=1)  #(4,256,256)
            stu2_seg_v, stu2_seg_i = torch.max(F.softmax(stu2_seg_logit[args.labeled_bs:], dim=1), dim=1)
            stu1e_seg_v, stu1e_seg_i = torch.max(F.softmax(stu1e_seg_logit[args.labeled_bs:], dim=1), dim=1)
            stu2e_seg_v, stu2e_seg_i = torch.max(F.softmax(stu2e_seg_logit[args.labeled_bs:], dim=1), dim=1)
            
            stu1_seg_v, stu1_seg_i = stu1_seg_v.unsqueeze(1), stu1_seg_i.unsqueeze(1)
            stu2_seg_v, stu2_seg_i = stu2_seg_v.unsqueeze(1), stu2_seg_i.unsqueeze(1)
            stu1e_seg_v, stu1e_seg_i = stu1e_seg_v.unsqueeze(1), stu1e_seg_i.unsqueeze(1)
            stu2e_seg_v, stu2e_seg_i = stu2e_seg_v.unsqueeze(1), stu2e_seg_i.unsqueeze(1)

            # detach logit -> for generating stablilization target 
            in_stu2_cons_logit = Variable(stu2_cons_logit[args.labeled_bs:].detach().data, requires_grad=False)
            tar_stu1_seg_logit = Variable(stu1_seg_logit[args.labeled_bs:].clone().detach().data, requires_grad=False)

            in_stu1_cons_logit = Variable(stu1_cons_logit[args.labeled_bs:].detach().data, requires_grad=False)
            tar_stu2_seg_logit = Variable(stu2_seg_logit[args.labeled_bs:].clone().detach().data, requires_grad=False)
            
            # generate target for each sample
            # stu1_stable 
            # 1st condition
            stu1_mask_1 = (stu1_seg_i == stu1e_seg_i)
            # 2nd condition
            stu1_mask_2 = stu1_mask_1 *  stu1_seg_v
            stu1e_mask_2 = stu1_mask_1 * stu1e_seg_v
            stu1_mask_3 = (stu1_mask_2 > args.stable_threshold)
            stu1e_mask_3 = (stu1e_mask_2 > args.stable_threshold)
            # finally mask
            stu1_mask = stu1_mask_3 + stu1e_mask_3 
            stu1_mask_inv = (stu1_mask == False)
            
            # stu2_stable                
            # 1st condition
            stu2_mask_1 = (stu2_seg_i == stu2e_seg_i)
            # 2nd condition
            stu2_mask_2 = stu2_mask_1 *  stu2_seg_v
            stu2e_mask_2 = stu2_mask_1 * stu2e_seg_v
            stu2_mask_3 = (stu2_mask_2 > args.stable_threshold)
            stu2e_mask_3 = (stu2e_mask_2 > args.stable_threshold)
            # finally mask
            stu2_mask = stu2_mask_3 + stu2e_mask_3
            stu2_mask_inv = (stu2_mask == False) 
            
            # stu1 and stu2 are stable
            mask_1 = stu1_mask * stu2_mask 
            # stu1 and stu2  eucliden_distance
            stu1_dis = eucliden_distance(stu1e_seg_logit[args.labeled_bs:], stu1_seg_logit[args.labeled_bs:])
            stu2_dis = eucliden_distance(stu2e_seg_logit[args.labeled_bs:], stu2_seg_logit[args.labeled_bs:])
            mask_stu1_dis = (stu1_dis > stu2_dis) * mask_1
            mask_stu1_dis_inv = (mask_stu1_dis == False)
            mask_stu2_dis = (stu2_dis > stu1_dis) * mask_1
            mask_stu2_dis_inv = (mask_stu2_dis == False)
            
            #stu2 to supervised stu1
            mask1_stu2 = mask_stu1_dis + stu2_mask * stu1_mask_inv
            mask1_stu2_inv = (mask1_stu2 == False)
            tar_stu2_seg_logit = tar_stu2_seg_logit * mask1_stu2 + in_stu1_cons_logit * mask1_stu2_inv
            
            # stu1 to supervised stu2
            mask1_stu1 = mask_stu2_dis + stu1_mask * stu2_mask_inv
            mask1_stu1_inv = (mask1_stu1 == False)
            tar_stu1_seg_logit = tar_stu1_seg_logit * mask1_stu1 + in_stu2_cons_logit * mask1_stu1_inv         
          
            # calculate stablization weight
            stabilization_weight = args.stabilization_scale * ramps.sigmoid_rampup(epoch_num, args.stabilization_rampup)
            stabilization_weight = (1-(args.labeled_bs / batch_size)) * stabilization_weight
            
            # stabilization loss for stu2 model
            stu2_stabilization_loss1 = stabilization_weight * stabilization_criterion(stu2_cons_logit[args.labeled_bs:], 
                                                                                      tar_stu1_seg_logit)
            stu2_stabilization_loss = torch.mean(stu2_stabilization_loss1)           
            stu2_loss += stu2_stabilization_loss
            
            # stabilization loss for stu1 model
            stu1_stabilization_loss1 = stabilization_weight * stabilization_criterion(stu1_cons_logit[args.labeled_bs:], 
                                                                                     tar_stu2_seg_logit)
            stu1_stabilization_loss = torch.mean(stu1_stabilization_loss1)            
            stu1_loss += stu1_stabilization_loss
            
            stu1_optimizer.zero_grad()
            stu1_loss.backward()
            stu1_optimizer.step()

            stu2_optimizer.zero_grad()
            stu2_loss.backward()
            stu2_optimizer.step()
            
            # adjust learning rate
            lr_stu1 = adjust_learning_rate(stu1_optimizer, epoch_num, i_batch, len(trainloader), max_epoch)
            lr_stu2 = adjust_learning_rate(stu2_optimizer, epoch_num, i_batch, len(trainloader), max_epoch)
            
            
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_stu1, iter_num)
            writer.add_scalar('info/stu1_loss', stu1_loss, iter_num)
            writer.add_scalar('info/stu2_loss', stu2_loss, iter_num)
            writer.add_scalar('info/stu1_loss_ce', stu1_seg_loss, iter_num)
            writer.add_scalar('info/stu2_loss_ce', stu2_seg_loss, iter_num)
            writer.add_scalar('info/stu1_stabilization_loss',
                              stu1_stabilization_loss, iter_num)
            writer.add_scalar('info/stu2_stabilization_loss',
                              stu2_stabilization_loss, iter_num)
            writer.add_scalar('info/stu1_consistency_loss',
                              stu1_consistency_loss, iter_num)
            writer.add_scalar('info/stu2_consistency_loss',
                              stu2_consistency_loss, iter_num) 
                              
            logging.info(
                'iteration %d : stu1_loss : %f, stu2_loss: %f' %
                (iter_num, stu1_loss.item(), stu2_loss.item()))
           
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs= torch.argmax(torch.softmax(
                    stu1_seg_logit, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
            if iter_num > 0 and iter_num % 200 == 0:
                stu1_model.eval()               
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    val_num = i_batch + (iter_num / 200) * len(db_val)
                    metric_i, writer = test_single_volume_dual(
                        sampled_batch["image"], sampled_batch["label"], stu1_model, val_num, writer, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(stu1_model.state_dict(), save_mode_path)
                    torch.save(stu1_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                stu1_model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(stu1_model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/home/xueli/semi_supervised/model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    log_path = snapshot_path+"/log.txt"
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
