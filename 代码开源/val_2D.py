import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)   #hausdorff distance
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, num, writer, classes, patch_size=[256, 256]):
    
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_net = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            #out_net = torch.sigmoid(net(input)).squeeze(0) #LITS
            #out_net[out_net <= 0.5] = 0  #LITS
            #out_net[out_net > 0.5] = 1   #LITS
            out = out_net.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            #pred = zoom(out.squeeze(0), (x / patch_size[0], y / patch_size[1]), order=0)  #LITS
            prediction[ind] = pred
        slice = torch.from_numpy(slice)
        label_slice = label[ind, :, :]        
        label_slice = torch.from_numpy(label_slice)
        writer.add_image('valid/Image', slice.unsqueeze(0), num+ind)   
        writer.add_image('valid/GroundTruth', label_slice.unsqueeze(0) * 50, num+ind)
        out_show = out_net.unsqueeze(0)
        #out_show = out_net # LITS
        writer.add_image('valid/Prediction',out_show * 50, num+ind)
                
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    # metric_list.append(calculate_metric_percase(prediction == 1, label == 1))
    return metric_list, writer


def test_single_volume_ds(image, label, net, num, writer, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            '''out_net = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)'''
            out_net = torch.sigmoid(output_main).squeeze(0) #LITS
            out_net[out_net <= 0.5] = 0  #LITS
            out_net[out_net > 0.5] = 1   #LITS
            out = out_net.cpu().detach().numpy()
            #pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            pred = zoom(out.squeeze(0), (x / patch_size[0], y / patch_size[1]), order=0)  #LITS
            prediction[ind] = pred
        slice = torch.from_numpy(slice)
        label_slice = label[ind, :, :]        
        label_slice = torch.from_numpy(label_slice)
        writer.add_image('valid/Image', slice.unsqueeze(0), num+ind)   
        writer.add_image('valid/GroundTruth', label_slice.unsqueeze(0) * 50, num+ind)
        #out_show = out_net.unsqueeze(0)
        out_show = out_net # LITS
        writer.add_image('valid/Prediction',out_show * 50, num+ind)                
    metric_list = []
    '''for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))'''
    metric_list.append(calculate_metric_percase(prediction == 1, label == 1))
    return metric_list, writer

def test_single_volume_dual(image, label, net, num, writer, classes, patch_size=[256, 256]):
    
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out, out2 = net(input)
            '''out_net = torch.argmax(torch.softmax(
                out, dim=1), dim=1).squeeze(0)'''
            out_net = torch.sigmoid(out).squeeze(0) #LITS
            out_net[out_net <= 0.5] = 0  #LITS
            out_net[out_net > 0.5] = 1   #LITS
            out = out_net.cpu().detach().numpy()
            #pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            pred = zoom(out.squeeze(0), (x / patch_size[0], y / patch_size[1]), order=0)  #LITS
            prediction[ind] = pred
        slice = torch.from_numpy(slice)
        label_slice = label[ind, :, :]        
        label_slice = torch.from_numpy(label_slice)
        writer.add_image('valid/Image', slice.unsqueeze(0), num+ind)   
        writer.add_image('valid/GroundTruth', label_slice.unsqueeze(0) * 50, num+ind)
        #out_show = out_net.unsqueeze(0)
        out_show = out_net  #LITS
        writer.add_image('valid/Prediction',out_show * 50, num+ind)               
    metric_list = []
    '''for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))'''
    metric_list.append(calculate_metric_percase(prediction == 1, label == 1))
    return metric_list, writer

def test_single_volume_dual_share_encoder(image, label, net, num, writer, classes, patch_size=[256, 256]):
    
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out, out2 = net(input,1)
            out_net = torch.argmax(torch.softmax(
                out, dim=1), dim=1).squeeze(0)
            '''out_net = torch.sigmoid(out).squeeze(0) #LITS
            out_net[out_net <= 0.5] = 0  #LITS
            out_net[out_net > 0.5] = 1   #LITS'''
            out = out_net.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            #pred = zoom(out.squeeze(0), (x / patch_size[0], y / patch_size[1]), order=0)  #LITS
            prediction[ind] = pred
        slice = torch.from_numpy(slice)
        label_slice = label[ind, :, :]        
        label_slice = torch.from_numpy(label_slice)
        writer.add_image('valid/Image', slice.unsqueeze(0), num+ind)   
        writer.add_image('valid/GroundTruth', label_slice.unsqueeze(0) * 50, num+ind)
        out_show = out_net.unsqueeze(0)
        #out_show = out_net  #LITS
        writer.add_image('valid/Prediction',out_show * 50, num+ind)               
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    #metric_list.append(calculate_metric_percase(prediction == 1, label == 1))
    return metric_list, writer