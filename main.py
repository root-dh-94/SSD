import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 32
boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])

#Create network
network = SSD(class_num,batch_size)
network.cuda()
cudnn.benchmark = True

val = False
if not args.test:
    dataset = COCO("data/data/train/images/", "data/data/train/annotations/", class_num, boxs_default, train = True, image_size=320, val = False)
    dataset_test = COCO("data/data/train/images/", "data/data/train/annotations/", class_num, boxs_default, train = False, image_size=320, val = True)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0, drop_last = True)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, image_name = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
        loss = avg_loss/avg_count
        train_losses.append(loss)
        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, loss))
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("train", image_name, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, epoch,"train")
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        visualize_pred("train", image_name, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, epoch,"train_nms")
        
        
        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_, image_name = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)

            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            avg_loss += loss_net.data
            avg_count += 1

            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        loss = avg_loss/avg_count
        val_losses.append(loss)
        print('[%d] val loss: %f' % (epoch, loss))
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("val", image_name, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, epoch, "test")
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        visualize_pred("val", image_name, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, epoch, "test_nms")
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')
    
    plt.plot(np.arange(1,epoch+1),train_losses,"g-",label = "train losses")
    plt.plot(np.arange(1,epoch+1),val_losses,"r-",label = "test_losses")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.yticks(np.arange(.09,.22,.04))
    plt.title("Losses across Epochs")
    plt.legend()
    plt.savefig("losses.png")

else:
    #TEST
    idxs = []
    dataset_test = COCO("data/data/test/images/", "data/data/train/annotations/", class_num, boxs_default, train = False, image_size=320, val = False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, image_name = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        
        visualize_pred("test", image_name, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, image_name, "test_set_nms")
        
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        
        visualize_pred("test", image_name, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, image_name, "test_set_nms")
        for idx, i in enumerate(pred_confidence[0:3]):
            if i[0] != 0 or i[1] != 0 or i[2] ! = 0:
                idxs.append(idx)
        
        with open('image_name.txt','w') as f:
            for idx in idxs:
                d1x,d1y,d1w,d1h = pred_box[idx]
                p1x,p1y,p1w,p1h = boxs_default[idx,0:4]
                x1_centre = p1w * d1x + p1x
                y1_centre = p1h * d1y + p1y
                w1 = p1w * np.exp(d1w)
                h1 = p1h * np.exp(d1h)
                x1_min = x1_centre - w1 / 2
                y1_min = y1_centre - h1 / 2
                x1_max = x1_centre + w1 / 2
                y1_max = y1_centre + h1 / 2
                id = np.argmax(pred_confidence[idx][0:3])
                f.write(str(id) + ' ' + str(x1_min) + ' ' + str(y1_min) + ' ' + str(w1) + ' ' + str(h1) + '\n')
        cv2.waitKey(1000)



