import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    pred_confidence = torch.reshape(pred_confidence, (-1,4))
    pred_box = torch.reshape(pred_box, (-1,4))
    ann_confidence = torch.reshape(ann_confidence, (-1,4))
    ann_box = torch.reshape(ann_box, (-1,4))

    #Separate boxes with objects from boxes without
    obj_idx = []
    noobj_idx = []
    for idx, val in pred_confidence:
        if val[0] >= .5 or val[1] >= .5 or val[2] >= .5:
            obj_idx.append(idx)
        else: noobj_idx.append(idx)
    loss_cls = F.cross_entropy(pred_confidence[obj_idx], ann_confidence[obj_idx]) + 3 * F.cross_entropy(pred_confidence[noobj_idx], ann_confidence[noobj_idx])

    loss_box = F.smooth_l1_loss(pred_box, ann_box)

    return loss_cls + loss_box


class SSD(nn.Module):

    def __init__(self, class_num, batch_size):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        self.batch_size = batch_size
        
        #TODO: define layers
        self.convBlock = nn.Sequential(nn.Conv2d(3,64,3,2,1,bias=True), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64,64,3,1,1,bias=True), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64,64,3,1,1,bias=True), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64,128,3,2,1,bias=True), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128,128,3,1,1,bias=True), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128,128,3,1,1,bias=True), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128,256,3,2,1,bias=True), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,256,3,1,1,bias=True), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,256,3,1,1,bias=True), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,512,3,2,1,bias=True), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512,512,3,1,1,bias=True), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512,512,3,1,1,bias=True), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512,256,3,2,1,bias=True), nn.BatchNorm2d(256), nn.ReLU())

        self.main_conv1 = nn.Sequential(nn.Conv2d(256,256,1,1,bias=True), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,256,3,2,1,bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.main_conv2 = nn.Sequential(nn.Conv2d(256,256,1,1,bias=True), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,256,3,1,bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.main_conv3 = nn.Sequential(nn.Conv2d(256,256,1,1,bias=True), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,256,3,1,bias=True), nn.BatchNorm2d(256), nn.ReLU())
        
        self.fork_main_conv = nn.Conv2d(256,16,1,1,bias=True)

        self.fork_branch_conv = nn.Conv2d(256,16,3,1,1,bias=True)
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        #x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.

        #TODO: define forward
        x = self.convBlock(x)

        x1 = self.main_conv1(x)
        x2_left1 = self.fork_branch_conv(x)
        x2_left1 = torch.reshape(x2_left1, (self.batch_size, 16, 100))
        x2_right1 = self.fork_branch_conv(x)
        x2_right1 = torch.reshape(x2_right1, (self.batch_size, 16, 100))

        x2_left2 = self.fork_branch_conv(x1)
        x2_left2 = torch.reshape(x2_left2, (self.batch_size, 16, 25))
        x2_right2 = self.fork_branch_conv(x1)
        x2_right2 = torch.reshape(x2_right2, (self.batch_size, 16, 25))
        x1 = self.main_conv2(x1)

        x2_left3 = self.fork_branch_conv(x1)
        x2_left3 = torch.reshape(x2_left3, (self.batch_size, 16, 9))
        x2_right3 = self.fork_branch_conv(x1)
        x2_right3 = torch.reshape(x2_right3, (self.batch_size, 16, 9))
        x1 = self.main_conv3(x1)

        x1left = self.fork_main_conv(x1)
        x1left = torch.reshape(x1left, (self.batch_size, 16, 1))
        x1right = self.fork_main_conv(x1)
        x1right = torch.reshape(x1right, (self.batch_size, 16, 1))

        bboxes = torch.cat((x2_left1,x2_left2,x2_left3,x1left),2)
        bboxes = torch.permute(bboxes, (0,2,1))
        bboxes = torch.reshape(bboxes, (self.batch_size, 540, 4))

        confidence = torch.cat((x2_right1,x2_right2,x2_right3,x1right),2)
        confidence = torch.permute(confidence, (0,2,1))
        confidence = torch.reshape(confidence, (self.batch_size, 540, 4))
        confidence = F.softmax(confidence, dim = 2)
        
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence,bboxes










