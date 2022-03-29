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
import numpy as np
import os
import cv2
import math
from PIL import Image

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    boxes = np.zeros((135,4,8))
    
    #generate boxes for grid 10
    for idx, grid in enumerate(layers):
        grid_size = 1 / grid
        grid_centre = grid_size / 2
        ssize = small_scale[idx]
        lsize = large_scale[idx]
        box_dims = [[ssize,ssize], [lsize,lsize], [lsize*math.sqrt(2),lsize/math.sqrt(2)], [lsize/math.sqrt(2),lsize*math.sqrt(2)]]
        boxes = diff_grids(boxes, grid_centre, box_dims)

    #clip boxes exceeding img limits and reshape to desired shape
    boxes = np.clip(boxes, 0, 1)
    boxes = np.reshape(boxes, (540,8))

    return boxes

def diff_grids(boxes, grid_centre, box_dims):
    #loop through each cell in size 10 grid
    for i in range(100):
        row = i // 10
        column = i % 10

        #set box dimensions for each one of 4 default boxes
        for j in range(4):
            boxes[i][j][0] = (column * 0.1) + grid_centre
            boxes[i][j][1] = (row * 0.1) + grid_centre
            boxes[i][j][2] = box_dims[j][0]
            boxes[i][j][3] = box_dims[j][1]
            boxes[i][j][4] = boxes[i][j][0] - boxes[i][j][2] / 2
            boxes[i][j][5] = boxes[i][j][1] - boxes[i][j][3] / 2
            boxes[i][j][6] = boxes[i][j][0] + boxes[i][j][2] / 2
            boxes[i][j][7] = boxes[i][j][0] + boxes[i][j][3] / 2
    
    return boxes

#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    x_centre = (x_min + x_max) / 2
    y_centre = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    for idx, val in enumerate(ious_true):
        if val:
            #update ann_box
            ann_box[idx][0] = (x_centre - boxs_default[idx][0]) / boxs_default[idx][2]
            ann_box[idx][1] = (y_centre - boxs_default[idx][1]) / boxs_default[idx][3]
            ann_box[idx][2] = math.log(width / boxs_default[idx][2])
            ann_box[idx][3] = math.log(height / boxs_default[idx][3])

            #update ann_confidence
            if cat_id == 0:
                ann_confidence[idx][0] = 1
                ann_confidence[idx][3] = 0
            
            elif cat_id == 1:
                ann_confidence[idx][1] = 1
                ann_confidence[idx][3] = 0

            elif cat_id == 2:
                ann_confidence[idx][2] = 1
                ann_confidence[idx][3] = 0

    if ious_true.sum() == 0:
        ious_true = np.argmax(ious)

        ann_box[ious_true][0] = (x_centre - boxs_default[ious_true][0]) / boxs_default[ious_true][2]
        ann_box[ious_true][1] = (y_centre - boxs_default[ious_true][1]) / boxs_default[ious_true][3]
        ann_box[ious_true][2] = math.log(width / boxs_default[ious_true][2])
        ann_box[ious_true][3] = math.log(height / boxs_default[ious_true][3])

        if cat_id == 0:
            ann_confidence[idx][0] = 1
            ann_confidence[idx][3] = 0
            
        elif cat_id == 1:
            ann_confidence[idx][1] = 1
            ann_confidence[idx][3] = 0

        elif cat_id == 2:
            ann_confidence[idx][2] = 1
            ann_confidence[idx][3] = 0
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    return ann_box, ann_confidence


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #Read txt
        with open(ann_name) as f:
            lines = f.readlines()
        
        #Load and open images
        image = Image.open(img_name)
        orig_w, orig_h = image.size[0], image.size[1]
        image = transforms.Resize(image,(self.image_size,self.image_size))

        #loop through lines in txt file
        for line in lines:
            box_coords = line.split(" ")
            class_id, x_min, y_min, x_max, y_max = box_coords
            y_max = y_max[:-2]
            class_id, x_min, y_min, x_max, y_max = int(class_id), float(x_min), float(y_min), float(x_max), float(y_max)

            #renormalize box values wrt resized dims
            x_min = x_min * orig_w / self.image_size
            x_max = x_max * orig_w / self.image_size
            y_min = y_min * orig_h / self.image_size
            y_max = y_max * orig_h / self.image_size
            
            ann_box,ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #to use function "match":
        #ann_box,ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        return image, ann_box, ann_confidence
