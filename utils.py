import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    
    #TODO: non maximum suppression
    max_class = np.argmax(confidence_[:,0:3], axis = 1)
    a = range(0,540)
    #a = confidence_.tolist()
    b = []

    while True:
        #find max prob
        max = np.amax(confidence_[a,0:3])
        if max>threshold:
            ious = []

            #find row/col(box) where max prob occurs
            row_idx = np.where(confidence_[a,0:3]==max)[0]
            column_idx = np.argmax(confidence_[row_idx])
            
            #assign max prob box to a separate list
            b.append(confidence_[a.pop(np.where(a==max)[0])])

            #check which other boxes have same class
            same_class_rows = np.where(max_class[a] == column_idx).tolist()

            #remove the identity case
            #for idx, val in same_class_rows:
                #if val == row_idx:
                    #same_class_rows.pop(idx)
                    #break

            #calculate box metrics for max prob box
            dx,dy,dw,dh = box_[row_idx]
            px,py,pw,ph = boxs_default[row_idx,0:4]
            x_centre = pw * dx + px
            y_centre = ph * dy + py
            w = pw * np.exp(dw)
            h = ph * np.exp(dh)
            x_min = x_centre - w / 2
            y_min = y_centre - h / 2
            x_max = x_centre + w / 2
            y_max = y_centre + h / 2
            pred = [[x_centre, y_centre, w, h, x_min, y_min, x_max, y_max]]

            #calculate box metrics for each other box of same class
            for idx in same_class_rows:
                #if idx == row_idx:
                    #continue

                d1x,d1y,d1w,d1h = box_[idx]
                p1x,p1y,p1w,p1h = boxs_default[idx,0:4]
                x1_centre = p1w * d1x + p1x
                y1_centre = p1h * d1y + p1y
                w1 = p1w * np.exp(d1w)
                h1 = p1h * np.exp(d1h)
                x1_min = x1_centre - w1 / 2
                y1_min = y1_centre - h1 / 2
                x1_max = x1_centre + w1 / 2
                y1_max = y1_centre + h1 / 2

                ious.append(iou(pred, x1_min, y1_min, x1_max, y1_max))

            #remove boxes that have a lot of overlap
            for idx, val in enumerate(ious):
                if val > overlap:
                    a.pop(np.where(a==same_class_rows[idx]))

        else: break









