import cv2
import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET

def selective_search(img, strategy):
    """
    @brief Selective search with different strategies
    @param img The input image
    @param strategy The strategy selected ['color', 'all']
    @retval bboxes Bounding boxes
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()
    ##################################################
    # TODO: For this part, please set the K as 200,  #
    #       sigma as 0.8 for the graph segmentation. #
    #       Use gs as the graph segmentation for ss  #
    #       to process after strategies are set.     #
    ##################################################
    gs.setK(200)
    gs.setSigma(0.8) 
    stra_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    stra_text = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    stra_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
    stra_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    stra_multi = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(stra_color, stra_text, stra_size, stra_fill)

    if strategy == 'color':
        stra = stra_color
    else:
        stra = stra_multi
       

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #ss.setBaseImage(img)
    ss.addImage(img) 
    ss.addGraphSegmentation(gs)
    ss.addStrategy(stra)
    
    
    #ss.switchToSelectiveSearchQuality()

    ##################################################
    # End of TODO                                    #
    ##################################################
    bboxes = ss.process()
    xyxy_bboxes = []

    for box in bboxes:
        x, y, w, h = box
        xyxy_bboxes.append([x, y, x+w, y + h])

    return xyxy_bboxes

def parse_annotation(anno_path):
    """
    @brief Parse annotation files for ground truth bounding boxes
    @param anno_path Path to the file
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()
    gt_bboxes = []
    for child in root:
        if child.tag == 'object':
            for grandchild in child:
                if grandchild.tag == "bndbox":
                    x0 = int(grandchild.find('xmin').text)
                    x1 = int(grandchild.find('xmax').text)
                    y0 = int(grandchild.find('ymin').text)
                    y1 = int(grandchild.find('ymax').text)
                    gt_bboxes.append([x0, y0, x1, y1])
    return gt_bboxes

def bb_intersection_over_union(boxA, boxB):
    """
    @brief compute the intersaction over union (IoU) of two given bounding boxes
    @param boxA numpy array (x_min, y_min, x_max, y_max)
    @param boxB numpy array (x_min, y_min, x_max, y_max)
    """
    ##################################################
    # TODO: Implement the IoU function               #
    ##################################################
    #box bound at intersection
    x_left = max(boxA[0], boxB[0])
    x_right = min(boxA[2], boxB[2])
    y_bottom = max(boxA[1], boxB[1])
    y_top = min(boxA[3], boxB[3])
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_sum = boxA_area + boxB_area

    #if no intersect
    if x_left >= x_right or y_top <= y_bottom:
        iou = 0
    else:
        intersect_area = (x_right - x_left) * (y_top - y_bottom)
        iou = intersect_area / (area_sum - intersect_area)

    ##################################################
    # End of TODO                                    #
    ##################################################
    return iou

def visualize(img, boxes, color):
    """
    @breif Visualize boxes
    @param img The target image
    @param boxes The box list
    @param color The color
    """
    for box in boxes:
        ##################################################
        # TODO: plot the rectangles with given color in  #
        #       the img for each box.                    #
        ##################################################
        cv2.rectangle(img, (box[0],box[1]), (box[2], box[3]), color, 1)
        

        ##################################################
        # End of TODO                                    #
        ##################################################
    return img


def main():
    parser = argparse.ArgumentParser()
    #change default between 'color' and 'all' depending on the strategy wanted
    #parser.add_argument('--strategy', type=str, default='color')
    parser.add_argument('--strategy', type=str, default='all')

    args =parser.parse_args()
    img_dir = './HW2_Data/JPEGImages'
    anno_dir = './HW2_Data/Annotations'
    thres = .5

    

    img_list = os.listdir(img_dir)
    num_hit = 0
    num_gt = 0

    for img_path in img_list:
        """
        Load the image file here through cv2.imread
        """
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)
        ##################################################
        # TODO: Load the image with OpenCV               #
        ##################################################
        img = cv2.imread(img_name)


        ##################################################
        # End of TODO                                    #
        ##################################################

        proposals = selective_search(img, args.strategy)
        gt_bboxes = parse_annotation(os.path.join(anno_dir, img_id + ".xml"))
        iou_bboxes = []  # proposals with IoU greater than 0.5

        ##################################################
        # TODO: For all the gt_bboxes in each image,     #
        #       please calculate the recall of the       #
        #       gt_bboxes according to the document.     #
        #       Store the bboxes with IoU >= 0.5         #
        #       If there is more than one proposal has   #
        #       IoU >= 0.5 with a same groundtruth bbox, #
        #       store the one with biggest IoU.          #
        ##################################################
        recall = 0.0
        overlap = 0.0
        all_gt = 0.0
        for box in gt_bboxes:
            IoU = 0.0
            curr_proposal = []
            all_gt = all_gt + 1
            idx = -1
           
            for proposal in proposals:
                curr_iou = bb_intersection_over_union(box, proposal)
               
                
                if curr_iou > IoU:
                    IoU = curr_iou
                    if curr_iou >= thres:
                        curr_proposal = proposal

            #iou_bboxes.append(curr_proposal)
            
            if IoU >= thres:
                overlap = overlap + 1
                iou_bboxes.append(curr_proposal)
                #print("add box")

        recall = overlap / all_gt
        print("Recall : " + img_id,recall)   
        print("overlap : " + img_id,overlap)  
        print("all_gt : " + img_id,all_gt)  
    

        ##################################################
        # End of TODO                                    #
        ##################################################
        
        vis_img = img.copy()
        vis_img = visualize(vis_img, gt_bboxes, (255, 0, 0)) 
        vis_img = visualize(vis_img, iou_bboxes, (0, 0, 255))
       

        proposals_img = img.copy()
        proposals_img = visualize(proposals_img, gt_bboxes, (255, 0, 0))  
        proposals_img = visualize(proposals_img, proposals, (0, 0, 255))
        
        ##################################################
        # TODO: (optional) You may use cv2 to visualize  #
        #       or save the image for report.            #
        ##################################################

        if args.strategy == 'color':
            output_vis = cv2.imwrite('//Users/Carolyn/Downloads/CSCI_677_HW2/vis_col_' + img_id + ".png",vis_img)
            output_proposal = cv2.imwrite('//Users/Carolyn/Downloads/CSCI_677_HW2/proposal_col_' + img_id + ".png",proposals_img)
            print("Vis Color Image written to file-system : ",output_vis)
            print("Proposal Color Image written to file-system : ",output_proposal)
        else:
            output_vis = cv2.imwrite('//Users/Carolyn/Downloads/CSCI_677_HW2/vis_all_' + img_id + ".png",vis_img)
            output_proposal = cv2.imwrite('//Users/Carolyn/Downloads/CSCI_677_HW2/proposal_all_' + img_id + ".png",proposals_img)
            print("Vis All Image written to file-system : ",output_vis)
            print("Proposal All Image written to file-system : ",output_proposal)              

        ##################################################
        # End of TODO                                    #
        ##################################################
        


if __name__ == "__main__":
    main()




