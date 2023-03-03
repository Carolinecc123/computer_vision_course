!pip install pyyaml==5.1
#get torch version
import torch
torch.__version__

#some import
import os, sys
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import PascalVOCDetectionEvaluator
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#download detectron and dataset
pip install detectron2 -f \https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
!tar -xvf VOCtrainval_06-Nov-2007.tar
!mv VOCdevkit datasets

#training for faster_rcnn_R_50_FPN_3x
cfg = get_cfg()
#choose the model faster_rcnn_R_50_FPN_3x
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = 'MyVOCTraining'
#choose the train dataset
cfg.DATASETS.TRAIN = ("voc_2007_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
#adjust learning rate, start with 0.00025 and finally pick 0.00025
cfg.SOLVER.BASE_LR = 0.00025 # pick a good LR
#adjust iteration number, start with 3000 and finally pick 4000
cfg.SOLVER.MAX_ITER = 4000 
#adjust batch size, start with 128 and finally pick 100
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 100 
#number of classes is 20
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

#train the model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()



#training for retinanet_R_50_FPN_3x
cfg = get_cfg()
#choose the model retinanet_R_50_FPN_3x
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = 'MyVOCTraining'
#choose the train dataset
cfg.DATASETS.TRAIN = ("voc_2007_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml") # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
#adjust learning rate, start with 0.00025 and finally pick 0.00025
cfg.SOLVER.BASE_LR = 0.00025 # pick a good LR
#adjust iteration number, start with 3000 and finally pick 4000
cfg.SOLVER.MAX_ITER = 4000
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.MODEL.RETINANET.NUM_CLASSES = 20  #20 number of classes

#train the model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

#load and show training data curve in tensorboard
#%load_ext tensorboard
%reload_ext tensorboard
import tensorflow as tf
import datetime, os
%tensorboard --logdir 'MyVOCTraining'

#path to the trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# set the testing threshold to be 0.7 for faster_rcnn_R_50_FPN_3x
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  
# set the testing threshold to be 0.7 for retinanet_R_50_FPN_3x
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

#get metadata
balloon_metadata = MetadataCatalog.get("voc_2007_train")

#randomly pick images from validation dataset for visualization
im = cv2.imread("datasets/VOC2007/JPEGImages/000060.jpg") 
im = cv2.imread("datasets/VOC2007/JPEGImages/000210.jpg") 
im = cv2.imread("datasets/VOC2007/JPEGImages/002427.jpg")
im = cv2.imread("datasets/VOC2007/JPEGImages/002613.jpg")

outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#visualize the image
v = Visualizer(im[:, :, ::-1],
                metadata=balloon_metadata, 
                scale=1, 
                instance_mode=ColorMode.IMAGE  
)

predictions = outputs["instances"].to("cpu")

out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#show the image
cv2_imshow(out.get_image()[:, :, ::-1])


from detectron2.evaluation import PascalVOCDetectionEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
#Pascal VOC evaluator
evaluator = PascalVOCDetectionEvaluator("voc_2007_val")
val_loader = build_detection_test_loader(cfg, "voc_2007_val")
#print the evaluation outcome
print(inference_on_dataset(predictor.model, val_loader, evaluator))