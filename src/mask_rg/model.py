import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data as td
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor
from src.mask_rg.engine import train_one_epoch, evaluate
import src.mask_rg.utils
import time
import datetime
import yaml
from collections import OrderedDict
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MaskRGNetwork(nn.Module):
    def __init__(self, config):
        super(MaskRGNetwork, self).__init__()

        self.training = False # check if this necessary

        # read configuration
        if config['model']['file'] == 'new':
            use_old_weights = False
            self.m_pretrained = False
        elif config['model']['file'] == 'pretrained':
            use_old_weights = False
            self.m_pretrained = True
        else:
            use_old_weights = True
            self.weigths_path = os.path.join(config['model']['path'], config['model']['file'])

        self.num_epochs = config['model']['settings']['epochs']
        self.lr = config['model']['settings']['learning_rate']
        self.batch_size = config['model']['settings']['batch_size']
        self.backbone = config['model']['settings']['backbone']
        self.bb_pretrained = config['model']['settings']['backbone_pretrained']
        is_cuda = config['model']['settings']['cuda_available']

        # uncomment for training!!!
        # self.train_indices = np.load(os.path.join(config['dataset']['path'], config['dataset']['train_indices']))
        # self.test_indices = np.load(os.path.join(config['dataset']['path'], config['dataset']['test_indices']))

        self.saving_path = config['saving']['path']
        self.saving_prefix = config['saving']['model_name']

        if is_cuda:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("CUDA device found!")
            else:
                print("WARNING: CUDA is not available!!! CPU will be used instead!")
                self.device = 'cpu'
        else:
            print("CPU will be used to train/evaluate the network!")
            self.device = 'cpu'

        # self.backbone = torchvision.models.resnet34(pretrained=True, progress=True)
        # self.anchors = AnchorGenerator(sizes=ANCHOR_SIZES)
        # self.backbone.out_channels = [512, 512, 512]

        if self.backbone == 'resnet50':
            if not use_old_weights:
                self.mask_r_cnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=self.m_pretrained, progress=True,
                                                                            pretrained_backbone=self.bb_pretrained)
            else:
                self.mask_r_cnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
                                                                                     progress=True,
                                                                                     pretrained_backbone=self.bb_pretrained)

        # get number of input features for the classifier
        self.input_features = self.mask_r_cnn.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one --> category-agnostic so number of classes=2
        self.mask_r_cnn.roi_heads.box_predictor = FastRCNNPredictor(self.input_features, 2)

        # now get the number of input features for the mask classifier
        self.input_features_mask = self.mask_r_cnn.roi_heads.mask_predictor.conv5_mask.in_channels

        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.mask_r_cnn.roi_heads.mask_predictor = MaskRCNNPredictor(self.input_features_mask, hidden_layer, 2)

        self.mask_r_cnn.to(self.device)
        # self.mask_r_cnn = MaskRCNN(backbone=self.backbone, num_classes=2, rpn_anchor_generator=self.anchors)

        # construct an optimizer
        self.params = [p for p in self.mask_r_cnn.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=0.00005)

        if use_old_weights:
            self.load_weights()


    def train_model(self):

        for epoch in range(self.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.mask_r_cnn, self.optimizer, self.data_loader, self.device, epoch, print_freq=5)
            # update the learning rate
            # self.lr_scheduler.step()
            self.optimizer.step()
            self._save_model('_epoch_no_' + str(epoch) + '_')


    def evaluate_model(self):
        # evaluate on the test dataset
        res = evaluate(self.mask_r_cnn, self.data_loader, device=self.device)
        return res

    def eval_single_img(self, img):
        # self.device = torch.device("cpu")

        self.mask_r_cnn.eval()
        with torch.no_grad():
            preds = self.mask_r_cnn(img)

        return preds

    def load_weights(self):
            # this should be called after the initialisation of the model
            self.mask_r_cnn.load_state_dict(torch.load(self.weigths_path))

    def set_data(self, data,  is_test=False):

        data_subset = td.Subset(data, data.train_indices) if not is_test else td.Subset(data, data.test_indices)

        # init a data loader either for training or testing
        # multiprocessing for data loading (num_workers) is not recommended for CUDA, so automatic memory pining
        # (pin_memory=True) is used instead.
        self.data_loader = td.DataLoader(data_subset, batch_size=self.batch_size, shuffle=False,
                                                       num_workers=0, pin_memory=True,
                                                       collate_fn=utils.collate_fn)

    def _save_model(self, string=None):
        t = time.time()
        timestamp = datetime.datetime.fromtimestamp(t)
        file_name = self.saving_prefix + string + timestamp.strftime('%Y-%m-%d.%H:%M:%S') + '.pth'
        torch.save(self.mask_r_cnn.state_dict(), os.path.join(self.saving_path, file_name))

    @staticmethod
    def print_boxes(image, input_tensor, score_threshold=0.75):
        boxes = input_tensor[0]['boxes']
        scores = input_tensor[0]['scores']

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        num_box=0
        for box in boxes:
            box_index = np.where(boxes == box)
            box_index = int(box_index[0][0])
            if scores[box_index] > score_threshold:
                # In the documentation it say bottom left corner, here i gave upper left corner for the correct box!!
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=3, edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)
                num_box += 1
        print('num boxes that have a score higher than .75 --> ', num_box)
        plt.show()

    @staticmethod
    def print_masks(image, input_tensor, score_threshold=0.75):
        masks = input_tensor[0]['masks']
        scores = input_tensor[0]['scores']
        num_pred = masks.shape[0]
        num_masks = 0
        all = np.zeros((1024, 1024), dtype=np.uint8)
        for mask in range(0, num_pred):
            if scores[mask] > score_threshold:
                #TODO if cuda, add a control here

                mask_arr = np.asarray(masks[mask].cpu().detach()).reshape((1024, 1024))
                mask_arr = np.where(mask_arr > score_threshold, 1, 0)
                all[np.where(mask_arr > 0)] = num_masks

                num_masks += 1
        plt.imshow(all)
        plt.show()
        print('num masks that have a score higher than .75 --> ', num_masks)


    def forward(self, input_img=None):
        pass
        # print("evaluation")
        # tt = T.ToTensor
        # input_tensor = tt(input_img)
        # tensor_list = [input_tensor]
        # self.mask_r_cnn.eval()
        # predictions = self.mask_r_cnn(tensor_list)
        # return predictions

