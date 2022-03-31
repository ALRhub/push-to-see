from src.mask_rg.model import MaskRGNetwork
import yaml
import numpy as np
import torchvision.transforms as T
import torch
from src.mask_rg.rewards import RewardGenerator


CONFIG_PATH = './model_config.yaml'

class MaskRG(object):

    def __init__(self, confidence_threshold=0.75, mask_threshold=0.75):
        # Make sure this configuration file is in your execution path
        with open(CONFIG_PATH) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        with torch.no_grad():
            self.model = MaskRGNetwork(configuration)
            self.model.load_weights()
            self.rg = RewardGenerator(confidence_threshold, mask_threshold)
            self.prediction = None
            self.gt = None

    def set_reward_generator(self, depth_image, gt_segmentation):

        depth_image = np.round(depth_image / 20).astype(np.uint8)
        depth_image = np.repeat(depth_image.reshape(1024, 1024, 1), 3, axis=2)
        with torch.no_grad():
            tt = T.ToTensor()
            depth_tensor = tt(depth_image)
            depth_tensor = depth_tensor.cuda()
        # print(depth_tensor.requires_grad)
            self.prediction = self.model.eval_single_img([depth_tensor])

        self.gt = gt_segmentation
        self.rg.set_element(self.prediction, self.gt)

    def get_current_rewards(self):
        return self.rg.get_reward()

    def print_segmentation(self, pred_ids):
        return self.rg.print_seg_diff(pred_ids)

    def print_masks(self):
        return self.rg.print_masks()
