import os
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils


class PushAndGraspDataset(object):

    def __init__(self, config):

        # read configuration
        self.root = config['dataset']['path']
        self.imgs_path = os.path.join(config['dataset']['path'], config['dataset']['images'])
        self.masks_path = os.path.join(config['dataset']['path'], config['dataset']['masks'])
        self.transforms = config['dataset']['data_augmentation']
        self.is_depth = config['dataset']['is_depth']

        self.train_indices = np.load(os.path.join(config['dataset']['path'], config['dataset']['train_indices']),
                                     allow_pickle=True)
        self.test_indices = np.load(os.path.join(config['dataset']['path'], config['dataset']['test_indices']),
                                    allow_pickle=True)


        # get a list of images and GTs
        self.imgs = list(sorted(os.listdir(self.imgs_path)))
        self.masks = list(sorted(os.listdir(self.masks_path)))

    def __getitem__(self, idx):

        # set getter path
        img_path = os.path.join(self.imgs_path, self.imgs[idx])
        mask_path = os.path.join(self.masks_path, self.masks[idx])

        # read an image
        if self.is_depth:

            img = np.load(img_path).astype(dtype=np.int16)
            # TODO take the normalization somewhere out
            # depth range --> 0 to 5000 1e-4meters
            # max depth = 5000 --> background (0.5m)
            # min depth = 3459 --> the minimum depth value in the all dataset (0.npy to 012250.npy)
            # img = np.round((img - 2960) / 8).astype(np.uint8)

            # all vals are in between 0 to 250
            img = np.round(img/20).astype(np.uint8)
            img = np.repeat(img.reshape(1024, 1024, 1), 3, axis=2)
        else:
            # TODO this conversion might create problems for depth when converting back, check!
            img = Image.open(img_path).convert("RGB")

        # read a GT mask and reformat
        mask = cv2.imread(mask_path)
        mask = np.asarray(mask)
        mask = mask[:, :, :1]
        mask = np.squeeze(mask, axis=2)
        # plt.imshow(mask)
        # plt.show()

        # list of objects - All data had been cleaned and obj_ids are in ordered
        obj_ids = np.unique(mask)
        # Remove background (i.e. 0)
        obj_ids = obj_ids[1:]

        # combine the masks TODO read height and weight from img
        num_objs = len(obj_ids)
        masks = np.zeros((num_objs, 1024, 1024), dtype=np.uint8)
        for i in range(num_objs):
            masks[i][np.where(mask == i+1)] = 1

        # get bounding box coordinates for each mask
        boxes = []
        for i in range(num_objs):
            # plt.imshow(masks[i])
            # plt.show()
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

            # fig, ax = plt.subplots(1)
            # ax.imshow(img)
            # In the documentation it say bottom left corner, here i gave upper left corner for the correct box!!
            # rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            # plt.show()

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # If there is no object on the scene, this will be used to ignore that instance during coco eval
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # this control is to solve 'loss is NaN error' during training which can be a result of an invalid box
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # TODO data augmentation
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # convert image to torch tensor
        # if self.is_depth:
        #     img = torch.from_numpy(img)
        #     asd=23
        # else:
        tt = T.ToTensor()
        img = tt(img)

        return img, target

    # def augment_data(train):
    #     transforms = [T.ToTensor()]
    #     if train:
    #         transforms.append(T.RandomHorizontalFlip(0.5))
    #     return T.Compose(transforms)

    def __len__(self):
        return len(self.imgs)