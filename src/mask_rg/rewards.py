import numpy as np
import matplotlib.pyplot as plt

NON_DETECTION_PUNISHMENT = -0.02
# EXTRA_DETECTION_PUNISHMENT = 0.0 #-0.01

color_space = np.asarray([#[78, 121, 167],  # blue
                          #[89, 161, 79],  # green
                         # [156, 117, 95],  # brown
                         # [242, 142, 43],  # orange
                          #[237, 201, 72],  # yellow
                          #[186, 176, 172],  # gray
                         # [255, 87, 89],  # red
                          #[176, 122, 161],  # purple
                          #[118, 183, 178],  # cyan
                          #[255, 157, 167],  # pink
                          [154, 205, 50],   # yellow green
                          [85, 107, 47],    # dark olive green
                          [107, 142, 35],   # olive drab
                          [124, 252, 0],    # lawn green
                          [127, 255, 0],    # chart reuse
                          [173, 255, 47],   # green yellow
                          [0, 100, 0],      # dark green
                          [0, 128, 0],      # green
                          [34, 139, 34],    # forest green
                          [0, 255, 0],      # lime
                          [50, 205, 50],    # lime green
                          [144, 238, 144],  # light green
                          [152, 251, 152],  # pale green
                          [143, 188, 143],  # dark sea green
                          [0, 250, 154],    # medium spring green
                          [0, 255, 127],    # spring green
                          [46, 139, 87],    # sea green
                          [0,255,255],    # cyan
                        [224,255,255],    # light cyan
                        [0,206,209],    # dark turquoise
                        [64,224,208],    # turquoise
                        [72,209,204],    # medium turquoise
                        [175,238,238],    # pale turquoise
                        [127,255,212],    # aqua marine
                        [176,224,230],    # powder blue
                        [95,158,160],    # cadet blue
                        [70,130,180],    # steel blue
                        [100,149,237],    # corn flower blue
                        [0,191,255],    # deep sky blue
                        [30,144,255],    # dodger blue
                        [173,216,230],    # light blue
                        [135,206,235],    # sky blue
                        [135,206,250],    # light sky blue
                        [25,25,112],    # midnight blue
                        [0,0,128],    # navy
                        [0,0,139],    # dark blue
                        [0,0,205],    # medium blue
                        [0,0,255],    # blue
                        [65,105,225],    # royal blue
                        [138,43,226],    # blue violet
                        [75,0,130],    # indigo
                        [72,61,139],    # dark slate blue
                          ], dtype=np.uint8)

color_space_red = np.asarray([[128, 0, 0],      # maroon
                              [139, 0, 0],      # dark red
                              [165, 42, 42],    # brown
                              [178, 34, 34], 	# firebrick
                              [220, 20, 60],    # crimson
                              [255, 0, 0],      # red
                              [255, 99, 71],    # tomato
                              [255, 127, 80],   # coral
                              [205, 92, 92],    # indian red
                              [240, 128, 128],  # light coral
                              [233, 150, 122],  # dark salmon
                              [250, 128, 114],  # salmon
                              [255, 160, 122],  # light salmon
                              [255, 69, 0],     # orange red
                              [255, 140, 0],    # dark orange
                              [255, 165, 0],    # orange
                              [160, 82, 45],    # sienna
                              [219, 112, 147]   # pale viaolet red
                              ], dtype=np.uint8)


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class RewardGenerator(object):

    def __init__(self, confidence_threshold=0.75, mask_threshold=0.75, device='cuda'):

        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold

        # self.set_element(pred_tensor, gt, confidence_threshold, mask_threshold, device)
        self.obj_ids = None
        self.num_objs = None
        self.height = 1024
        self.width = 1024
        self.gts = None
        self.scores = None
        self.masks = None
        self.boxes = None
        self.num_pred = None
        self.valid_masks = []
        self.valid_boxes = []
        self.num_valid_pred = 0

    def set_element(self, pred_tensor, gt):
        # list of objects - All data had been cleaned and obj_ids are in ordered
        obj_ids = np.unique(gt)
        # Remove background (i.e. 0)
        self.obj_ids = obj_ids[1:]

        # combine the masks
        self.num_objs = len(self.obj_ids)
        self.height = gt.shape[0]
        self.width = gt.shape[1]
        self.gts = np.zeros((self.num_objs, self.height, self.width), dtype=np.uint8)
        for i in range(self.num_objs):
            self.gts[i][np.where(gt == i + 1)] = 1

        self.scores = pred_tensor[0]['scores']
        self.masks = pred_tensor[0]['masks']
        self.boxes = pred_tensor[0]['boxes']
        self.num_pred = self.masks.shape[0]
        self.valid_masks = []
        self.valid_boxes = []
        self.num_valid_pred = 0

        for pred_no in range(0, self.num_pred):
            if self.scores[pred_no] > self.confidence_threshold:
                self.valid_masks.append(self.masks[pred_no])
                self.valid_boxes.append(self.boxes[pred_no])
                self.num_valid_pred += 1

    def f1_reward(self):
        pass

    def object_wise(self):
        compare = 0
        return compare

    @staticmethod
    def _jaccard(gt, pred):
        intersection = gt * pred
        union = (gt + pred) / 2
        area_i = np.count_nonzero(intersection)
        area_u = np.count_nonzero(union)
        if area_u > 0:
            iou = area_i / area_u
        else:
            return [0, 0, 0]
        return [iou, area_i, area_u]

    def get_reward(self):
        res = self._detect_mask_id()
        # self.print_masks()
        # if self.num_valid_pred - self.num_objs > 0:
        #     extra_pun = (self.num_valid_pred - self.num_objs) * EXTRA_DETECTION_PUNISHMENT
        # else:
        #     extra_pun = 0
        if self.num_objs > 0:  # and len(res) <= self.num_objs:
            sum_iou = np.sum(res, axis=0)
            sum_iou = sum_iou[1]
            reward_iou = sum_iou / self.num_objs

            true_detections = np.count_nonzero(res, axis=0)
            true_detections = true_detections[1]
            num_non_detected = self.num_objs - true_detections
            reward = reward_iou + num_non_detected * NON_DETECTION_PUNISHMENT #+ extra_pun

            print(BColors.HEADER + '--------------------------- MASK-RG RETURNS ----------------------------' + BColors.ENDC)
            print('Number of objects in GT --> ', self.num_objs)
            print('Number of possible objects (confidence threshold {:.2f}) --> {:d}'.format(self.confidence_threshold,
                                                                                           self.num_valid_pred))
            print('Number of correct detections (masking threshold {:.2f}) --> {:d}'.format(self.mask_threshold,
                                                                                              true_detections))
            print('NEGATIVE SCORE: --> {:.2f} (cnst {:.2f} x non-detected '.format(num_non_detected * NON_DETECTION_PUNISHMENT,
                                                                        NON_DETECTION_PUNISHMENT) + BColors.FAIL +
                                                                        ' {:d}'.format(num_non_detected) +  BColors.ENDC + ')')

            print('POSITIVE SCORE: --> {:.6f} (Matching IoU)'.format(reward_iou))
            # print('Non detection ({:.3f}) + extra pun ({:.3f}) = '
            #       '{:.3f}'.format(num_non_detected * NON_DETECTION_PUNISHMENT,
            #                       extra_pun, num_non_detected * NON_DETECTION_PUNISHMENT + extra_pun))
            print(BColors.HEADER + 'TOTAL RETURN  --> {:.6f}'.format(reward))
            print('------------------------------------------------------------------------' + BColors.ENDC)
            # count 255-254 and diff
        else:
            # predictions are more than actual objects in the scene
            print("CHECK!")
            return -1
            pass
        # return res, reward, [self.num_objs, num_non_detected, num_non_detected / self.num_objs]
        return res, reward, [self.num_objs, num_non_detected, self.num_valid_pred]

    def _detect_mask_id(self):

        results = []
        pred_order = np.zeros((self.num_objs, 2), dtype=np.float32)

        for idx_gt, gt in enumerate(self.gts):
            for inx_pred, mask in enumerate(self.valid_masks):
                pred_arr = np.asarray(mask.cpu().detach()).reshape((self.height, self.width))
                pred_arr = np.where(pred_arr > self.mask_threshold, 1, 0)
                res = self._jaccard(gt, pred_arr)
                if res[0] > 0:
                    results.append([inx_pred, res[0]])

            if not results == []:
                res_arr = np.asarray(results)
                max_index = np.argmax(res_arr, axis=0)
                max_index = max_index[1]
                if res_arr[max_index][1] > self.mask_threshold:
                    pred_order[idx_gt] = res_arr[max_index]
                else:
                    pred_order[idx_gt] = [255, 0]
            else:
                pred_order[idx_gt] = [254, 0]
            results = []
        return pred_order

    def print_masks(self):
        # masks = input_tensor[0]['masks']
        # scores = input_tensor[0]['scores']
        # num_pred = masks.shape[0]
        num_masks = 0
        all = np.zeros((self.height, self.width), dtype=np.uint8)
        for mask in range(self.num_pred):
            if self.scores[mask] > self.confidence_threshold:
                # TODO if cuda, add a control here
                mask_arr = np.asarray(self.masks[mask].cpu().detach()).reshape((self.height, self.width))
                mask_arr = np.where(mask_arr > self.mask_threshold, 1, 0)
                all[np.where(mask_arr > 0)] = num_masks
                num_masks += 1
        # plt.imshow(all)
        # plt.show()
        # print('num masks that have a score higher than %.02f --> ' % self.confidence_threshold, num_masks)
        return all

    def print_seg_diff(self, order):
        img_err_masks = np.zeros([self.height, self.width, 3], dtype=np.uint8)
        for idx, curr_gt in enumerate(self.gts):
            if not (int(order[idx][0]) == 254 or int(order[idx][0]) == 255):
                curr_pred = np.asarray(self.masks[int(order[idx][0])].cpu().detach()).reshape((self.height, self.width))
                curr_pred = np.where(curr_pred > self.mask_threshold, 1, 0)

                compare = curr_pred != curr_gt
                img_err_masks[compare] = color_space[np.abs(41-idx)]
                # plt.imshow(img_err_masks)
                # plt.imshow(curr_gt)
                # plt.imshow(curr_pred)
            else:
                img_err_masks[np.asarray(curr_gt, dtype=np.bool)] = color_space_red[np.abs(17-idx)]
                # plt.imshow(curr_gt)
                # plt.imshow(img_err_masks)
        return img_err_masks
