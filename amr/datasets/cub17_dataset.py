import os
import cv2
import scipy
import copy
from glob import glob
from PIL import Image 
import numpy as np
import torch
import hydra
from yacs.config import CfgNode
from omegaconf import DictConfig
from tqdm import tqdm
import json

from .utils import get_cub17_example, get_example

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train
        self.focal_length = cfg.AVES.get("FOCAL_LENGTH", 2167)
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.augm_config = cfg.DATASETS.CONFIG

        self.init_load()

    def __len__(self):
        raise NotImplementedError("Dataset should implement __len__ method")
    
    def __getitem__(self, idx):
        raise NotImplementedError("Dataset should implement __getitem__ method")
    
    def init_load(self):
        raise NotImplementedError("Dataset should implement init_load method")
    

class CUBDataset(BaseDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)

    def init_load(self):
        self.root_image = self.cfg.DATASETS.CUB.ROOT_IMAGE
        self.annotations = scipy.io.loadmat(self.cfg.DATASETS.CUB.ANNOTATIONS.TRAIN if self.is_train else self.cfg.DATASETS.CUB.ANNOTATIONS.TEST,
                                            squeeze_me=True, struct_as_record=False)['images']
        self.keypoint_mapping = np.array([6, 10, 5, 4, 9, 0, 14,
                                          3, 1, -1, -1, 7, -1, 11, 13,
                                          -1, -1, -1])
        
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.root_image, "images", str(annotation.rel_path))
        image = np.array(Image.open(img_path).convert("RGB"))

        mask = annotation.mask * 255
        keypoint_2d = annotation.parts.T.astype(np.float32)
        # (-1 means adjust the begining of index to 0)
        visible = keypoint_2d[:, 2] > 0
        keypoint_2d[visible, :2] -= 1
        keypoint_2d = keypoint_2d[self.keypoint_mapping, :]
        keypoint_2d[self.keypoint_mapping == -1] = np.array([0, 0, 0], dtype=np.float32)
        keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        # x1, y1, x2, y2
        bbox = np.array([annotation.bbox.x1, annotation.bbox.y1, annotation.bbox.x2, annotation.bbox.y2], dtype=np.float32) - 1 
        center = np.array([(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2])
        ori_keypoint_2d = keypoint_2d.copy()
        center_x, center_y = center[0], center[1]
        bbox_size = max([bbox[2]-bbox[0], bbox[3]-bbox[1]])

        aves_params = {'global_orient': np.zeros((3), dtype=np.float32),
                       'pose': np.zeros((34*3), dtype=np.float32),  # align with SMAL
                       'betas': np.zeros((41), dtype=np.float32),  # align with SMAL
                       'transl': np.zeros((3), dtype=np.float32),
                       'bone': np.ones((24), dtype=np.float32),
                       }
        has_aves_params = {'global_orient': np.array(0, dtype=np.float32),
                           'pose': np.array(0, dtype=np.float32),
                           'betas': np.array(0, dtype=np.float32),
                           'transl': np.array(0, dtype=np.float32),
                           'bone': np.array(0, dtype=np.float32),
                           }
        aves_params_is_axis_angle = {'global_orient': True,
                                     'pose': True,
                                     'betas': False,
                                     'transl': False,
                                     'bone': False,
                                     }
        augm_config = copy.deepcopy(self.augm_config)
        img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, aves_params, has_aves_params, img_size, trans, img_border_mask = get_example(
            img_rgba,
            center_x, center_y,
            bbox_size, bbox_size,
            keypoint_2d, keypoint_3d,
            aves_params, has_aves_params,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=False, return_trans=True,
            use_skimage_antialias=False,
            border_mode=cv2.BORDER_CONSTANT,
        )
        img_patch = (img_patch_rgba[:3, :, :])
        mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)
        
        keypoints_2d = np.concatenate([keypoints_2d, 
                                       np.zeros((26-keypoints_2d.shape[0], keypoints_2d.shape[1]), dtype=np.float32)], 
                                       axis=0)
        keypoints_3d = np.zeros((keypoints_2d.shape[0], 4), dtype=np.float32)
        ori_keypoint_2d = np.concatenate([ori_keypoint_2d, 
                                          np.zeros((26-ori_keypoint_2d.shape[0], ori_keypoint_2d.shape[1]), dtype=np.float32)], 
                                          axis=0)
        item = {'img': img_patch,
                'mask': mask_patch,
                'keypoints_2d': keypoints_2d,
                'keypoints_3d': keypoints_3d,
                'orig_keypoints_2d': ori_keypoint_2d,
                'box_center': np.array(center.copy(), dtype=np.float32),
                'box_size': float(bbox_size),
                'img_size': np.array(1.0 * img_size[::-1].copy(), dtype=np.float32),
                'smal_params': aves_params,
                'has_smal_params': has_aves_params,
                'smal_params_is_axis_angle': aves_params_is_axis_angle,
                '_trans': trans,
                'focal_length': np.array([self.focal_length, self.focal_length], dtype=np.float32),
                'category': np.array(int(annotation.rel_path.split('/')[0].split('.')[0]) + 6, dtype=np.int32),
                'supercategory': np.array(6, dtype=np.int32),
                "img_border_mask": img_border_mask.astype(np.float32),
                "has_mask": np.array(1, dtype=np.float32)}
        return item

    def __len__(self):
        return len(self.annotations)
        