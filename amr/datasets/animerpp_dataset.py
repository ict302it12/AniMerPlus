import torch
from torch.utils.data import ConcatDataset
from .animal3d_dataset import *
from .cub17_dataset import *
from yacs.config import CfgNode
from ..utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AniMerPlusPlusDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode):
        datasets = []
        weights = []

        dataset_configs = cfg.DATASETS
        if dataset_configs.ANIMAL3D.WEIGHT > 0.:
            self.animal3d_dataset = Train3DDataset(cfg, is_train=True, 
                                                   root_image=dataset_configs.ANIMAL3D.ROOT_IMAGE, 
                                                   json_file=dataset_configs.ANIMAL3D.JSON_FILE.TRAIN)
            datasets.append(self.animal3d_dataset)
            weights.extend([dataset_configs.ANIMAL3D.WEIGHT] * len(self.animal3d_dataset))
            log.info("Animal3D Dataset loading finish, weight: {}".format(dataset_configs.ANIMAL3D.WEIGHT))
        
        if dataset_configs.CUB.WEIGHT > 0:
            cub_dataset = CUBDataset(cfg, is_train=True)
            datasets.append(cub_dataset)
            weights.extend([dataset_configs.CUB.WEIGHT] * len(cub_dataset))
            log.info("CUB Dataset loading finish, weight: {}".format(dataset_configs.CUB.WEIGHT))
        
        if dataset_configs.CTRLAVES3D.WEIGHT > 0:
            ctrlaves3d_dataset = Train3DDataset(cfg, is_train=True, 
                                                root_image=dataset_configs.CTRLAVES3D.ROOT_IMAGE, 
                                                json_file=dataset_configs.CTRLAVES3D.JSON_FILE.TRAIN)
            datasets.append(ctrlaves3d_dataset)
            weights.extend([dataset_configs.CTRLAVES3D.WEIGHT] * len(ctrlaves3d_dataset))
            log.info("CTRLAVES3D Dataset loading finish, weight: {}".format(dataset_configs.CTRLAVES3D.WEIGHT))

        # Concatenate all enabled datasets
        if datasets:
            self.dataset = ConcatDataset(datasets)
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            raise ValueError("No datasets enabled in the configuration.")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
