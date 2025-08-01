import torch
import pickle
import pytorch_lightning as pl
from torchvision.utils import make_grid
from typing import Dict
from pytorch3d.transforms import matrix_to_axis_angle
from yacs.config import CfgNode
from ..utils import MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from ..utils.mesh_renderer import SilhouetteRenderer
from .backbones import create_backbone
from .heads.classifier_head import ClassTokenHead
from .heads import build_aves_head, build_smal_head
from .losses import (Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, SupConLoss,
                    PoseBonePriorLoss, SilhouetteLoss, ShapePriorLoss, PosePriorLoss)
from .aves_warapper import AVES
from .smal_warapper import SMAL


log = get_pylogger(__name__)


class AniMerPlusPlus(pl.LightningModule):
    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup AVES-HMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            state_dict = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu', weights_only=True)
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
    
        # Create AVES head
        self.aves_head = build_aves_head(cfg)

        # Create SMAL head
        self.smal_head = build_smal_head(cfg)

        self.class_token_head = ClassTokenHead(**cfg.MODEL.get("CLASS_TOKEN_HEAD", dict()))

        # Define loss functions
        # common loss
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.supcon_loss = SupConLoss()
        self.parameter_loss = ParameterLoss()
        # aves loss
        self.posebone_prior_loss = PoseBonePriorLoss(path_prior=cfg.AVES.POSE_PRIOR_PATH)
        self.mask_loss = SilhouetteLoss()
        # smal loss
        self.shape_prior_loss = ShapePriorLoss(path_prior=cfg.SMAL.SHAPE_PRIOR_PATH)
        self.pose_prior_loss = PosePriorLoss(path_prior=cfg.SMAL.POSE_PRIOR_PATH)
        # Instantiate AVES model
        aves_model_path = cfg.AVES.MODEL_PATH
        aves_cfg = torch.load(aves_model_path, weights_only=True)
        self.aves = AVES(**aves_cfg)

        # Instantiate SMAL model
        smal_model_path = cfg.SMAL.MODEL_PATH
        with open(smal_model_path, 'rb') as f:
            smal_cfg = pickle.load(f, encoding="latin1")
        self.smal = SMAL(**smal_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.aves_mesh_renderer = MeshRenderer(self.cfg, faces=aves_cfg['F'].numpy())
            self.smal_mesh_renderer = MeshRenderer(self.cfg, faces=self.smal.faces.numpy())
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Only appling for AVES training
        self.aves_silouette_render = SilhouetteRenderer(size=self.cfg.MODEL.IMAGE_SIZE,
                                                        focal=self.cfg.AVES.get("FOCAL_LENGTH", 2167),
                                                        device='cuda')

        self.automatic_optimization = False

    def get_parameters(self):
        all_params = list(self.aves_head.parameters())
        all_params += list(self.backbone.parameters())
        all_params += list(self.smal_head.parameters())
        all_params += list(self.class_token_head.parameters())
        return all_params

    def configure_optimizers(self):
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]
        
        if "vit" in self.cfg.MODEL.BACKBONE.TYPE:
            optimizer = torch.optim.AdamW(params=param_groups,
                                          weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        else:
            optimizer = torch.optim.Adam(params=param_groups,
                                         weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        return optimizer
    
    def forward_backbone(self, batch: Dict):
        x = batch['img']
        dataset_source = batch["supercategory"] < 5  # bird for index 0
        # Compute conditioning features using the backbone
        if self.cfg.MODEL.BACKBONE.TYPE in ["vith"]:
            conditioning_feats, cls = self.backbone(x[:, :, :, 32:-32])  # [256, 192]
        elif self.cfg.MODEL.BACKBONE.TYPE in ["vithmoe"]:
            conditioning_feats, cls = self.backbone(x[:, :, :, 32:-32], dataset_source=dataset_source.type(torch.long))
        else:
            conditioning_feats = self.backbone(x)
            cls = None
        return conditioning_feats, cls

    def forward_one_parametric_model(self, 
                                     focal_length: torch.tensor, 
                                     features: torch.tensor,
                                     head: torch.nn.Module,
                                     parametric_model: torch.nn.Module,):
        """
        Run a forward step of one parametric model.
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        batch_size = features.shape[0]
        pred_params, pred_cam, _ = head(features)
        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_params'] = {k: v.clone() for k, v in pred_params.items()}

        # Compute camera translation
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)], dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_params['global_orient'] = pred_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_params['pose'] = pred_params['pose'].reshape(batch_size, -1, 3, 3)
        pred_params['betas'] = pred_params['betas'].reshape(batch_size, -1)
        pred_params['bone'] = pred_params['bone'].reshape(batch_size, -1) if 'bone' in pred_params else None
        parametric_model_output = parametric_model(**pred_params, pose2rot=False)

        pred_keypoints_3d = parametric_model_output.joints
        pred_vertices = parametric_model_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]
        device = x.device
        dataset_source = (batch["supercategory"] < 5)  # bird for index 0

        features, cls = self.forward_backbone(batch)

        output = dict()
        output['cls_feats'] = self.class_token_head(cls) if self.cfg.MODEL.BACKBONE.get("USE_CLS", False) else None
        
        num_aves = (batch_size - dataset_source.sum()).item()
        if num_aves:
            output['aves_output'] = self.forward_one_parametric_model(batch['focal_length'][~dataset_source],
                                                                     features[~dataset_source], 
                                                                     self.aves_head, 
                                                                     self.aves)
            # Only specific to AVES training
            output['aves_output']['pred_mask'] = self.aves_silouette_render(output['aves_output']['pred_vertices']+output['aves_output']['pred_cam_t'].unsqueeze(1), 
                                                 faces=self.aves.face.unsqueeze(0).repeat(batch_size-dataset_source.sum().item(), 1, 1).to(device))
        
        num_smal = dataset_source.sum().item()
        if num_smal:
            output['smal_output'] = self.forward_one_parametric_model(batch['focal_length'][dataset_source], 
                                                                      features[dataset_source], 
                                                                      self.smal_head, 
                                                                      self.smal)
        return output
    
    def compute_aves_loss(self, batch: Dict, output: Dict) -> torch.Tensor:
        """
        Compute AVES losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        dataset_source = (batch["supercategory"] > 5)

        pred_params = output['pred_params']
        pred_mask = output['pred_mask']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_params['pose'].shape[0]
        
        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d'][dataset_source][:, :18]
        gt_keypoints_3d = batch['keypoints_3d'][dataset_source][:, :18]
        gt_mask = batch['mask'][dataset_source]
        gt_params = {k: v[dataset_source] for k,v in batch['smal_params'].items()}
        has_params = {k: v[dataset_source] for k,v in batch['has_smal_params'].items()}
        is_axis_angle = {k: v[dataset_source] for k,v in batch['smal_params_is_axis_angle'].items()}

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)
        loss_mask = self.mask_loss(pred_mask, gt_mask)

        # Compute loss on AVES parameters
        loss_params = {}
        for k, pred in pred_params.items():
            gt = gt_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_params[k]
            if k == "betas":
                loss_params[k] = self.parameter_loss(pred.reshape(batch_size, -1),
                                                     gt[:, :15].reshape(batch_size, -1),
                                                     has_gt)
                loss_params[k+"_re"] = torch.sum(pred[has_gt.bool()] ** 2) + torch.sum(pred[has_gt.bool()] ** 2) * 0.5
            elif k == "bone":
                loss_params[k] = self.parameter_loss(pred.reshape(batch_size, -1),
                                                     gt.reshape(batch_size, -1),
                                                     has_gt)
                loss_params[k+"_re"] = self.posebone_prior_loss.l2_loss(pred, self.posebone_prior_loss.bone_mean, 1 - has_gt) + \
                                       self.posebone_prior_loss.l2_loss(pred, self.posebone_prior_loss.bone_mean, has_gt) * 0.02
            elif k == "pose":
                loss_params[k] = self.parameter_loss(pred.reshape(batch_size, -1),
                                                     gt[:, :24].reshape(batch_size, -1),
                                                     has_gt)
                pose_axis_angle = matrix_to_axis_angle(pred)
                loss_params[k+"_re"] = self.posebone_prior_loss.l2_loss(pose_axis_angle.reshape(batch_size, -1), self.posebone_prior_loss.pose_mean, 1 - has_gt) + \
                                       self.posebone_prior_loss.l2_loss(pose_axis_angle.reshape(batch_size, -1), self.posebone_prior_loss.pose_mean, has_gt) * 0.02
            else:
                loss_params[k] = self.parameter_loss(pred.reshape(batch_size, -1), 
                                                     gt.reshape(batch_size, -1),
                                                     has_gt)
        
        loss_config = self.cfg.LOSS_WEIGHTS.AVES
        loss = loss_config['KEYPOINTS_3D'] * loss_keypoints_3d + \
               loss_config['KEYPOINTS_2D'] * loss_keypoints_2d + \
               sum([loss_params[k] * loss_config[k.upper()] for k in loss_params]) + \
               loss_config['MASK'] * loss_mask

        losses = dict(loss_aves=loss.detach(),
                      loss_aves_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_aves_keypoints_3d=loss_keypoints_3d.detach(),
                      loss_aves_mask=loss_mask.detach(),
                      )
        for k, v in loss_params.items():
            losses['loss_aves_' + k] = v.detach()
        
        return loss, losses
    
    def compute_smal_loss(self, batch: Dict, output: Dict) -> torch.Tensor:
        """
        Compute SMAL losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
        Returns:
            torch.Tensor : Total loss for current batch
        """
        dataset_source = (batch["supercategory"] < 5)

        pred_params = output['pred_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_params['pose'].shape[0]

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d'][dataset_source]
        gt_keypoints_3d = batch['keypoints_3d'][dataset_source]
        gt_params = {k: v[dataset_source] for k,v in batch['smal_params'].items()}
        has_params = {k: v[dataset_source] for k,v in batch['has_smal_params'].items()}
        is_axis_angle = {k: v[dataset_source] for k,v in batch['smal_params_is_axis_angle'].items()}

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # Compute loss on SMAL parameters
        loss_smal_params = {}
        for k, pred in pred_params.items():
            gt = gt_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_params[k]
            if k == "betas":
                loss_smal_params[k] = self.parameter_loss(pred.reshape(batch_size, -1),
                                                            gt.reshape(batch_size, -1),
                                                            has_gt) + \
                                      self.shape_prior_loss(pred, batch["category"][dataset_source], has_gt)
            elif k == "bone":
                continue
            else:
                loss_smal_params[k] = self.parameter_loss(pred.reshape(batch_size, -1),
                                                                gt.reshape(batch_size, -1),
                                                                has_gt) + \
                                        self.pose_prior_loss(torch.cat((pred_params["global_orient"],
                                                                        pred_params["pose"]),
                                                                        dim=1), has_gt) / 2.
        
        loss_config = self.cfg.LOSS_WEIGHTS.SMAL
        loss = loss_config['KEYPOINTS_3D'] * loss_keypoints_3d + \
               loss_config['KEYPOINTS_2D'] * loss_keypoints_2d + \
               sum([loss_smal_params[k] * loss_config[k.upper()] for k in loss_smal_params])

        losses = dict(loss_smal=loss.detach(),
                      loss_smal_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_smal_keypoints_3d=loss_keypoints_3d.detach(),
                      )
        for k, v in loss_smal_params.items():
            losses['loss_smal_' + k] = v.detach()

        return loss, losses

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        x = batch['img']
        device, dtype = x.device, x.dtype
        if 'aves_output' in output:
            loss_aves, losses_aves = self.compute_aves_loss(batch, output['aves_output'])
        else:
            loss_aves, losses_aves = torch.tensor(0.0, device=device, dtype=dtype), {}
        if 'smal_output' in output:
            loss_smal, losses_smal = self.compute_smal_loss(batch, output['smal_output'])
        else:
            loss_smal, losses_smal = torch.tensor(0.0, device=device, dtype=dtype), {}
        loss_supcon = self.supcon_loss(output['cls_feats'], labels=batch['category']) if self.cfg.MODEL.BACKBONE.get("USE_CLS", False) \
                      else torch.tensor(0.0, device=device, dtype=dtype)
        loss = loss_aves + loss_smal + loss_supcon * self.cfg.LOSS_WEIGHTS['SUPCON']

        # Saving loss
        losses = {}
        losses['loss'] = loss.detach()
        losses['loss_supcon'] = loss_supcon.detach()
        for k, v in losses_aves.items():
            losses[k] = v.detach()
        for k, v in losses_smal.items():
            losses[k] = v.detach()
        output['losses'] = losses
        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True,
                            write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        masks = batch['mask']
        # mul std then add mean
        images = (images) * (torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1))
        images = (images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1))
        masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)

        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode + '/' + loss_name, val.detach().item(), step_count)
            if train is False:
                for metric_name, val in output['metric'].items():
                    summary_writer.add_scalar(mode + '/' + metric_name, val, step_count)
        
        rend_imgs = []
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)
        dataset_source = (batch["supercategory"] < 5)[:num_images]  # bird for index 0

        num_aves = (num_images - dataset_source[:num_images].sum()).item()
        if num_aves:
            rend_imgs_aves = self.aves_mesh_renderer.visualize_tensorboard( output['aves_output']['pred_vertices'][:num_aves].detach().cpu().numpy(),
                                                                            output['aves_output']['pred_cam_t'][:num_aves].detach().cpu().numpy(),
                                                                            images[:num_images][~dataset_source].cpu().numpy(),
                                                                            self.cfg.AVES.get("FOCAL_LENGTH", 2167),
                                                                            output['aves_output']['pred_keypoints_2d'][:num_aves].detach().cpu().numpy(),
                                                                            gt_keypoints_2d[:num_images][~dataset_source][:, :18].cpu().numpy(),
                                                                            )
            rend_imgs.extend(rend_imgs_aves)

        num_smal = dataset_source[:num_images].sum().item()
        if num_smal:
            rend_imgs_smal = self.smal_mesh_renderer.visualize_tensorboard( output['smal_output']['pred_vertices'][:num_smal].detach().cpu().numpy(),
                                                                            output['smal_output']['pred_cam_t'][:num_smal].detach().cpu().numpy(),
                                                                            images[:num_images][dataset_source].cpu().numpy(),
                                                                            self.cfg.SMAL.get("FOCAL_LENGTH", 1000),
                                                                            output['smal_output']['pred_keypoints_2d'][:num_smal].detach().cpu().numpy(),
                                                                            gt_keypoints_2d[:num_images][dataset_source].cpu().numpy(),
                                                                            )
            rend_imgs.extend(rend_imgs_smal)

        rend_imgs = make_grid(rend_imgs, nrow=5, padding=2)
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, rend_imgs, step_count)

        return rend_imgs

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step(self, batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            batch (Dict): Dictionary containing {'img', 'mask', 'keypoints_2d', 'keypoints_3d', 'orig_keypoints_2d',
                                                 'aves_params', 'aves_params_is_axis_angle', 'focal_length'}
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = batch['img']
        optimizer = self.optimizers(use_pl_optimizer=True)

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL,
                                                error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        optimizer.step()
        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
                 batch_size=batch_size, sync_dist=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        pass

