# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .dyamr import DyAMR
from .stamr import STAMR
from .smooth_amr import SmoothAMR

import einops
import torch
from typing import Dict, Tuple
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


class AMRVideoPredictor(DyAMR):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
    def forward_image(self, img: torch.Tensor) -> torch.Tensor:
        feat, cls = self.backbone(img[:, :, :, 32:-32])
        return feat
    
    def init_state(self, num_frames: int):
        init_cond_frames, frame_not_in_cond = self.select_cond_frame(num_frames, start_frame_idx=0)
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        return init_cond_frames, frame_not_in_cond, output_dict
    
    def forward(self, item: Dict, state: Dict):
        """
        Args:
            item (Dict): model input data
            state (Dict): conditional outputs, non-conditional outputs, 
                          frame idx, condition frame idx and non-condition frame idx
        """
        # Step0: get input data
        img = item['img']
        output_dict = state['output_dict']
        frame_idx = state['frame_idx']
        init_cond_frames, frame_not_in_cond = state['init_cond_frames'], state['frame_not_in_cond']
        
        # Step1: prepare for tracking 
        feat = self.forward_image(img)
        pos_emb = self.position_encoding(feat)
        feat_size = feat.shape[-2:]
        # (N, C, H, W) -> (N, C, HW) -> (HW, N, C)
        feat = feat.flatten(2).permute(2, 0, 1)  
        pos_emb = pos_emb.flatten(2).permute(2, 0, 1)
        
        # Step2: tracking
        current_out = self.track_step(
        input_batch=item,
        frame_idx=frame_idx, 
        is_init_cond_frame=frame_idx in init_cond_frames,
        current_vision_feats=feat,
        current_vision_pos_embeds=pos_emb,
        feat_sizes=feat_size,
        output_dict=output_dict,
        num_frames=len(frame_not_in_cond)+len(init_cond_frames),
        )
        
        # Step3: postprocess. Append the output, depending on whether it's a conditioning frame
        add_output_as_cond_frame = frame_idx in init_cond_frames
        if add_output_as_cond_frame:
            output_dict["cond_frame_outputs"][frame_idx] = current_out
        else:
            output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            
        # if len(output_dict["non_cond_frame_outputs"]) > self.cfg.TRAIN.NUM_MEMORY:
        #     del output_dict["non_cond_frame_outputs"][frame_idx - self.cfg.TRAIN.NUM_MEMORY]
        return output_dict, current_out
        

class STAMRVideoPredictor(STAMR):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
    def forward_image(self, img: torch.Tensor) -> torch.Tensor:
        feat, cls = self.backbone(img[:, :, :, 32:-32])
        return feat
    
    def forward(self, item: Dict):
        """
        Args:
            item (Dict): model input data
            state (Dict): conditional outputs, non-conditional outputs, 
                          frame idx, condition frame idx and non-condition frame idx
        """
        # Step0: get input data
        img = item['img']  # (T, C, H, W) where T is the training window size
        T = img.shape[0]
        
        # Step1: obtain the spatial features
        feat = self.forward_image(img)
        h, w = feat.shape[-2:]
        
        # Step2: obtain the temporal features
        feat = einops.rearrange(feat, 't c h w -> (h w) t c', t=T)
        feat = self.st_moulde(feat)
        feat = einops.rearrange(feat, '(h w) t c -> t c h w', h=h, w=w)
        
        # Step3: decode the features
        pred_smal_params, pred_cam, _ = self.smal_head(feat)
        
        # Step4: smooth the motion
        pred_pose = torch.cat([pred_smal_params['global_orient'], pred_smal_params['pose']], dim=1)
        pred_pose = matrix_to_rotation_6d(pred_pose.view(-1, 3, 3)).view(T, -1)
        pred_pose = einops.rearrange(pred_pose, '(b t) c -> b t c', t=T)
        pred_pose = self.motion_module(pred_pose)
        pred_pose = pred_pose.view(-1, 6)
        # (Tx35)x6 -> Tx35x3x3
        pred_pose = rotation_6d_to_matrix(pred_pose).view(T, -1, 3, 3)
        pred_smal_params['global_orient'] = pred_pose[:, [0]]
        pred_smal_params['pose'] = pred_pose[:, 1:]
        
        # Step5: postprocess
        output_dict = dict()
        output_dict["pred_cam"] = pred_cam
        output_dict["pred_smal_params"] = pred_smal_params
        focal_length = item['focal_length'].reshape(-1, 2)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)], dim=-1)
        output_dict['pred_cam_t'] = pred_cam_t
        output_dict['focal_length'] = focal_length

        smal_output = self.smal(**pred_smal_params, pose2rot=False)
        output_dict['pred_vertices'] = smal_output.vertices.reshape(T, -1, 3)
        return output_dict
        
        
class SmoothVideoPredictor(SmoothAMR):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
    def forward_image(self, img: torch.Tensor) -> torch.Tensor:
        feat, cls = self.backbone(img[:, :, :, 32:-32])
        return feat
    
    def forward_preprocess(self, item: Dict) -> Dict:
        output = dict()
        # Step1: forward image to obtain the spatial features
        img = item['img']
        feat = self.forward_image(img)
        # Step2: forward the spatial features to obtain SMAL parameters
        pred_smal_params, pred_cam, _ = self.smal_head(feat)
        output['pred_cam'] = pred_cam
        output['pred_smal_params'] = {k: v.clone() for k, v in pred_smal_params.items()}
        # Step3: forward SMAL to obtain the mesh
        output['pred_vertices'] = self.forward_smal(pred_smal_params)
        return output
        
    def forward_smal(self, pred_smal_params: Dict):
        smal_output = self.smal(**pred_smal_params, pose2rot=False)
        pred_vertices = smal_output.vertices.reshape(smal_output.vertices.shape[0], -1, 3)
        return pred_vertices
    
    def forward(self, noise_pred: Dict):
        """
        Args:
            noise_pred: dict containing the noise predictions
        """
        rotation_matrix = torch.cat((noise_pred["global_orient"], noise_pred["pose"]), dim=1)
        data_len = len(rotation_matrix)
        
        slide_6d = self.sequence_to_slide_window(rotation_matrix)  # (N, T, C)
        slide_6d = slide_6d.permute(0, 2, 1)  # (N, C, T)
        slide_6d = self.smooth_net(slide_6d).permute(0, 2, 1)
        seq_6d = self.slide_window_to_sequence(slide_6d, 1, self.cfg.MODEL.SMOOTH_NET.SLIDE_WINDOW_SIZE)  # (N, 35*6)
        rotation_matrix = rotation_6d_to_matrix(seq_6d.view(data_len, -1, 6))
        pred_smal_params = {
            "global_orient": rotation_matrix[:, [0], ...],
            "pose": rotation_matrix[:, 1:, ...],
            "betas": noise_pred['betas'],
        }
        pred_vertices = self.forward_smal(pred_smal_params)
        output_dict = {
            "pred_smal_params": pred_smal_params,
            "pred_vertices": pred_vertices,
        }
        return output_dict
        
    def sequence_to_slide_window(self, pose: torch.Tensor):
        """
        Slide the window to obtain the pose
        Args:
            pose (torch.Tensor): (N, 35, 3, 3)

        Returns:
            pose (torch.Tensor): Slide window 6d pose (N, window_size, 35*6)
        """
        slide_window_size = self.cfg.MODEL.SMOOTH_NET.SLIDE_WINDOW_SIZE
        data_len = len(pose)
        device = pose.device
        
        pose = matrix_to_rotation_6d(pose).view(data_len, -1)
        if slide_window_size <= data_len:
            start_idx = torch.arange(0, data_len-slide_window_size+1)
            pose_data = []
            for idx in start_idx:
                pose_data.append(pose[idx:idx+slide_window_size, :])
            
            pose = torch.stack(pose_data, dim=0)
        else:
            pose = torch.cat((
                pose,
                torch.zeros(
                    tuple((self.slide_window_size-data_len, )) +
                    tuple(pose.shape[1:]))),
                                     axis=0)[None, :]
        return pose
    
    def slide_window_to_sequence(self, slide_window, window_step, window_size):
        output_len=(slide_window.shape[0]-1)*window_step+window_size
        sequence = [[] for i in range(output_len)]

        for i in range(slide_window.shape[0]):
            for j in range(window_size):
                sequence[i * window_step + j].append(slide_window[i, j, ...])

        for i in range(output_len):
            sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)

        sequence = torch.stack(sequence)

        return sequence
    

class SmoothVideoPredictor(torch.nn.Module):
    def __init__(self, cfg, AniMer, smooth_net):
        super().__init__(cfg)
        self.cfg = cfg
        self.AniMer = AniMer
        self.smooth_net = smooth_net
        
    def forward_image(self, img: torch.Tensor) -> torch.Tensor:
        feat, cls = self.AniMer.backbone(img[:, :, :, 32:-32])
        return feat
    
    def forward_preprocess(self, item: Dict) -> Dict:
        output = dict()
        # Step1: forward image to obtain the spatial features
        img = item['img']
        feat = self.forward_image(img)
        # Step2: forward the spatial features to obtain SMAL parameters
        pred_smal_params, pred_cam, _ = self.AniMer.smal_head(feat)
        output['pred_cam'] = pred_cam
        output['pred_smal_params'] = {k: v.clone() for k, v in pred_smal_params.items()}
        # Step3: forward SMAL to obtain the mesh
        output['pred_vertices'] = self.forward_smal(pred_smal_params)
        return output
        
    def forward_smal(self, pred_smal_params: Dict):
        smal_output = self.AniMer.smal(**pred_smal_params, pose2rot=False)
        pred_vertices = smal_output.vertices.reshape(smal_output.vertices.shape[0], -1, 3)
        return pred_vertices
    
    def forward(self, noise_pred: Dict):
        """
        Args:
            noise_pred: dict containing the noise predictions
        """
        rotation_matrix = torch.cat((noise_pred["global_orient"], noise_pred["pose"]), dim=1)
        data_len = len(rotation_matrix)
        
        slide_6d = self.sequence_to_slide_window(rotation_matrix)  # (N, T, C)
        slide_6d = slide_6d.permute(0, 2, 1)  # (N, C, T)
        slide_6d = self.smooth_net(slide_6d).permute(0, 2, 1)
        seq_6d = self.slide_window_to_sequence(slide_6d, 1, self.cfg.MODEL.SMOOTH_NET.SLIDE_WINDOW_SIZE)  # (N, 35*6)
        rotation_matrix = rotation_6d_to_matrix(seq_6d.view(data_len, -1, 6))
        pred_smal_params = {
            "global_orient": rotation_matrix[:, [0], ...],
            "pose": rotation_matrix[:, 1:, ...],
            "betas": noise_pred['betas'],
        }
        pred_vertices = self.forward_smal(pred_smal_params)
        output_dict = {
            "pred_smal_params": pred_smal_params,
            "pred_vertices": pred_vertices,
        }
        return output_dict
        
    def sequence_to_slide_window(self, pose: torch.Tensor):
        """
        Slide the window to obtain the pose
        Args:
            pose (torch.Tensor): (N, 35, 3, 3)

        Returns:
            pose (torch.Tensor): Slide window 6d pose (N, window_size, 35*6)
        """
        slide_window_size = self.cfg.MODEL.SMOOTH_NET.SLIDE_WINDOW_SIZE
        data_len = len(pose)
        device = pose.device
        
        pose = matrix_to_rotation_6d(pose).view(data_len, -1)
        if slide_window_size <= data_len:
            start_idx = torch.arange(0, data_len-slide_window_size+1)
            pose_data = []
            for idx in start_idx:
                pose_data.append(pose[idx:idx+slide_window_size, :])
            
            pose = torch.stack(pose_data, dim=0)
        else:
            pose = torch.cat((
                pose,
                torch.zeros(
                    tuple((self.slide_window_size-data_len, )) +
                    tuple(pose.shape[1:]))),
                                     axis=0)[None, :]
        return pose
    
    def slide_window_to_sequence(self, slide_window, window_step, window_size):
        output_len=(slide_window.shape[0]-1)*window_step+window_size
        sequence = [[] for i in range(output_len)]

        for i in range(slide_window.shape[0]):
            for j in range(window_size):
                sequence[i * window_step + j].append(slide_window[i, j, ...])

        for i in range(output_len):
            sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)

        sequence = torch.stack(sequence)

        return sequence  