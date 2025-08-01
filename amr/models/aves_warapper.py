import torch
import torch.nn.functional as F
from dataclasses import dataclass
from smplx.utils import ModelOutput
from typing import Optional, NewType
from pytorch3d.transforms import axis_angle_to_matrix


Tensor = NewType('Tensor', torch.Tensor)
@dataclass
class AVESOutput(ModelOutput):
    betas: Optional[Tensor] = None
    pose: Optional[Tensor] = None
    bone: Optional[Tensor] = None


class LBS(torch.nn.Module):
    '''
    Implementation of linear blend skinning, with additional bone and scale
    Input:
        V (BN, V, 3): vertices to pose and shape
        pose (BN, J, 3, 3) or (BN, J, 3): pose in rot or axis-angle
        bone (BN, K): allow for direct change of relative joint distances
        scale (1): scale the whole kinematic tree
    '''
    def __init__(self, J, parents, weights):
        super(LBS, self).__init__()
        self.n_joints = J.shape[1]
        self.register_buffer('h_joints', F.pad(J.unsqueeze(-1), [0,0,0,1], value=0))
        self.register_buffer('kin_tree', torch.cat([J[:,[0], :], J[:, 1:]-J[:, parents[1:]]], dim=1).unsqueeze(-1))
        
        self.register_buffer('parents', parents)
        self.register_buffer('weights', weights[None].float())
        
    def __call__(self, V, pose, bone, scale, to_rotmats=False):
        batch_size = len(V)
        device = pose.device
        V = F.pad(V.unsqueeze(-1), [0,0,0,1], value=1)
        kin_tree = (scale*self.kin_tree) * bone[:, :, None, None]
        if to_rotmats:
            pose = axis_angle_to_matrix(pose.view(-1, 3))
        pose = pose.view([batch_size, -1, 3, 3])
        T = torch.zeros([batch_size, self.n_joints, 4, 4]).float().to(device)
        T[:, :, -1, -1] = 1
        T[:, :, :3, :] = torch.cat([pose, kin_tree], dim=-1)
        T_rel = [T[:, 0]]
        for i in range(1, self.n_joints):
            T_rel.append(T_rel[self.parents[i]] @ T[:, i])
        T_rel = torch.stack(T_rel, dim=1)
        T_rel[:,:,:,[-1]] -= T_rel.clone() @ (self.h_joints*scale)
        T_ = self.weights @ T_rel.view(batch_size, self.n_joints, -1)
        T_ = T_.view(batch_size, -1, 4, 4)
        V = T_ @ V
        return V[:, :, :3, 0]


class AVES(torch.nn.Module):
    def __init__(self, **kwargs):
        super(AVES, self).__init__()
        # kinematic tree, and map to keypoints from vertices
        self.register_buffer('kintree_table', kwargs['kintree_table'])
        self.register_buffer('parents', kwargs['kintree_table'][0])
        self.register_buffer('weights', kwargs['weights'])
        self.register_buffer('vert2kpt', kwargs['vert2kpt'])
        self.register_buffer('face', kwargs['F'])

        # mean shape and default joints
        rot = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float32)
        rot = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32) @ rot
        # rot = torch.eye(3, dtype=torch.float32)
        V = (rot @ kwargs['V'].T).T.unsqueeze(0)
        J = (rot @ kwargs['J'].T).T.unsqueeze(0)
        self.register_buffer('V', V)
        self.register_buffer('J', J)
        self.LBS = LBS(self.J, self.parents, self.weights)
        
        # pose and bone prior
        self.register_buffer('p_m', kwargs['pose_mean'])
        self.register_buffer('b_m', kwargs['bone_mean'])
        self.register_buffer('p_cov', kwargs['pose_cov'])
        self.register_buffer('b_cov', kwargs['bone_cov'])

        # standardized blend shape basis
        B = kwargs['Beta']
        sigma = kwargs['Beta_sigma']
        B = B * sigma[:,None,None]
        self.register_buffer('B', B)

        # PCA coefficient that is optimized to match the original template shape
        ### so in the __call__ funciton, if beta is set to self.beta_original,
        ### it will return the template shape from ECCV2020 (marcbadger/avian-mesh). 
        self.register_buffer('beta_original', kwargs['beta_original'])
        
    def __call__(self, global_orient, pose, bone, transl=None,
                 scale=1, betas=None, pose2rot=False, **kwargs):
        '''
        Input:
            global_pose [bn, 3] tensor for batched global_pose on root joint
            body_pose   [bn, 72] tensor for batched body pose
            bone_length [bn, 24] tensor for bone length; the bone variable 
                                 captures non-rigid joint articulation in this model

            beta [bn, 15] shape PCA coefficients
            If beta is None, it will return the mean shape
            If beta is self.beta_original, it will return the orignial tempalte shape

        '''
        device = global_orient.device
        batch_size = global_orient.shape[0]
        V = self.V.repeat([batch_size, 1, 1]) * scale
        J = self.J.repeat([batch_size, 1, 1]) * scale

        # multi-bird shape space
        if betas is not None:
            V = V + torch.einsum('bk, kmn->bmn', betas, self.B)

        # concatenate bone and pose
        bone = torch.cat([torch.ones([batch_size, 1]).to(device), bone], dim=1)
        pose = torch.cat([global_orient, pose], dim=1)

        # LBS          
        verts = self.LBS(V, pose, bone, scale, to_rotmats=pose2rot)
        if transl is not None:
            verts = verts + transl[:, None, :]

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = torch.einsum('bni,kn->bki', verts, self.vert2kpt)
        
        output = AVESOutput(
            vertices=verts,
            joints=keypoints,
            betas=betas,
            global_orient=global_orient,
            pose=pose,
            bone=bone,
            transl=transl,
            full_pose=None,
        )
        return output
