import torch.nn as nn
import torch
import einops
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder


def build_aves_head(cfg):
    aves_head_type = cfg.MODEL.AVES_HEAD.get('TYPE', 'transformer_decoder')
    if aves_head_type == 'transformer_decoder':
        return AVESTransformerDecoderHead(cfg)
    elif aves_head_type == 'mlp_decoder':
        return ThetaRegressor(cfg)
    else:
        raise ValueError('Unknown AVES head type: {}'.format(aves_head_type))


class LinearModel(nn.Module):
    '''
    Args:
        fc_layers: a list of neuron count, such as [2133, 1024, 1024, 85]
        use_dropout: a list of bool define use dropout or not for each layer, such as [True, True, False]
        drop_prob: a list of float defined the drop prob, such as [0.5, 0.5, 0]
        use_ac_func: a list of bool define use active function or not, such as [True, True, False]
    '''
    def __init__(self, fc_layers, dropout, use_ac_func):
        super(LinearModel, self).__init__()
        self.fc_layers = fc_layers
        self.drop_prob = dropout
        self.use_ac_func = use_ac_func
        self.create_layers()

    def create_layers(self):
        l_fc_layer = len(self.fc_layers)
        l_drop_porb = len(self.drop_prob)
        l_use_ac_func = len(self.use_ac_func)

        self.fc_blocks = nn.Sequential()
        
        for _ in range(l_fc_layer - 1):
            self.fc_blocks.add_module(
                name = 'regressor_fc_{}'.format(_),
                module = nn.Linear(in_features = self.fc_layers[_], out_features = self.fc_layers[_ + 1])
            )
            
            if _ < l_use_ac_func and self.use_ac_func[_]:
                self.fc_blocks.add_module(
                    name = 'regressor_af_{}'.format(_),
                    module = nn.ReLU()
                )
            
            if _ < l_drop_porb and self.drop_prob[_]:
                self.fc_blocks.add_module(
                    name = 'regressor_fc_dropout_{}'.format(_),
                    module = nn.Dropout(p=self.drop_prob[_])
                )

    def forward(self, inputs):
        msg = 'the base class [LinearModel] is not callable!'
        raise NotImplementedError(msg)


class ThetaRegressor(LinearModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.AVES_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.AVES.NUM_JOINTS + 1)
        self.npose = npose

        mlp_args = cfg.MODEL.AVES_HEAD.MLP_DECODER
        super(ThetaRegressor, self).__init__(**mlp_args)

        init_cam = torch.tensor([[0.15, 0, 0]], dtype=torch.float32)
        init_pose = torch.zeros(size=(1, npose), dtype=torch.float32)
        init_bone = torch.ones(size=(1, 24), dtype=torch.float32)
        init_betas = torch.zeros(size=(1, 15), dtype=torch.float32)
        init_theta = torch.cat([init_cam, init_pose, init_bone, init_betas], dim=1)
        self.register_buffer('init_theta', init_theta)

    def forward(self, inputs):
        """
        Args:
            inputs: N x dim_feature
        """
        thetas = []
        batch_size = inputs.shape[0]
        theta = self.init_theta.expand(batch_size, -1)
        for _ in range(self.cfg.MODEL.AVES_HEAD.get('IEF_ITERS', 1)):
            total_inputs = torch.cat([inputs, theta], 1)
            theta = theta + self.fc_blocks(total_inputs)
            thetas.append(theta)

        pred_cam, pred_pose, pred_bone, pred_betas = torch.split(theta, [3, self.npose, 24, 15], dim=1)
        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]
        pred_pose = joint_conversion_fn(pred_pose).view(batch_size, self.cfg.AVES.NUM_JOINTS + 1, 3, 3)
        pred_aves_params = {'global_orient': pred_pose[:, [0]],
                            'pose': pred_pose[:, 1:],
                            'betas': pred_betas,
                            'bone': pred_bone,
                            }
        thetas = [torch.split(theta, [3, self.npose, 24, 15], dim=1) for theta in thetas]
        pred_aves_params_list = dict()
        pred_aves_params_list['pose'] = torch.cat(
            [joint_conversion_fn(pbp[1]).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in thetas], dim=0)
        pred_aves_params_list['bone'] = torch.cat([pbp[2] for pbp in thetas], dim=0)
        pred_aves_params_list['betas'] = torch.cat([pbp[3] for pbp in thetas], dim=0)
        return pred_aves_params, pred_cam, pred_aves_params_list
    

class AVESTransformerDecoderHead(nn.Module):
    """ Cross-attention based AVES Transformer decoder
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.AVES_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.AVES.NUM_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.AVES_HEAD.get('TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = {**transformer_args, **dict(cfg.MODEL.AVES_HEAD.TRANSFORMER_DECODER)}
        
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim = transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decbone = nn.Linear(dim, 24)
        self.decshape = nn.Linear(dim, 15)
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.AVES_HEAD.get("USE_RES", False):
            data_prior = torch.load(cfg.AVES.POSE_PRIOR_PATH, weights_only=True)
            init_pose = matrix_to_rotation_6d(
                axis_angle_to_matrix(data_prior['pose_mean'].unsqueeze(0).reshape(1, -1, 3))
                ).view(1, -1) if self.joint_rep_type == '6d' else data_prior['pose_mean'].unsqueeze(0)
            init_pose = torch.cat((torch.zeros((1, self.joint_rep_dim), dtype=torch.float32), init_pose), dim=1)
            init_bone = data_prior['bone_mean'].unsqueeze(0)
        else:
            init_pose = torch.zeros(size=(1, npose), dtype=torch.float32)
            init_bone = torch.ones(size=(1, 24), dtype=torch.float32)
        init_betas = torch.zeros(size=(1, 15), dtype=torch.float32)
        init_cam = torch.tensor([[0.15, 0, 0]], dtype=torch.float32)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_bone', init_bone)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_pose = self.init_pose.expand(batch_size, -1)
        init_bone = self.init_bone.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_bone = init_bone
        pred_betas = init_betas
        pred_cam = init_cam
        pred_pose_list = []
        pred_bone_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.cfg.MODEL.AVES_HEAD.get('IEF_ITERS', 3)):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_pose, pred_bone, pred_betas, pred_cam], dim=1)[:, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)  # (B, C)

            # Readout from token_out
            pred_pose = self.decpose(token_out) + pred_pose
            pred_bone = self.decbone(token_out) + pred_bone
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam
            pred_pose_list.append(pred_pose)
            pred_bone_list.append(pred_bone)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_aves_params_list = {}
        pred_aves_params_list['pose'] = torch.cat(
            [joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_pose_list], dim=0)
        pred_aves_params_list['bone'] = torch.cat(pred_bone_list, dim=0)
        pred_aves_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_aves_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_pose = joint_conversion_fn(pred_pose).view(batch_size, self.cfg.AVES.NUM_JOINTS + 1, 3, 3)

        pred_aves_params = {'global_orient': pred_pose[:, [0]],
                            'pose': pred_pose[:, 1:],
                            'bone': pred_bone,
                            'betas': pred_betas,
                            }
        return pred_aves_params, pred_cam, pred_aves_params_list
