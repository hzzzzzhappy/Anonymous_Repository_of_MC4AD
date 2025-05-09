import torch
import torch.nn as nn
import MinkowskiEngine as ME
from network.Mink import Mink_unet as unet3d
import numpy as np
import open3d as o3d

class PONet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PONet, self).__init__()
        self.backbone = unet3d(in_channels=in_channels, out_channels=out_channels, arch='MinkUNet34C')
        self.linear_offset = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Linear(16, 6, bias=True)
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self,  feat_voxel, xyz_voxel, v2p_v1):
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_v1]
        pred_offset = self.linear_offset(point_feat)

        return pred_offset

    def test_inference(self, feat_voxel, xyz_voxel, v2p_v1):
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_v1]
        pred_offset = self.linear_offset(point_feat)

        return pred_offset

def model_fn(batch, model, cfg):
    batch_size = cfg.batch_size
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    v2p_index = batch['v2p_index']
    batch_count = batch['batch_count']
    pred_offset = model(feat_voxel, xyz_voxel,  v2p_index)

    gt_offsets = batch['batch_offset'].cuda()
    ####################################################################
    joint_offset = 0.5 * (pred_offset[:,:3] + pred_offset[:,3:6])
    pt_diff = joint_offset - gt_offsets  # [N, 3] float32  :l1 distance between gt and pred
    
    delta_in = pred_offset[:, :3]  # [N,3]
    delta_ex = pred_offset[:, 3:6]  # [N,3]
    sym_error = delta_in - delta_ex  
    abs_error = torch.abs(sym_error)  # [N,3]
    
    norm_in = torch.norm(delta_in, p=2, dim=1, keepdim=True)  # [N,1]
    norm_ex = torch.norm(delta_ex, p=2, dim=1, keepdim=True)  # [N,1]
    denominator = (norm_in + norm_ex + 1e-6)  
    magnitude_loss = torch.sum(abs_error, dim=-1) / denominator.squeeze()  # [N]
    magnitude_loss = torch.mean(magnitude_loss) 
    
    unit_in = delta_in / (norm_in + 1e-8) 
    unit_ex = delta_ex / (norm_ex + 1e-8) 
    direction_diff = - (unit_in * unit_ex).sum(-1)  
    direction_loss = torch.mean(direction_diff)
    
    sym_loss = 0.7 * magnitude_loss + 0.3 * direction_loss
    ####################################################################
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # [N]    float32  :sum l1
    valid = torch.ones(pt_dist.shape[0]).cuda()  # # get valid num
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)  # # avg

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # [N]    float32  :norm
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)  # [N, 3] float32  :unit vector
    pt_offsets_norm = torch.norm(joint_offset, p=2, dim=1)  # [N]    float32  :norm
    pt_offsets = joint_offset / (pt_offsets_norm.unsqueeze(-1) + 1e-8)  # [N, 3] float32  :unit vector
    direction_diff = - (gt_offsets_ * pt_offsets).sum(-1)  # [N]    float32  :direction diff (cos)
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)  # # avg
    loss = offset_norm_loss + offset_dir_loss + 0.1 * sym_loss

    with torch.no_grad():
        pred = {}

        visual_dict = {}
        visual_dict['loss'] = loss.item()

        meter_dict = {}
        meter_dict['loss'] = (loss.item(), pred_offset.shape[0])

    return loss, pred, visual_dict, meter_dict

def eval_fn(batch, model):
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    v2p_index = batch['v2p_index']

    with torch.no_grad():
        pred_offset = model.test_inference(feat_voxel, xyz_voxel, v2p_index)
        joint_offset = 0.5 * (pred_offset[:,:3] + pred_offset[:,3:6])
    sample_score = torch.mean(torch.sum(torch.abs(joint_offset.detach().cpu()), dim=-1))

    return sample_score, joint_offset