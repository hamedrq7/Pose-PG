import sys 
import os 
sys.path.insert(1, os.getcwd())

# Copyright (c) Open-MMLab. All rights reserved.

import os.path as osp

import warnings

import torch

from torch.nn import functional as F

# from mmcv.parallel import is_module_wrapper
# from mmcv.runner import get_dist_info

import numpy as np
import re
import copy

ENV_MMCV_HOME = 'MMCV_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'



# def load_state_dict(module, state_dict, strict=False, logger=None):
#     """Load state_dict to a module.

#     This method is modified from :meth:`torch.nn.Module.load_state_dict`.
#     Default value for ``strict`` is set to ``False`` and the message for
#     param mismatch will be shown even if strict is False.

#     Args:
#         module (Module): Module that receives the state_dict.
#         state_dict (OrderedDict): Weights.
#         strict (bool): whether to strictly enforce that the keys
#             in :attr:`state_dict` match the keys returned by this module's
#             :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
#         logger (:obj:`logging.Logger`, optional): Logger to log the error
#             message. If not specified, print function will be used.
#     """
#     unexpected_keys = []
#     all_missing_keys = []
#     err_msg = []

#     metadata = getattr(state_dict, '_metadata', None)
#     state_dict = state_dict.copy()
#     if metadata is not None:
#         state_dict._metadata = metadata

#     # use _load_from_state_dict to enable checkpoint version control
#     def load(module, prefix=''):
#         # recursively check parallel module in case that the model has a
#         # complicated structure, e.g., nn.Module(nn.Module(DDP))
#         if is_module_wrapper(module):
#             module = module.module
#         local_metadata = {} if metadata is None else metadata.get(
#             prefix[:-1], {})
#         module._load_from_state_dict(state_dict, prefix, local_metadata, True,
#                                      all_missing_keys, unexpected_keys,
#                                      err_msg)
#         for name, child in module._modules.items():
#             if child is not None:
#                 load(child, prefix + name + '.')

#     load(module)
#     load = None  # break load->load reference cycle

#     # ignore "num_batches_tracked" of BN layers
#     missing_keys = [
#         key for key in all_missing_keys if 'num_batches_tracked' not in key
#     ]

#     if unexpected_keys:
#         err_msg.append('unexpected key in source '
#                        f'state_dict: {", ".join(unexpected_keys)}\n')
#     if missing_keys:
#         err_msg.append(
#             f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

#     rank, _ = get_dist_info()
#     if len(err_msg) > 0 and rank == 0:
#         err_msg.insert(
#             0, 'The model and loaded state dict do not match exactly\n')
#         err_msg = '\n'.join(err_msg)
#         if strict:
#             raise RuntimeError(err_msg)
#         elif logger is not None:
#             logger.warning(err_msg)
#         else:
#             print(err_msg)


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    # if filename.startswith('modelzoo://'):
    #     warnings.warn('The URL scheme of "modelzoo://" is deprecated, please '
    #                   'use "torchvision://" instead')
    #     model_urls = get_torchvision_models()
    #     model_name = filename[11:]
    #     checkpoint = load_url_dist(model_urls[model_name])
    # elif filename.startswith('torchvision://'):
    #     model_urls = get_torchvision_models()
    #     model_name = filename[14:]
    #     checkpoint = load_url_dist(model_urls[model_name])
    # elif filename.startswith('open-mmlab://'):
    #     model_urls = get_external_models()
    #     model_name = filename[13:]
    #     deprecated_urls = get_deprecated_model_names()
    #     if model_name in deprecated_urls:
    #         warnings.warn(f'open-mmlab://{model_name} is deprecated in favor '
    #                       f'of open-mmlab://{deprecated_urls[model_name]}')
    #         model_name = deprecated_urls[model_name]
    #     model_url = model_urls[model_name]
    #     # check if is url
    #     if model_url.startswith(('http://', 'https://')):
    #         checkpoint = load_url_dist(model_url)
    #     else:
    #         filename = osp.join(_get_mmcv_home(), model_url)
    #         if not osp.isfile(filename):
    #             raise IOError(f'{filename} is not a checkpoint file')
    #         checkpoint = torch.load(filename, map_location=map_location)
    # elif filename.startswith('mmcls://'):
    #     model_urls = get_mmcls_models()
    #     model_name = filename[8:]
    #     checkpoint = load_url_dist(model_urls[model_name])
    #     checkpoint = _process_mmcls_checkpoint(checkpoint)
    # elif filename.startswith(('http://', 'https://')):
    #     checkpoint = load_url_dist(filename)
    # elif filename.startswith('pavi://'):
    #     model_path = filename[7:]
    #     checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    # elif filename.startswith('s3://'):
    #     checkpoint = load_fileclient_dist(
    #         filename, backend='ceph', map_location=map_location)
    # else:
    #     if not osp.isfile(filename):
    #         raise IOError(f'{filename} is not a checkpoint file')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint

def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None,
                    patch_padding='pad',
                    part_features=None
                    ):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        patch_padding (str): 'pad' or 'bilinear' or 'bicubic', used for interpolate patch embed from 14x14 to 16x16

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    # rank, _ = get_dist_info()
    from torch import distributed as dist
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if 'patch_embed.proj.weight' in state_dict:
        proj_weight = state_dict['patch_embed.proj.weight']
        orig_size = proj_weight.shape[2:]
        current_size = model.patch_embed.proj.weight.shape[2:]
        padding_size = current_size[0] - orig_size[0]
        padding_l = padding_size // 2
        padding_r = padding_size - padding_l
        if orig_size != current_size:
            if 'pad' in patch_padding:
                proj_weight = torch.nn.functional.pad(proj_weight, (padding_l, padding_r, padding_l, padding_r))
            elif 'bilinear' in patch_padding:
                proj_weight = torch.nn.functional.interpolate(proj_weight, size=current_size, mode='bilinear', align_corners=False)
            elif 'bicubic' in patch_padding:
                proj_weight = torch.nn.functional.interpolate(proj_weight, size=current_size, mode='bicubic', align_corners=False)
            state_dict['patch_embed.proj.weight'] = proj_weight

    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        H, W = model.patch_embed.patch_shape
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        if rank == 0:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed
    
    new_state_dict = copy.deepcopy(state_dict)
    if part_features is not None:
        current_keys = list(model.state_dict().keys())
        for key in current_keys:
            if "mlp.experts" in key:
                source_key = re.sub(r'experts.\d+.', 'fc2.', key)
                new_state_dict[key] = state_dict[source_key][-part_features:]
            elif 'fc2' in key:
                new_state_dict[key] = state_dict[key][:-part_features]

    # load state_dict
    model.load_state_dict(new_state_dict, strict=False)
    # load_state_dict(model, new_state_dict, strict, logger)
    return checkpoint


# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn

# from .utils import load_checkpoint
# from mmcv_custom.checkpoint import load_checkpoint

class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def init_weights(self, pretrained=None, patch_padding='pad', part_features=None):
        """Init backbone weights.

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes backbone weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger, patch_padding=patch_padding, part_features=part_features)
        elif pretrained is None:
            # use default initializer or customized initializer in subclasses
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')

    @abstractmethod
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """

########################################################################################################################################################################
# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

# from ..builder import BACKBONES
# from .base_backbone import BaseBackbone

def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos
    
    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# @BACKBONES.register_module()
class ViT(BaseBackbone):

    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()



######################################################################################################################################
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn

# from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps


class TopdownHeatmapBaseHead(nn.Module):
    """Base class for top-down heatmap heads.

    All top-down heatmap heads should subclass it.
    All subclass should overwrite:

    Methods:`get_loss`, supporting to calculate loss.
    Methods:`get_accuracy`, supporting to calculate accuracy.
    Methods:`forward`, supporting to forward model.
    Methods:`inference_model`, supporting to inference model.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_loss(self, **kwargs):
        """Gets the loss."""

    @abstractmethod
    def get_accuracy(self, **kwargs):
        """Gets the accuracy."""

    @abstractmethod
    def forward(self, **kwargs):
        """Forward function."""

    @abstractmethod
    def inference_model(self, **kwargs):
        """Inference function."""

    def decode(self, img_metas, output, **kwargs):
        return 
        # """Decode keypoints from heatmaps.

        # Args:
        #     img_metas (list(dict)): Information about data augmentation
        #         By default this includes:

        #         - "image_file: path to the image file
        #         - "center": center of the bbox
        #         - "scale": scale of the bbox
        #         - "rotation": rotation of the bbox
        #         - "bbox_score": score of bbox
        #     output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        # """
        # batch_size = len(img_metas)

        # if 'bbox_id' in img_metas[0]:
        #     bbox_ids = []
        # else:
        #     bbox_ids = None

        # c = np.zeros((batch_size, 2), dtype=np.float32)
        # s = np.zeros((batch_size, 2), dtype=np.float32)
        # image_paths = []
        # score = np.ones(batch_size)
        # for i in range(batch_size):
        #     c[i, :] = img_metas[i]['center']
        #     s[i, :] = img_metas[i]['scale']
        #     image_paths.append(img_metas[i]['image_file'])

        #     if 'bbox_score' in img_metas[i]:
        #         score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
        #     if bbox_ids is not None:
        #         bbox_ids.append(img_metas[i]['bbox_id'])

        # preds, maxvals = keypoints_from_heatmaps(
        #     output,
        #     c,
        #     s,
        #     unbiased=self.test_cfg.get('unbiased_decoding', False),
        #     post_process=self.test_cfg.get('post_process', 'default'),
        #     kernel=self.test_cfg.get('modulate_kernel', 11),
        #     valid_radius_factor=self.test_cfg.get('valid_radius_factor',
        #                                           0.0546875),
        #     use_udp=self.test_cfg.get('use_udp', False),
        #     target_type=self.test_cfg.get('target_type', 'GaussianHeatmap'))

        # all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        # all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        # all_preds[:, :, 0:2] = preds[:, :, 0:2]
        # all_preds[:, :, 2:3] = maxvals
        # all_boxes[:, 0:2] = c[:, 0:2]
        # all_boxes[:, 2:4] = s[:, 0:2]
        # all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        # all_boxes[:, 5] = score

        # result = {}

        # result['preds'] = all_preds
        # result['boxes'] = all_boxes
        # result['image_paths'] = image_paths
        # result['bbox_ids'] = bbox_ids

        # return result

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
    

######################################################################################################################################
def flip_back(output_flipped, flip_pairs, target_type='GaussianHeatmap'):
    """Flip the flipped heatmaps back to the original form.

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output_flipped (np.ndarray[N, K, H, W]): The output heatmaps obtained
            from the flipped images.
        flip_pairs (list[tuple()): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        target_type (str): GaussianHeatmap or CombinedTarget

    Returns:
        np.ndarray: heatmaps that flipped back to the original image
    """
    assert output_flipped.ndim == 4, \
        'output_flipped should be [batch_size, num_keypoints, height, width]'
    shape_ori = output_flipped.shape
    channels = 1
    if target_type.lower() == 'CombinedTarget'.lower():
        channels = 3
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels,
                                            shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally
    output_flipped_back = output_flipped_back[..., ::-1]
    return output_flipped_back


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    # if not inspect.isclass(class_type):
    #     raise TypeError(
    #         f'class_type must be a type, but got {type(class_type)}')
    # if hasattr(class_type, '_abbr_'):
    #     return class_type._abbr_
    # if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
    #     return 'in'
    # elif issubclass(class_type, _BatchNorm):
    #     return 'bn'
    if issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    # if layer_type not in CONV_LAYERS:
    #     raise KeyError(f'Unrecognized norm type {layer_type}')
    # else:
    #     conv_layer = CONV_LAYERS.get(layer_type)

    # layer = conv_layer(*args, **kwargs, **cfg_)
    layer = nn.Conv2d(*args, **kwargs, **cfg_)
    return layer

def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        (str, nn.Module): The first element is the layer name consisting of
            abbreviation and postfix, e.g., bn1, gn. The second element is the
            created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    # if layer_type not in NORM_LAYERS:
    #     raise KeyError(f'Unrecognized norm type {layer_type}')

    # norm_layer = NORM_LAYERS.get(layer_type)
    norm_layer=nn.BatchNorm2d

    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

def build_upsample_layer(cfg, *args, **kwargs):
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
                deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    # if layer_type not in UPSAMPLE_LAYERS:
    #     raise KeyError(f'Unrecognized upsample type {layer_type}')
    # else:
    #     upsample = UPSAMPLE_LAYERS.get(layer_type)
    upsample = nn.ConvTranspose2d

    if upsample is nn.Upsample:
        cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer
#########################

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
# from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
#                       constant_init, normal_init)

# # from mmpose.core.evaluation import pose_pck_accuracy
# # from mmpose.core.post_processing import flip_back
# # from mmpose.models.builder import build_loss
# from mmpose.models.utils.ops import resize
# # from ..builder import HEADS
import torch.nn.functional as F
# from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
# from losses.loss import JointsMSELoss

# @HEADS.register_module()
class TopdownHeatmapSimpleHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                #  loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None,
                 upsample=0,):
        super().__init__()

        self.in_channels = in_channels
        # self.loss = JointsMSELoss(True)
        self.upsample = upsample

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        return 
        # losses = dict()

        # assert not isinstance(self.loss, nn.Sequential)
        # assert target.dim() == 4 and target_weight.dim() == 3
        # losses['heatmap_loss'] = self.loss(output, target, target_weight)

        # return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        return 
        # accuracy = dict()

        # if self.target_type == 'GaussianHeatmap':
        #     _, avg_acc, _ = pose_pck_accuracy(
        #         output.detach().cpu().numpy(),
        #         target.detach().cpu().numpy(),
        #         target_weight.detach().cpu().numpy().squeeze(-1) > 0)
        #     accuracy['acc_pose'] = float(avg_acc)

        # return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """
        return
        # if input_transform is not None:
        #     assert input_transform in ['resize_concat', 'multiple_select']
        # self.input_transform = input_transform
        # self.in_index = in_index
        # if input_transform is not None:
        #     assert isinstance(in_channels, (list, tuple))
        #     assert isinstance(in_index, (list, tuple))
        #     assert len(in_channels) == len(in_index)
        #     if input_transform == 'resize_concat':
        #         self.in_channels = sum(in_channels)
        #     else:
        #         self.in_channels = in_channels
        # else:
        #     assert isinstance(in_channels, int)
        #     assert isinstance(in_index, int)
        #     self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        return inputs
    
        # if not isinstance(inputs, list):
        #     if not isinstance(inputs, list):
        #         if self.upsample > 0:
        #             inputs = resize(
        #                 input=F.relu(inputs),
        #                 scale_factor=self.upsample,
        #                 mode='bilinear',
        #                 align_corners=self.align_corners
        #                 )
        #     return inputs

        # if self.input_transform == 'resize_concat':
        #     inputs = [inputs[i] for i in self.in_index]
        #     upsampled_inputs = [
        #         resize(
        #             input=x,
        #             size=inputs[0].shape[2:],
        #             mode='bilinear',
        #             align_corners=self.align_corners) for x in inputs
        #     ]
        #     inputs = torch.cat(upsampled_inputs, dim=1)
        # elif self.input_transform == 'multiple_select':
        #     inputs = [inputs[i] for i in self.in_index]
        # else:
        #     inputs = inputs[self.in_index]

        # return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    # def init_weights(self):
    #     """Initialize model weights."""
    #     for _, m in self.deconv_layers.named_modules():
    #         if isinstance(m, nn.ConvTranspose2d):
    #             normal_init(m, std=0.001)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             constant_init(m, 1)
    #     for m in self.final_layer.modules():
    #         if isinstance(m, nn.Conv2d):
    #             normal_init(m, std=0.001, bias=0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             constant_init(m, 1)

class VitPose(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = ViT(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.1,
        )
        self.keypoint_head = TopdownHeatmapSimpleHead(
            in_channels=384,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4,4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=17,
            # loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        )
    
    def get_joint_prototypes(self):
        return self.keypoint_head.final_layer
    
    def forward(self, x):
        backbone_out = self.backbone(x)
        # print(backbone_out.shape)

        return self.keypoint_head(backbone_out)
    
####### Wrapper model? 
if __name__ == "__main__":
    print('asd')
    # backbone = ViT(
    #     img_size=(256, 192),
    #     patch_size=16,
    #     embed_dim=384,
    #     depth=12,
    #     num_heads=12,
    #     ratio=1,
    #     use_checkpoint=False,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     drop_path_rate=0.1,
    # )
    # print(backbone)
    # head = TopdownHeatmapSimpleHead(
    #     in_channels=384,
    #     num_deconv_layers=2,
    #     num_deconv_filters=(256, 256),
    #     num_deconv_kernels=(4,4),
    #     extra=dict(final_conv_kernel=1, ),
    #     out_channels=17,
    #     # loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    # )
    # print(head)

    model = VitPose()
    print(model)
    from torchinfo import summary
    summary(model, (1, 3, 256, 192))
    # print('my model')
    # for k, v in model.state_dict().items():``
    #     print(k, v.shape)


    model_path = 'D:/Pose/vitpose_small.pth'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    # print('saved model', )
    # for k, v in checkpoint['state_dict'].items():
    #     print(k, v.shape)

    from misc.general_utils import load_pretrained
    model = load_pretrained(model, model_path, 'cpu')
