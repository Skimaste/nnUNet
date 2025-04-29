from typing import Union, Type, List, Tuple
import numpy as np
import torch
from torchviz import make_dot
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
#from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
#from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
#from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from nnunetv2.nets.encoder_and_decoder.residual_encoder_dropout import ResidualEncoderDropout
from nnunetv2.nets.encoder_and_decoder.decoder_dropout import UNetDecoderDropout
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd, Dropout2d, Dropout3d
# from mamba_ssm import Mamba
from torch.cuda.amp import autocast


class ResidualEncoderDropoutUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 skip_dropout_layers: int = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoderDropout(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels, skip_dropout_layers=skip_dropout_layers)
        self.decoder = UNetDecoderDropout(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
def print_model_parameters(model):
    print(f"{'Module':<50} {'# Parameters':>15}")
    print("="*65)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name:<50} {param_count:>15}")
    print("="*65)
    print(f"{'Total':<50} {total_params:>15}")

if __name__ == '__main__':
    data = torch.rand((1, 3, 128, 160))  # [batch_size, channels, height, width]


    model = ResidualEncoderDropoutUNet(
        input_channels=3,
        n_stages=3,
        features_per_stage=[32, 64, 128],
        conv_op=nn.Conv2d,
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        n_blocks_per_stage=[2, 2,  2],
        num_classes=2,
        n_conv_per_stage_decoder=[1, 1],
        conv_bias=True,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5},
        nonlin=nn.ReLU,
        nonlin_kwargs={"inplace": True},
        skip_dropout_layers=0,
        deep_supervision=True,
        block=BasicBlockD,  # or BottleneckD if you want
        bottleneck_channels=[16, 32, 64],
        stem_channels=16)
    # Forward pass (debugging) 3547142, 3547142
    #model.train()
    for name, module in model.named_modules():
        if 'Dropout' in module.__class__.__name__:
            print(name, module)

    output = model(data)

    #make_dot(output[0], params=dict(model.named_parameters())).render("model_graph", format="png")
    #print_model_parameters(model)


