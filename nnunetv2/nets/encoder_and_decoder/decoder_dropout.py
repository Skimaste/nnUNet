import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd, Dropout2d, Dropout3d

from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BottleneckD, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from nnunetv2.nets.encoder_and_decoder.simple_conv_blocks import StackedConvBlocksDropoutFirst
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
#from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from nnunetv2.nets.encoder_and_decoder.residual_encoder_dropout import ResidualEncoderDropout
#from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder


class UNetDecoderDropout(nn.Module):
    def __init__(self,
                 encoder: ResidualEncoderDropout,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        skip_dropout_layers = encoder.skip_dropout_layers if dropout_op is not None else 0


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            if skip_dropout_layers is not None and s < n_stages_encoder - skip_dropout_layers:
                
                stage = StackedConvBlocksDropoutFirst(
                    n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first
                )
            else:
                stage = StackedConvBlocks(
                    n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    None,
                    None,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first
                )
            stages.append(stage)

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


if __name__ == '__main__':
    data = torch.rand((1, 3, 128, 160))  # [batch_size, channels, height, width]

    # Create encoder
    encoder = ResidualEncoderDropout(
        input_channels=3,
        n_stages=5,
        features_per_stage=(2, 4, 6, 8, 10),
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=((1, 1), 2, (2, 2), (2, 2), (2, 2)),
        n_blocks_per_stage=2,
        conv_bias=False,
        norm_op=nn.BatchNorm2d,
        dropout_op=Dropout2d,
        dropout_op_kwargs={'p': 0.2},
        nonlin=nn.ReLU,
        nonlin_kwargs=None,
        block=BasicBlockD,
        bottleneck_channels=None,
        return_skips=True,
        disable_default_stem=False,
        stem_channels=7,
        pool_type='conv',
        stochastic_depth_p=0.0,
        squeeze_excitation=False,
        squeeze_excitation_reduction_ratio=1. / 16,
        skip_dropout_layers=2
    )

    # Create decoder
    decoder = UNetDecoderDropout(
        encoder=encoder,
        num_classes=3,  # or however many classes you want
        n_conv_per_stage=2,
        deep_supervision=False,
        nonlin_first=False,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=Dropout2d,
        dropout_op_kwargs={'p': 0.2},
        nonlin=nn.ReLU,
        nonlin_kwargs=None,
        conv_bias=False
    )

    # Forward pass (debugging)

    output = decoder(encoder(data))

    print(f"Output shape: {output.shape}")
