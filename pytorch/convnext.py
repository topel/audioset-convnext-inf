import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from pytorch_utils import do_mixup, interpolate, pad_framewise_output

from timm_weight_init import trunc_normal_

# try:
#     from DCLS.construct.modules.Dcls import  Dcls2d as cDcls2d
# except:
#     from DCLS.construct.modules import  Dcls2d as cDcls2d

from augmentations import SpeedPerturbation, pydub_augment, roll_augment

#### ConvNext #####
# remark: DropPath is only usable if channels_in == channels_out
# I modified the convNext blocks to have the option to use
# channels_in != channels_out

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'



class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 use_pydub_augment=False,
                 use_roll_augment=False,
                 use_speed_perturb=False,
                 use_torchaudio=False,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        sample_rate=32000
        window_size=1024
        hop_size=320
        # mel_bins=64
        mel_bins=224
        fmin=50
        fmax=14000

        self.use_torchaudio = use_torchaudio
        if not use_torchaudio:
            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                     win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
                                                     freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                     n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
                                                     freeze_parameters=True)

        # Spec augmenter
        # freq_drop_width=8
        freq_drop_width=28 # 28 = 8*224//64, in order to be the same as the nb of bins dropped in Cnn14
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=freq_drop_width, freq_stripes_num=2)

        self.use_pydub_augment = use_pydub_augment
        self.use_roll_augment = use_roll_augment
        
        self.use_speed_perturb = use_speed_perturb
        if self.use_speed_perturb:
            self.speed_perturb = SpeedPerturbation(rates=(0.5, 1.5), p=0.5)
        
        self.bn0 = nn.BatchNorm2d(224)

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=(4, 4), stride=(4, 4)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head_audioset = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head_audioset.weight.data.mul_(head_init_scale)
        self.head_audioset.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        # pass
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            # print(x.size())
            x = self.downsample_layers[i](x)
            # print(x.size())
            x = self.stages[i](x)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        # print(x.size())
            
        return self.norm(x) # global average+max pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, mixup_lambda=None):

        if self.training and self.use_pydub_augment:
            x = pydub_augment(x)

        if self.training and self.use_roll_augment:
            x = roll_augment(x)
        
        if self.training and self.use_speed_perturb:
            x = self.speed_perturb(x)

        if not self.use_torchaudio:
            
            x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        # print("x.size()", x.size())
        # with torchaudio: torch.Size([128, 1, 994, 224])
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        # print(x.size())

        x = self.forward_features(x)
        # print("after forward_features", x.size())
        
        # embedding = F.dropout(x, p=0.5, training=self.training)

        x = self.head_audioset(x)
        
        logits = x
        # print(x.size())
        clipwise_output = torch.sigmoid(logits)
        # print(x.size())

        # output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        output_dict = {'clipwise_output': clipwise_output, 'clipwise_logits': logits}

        return output_dict
        
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# atto: 3545095
# femto: 5037279
# pico: 8805007
# nano: 1921982
# tiny: 28228143

model_urls = {
    "convnext_atto_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_d2-01bb0f51.pth",
    "convnext_femto_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth",
    "convnext_pico_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth",
    "convnext_nano_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth",
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

# Thomas: download models to /linkhome/rech/geniri01/uzj43um/.cache/torch/hub/checkpoints/

def convnext_nano(pretrained=False, strict=False, in_22k=False, drop_path_rate=0.1, after_stem_dim=[56], use_speed_perturb=False, use_pydub_augment=False,
                              use_roll_augment=False, **kwargs):
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model = ConvNeXt(in_chans=1, num_classes=527, depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], drop_path_rate=drop_path_rate, use_speed_perturb=use_speed_perturb,
                     use_pydub_augment=use_pydub_augment, use_roll_augment=use_roll_augment, **kwargs)
    if pretrained:
        url = model_urls['convnext_nano_22k'] if in_22k else model_urls['convnext_nano_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint, strict=strict)

    if len(after_stem_dim)<2:
        if after_stem_dim[0]==56:
            stem_audioset = nn.Conv2d(1, 80, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
        elif after_stem_dim[0]==112:
            stem_audioset = nn.Conv2d(1, 80, kernel_size=(9, 2), stride=(9,2), padding=(4, 0))
        else:
            print("ERROR: after_stem_dim can be set to 56 or 112 or [252,56]")
            return None
    else:
        if after_stem_dim==[252,56]:
            stem_audioset = nn.Conv2d(1, 80, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0))
        elif after_stem_dim==[504,28]:
            stem_audioset = nn.Conv2d(1, 80, kernel_size=(4, 8), stride=(2, 8), padding=(5, 0))
        elif after_stem_dim==[504,56]:
            stem_audioset = nn.Conv2d(1, 80, kernel_size=(4, 4), stride=(2, 4), padding=(5, 0))  
        else:
            print("ERROR: after_stem_dim can be set to 56 or 112 or [252,56]")
            return None

        # stem_audioset = nn.Conv2d(1, 80, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
        #### stem_audioset = nn.Conv2d(1, 80, kernel_size=(18, 9), stride=(18,1), padding=(9,0))
    trunc_normal_(stem_audioset.weight, std=.02)
    nn.init.constant_(stem_audioset.bias, 0)
    model.downsample_layers[0][0] = stem_audioset

    return model


def convnext_tiny(pretrained=False, strict=False, in_22k=False, drop_path_rate=0.1, after_stem_dim=[56], use_speed_perturb=False, use_pydub_augment=False,
                              use_roll_augment=False, **kwargs):
    model = ConvNeXt(in_chans=1, num_classes=527, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate, use_speed_perturb=use_speed_perturb,
                     use_pydub_augment=use_pydub_augment, use_roll_augment=use_roll_augment, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=strict)
    # stem_audioset = nn.Conv2d(1, 96, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
    if len(after_stem_dim)<2:
        if after_stem_dim[0]==56:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
        elif after_stem_dim[0]==112:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(9, 2), stride=(9,2), padding=(4, 0))
        else:
            print("ERROR: after_stem_dim can be set to 56 or 112 or [252,56]")
            return None
    else:
        if after_stem_dim==[252,56]:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0))
        elif after_stem_dim==[504,28]:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(4, 8), stride=(2, 8), padding=(5, 0))
        elif after_stem_dim==[504,56]:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(2, 4), padding=(5, 0))  
        else:
            print("ERROR: after_stem_dim can be set to 56 or 112 or [252,56]")
            return None
    trunc_normal_(stem_audioset.weight, std=.02)
    nn.init.constant_(stem_audioset.bias, 0)
    model.downsample_layers[0][0] = stem_audioset
    return model


def convnext_small(pretrained=False, strict=False, in_22k=False, drop_path_rate=0.1, after_stem_dim=[56], use_speed_perturb=False, use_pydub_augment=False,
                              use_roll_augment=False, **kwargs):
    model = ConvNeXt(in_chans=1, num_classes=527, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,**kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=strict)
        if len(after_stem_dim)<2:
          if after_stem_dim[0]==56:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
          elif after_stem_dim[0]==112:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(9, 2), stride=(9,2), padding=(4, 0))
        else:
          if after_stem_dim==[252,56]:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0))

        trunc_normal_(stem_audioset.weight, std=.02)
        nn.init.constant_(stem_audioset.bias, 0)
        model.downsample_layers[0][0] = stem_audioset
    return model


def convnext_base(pretrained=False, strict=False, in_22k=False, drop_path_rate=0.1, after_stem_dim=[56], **kwargs):
    model = ConvNeXt(in_chans=1, num_classes=527, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=drop_path_rate, **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=strict)
        if len(after_stem_dim)<2:
          if after_stem_dim[0]==56:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
          elif after_stem_dim[0]==112:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(9, 2), stride=(9,2), padding=(4, 0))
        else:
          if after_stem_dim==[252,56]:
            stem_audioset = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0))

        trunc_normal_(stem_audioset.weight, std=.02)
        nn.init.constant_(stem_audioset.bias, 0)
        model.downsample_layers[0][0] = stem_audioset
    
    return model

def convnext_atto(pretrained=False, strict=False, in_22k=False, drop_path_rate=0.1, **kwargs):
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model = ConvNeXt(in_chans=1, num_classes=527, depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], drop_path_rate=drop_path_rate, **kwargs)
    if pretrained:
        url = model_urls['convnext_atto_22k'] if in_22k else model_urls['convnext_atto_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint, strict=strict)
        stem_audioset = nn.Conv2d(1, 40, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
        # stem_audioset = nn.Conv2d(1, 96, kernel_size=(18, 9), stride=(18,1), padding=(9,0))
    trunc_normal_(stem_audioset.weight, std=.02)
    nn.init.constant_(stem_audioset.bias, 0)
    model.downsample_layers[0][0] = stem_audioset

    return model


def convnext_femto(pretrained=False, strict=False, in_22k=False, drop_path_rate=0.1, **kwargs):
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model = ConvNeXt(in_chans=1, num_classes=527, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], drop_path_rate=drop_path_rate, **kwargs)
    if pretrained:
        url = model_urls['convnext_femto_22k'] if in_22k else model_urls['convnext_femto_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint, strict=strict)
        stem_audioset = nn.Conv2d(1, 48, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
        # stem_audioset = nn.Conv2d(1, 96, kernel_size=(18, 9), stride=(18,1), padding=(9,0))
        trunc_normal_(stem_audioset.weight, std=.02)
        nn.init.constant_(stem_audioset.bias, 0)
        model.downsample_layers[0][0] = stem_audioset

    return model


def convnext_pico(pretrained=False, strict=False, in_22k=False, drop_path_rate=0.1, **kwargs):
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model = ConvNeXt(in_chans=1, num_classes=527, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], drop_path_rate=drop_path_rate, **kwargs)
    if pretrained:
        url = model_urls['convnext_pico_22k'] if in_22k else model_urls['convnext_pico_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint, strict=strict)
        stem_audioset = nn.Conv2d(1, 64, kernel_size=(18, 4), stride=(18,4), padding=(9, 0))
        # stem_audioset = nn.Conv2d(1, 96, kernel_size=(18, 9), stride=(18,1), padding=(9,0))
        trunc_normal_(stem_audioset.weight, std=.02)
        nn.init.constant_(stem_audioset.bias, 0)
        model.downsample_layers[0][0] = stem_audioset

    return model



# def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
#         model.load_state_dict(checkpoint["model"])
#     return model

# def convnext_small(pretrained=False,in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model

# def convnext_base(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model

# def convnext_large(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model

# def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
#     if pretrained:
#         assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
#         url = model_urls['convnext_xlarge_22k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model

