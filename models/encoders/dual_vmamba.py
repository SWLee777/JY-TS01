import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import logging
from models.encoders.vmamba import Backbone_VSSM

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    from mamba_ssm.modules.mamba_simple import Mamba

    HAS_MAMBA = True
except Exception as e:
    HAS_MAMBA = False
    Mamba = None
    MAMBA_IMPORT_ERROR = e
    logger.warning(
        f"mamba_ssm import failed: {e}; high-frequency Mamba construction will fail."
    )


class AvgFreqDecompose(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        low = self.pool(x_bchw)
        high = x_bchw - low
        low = low.permute(0, 2, 3, 1).contiguous()
        high = high.permute(0, 2, 3, 1).contiguous()
        return low, high


class CrossAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, token_drop_rate=0.2
    ):
        super(CrossAttention, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.token_drop_rate = token_drop_rate

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)

        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape

        q1 = (
            self.q1(x1)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        q2 = (
            self.q2(x2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        q_fused = q1 + q2

        k1, v1 = (
            self.kv1(x1)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        k2, v2 = (
            self.kv2(x2)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )

        k_cat = torch.cat([k1, k2], dim=-2)
        v_cat = torch.cat([v1, v2], dim=-2)

        if self.training and self.token_drop_rate > 0:
            L_cat = k_cat.shape[2]
            keep_L = int(L_cat * (1 - self.token_drop_rate))

            noise = torch.rand(B, L_cat, device=k_cat.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :keep_L]

            ids_keep_expanded = (
                ids_keep.unsqueeze(1)
                .unsqueeze(-1)
                .expand(-1, self.num_heads, -1, C // self.num_heads)
            )
            k_cat = torch.gather(k_cat, dim=2, index=ids_keep_expanded)
            v_cat = torch.gather(v_cat, dim=2, index=ids_keep_expanded)

        ctx = (k_cat.transpose(-2, -1) @ v_cat) * self.scale
        ctx = ctx.softmax(dim=-2)

        x_out = (q_fused @ ctx).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x_out, x_out


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

        self.cross_attn = CrossAttention(
            dim // reduction, num_heads=num_heads, token_drop_rate=0.2
        )
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)

        v1, v2 = self.cross_attn(u1, u2)

        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class LowFreqTransformerFusion(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=4,
        mlp_ratio=4.0,
        num_layers=1,
        pool_size=16,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size

        self.cross_path = CrossPath(dim=hidden_dim, reduction=1, num_heads=num_heads)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )

    def forward(self, rgb_low, x_low):

        B, H, W, C = rgb_low.shape

        rgb_bchw = rgb_low.permute(0, 3, 1, 2).contiguous()
        x_bchw = x_low.permute(0, 3, 1, 2).contiguous()

        ph = min(self.pool_size, H)
        pw = min(self.pool_size, W)

        rgb_small = (
            F.adaptive_avg_pool2d(rgb_bchw, (ph, pw)).permute(0, 2, 3, 1).contiguous()
        )
        x_small = (
            F.adaptive_avg_pool2d(x_bchw, (ph, pw)).permute(0, 2, 3, 1).contiguous()
        )

        tokens_rgb = rgb_small.view(B, ph * pw, C)
        tokens_x = x_small.view(B, ph * pw, C)

        out_rgb, out_x = self.cross_path(tokens_rgb, tokens_x)

        fused_tokens = torch.cat([out_rgb, out_x], dim=-1)
        fused_tokens = self.out_proj(fused_tokens)

        feat_small = fused_tokens.view(B, ph, pw, C).permute(0, 3, 1, 2).contiguous()
        feat_up = F.interpolate(
            feat_small, size=(H, W), mode="bilinear", align_corners=False
        )
        feat_up = feat_up.permute(0, 2, 3, 1).contiguous()

        return feat_up


class HighFreqSequenceBlock(nn.Module):

    def __init__(self, hidden_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        if not HAS_MAMBA:
            raise ImportError(
                "High-frequency fusion requires mamba_ssm. Install a build "
                "matching the active PyTorch and CUDA versions."
            ) from MAMBA_IMPORT_ERROR
        self.mixer = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x):

        residual = x
        x_norm = self.norm1(x)

        x_mix = self.mixer(x_norm)

        x = residual + x_mix
        x = x + self.ffn(self.norm2(x))
        return x


class HighFreqMambaFusion(nn.Module):

    def __init__(self, hidden_dim, num_layers=1):
        super().__init__()

        self.in_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.mamba_h = HighFreqSequenceBlock(hidden_dim=hidden_dim)
        self.mamba_v = HighFreqSequenceBlock(hidden_dim=hidden_dim)
        self.mamba_d = HighFreqSequenceBlock(hidden_dim=hidden_dim)

        self.out_norm = nn.LayerNorm(hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, rgb_high, x_high):

        feat = torch.cat([rgb_high, x_high], dim=-1)
        feat = self.in_proj(feat)

        B, H, W, C = feat.shape

        x = feat.permute(0, 3, 1, 2).contiguous()

        x_h = x.flatten(2).transpose(1, 2).contiguous()
        out_h = self.mamba_h(x_h)

        x_v = x.transpose(2, 3).flatten(2).transpose(1, 2).contiguous()
        out_v = self.mamba_v(x_v)

        out_v = out_v.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)
        out_v = out_v.flatten(2).transpose(1, 2).contiguous()

        x_d = torch.flip(x, dims=[2, 3]).flatten(2).transpose(1, 2).contiguous()
        out_d = self.mamba_d(x_d)

        out_d = out_d.transpose(1, 2).reshape(B, C, H, W)
        out_d = torch.flip(out_d, dims=[2, 3]).flatten(2).transpose(1, 2).contiguous()

        out_fused = torch.cat([out_h, out_v, out_d], dim=-1)

        tokens = self.out_proj(self.out_norm(out_fused))

        feat_out = tokens.view(B, H, W, C)
        return feat_out


class MyFreqFusionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=4,
        low_pool_size=16,
        low_layers=1,
        high_layers=1,
        return_evidence=False,
        fusion_mode="full",
    ):
        super().__init__()
        self.return_evidence = bool(return_evidence)
        self.fusion_mode = str(fusion_mode)
        valid_modes = {
            "full",
            "high_only",
            "low_only",
            "simple",
            "linear_only",
        }
        if self.fusion_mode not in valid_modes:
            raise ValueError(
                f"Unsupported fusion_mode={self.fusion_mode!r}; "
                f"expected one of {sorted(valid_modes)}"
            )

        self.decompose_rgb = AvgFreqDecompose(kernel_size=3)
        self.decompose_x = AvgFreqDecompose(kernel_size=3)

        self.low_fuse = None
        if self.fusion_mode in {"full", "low_only"}:
            self.low_fuse = LowFreqTransformerFusion(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                pool_size=low_pool_size,
            )

        self.high_fuse = None
        if self.fusion_mode in {"full", "high_only"}:
            self.high_fuse = HighFreqMambaFusion(
                hidden_dim=hidden_dim,
                num_layers=high_layers,
            )

        self.merge = None
        if self.fusion_mode == "full":
            self.merge = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        self.simple_fuse = None
        if self.fusion_mode == "simple":
            self.simple_fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        elif self.fusion_mode == "linear_only":

            self.simple_fuse = nn.Linear(hidden_dim * 2, hidden_dim)

        self.res_scale = (
            None if self.fusion_mode == "linear_only" else nn.Parameter(torch.zeros(1))
        )

    def forward(self, rgb_feat, x_feat):

        rgb_low, rgb_high = self.decompose_rgb(rgb_feat)
        x_low, x_high = self.decompose_x(x_feat)

        low_fused = None
        high_fused = None
        if self.low_fuse is not None:
            low_fused = self.low_fuse(rgb_low, x_low)
        if self.high_fuse is not None:
            high_fused = self.high_fuse(rgb_high, x_high)

        if self.fusion_mode == "full":
            fused = self.merge(torch.cat([low_fused, high_fused], dim=-1))
        elif self.fusion_mode == "high_only":
            fused = high_fused
        elif self.fusion_mode == "low_only":
            fused = low_fused
        else:
            fused = self.simple_fuse(torch.cat([rgb_feat, x_feat], dim=-1))

        if self.res_scale is not None:
            fused = fused + self.res_scale * (rgb_feat + x_feat)

        if self.return_evidence:

            high_reference = (
                high_fused if high_fused is not None else 0.5 * (rgb_high + x_high)
            )
            return fused, rgb_high, x_high, high_reference
        return fused


class RGBXTransformer(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        norm_layer=nn.LayerNorm,
        depths=[2, 2, 27, 2],
        dims=96,
        pretrained=None,
        mlp_ratio=4.0,
        downsample_version="v1",
        ape=False,
        img_size=[480, 640],
        patch_size=4,
        drop_path_rate=0.2,
        return_high_evidence=False,
        fusion_mode="full",
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.ape = ape
        self.return_high_evidence = bool(return_high_evidence)
        self.fusion_mode = str(fusion_mode)

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )

        self.freq_fusion = nn.ModuleList(
            [
                MyFreqFusionBlock(
                    hidden_dim=dims * (2**i),
                    num_heads=4,
                    low_pool_size=16,
                    low_layers=1,
                    high_layers=1,
                    return_evidence=(self.return_high_evidence and i < 2),
                    fusion_mode=self.fusion_mode,
                )
                for i in range(4)
            ]
        )

        if self.ape:
            self.patches_resolution = [
                img_size[0] // patch_size,
                img_size[1] // patch_size,
            ]
            self.absolute_pos_embed = nn.ParameterList()
            self.absolute_pos_embed_x = nn.ParameterList()

            for i_layer in range(len(depths)):
                input_resolution = (
                    self.patches_resolution[0] // (2**i_layer),
                    self.patches_resolution[1] // (2**i_layer),
                )
                dim = int(dims * (2**i_layer))

                absolute_pos_embed = nn.Parameter(
                    torch.zeros(1, dim, input_resolution[0], input_resolution[1])
                )
                trunc_normal_(absolute_pos_embed, std=0.02)

                absolute_pos_embed_x = nn.Parameter(
                    torch.zeros(1, dim, input_resolution[0], input_resolution[1])
                )
                trunc_normal_(absolute_pos_embed_x, std=0.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        outs_fused = []
        high_evidence = {}

        outs_rgb = self.vssm(x_rgb)
        outs_x = self.vssm(x_e)

        for i in range(4):
            if self.ape:
                out_rgb = (
                    self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                )
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]

            fusion_output = self.freq_fusion[i](
                out_rgb.permute(0, 2, 3, 1).contiguous(),
                out_x.permute(0, 2, 3, 1).contiguous(),
            )

            if self.return_high_evidence and i < 2:
                fused_bhwc, rgb_h_bhwc, x_h_bhwc, fused_h_bhwc = fusion_output
                scale = "1_4" if i == 0 else "1_8"
                high_evidence[scale] = {
                    "rgb": rgb_h_bhwc.permute(0, 3, 1, 2).contiguous(),
                    "depth": x_h_bhwc.permute(0, 3, 1, 2).contiguous(),
                    "fused": fused_h_bhwc.permute(0, 3, 1, 2).contiguous(),
                }
            else:
                fused_bhwc = fusion_output

            outs_fused.append(fused_bhwc.permute(0, 3, 1, 2).contiguous())

        if self.return_high_evidence:
            return {
                "features": outs_fused,
                "high_evidence": high_evidence,
            }
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out


class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained="pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth",
            mlp_ratio=0.0,
            downsample_version="v1",
            drop_path_rate=0.2,
            **kwargs,
        )


class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained="pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth",
            mlp_ratio=0.0,
            downsample_version="v1",
            drop_path_rate=0.3,
            **kwargs,
        )


class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained="pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth",
            mlp_ratio=0.0,
            downsample_version="v1",
            drop_path_rate=0.6,
            **kwargs,
        )
