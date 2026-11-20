import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class EncoderDecoder(nn.Module):

    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction="mean", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        if cfg is None:
            raise ValueError("cfg is required")
        if cfg.decoder != "ResidualDecoder":
            raise ValueError(
                "This minimal release supports decoder='ResidualDecoder' only"
            )

        self.norm_layer = norm_layer
        self.criterion = criterion
        self.decoder_aux_weights = dict(
            getattr(
                cfg,
                "decoder_aux_weights",
                {"prototype_1_8": 0.1, "prototype_1_4": 0.1},
            )
        )

        self._build_backbone(cfg, norm_layer)
        self._build_decoder(cfg, norm_layer)

        if self.criterion is not None:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def _build_backbone(self, cfg, norm_layer):
        if cfg.backbone not in {"sigma_tiny", "sigma_small", "sigma_base"}:
            raise ValueError(
                "This minimal release supports sigma_tiny, sigma_small, "
                "or sigma_base only"
            )

        from .encoders import dual_vmamba

        backbone_name = cfg.backbone.replace("sigma", "vssm")
        backbone = getattr(dual_vmamba, backbone_name)
        self.channels = (
            [128, 256, 512, 1024]
            if cfg.backbone == "sigma_base"
            else [96, 192, 384, 768]
        )
        self.backbone = backbone(
            return_high_evidence=True,
            fusion_mode=getattr(cfg, "encoder_fusion_mode", "full"),
            use_checkpoint=bool(
                getattr(
                    cfg,
                    "backbone_use_checkpoint",
                    getattr(cfg, "use_checkpoint", False),
                )
            ),
        )

    def _build_decoder(self, cfg, norm_layer):
        from .decoders.ResidualDecoder import ResidualEvidenceDecoder

        self.decode_head = ResidualEvidenceDecoder(
            img_size=[cfg.image_height, cfg.image_width],
            in_channels=self.channels,
            num_classes=cfg.num_classes,
            embed_dim=self.channels[0],
            deep_supervision=False,
            residual_mode=getattr(cfg, "decoder_residual_mode", "orthogonal"),
            router_mode=getattr(cfg, "decoder_router_mode", "prototype"),
            recovery_scales=getattr(cfg, "decoder_recovery_scales", ("1_8", "1_4")),
            collect_diagnostics=getattr(cfg, "decoder_collect_diagnostics", True),
            use_checkpoint=bool(
                getattr(
                    cfg,
                    "decoder_use_checkpoint",
                    getattr(cfg, "use_checkpoint", False),
                )
            ),
        )

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            if not hasattr(self.backbone, "init_weights"):
                raise AttributeError(
                    "The selected VMamba backbone cannot load a custom "
                    "cfg.pretrained_model"
                )
            self.backbone.init_weights(pretrained=pretrained)

        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )
        self.decode_head.reset_evidence_parameters()

    @staticmethod
    def _extract_features(encoder_output):
        features = encoder_output.get("features")
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError("VMamba must return a four-level feature pyramid")
        return features

    @staticmethod
    def _resize_label(label, size):
        if label.shape[-2:] == size:
            return label.long()
        return (
            F.interpolate(label.unsqueeze(1).float(), size=size, mode="nearest")
            .squeeze(1)
            .long()
        )

    def encode_decode(self, rgb, modal_x):
        input_size = rgb.shape[-2:]
        encoder_output = self.backbone(rgb, modal_x)
        decoder_output = self.decode_head(encoder_output)

        if not isinstance(decoder_output, dict):
            raise TypeError("ResidualEvidenceDecoder must return a dictionary")
        main_logits = decoder_output.get("logits")
        auxiliary_logits = decoder_output.get("aux_logits", {})
        if not torch.is_tensor(main_logits):
            raise ValueError("Decoder output is missing tensor 'logits'")
        if not isinstance(auxiliary_logits, dict):
            raise TypeError("Decoder 'aux_logits' must be a dictionary")

        main_logits = F.interpolate(
            main_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return main_logits, auxiliary_logits

    def forward(self, rgb, modal_x, label=None):
        main_logits, auxiliary_logits = self.encode_decode(rgb, modal_x)
        if label is None:
            return main_logits

        loss = self.criterion(main_logits, label.long())
        for name, logits in auxiliary_logits.items():
            weight = float(self.decoder_aux_weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            target = self._resize_label(label, logits.shape[-2:])
            loss = loss + weight * self.criterion(logits, target)
        return loss
