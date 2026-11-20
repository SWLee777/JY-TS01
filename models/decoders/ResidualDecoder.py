import torch
import torch.nn as nn
import torch.nn.functional as F

from .MambaDecoder import FinalUpsample_X4, Mamba_up, PatchExpand


def compute_orthogonal_residual(modality_feature, fused_feature, eps=1e-6):

    if modality_feature.ndim != 4 or fused_feature.ndim != 4:
        raise ValueError("Orthogonal residual inputs must be BCHW tensors.")
    if modality_feature.shape != fused_feature.shape:
        raise ValueError(
            "Orthogonal residual inputs must have identical shapes, got "
            f"{tuple(modality_feature.shape)} and {tuple(fused_feature.shape)}."
        )

    output_dtype = modality_feature.dtype
    compute_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    modality = modality_feature.to(dtype=compute_dtype)
    fused = fused_feature.to(dtype=compute_dtype)
    numerator = torch.sum(modality * fused, dim=1, keepdim=True)
    denominator = torch.sum(fused.square(), dim=1, keepdim=True).add(eps)
    residual = modality - numerator.div(denominator) * fused
    return residual.to(dtype=output_dtype)


def compute_pair_orthogonal_residual(
    rgb_feature, depth_feature, fused_feature, eps=1e-6
):

    if not (rgb_feature.shape == depth_feature.shape == fused_feature.shape):
        raise ValueError("RGB, Depth, and fused evidence must share one shape.")
    if rgb_feature.ndim != 4:
        raise ValueError("Evidence tensors must be BCHW tensors.")

    output_dtype = rgb_feature.dtype
    compute_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    rgb = rgb_feature.to(dtype=compute_dtype)
    depth = depth_feature.to(dtype=compute_dtype)
    fused = fused_feature.to(dtype=compute_dtype)
    denominator = torch.sum(fused.square(), dim=1, keepdim=True).add(eps)

    rgb_projection = (
        torch.sum(rgb * fused, dim=1, keepdim=True).div(denominator) * fused
    )
    depth_projection = (
        torch.sum(depth * fused, dim=1, keepdim=True).div(denominator) * fused
    )
    return (
        (rgb - rgb_projection).to(dtype=rgb_feature.dtype),
        (depth - depth_projection).to(dtype=depth_feature.dtype),
    )


def _group_count(channels, max_groups=32):
    for groups in (max_groups, 16, 8, 4, 2, 1):
        if groups <= channels and channels % groups == 0:
            return groups
    return 1


class SemanticPrototypeRouter(nn.Module):

    def __init__(self, channels, num_classes, projection_ratio=4, eps=1e-6):
        super().__init__()
        route_channels = max(channels // projection_ratio, 16)
        hidden_channels = max(route_channels, 32)
        self.eps = float(eps)

        self.class_head = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.semantic_projection = nn.Conv2d(
            channels, route_channels, kernel_size=1, bias=False
        )
        self.prototype_projection = nn.Conv2d(
            channels, route_channels, kernel_size=1, bias=False
        )
        self.rgb_projection = nn.Conv2d(
            channels, route_channels, kernel_size=1, bias=False
        )
        self.depth_projection = nn.Conv2d(
            channels, route_channels, kernel_size=1, bias=False
        )

        route_input_channels = route_channels * 4 + 5
        groups = _group_count(hidden_channels)
        self.route_features = nn.Sequential(
            nn.Conv2d(
                route_input_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.route_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)

    def reset_output_biases(self):
        if self.class_head.bias is not None:
            nn.init.zeros_(self.class_head.bias)

        nn.init.zeros_(self.route_head.weight)
        if self.route_head.bias is not None:
            nn.init.zeros_(self.route_head.bias)

    def _build_prototypes(self, semantic, class_probability):
        batch, channels, height, width = semantic.shape
        semantic_tokens = semantic.flatten(2)
        probability_tokens = class_probability.flatten(2)

        class_mass = probability_tokens.sum(dim=-1, keepdim=True).add(self.eps)
        prototypes = torch.bmm(probability_tokens, semantic_tokens.transpose(1, 2)).div(
            class_mass
        )
        pixel_prototype = torch.bmm(
            prototypes.transpose(1, 2), probability_tokens
        ).reshape(batch, channels, height, width)
        return prototypes, pixel_prototype

    def forward(self, semantic, rgb_residual, depth_residual):
        if not (semantic.shape == rgb_residual.shape == depth_residual.shape):
            raise ValueError(
                "Semantic state and modality residuals must share one BCHW shape."
            )

        class_logits = self.class_head(semantic)
        class_probability = F.softmax(class_logits, dim=1)
        _, pixel_prototype = self._build_prototypes(semantic, class_probability)

        semantic_route = F.normalize(
            self.semantic_projection(semantic), p=2, dim=1, eps=self.eps
        )
        prototype_route = F.normalize(
            self.prototype_projection(pixel_prototype),
            p=2,
            dim=1,
            eps=self.eps,
        )
        rgb_route = F.normalize(
            self.rgb_projection(rgb_residual), p=2, dim=1, eps=self.eps
        )
        depth_route = F.normalize(
            self.depth_projection(depth_residual), p=2, dim=1, eps=self.eps
        )

        rgb_similarity = torch.sum(prototype_route * rgb_route, dim=1, keepdim=True)
        depth_similarity = torch.sum(prototype_route * depth_route, dim=1, keepdim=True)
        cross_modal_agreement = torch.sum(rgb_route * depth_route, dim=1, keepdim=True)
        rgb_energy = torch.log1p(rgb_residual.square().mean(dim=1, keepdim=True).sqrt())
        depth_energy = torch.log1p(
            depth_residual.square().mean(dim=1, keepdim=True).sqrt()
        )

        route_input = torch.cat(
            [
                semantic_route,
                prototype_route,
                rgb_route,
                depth_route,
                rgb_similarity,
                depth_similarity,
                cross_modal_agreement,
                rgb_energy,
                depth_energy,
            ],
            dim=1,
        )

        route_weights = F.softmax(
            self.route_head(self.route_features(route_input)), dim=1
        )
        return route_weights, class_logits


class PlainConvRouter(nn.Module):

    def __init__(self, channels):
        super().__init__()
        hidden_channels = max(channels // 2, 32)
        groups = _group_count(hidden_channels)
        self.route_features = nn.Sequential(
            nn.Conv2d(
                channels * 3 + 2,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.route_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)

    def reset_output_biases(self):
        nn.init.zeros_(self.route_head.weight)
        if self.route_head.bias is not None:
            nn.init.zeros_(self.route_head.bias)

    def forward(self, semantic, rgb_residual, depth_residual):
        if not (semantic.shape == rgb_residual.shape == depth_residual.shape):
            raise ValueError(
                "Semantic state and modality residuals must share one BCHW shape."
            )
        rgb_energy = torch.log1p(rgb_residual.square().mean(dim=1, keepdim=True).sqrt())
        depth_energy = torch.log1p(
            depth_residual.square().mean(dim=1, keepdim=True).sqrt()
        )
        route_input = torch.cat(
            [semantic, rgb_residual, depth_residual, rgb_energy, depth_energy],
            dim=1,
        )
        route_weights = F.softmax(
            self.route_head(self.route_features(route_input)), dim=1
        )
        return route_weights, None


class ResidualEvidenceDecoder(nn.Module):

    def __init__(
        self,
        img_size=(480, 640),
        in_channels=(96, 192, 384, 768),
        num_classes=40,
        embed_dim=96,
        patch_size=4,
        depths=(4, 4, 4, 4),
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        deep_supervision=False,
        residual_mode="orthogonal",
        router_mode="prototype",
        recovery_scales=("1_8", "1_4"),
        collect_diagnostics=True,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        if len(in_channels) != 4 or len(depths) != 4:
            raise ValueError("ResidualEvidenceDecoder requires four feature scales.")
        if deep_supervision:
            raise ValueError(
                "Generic deep supervision is disabled for ResidualEvidenceDecoder; "
                "use its two prototype auxiliary heads instead."
            )

        self.num_classes = int(num_classes)
        self.num_layers = len(depths)
        self.mlp_ratio = float(mlp_ratio)
        self.patch_size = int(patch_size)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.eps = float(eps)
        self.residual_mode = str(residual_mode)
        self.router_mode = str(router_mode)
        valid_residual_modes = {"direct", "difference", "orthogonal"}
        valid_router_modes = {"none", "conv", "prototype"}
        if self.residual_mode not in valid_residual_modes:
            raise ValueError(f"Unsupported residual mode: {self.residual_mode}.")
        if self.router_mode not in valid_router_modes:
            raise ValueError(f"Unsupported router mode: {self.router_mode}.")
        ordered_scales = tuple(
            scale for scale in ("1_8", "1_4") if scale in recovery_scales
        )
        if not ordered_scales or len(ordered_scales) != len(set(recovery_scales)):
            raise ValueError(
                "Recovery scales must be a non-empty subset of ('1_8', '1_4')."
            )
        self.recovery_scales = ordered_scales
        self.patches_resolution = [
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        ]

        scale_channels = {"1_4": in_channels[0], "1_8": in_channels[1]}
        if self.router_mode == "prototype":
            self.routers = nn.ModuleDict(
                {
                    scale: SemanticPrototypeRouter(
                        scale_channels[scale], num_classes, eps=self.eps
                    )
                    for scale in self.recovery_scales
                }
            )
        elif self.router_mode == "conv":
            self.routers = nn.ModuleDict(
                {
                    scale: PlainConvRouter(scale_channels[scale])
                    for scale in self.recovery_scales
                }
            )
        else:
            self.routers = nn.ModuleDict()
        self.beta = nn.ParameterDict(
            {scale: nn.Parameter(torch.zeros(1)) for scale in self.recovery_scales}
        )
        self.prototype_logits = {}
        self.route_stats = {}

        drop_paths = [
            value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        self.layers_up = nn.ModuleList()
        for layer_index in range(self.num_layers):
            depth_index = self.num_layers - 1 - layer_index
            dim = int(embed_dim * 2**depth_index)
            resolution = (
                self.patches_resolution[0] // (2**depth_index),
                self.patches_resolution[1] // (2**depth_index),
            )
            if layer_index == 0:
                layer = PatchExpand(
                    input_resolution=resolution,
                    dim=dim,
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                layer = Mamba_up(
                    dim=dim,
                    input_resolution=resolution,
                    depth=depths[depth_index],
                    mlp_ratio=self.mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path=drop_paths[
                        sum(depths[:depth_index]) : sum(depths[: depth_index + 1])
                    ],
                    norm_layer=norm_layer,
                    upsample=(
                        PatchExpand if layer_index < self.num_layers - 1 else None
                    ),
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer)

        self.norm_up = norm_layer(embed_dim)
        self.up = FinalUpsample_X4(
            input_resolution=(
                img_size[0] // patch_size,
                img_size[1] // patch_size,
            ),
            patch_size=patch_size,
            dim=embed_dim,
        )
        self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=False)
        self.reset_evidence_parameters()

    def reset_evidence_parameters(self):
        with torch.no_grad():
            for beta in self.beta.values():
                beta.zero_()
        for router in self.routers.values():
            router.reset_output_biases()

    @staticmethod
    def _unpack_inputs(inputs):
        if isinstance(inputs, dict):
            features = inputs.get("features")
            evidence = inputs.get("high_evidence")
        elif isinstance(inputs, tuple) and len(inputs) == 4:
            features, rgb_list, depth_list, fused_list = inputs
            evidence = {
                "1_4": {
                    "rgb": rgb_list[0],
                    "depth": depth_list[0],
                    "fused": fused_list[0],
                },
                "1_8": {
                    "rgb": rgb_list[1],
                    "depth": depth_list[1],
                    "fused": fused_list[1],
                },
            }
        else:
            raise TypeError(
                "ResidualEvidenceDecoder expects structured encoder evidence."
            )

        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError("Expected four fused encoder feature scales.")
        if not isinstance(evidence, dict):
            raise ValueError("Encoder high-frequency evidence is missing.")
        return features, evidence

    @staticmethod
    def _resize_if_needed(feature, size):
        if feature.shape[-2:] == size:
            return feature
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def _inject_evidence(self, scale, semantic_bhwc, evidence):
        if scale not in evidence:
            raise KeyError(f"Encoder did not return {scale} evidence.")
        stage_evidence = evidence[scale]
        required = ("rgb", "depth", "fused")
        if any(key not in stage_evidence for key in required):
            raise KeyError(f"Incomplete {scale} RGB/Depth/fused evidence.")

        semantic = semantic_bhwc.permute(0, 3, 1, 2).contiguous()
        target_size = semantic.shape[-2:]
        rgb_high = self._resize_if_needed(stage_evidence["rgb"], target_size)
        depth_high = self._resize_if_needed(stage_evidence["depth"], target_size)
        fused_reference = self._resize_if_needed(
            stage_evidence["fused"], target_size
        ).detach()

        if self.residual_mode == "direct":
            rgb_residual, depth_residual = rgb_high, depth_high
        elif self.residual_mode == "difference":
            rgb_residual = rgb_high - fused_reference
            depth_residual = depth_high - fused_reference
        else:
            rgb_residual, depth_residual = compute_pair_orthogonal_residual(
                rgb_high, depth_high, fused_reference, eps=self.eps
            )

        if self.router_mode == "none":
            selected_evidence = 0.5 * (rgb_residual + depth_residual)
            route_weights = semantic.new_zeros(
                semantic.shape[0], 3, *semantic.shape[-2:]
            )
            route_weights[:, 1:].fill_(0.5)
            class_logits = None
        else:
            route_weights, class_logits = self.routers[scale](
                semantic, rgb_residual, depth_residual
            )
            _, rgb_weight, depth_weight = route_weights.chunk(3, dim=1)
            selected_evidence = (
                rgb_weight * rgb_residual + depth_weight * depth_residual
            )

        reject_weight, rgb_weight, depth_weight = route_weights.chunk(3, dim=1)
        alpha = torch.tanh(self.beta[scale]).to(dtype=semantic.dtype)
        recovered = semantic + alpha * selected_evidence

        if class_logits is not None:
            self.prototype_logits[scale] = class_logits
        if self.collect_diagnostics:
            entropy = -torch.sum(
                route_weights.clamp_min(self.eps)
                * route_weights.clamp_min(self.eps).log(),
                dim=1,
            ).mean()
            self.route_stats[scale] = torch.stack(
                [
                    reject_weight.mean(),
                    rgb_weight.mean(),
                    depth_weight.mean(),
                    entropy,
                    rgb_residual.square().mean().sqrt(),
                    depth_residual.square().mean().sqrt(),
                ]
            ).detach()
        return recovered.permute(0, 2, 3, 1).contiguous()

    def _run_recovery_stage(self, layer, semantic_input, scale, evidence):
        semantic_state = layer.forward_blocks(semantic_input)
        recovered_state = self._inject_evidence(scale, semantic_state, evidence)
        if layer.upsample is not None:
            return layer.upsample(recovered_state)
        return recovered_state

    def forward_up_features(self, inputs):
        features, evidence = self._unpack_inputs(inputs)
        self.prototype_logits = {}
        self.route_stats = {}

        for index, layer in enumerate(self.layers_up):
            skip = features[3 - index]
            if index == 0:
                y = layer(skip.permute(0, 2, 3, 1).contiguous())
                continue

            y = (
                F.interpolate(
                    y.permute(0, 3, 1, 2).contiguous(),
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            semantic_input = y + skip.permute(0, 2, 3, 1).contiguous()

            if index == 2 and "1_8" in self.recovery_scales:
                y = self._run_recovery_stage(layer, semantic_input, "1_8", evidence)
            elif index == 3 and "1_4" in self.recovery_scales:
                y = self._run_recovery_stage(layer, semantic_input, "1_4", evidence)
            else:
                y = layer(semantic_input)

        return self.norm_up(y)

    def forward(self, inputs):
        decoded = self.forward_up_features(inputs)
        final_logits = self.up_x4(decoded)
        auxiliary_logits = {
            f"prototype_{scale}": logits
            for scale, logits in self.prototype_logits.items()
        }
        return {"logits": final_logits, "aux_logits": auxiliary_logits}

    def diagnostic_summary(self):
        summary = {}
        for scale, beta in self.beta.items():
            summary[f"alpha_{scale}"] = torch.tanh(beta).detach().item()
            if scale in self.route_stats:
                values = self.route_stats[scale]
                summary[f"reject_{scale}"] = values[0].item()
                summary[f"rgb_{scale}"] = values[1].item()
                summary[f"depth_{scale}"] = values[2].item()
                summary[f"entropy_{scale}"] = values[3].item()
                summary[f"rgb_residual_{scale}"] = values[4].item()
                summary[f"depth_residual_{scale}"] = values[5].item()
        return summary

    def up_x4(self, feature):
        batch, height, width, _ = feature.shape
        feature = self.up(feature)
        feature = feature.reshape(
            batch,
            self.patch_size * height,
            self.patch_size * width,
            -1,
        )
        feature = feature.permute(0, 3, 1, 2).contiguous()
        return self.output(feature)
