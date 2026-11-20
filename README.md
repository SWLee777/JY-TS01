# FDOER-Net

Minimal, self-contained source release for **FDOER-Net**, a frequency-decoupled cross-modal fusion and orthogonal evidence recovery network for RGB-X semantic segmentation.

中文说明：[README_zh-CN.md](README_zh-CN.md)

The repository includes only the execution path used by FDOER-Net:

- shared RGB-X VMamba-T encoder;
- FD-Fusion with JQCMA and Tri-Scan Mamba branches;
- post-Mamba orthogonal evidence recovery decoder;
- class-prototype-conditioned evidence routing;
- NYU Depth V2, MFNet, and PST900 configurations;
- training and evaluation entry points;
- official and corrected evaluator implementations.

Unrelated Sigma backbones, alternative decoders, hyperparameter-search scripts, server automation, logs, and qualitative outputs are not included.

## Requirements

The reference environment is:

```text
Python        3.9.25
PyTorch       2.2.1+cu118
CUDA          11.8
timm          0.4.12
mamba_ssm     2.2.2
causal_conv1d 1.4.0
einops        0.8.2
```

Create an environment and install dependencies:

```bash
conda create -n fdoer python=3.9
conda activate fdoer

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r requirements-model.txt
```

Install the bundled selective-scan extension:

```bash
cd models/encoders/selective_scan
pip install .
cd ../../..
```

## Required External Assets

### Pretrained VMamba-T

Download the VMamba-T checkpoint and place it at:

```text
pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth
```

The source follows VMamba-T dimensions `[96, 192, 384, 768]` and encoder depths `[2, 2, 9, 2]`.

### Datasets

Arrange each dataset as:

```text
datasets/<DatasetName>/
├── RGB/
├── <X modality folder>/
├── Label/
├── train.txt
└── test.txt
```

Supported configurations:

| Config key | Dataset |
|---|---|
| `nyu` | NYU Depth V2 |
| `mfnet` | MFNet |
| `pst` | PST900 |

Update the paths in:

```text
configs/config_nyu.py
configs/config_MFNet.py
configs/config_pst900.py
```

For NYU Depth V2, the public-official test list is `test2.txt` with 654 images.

## Model

### FD-Fusion

At every encoder scale, RGB and X features are split into low- and high-frequency components:

```text
low  = AvgPool3x3(feature)
high = feature - low
```

The low-frequency path uses joint-query cross-modal attention:

```text
Q = Q_rgb + Q_x
K = concat(K_rgb, K_x)
V = concat(V_rgb, V_x)
```

The high-frequency path uses three parameter-independent Mamba scans:

```text
horizontal: left-to-right row-major order
vertical:   top-to-bottom column-major order
reverse:    bottom-right-to-top-left order
```

### OER

OER is applied at decoder resolutions `1/8` and `1/4`. RGB and X high-frequency features are projected into the orthogonal complement of the fused high-frequency reference:

```text
r_rgb = h_rgb - proj(h_rgb, h_fused)
r_x   = h_x   - proj(h_x,   h_fused)
```

The fused reference is detached for this projection. A semantic class head produces class probabilities and prototypes. The router selects one of:

```text
reject | RGB evidence | X evidence
```

The selected evidence is injected with a zero-initialized gate:

```text
s_recovered = s + tanh(beta) * selected_evidence
```

The `1/8` and `1/4` prototype heads use auxiliary cross entropy with weight `0.1`.

## Training

Example for NYU Depth V2 on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=1 \
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29502 \
  train.py \
  -p 29502 \
  -d 0,1 \
  -n nyu \
  --seed 3407 \
  --checkpointing
```

The final model settings are already enabled in the supplied configurations:

```python
C.backbone = "sigma_tiny"
C.decoder = "ResidualDecoder"
C.encoder_fusion_mode = "full"
C.decoder_residual_mode = "orthogonal"
C.decoder_router_mode = "prototype"
C.decoder_recovery_scales = ("1_8", "1_4")
C.decoder_aux_weights = {
    "prototype_1_8": 0.1,
    "prototype_1_4": 0.1,
}
```

## Evaluation

Official NYU Depth V2 evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python eval.py \
  -n nyu \
  -e 500 \
  -d 0,1 \
  --eval-protocol official \
  --seed 3407
```

The `official` option reproduces the convention used for paper comparisons. The `corrected` option is retained only for protocol auditing. Do not mix outputs from the two modes.

## Reference Result

NYU Depth V2 official result over three seeds:

| Seed | mIoU (%) |
|---|---:|
| 3407 | 54.227 |
| 0 | 55.057 |
| 42 | 54.119 |
| **Mean ± SD** | **54.468 ± 0.513** |

## Repository Layout

```text
.
├── configs/                  # NYU, MFNet, and PST900 configurations
├── dataloader/               # RGB-X data pipeline
├── engine/                   # trainer and evaluator
├── models/
│   ├── builder.py            # FDOER-only model assembly
│   ├── encoders/
│   │   ├── vmamba.py         # required generic VMamba backbone
│   │   ├── dual_vmamba.py    # FDOER encoder and FD-Fusion
│   │   └── selective_scan/   # required VMamba CUDA extension
│   └── decoders/
│       ├── MambaDecoder.py   # required patch/Mamba upsampling primitives
│       └── ResidualDecoder.py
├── utils/                    # training, metrics, transforms, logging
├── train.py
└── eval.py
```

## Attribution

The execution scaffold and generic VMamba implementation derive from the MIT-licensed [Sigma](https://github.com/zifuwan/Sigma) project. The original license notice is retained in `LICENSE`; additional dependency attribution is provided in `THIRD_PARTY_NOTICES.md`.

