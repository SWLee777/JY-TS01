# FDOER-Net 最小可运行源码

本仓库是 **FDOER-Net-T 主模型**的最小可运行版本，用于 RGB-X 多模态语义分割。它保留了模型训练和评测实际使用的代码，删除了 Sigma 中与本模型无关的 backbone、decoder、实验脚本和运行产物。

## 包含内容

- 共享权重的 RGB-X VMamba-T 编码器；
- FD-Fusion 高低频跨模态融合；
- JQCMA 低频联合查询注意力；
- Tri-Scan Mamba 高频特征建模；
- OER 正交证据恢复；
- 类别原型条件的三路证据路由；
- NYU Depth V2、MFNet、PST900 配置；
- 训练入口、official/corrected 评测入口和必要工具代码。

## 参考环境

```text
Python        3.9.25
PyTorch       2.2.1+cu118
CUDA          11.8
timm          0.4.12
mamba_ssm     2.2.2
causal_conv1d 1.4.0
einops        0.8.2
```

创建环境：

```bash
conda create -n fdoer python=3.9
conda activate fdoer

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r requirements-model.txt
```

安装仓库内附带的 selective-scan 扩展：

```bash
cd models/encoders/selective_scan
pip install .
cd ../../..
```

## 需要单独准备的文件

### VMamba-T 预训练权重

将权重放到：

```text
pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth
```

压缩包不包含模型权重。

### 数据集

数据目录格式如下：

```text
datasets/<DatasetName>/
├── RGB/
├── <X modality folder>/
├── Label/
├── train.txt
└── test.txt
```

支持的配置：

| 配置名称 | 数据集 |
|---|---|
| `nyu` | NYU Depth V2 |
| `mfnet` | MFNet |
| `pst` | PST900 |

根据实际数据位置修改：

```text
configs/config_nyu.py
configs/config_MFNet.py
configs/config_pst900.py
```

NYU Depth V2 的 official 测试列表使用 `test2.txt`，共 654 张图像。

## 模型结构

### FD-Fusion

每个尺度首先进行频率分解：

```text
low  = AvgPool3x3(feature)
high = feature - low
```

低频分支采用 JQCMA：

```text
Q = Q_rgb + Q_x
K = concat(K_rgb, K_x)
V = concat(V_rgb, V_x)
```

高频分支执行三条参数独立的 Mamba 扫描：

```text
水平扫描：从左到右逐行
垂直扫描：从上到下逐列
反向扫描：从右下到左上
```

### OER

OER 仅用于解码器的 `1/8` 和 `1/4` 尺度。RGB 与 X 的高频特征会去掉融合高频参考中已经解释的分量：

```text
r_rgb = h_rgb - proj(h_rgb, h_fused)
r_x   = h_x   - proj(h_x,   h_fused)
```

融合高频参考执行 stop-gradient。类别原型路由在以下三种选择中做逐像素决策：

```text
拒绝 | RGB 证据 | X 证据
```

证据通过零初始化门控回注：

```text
s_recovered = s + tanh(beta) * selected_evidence
```

`1/8` 和 `1/4` 的类别原型头均使用权重为 `0.1` 的辅助交叉熵。

## 训练

NYU Depth V2 双卡训练示例：

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

提供给本发布版的主模型配置已经设置为：

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

## 评测

NYU Depth V2 official 评测：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python eval.py \
  -n nyu \
  -e 500 \
  -d 0,1 \
  --eval-protocol official \
  --seed 3407
```

`official` 用于论文主结果对比；`corrected` 仅用于评测协议敏感性分析。两套结果不能混在同一张表中。

## 参考结果

NYU Depth V2 official 三随机种子结果：

| 随机种子 | mIoU (%) |
|---|---:|
| 3407 | 54.227 |
| 0 | 55.057 |
| 42 | 54.119 |
| **均值 ± 标准差** | **54.468 ± 0.513** |

## 来源与许可

训练框架和通用 VMamba 实现来自 MIT 许可的 [Sigma](https://github.com/zifuwan/Sigma) 项目。原始许可证保存在 `LICENSE`，依赖说明见 `THIRD_PARTY_NOTICES.md`。

