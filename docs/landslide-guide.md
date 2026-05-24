# DA-TransUNet 滑坡分割迁移指南

将医学图像分割模型 DA-TransUNet 迁移到遥感滑坡识别。

论文: 物探化探计算技术 2025 | 仓库: github.com/SUN-1024/DA-TransUnet

## 模型简介

DA-TransUNet 集成双重注意力机制（空间注意力 + 通道注意力）与 TransUNet 架构。在 GDCLD 全球同震滑坡数据集预训练，迁移到毕节地区滑坡数据后 IoU 达 79.31%。

## 环境配置

```bash
git clone https://github.com/SUN-1024/DA-TransUnet.git
cd DA-TransUnet
pip install -r requirements.txt
```

## 数据预处理

### GDCLD 数据集

全球同震滑坡数据库 (Zenodo doi:10.5281/zenodo.11369484)，覆盖 38 次强震，40 万+ 滑坡样本。

```python
# 裁剪指定区域子集
import rasterio, numpy as np

def crop_region(src_path, dst_path, bbox):
    """按地理边界框裁剪滑坡数据集"""
    with rasterio.open(src_path) as src:
        window = rasterio.windows.from_bounds(*bbox, src.transform)
        data = src.read(window=window)
        # 保存裁剪结果...
```

### 正负样本均衡

滑坡识别是典型的正负样本不均衡问题——滑坡区域仅占影像 ~5%。建议：

```python
# 负样本降采样
from sklearn.utils import resample
neg_indices = resample(neg_indices, n_samples=len(pos_indices), random_state=42)
```

## 训练

```bash
python train.py --dataset gdcld --epochs 200 --batch_size 8
```

关键参数：
- `--dataset`: 数据集选择 (gdcld / bijie)
- `--attention_type`: dual / spatial / channel
- `--pretrained`: 是否加载预训练权重

## 双注意力模块原理

- **空间注意力 (Spatial Attention)**: 关注"滑坡在图像中的位置"
- **通道注意力 (Channel Attention)**: 关注"哪个波段对滑坡最敏感"

两者并联后与 TransUNet 编码器输出融合进入解码器。

## 滑坡区域适配建议

1. **数据增强**: 加入旋转、翻转、弹性变形模拟不同坡向
2. **多源特征**: 加入 DEM 坡度/坡向作为额外输入通道
3. **后处理**: CRF 精修边界，去除孤立误检像素
