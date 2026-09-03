# ZeroTrain-FPP

面向单帧条纹投影三维测量的双目相位恢复研究。项目以 UDPR/UCNNet 为
baseline，探索**序数条纹级推断、有界相位残差和置信度物理约束**，目标是
在不使用目标场景相位标签的条件下，提高绝对相位恢复的可靠性与效率。

> 当前状态：官方数据读取、标定解析、Baseline 官方输出评估和两阶段模型
> 骨架已经完成；完整物理损失、训练流程及消融实验仍在开发。README 中的
> “创新点”是待实验验证的研究假设，不代表已经取得性能提升。

## 项目概览

条纹投影轮廓术通过投影仪向物体表面投射正弦条纹，从相机采集图像中恢复
绝对相位，并进一步重建三维形貌。传统高精度方案通常需要多幅相移条纹和
Gray Code，不适用于高速运动场景；单帧方案采集速度快，但面临包裹相位误差
以及 \(2\pi\) 周期模糊。

本项目研究的问题是：

> 如何利用双相机之间的相位与几何关系，从左右单帧条纹中恢复可靠的绝对
> 相位，同时降低逐场景优化的计算成本？

推荐课题名称：

- 中文：基于序数条纹级推断与置信度物理约束的单帧双目相位恢复
- 英文：Confidence-Guided Stereo Phase Retrieval with Ordinal
  Fringe-Order Inference

## 方法

### Baseline：UDPR/UCNNet

Baseline 来自论文 *Untrained deep learning-based phase retrieval for
fringe projection profilometry*。它针对每个待测场景优化共享权重的双目
网络，不依赖当前场景的相位标签。

第一阶段根据左右单幅条纹 \(I_L,I_R\) 和 WFT 粗包裹相位
\(\phi_L^c,\phi_R^c\) 预测条纹级 \(K_L,K_R\)：

\[
\Phi_i^c=\phi_i^c+2\pi K_i.
\]

第二阶段根据条纹和粗绝对相位 \(\Phi_i^c\) 恢复精细绝对相位。优化过程
采用三种场景无关约束：

- 相位一致性：同一三维点在左右视图中对应相同投影相位；
- 结构一致性：左右对应图像区域具有相似的局部结构；
- 三维一致性：双目系统与相机—投影仪系统给出的几何对应一致。

### Baseline 局限

1. 条纹级本质上是离散、有序变量，但 Baseline 将其作为连续量回归，无法
   显式描述候选条纹级的歧义；
2. 暗区、饱和区、遮挡区和弱纹理区可靠程度不同，固定损失权重容易受到错误
   对应点干扰；
3. 每个场景从随机参数开始优化，计算成本较高；
4. 第二阶段直接处理大范围绝对相位，未充分利用“细化粗相位”的任务先验。

### 总体方案

```text
左右单幅条纹
      │
      ▼
WFT 粗包裹相位
      │
      ▼
Stage I：序数条纹级概率推断 ──► 条纹级不确定度
      │
      ▼
粗绝对相位 Φᶜ = φᶜ + 2πK
      │
      ▼
Stage II：[-π, π] 有界相位残差 ──► 相位置信度
      │
      ▼
置信度加权的双目物理一致性优化
      │
      ▼
精细绝对相位与三维重建
```

## 拟验证的创新点

### 1. 序数概率条纹级推断

对候选条纹级建立概率分布：

\[
p(k\mid I,\phi^c),\qquad k\in[K_{\min},K_{\max}],
\]

并通过 soft-argmax 得到可微预测：

\[
\hat K=\sum_k k\,p(k).
\]

概率分布的归一化熵用于描述条纹级不确定度：

\[
U_K=-\frac{1}{\log N}\sum_kp(k)\log p(k).
\]

相较于无约束连续回归，该表示能够利用条纹级的离散和有序属性，并同时给出
预测值与可信程度。

### 2. 有界绝对相位残差

第二阶段只预测粗相位的周期内修正量：

\[
\hat\Phi=\Phi^c+\Delta\Phi,\qquad
\Delta\Phi=\pi\tanh(z)\in[-\pi,\pi].
\]

该设计缩小了网络输出空间，并通过物理边界减少异常的跨周期相位修正。

### 3. 置信度引导的物理约束

计划融合有效区域、曝光质量、局部纹理、条纹级熵和相位方差：

\[
C=C_{\mathrm{mask}}C_{\mathrm{intensity}}C_{\mathrm{texture}}
C_{\mathrm{order}}C_{\mathrm{phase}},
\]

并使用置信度加权的鲁棒损失：

\[
\mathcal L_{\mathrm{phy}}=
\frac{\sum_x C(x)\rho(r(x))}
{\sum_x C(x)+\epsilon}.
\]

目标是降低遮挡、异常曝光和弱纹理像素对相位、结构及三维一致性优化的干扰。

### 4. 轻量测试时自适应

后续将探索共享初始化和低秩 Adapter：测试时冻结主干，只更新少量场景相关
参数，并根据物理残差提前停止。引入共享初始化后，方法应准确描述为
“无目标域标签的测试时自适应”，而不是“完全无训练”。

## 官方数据

本项目使用 Baseline 论文公开的约 7.5 GB 官方数据。数据不包含在 Git
仓库中，请将其放置在项目根目录的 `Dataset/` 下。

| 数据部分 | 场景数 | 论文实验设置 |
|---|---:|---|
| section 3.2.1 | 100 | 充足、相似场景 |
| section 3.2.2 | 15 | 有限场景 |
| section 3.2.3 | 75 | 树皮、叶片等跨类别场景 |

每个公开场景包含左右单幅条纹、WFT 粗包裹相位、条纹级真值、粗绝对相位、
绝对相位真值、有效 mask、官方两阶段输出以及系统标定参数。图像分辨率为
\(640\times480\)。

公开文件共包含 190 个论文实验场景，并不是论文提到的全部 675 个采集场景。
因此，数据划分和实验结论必须以实际公开文件为准。

预期目录结构：

```text
Dataset/
├── section 3.2.1/
│   ├── calibration parameter/
│   ├── fringe order range/
│   ├── data-stage1/
│   └── data-stage2/
├── section 3.2.2/
└── section 3.2.3/
```

## 已完成工作

- 兼容 classic MAT、MATLAB v7.3 与普通 HDF5；
- 将 MATLAB 保存的 `640×480` 数组恢复为 Python 的 `[H,W]=[480,640]`；
- 严格配对左右视图、两阶段输入、真值、mask 和官方输出；
- 解析相机—相机及相机—投影仪标定，并导出常规列向量/OpenCV 形式
  \(K,R,t\)；
- 在有效 mask 内复算 MAE、RMSE、Bad-pixel rate 和条纹级准确率；
- 实现左右视图权重共享的两阶段网络；
- 实现 56 个候选条纹级、soft-argmax、熵不确定度、像素级候选范围和
  \([-\pi,\pi]\) 相位残差。

核心文件：

| 文件 | 作用 |
|---|---|
| `fringe_repair/udpr_io.py` | 官方数据与标定解析 |
| `fringe_repair/udpr_metrics.py` | Baseline 指标统计 |
| `fringe_repair/udpr_models.py` | 双目两阶段模型 |
| `scripts/evaluate_udpr_baseline.py` | 官方输出复现入口 |
| `tests/test_udpr.py` | 数据、标定和模型测试 |

## Baseline 复现结果

以下结果由官方保存的 Stage-II 输出计算，而不是当前创新模型的训练结果：

| 数据部分 | 样本数 | 双视图 MAE | 双视图 RMSE |
|---|---:|---:|---:|
| section 3.2.1 | 100 | 0.1098 rad | 0.3861 rad |
| section 3.2.2 | 15 | 0.0944 rad | 0.3427 rad |
| section 3.2.3 | 75 | 0.0813 rad | 0.2939 rad |

结果与论文报告的约 \(0.10/0.08/0.07\) rad 处于相同量级。差异可能来自
逐场景与逐像素平均方式、视图选择或未公开后处理，因此当前结论是“评估
链路正确且结果同量级”，不是“逐位复现论文数值”。

完整统计见 `results/udpr_official_baseline.json`。

## 快速开始

### 1. 安装

建议使用 Python 3.11 和 PyTorch 2.4 或更高版本：

```bash
git clone https://github.com/YeLuo-123/ZeroTrain-FPP.git
cd ZeroTrain-FPP

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision
.venv/bin/pip install -r requirements.txt
```

### 2. 复现官方输出指标

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_udpr_baseline.py \
  --root Dataset \
  --section all \
  --stage all \
  --output results/udpr_official_baseline.json
```

只评估一个子集和阶段：

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_udpr_baseline.py \
  --root Dataset --section 3.2.3 --stage 2
```

### 3. 加载数据和标定

```python
from fringe_repair.udpr_io import OfficialUDPRDataset

dataset = OfficialUDPRDataset(
    "Dataset",
    section="3.2.1",
    stage=1,
    include_output=True,
)

sample = dataset[0]
calibration = dataset.calibration("leftcamera_rightcamera")
lower, upper = dataset.fringe_order_bounds("left")

print(sample["fringe_left"].shape)       # torch.Size([480, 640])
print(calibration.camera_matrix_1.shape) # (3, 3)
```

### 4. 运行测试

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

若本地没有官方 `Dataset/`，依赖真实数据的测试会自动跳过。

## 后续实验

计划对比 WFT/TPR、原始 UCNNet、连续条纹级回归、序数条纹级模型、完整
置信度方法和 Adapter 测试时自适应。主要评价指标包括：

- 相位 MAE、RMSE、Bad-0.1、Bad-0.2 和 Bad-0.5；
- 条纹级准确率、\(2\pi\) 跳变率及不确定度校准；
- 边缘、弱纹理、暗区和饱和区的分区域误差；
- 单场景优化迭代数、耗时、参数量和显存；
- 标定三角测量后的三维 RMSE 与 Chamfer Distance。

关键消融：

| 消融设置 | 验证目标 |
|---|---|
| 连续回归 → 序数概率推断 | 条纹级表示 |
| 去掉条纹级熵 | 不确定度作用 |
| 直接绝对相位 → 有界残差 | 相位细化方式 |
| 固定权重 → 置信度加权 | 鲁棒物理损失 |
| 随机初始化 → 共享初始化 | 收敛速度 |
| 全参数优化 → Adapter | 计算效率 |

## 当前边界

- 尚未完整重新训练原论文 UCNNet，当前 Baseline 数值来自官方保存输出；
- 创新网络已完成前向结构，置信度物理损失、训练入口和 Adapter 尚未完成；
- 当前结果证明数据及评价链路正确，不能据此宣称创新方法优于 Baseline；
- 官方公开场景数量有限，后续实验必须避免跨场景数据泄漏；
- 三维精度需要使用真实标定进行三角测量，不能用相位线性缩放代理毫米误差。

## 辅助四步条纹修复模块

仓库仍保留早期四步相移条纹破损实验代码，包括合成退化、传统修复、
U-Net/Pix2Pix 和物理双头网络。该模块使用统一 NPZ 格式：

```text
fringe [4,H,W], phase [H,W], depth [H,W], valid [H,W]
```

它用于辅助验证相位损失和代码组件，不属于当前 UDPR 官方数据的主实验协议，
也不能把单帧/双目数据重新解释成四步相移序列。相关入口为 `train.py`、
`test.py` 和 `scripts/make_synthetic.py`。

## 引用与致谢

本项目基于以下工作开展：

> H. Yu, X. Chen, R. Huang, et al., “Untrained deep learning-based phase
> retrieval for fringe projection profilometry,” *Optics and Lasers in
> Engineering*, vol. 164, 107483, 2023.

如本项目对你的研究有帮助，请同时引用原始 UDPR/UCNNet 论文及其官方数据。
