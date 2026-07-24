# 物理约束相移条纹断裂修复：可复现实验

## 官方 UDPR/UCNNet 数据

`Dataset/` 是论文 *Untrained deep learning-based phase retrieval for
fringe projection profilometry* 的官方数据，而不是四步相移序列。新增代码将
该协议与下文原有的四步修复实验严格分开：

- `fringe_repair/udpr_io.py`：兼容 classic MAT、MATLAB v7.3/HDF5，配对
  左右视图、两阶段输入、真值、mask 和官方输出；同时解析相机—相机及
  相机—投影仪标定；
- `scripts/evaluate_udpr_baseline.py`：在有效 mask 内复算官方输出的 MAE、
  RMSE、阈值错误率及 stage-1 条纹级准确率；
- `fringe_repair/udpr_models.py`：共享权重的双目两阶段模型。第一阶段使用
  序数分布和 soft-argmax 预测条纹级及不确定度，第二阶段预测限制在
  `[-pi, pi]` 内的绝对相位残差及置信度。

复现全部三个官方子集：

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_udpr_baseline.py \
  --root Dataset --output results/udpr_official_baseline.json
```

加载一个阶段的数据和标定：

```python
from fringe_repair.udpr_io import OfficialUDPRDataset

data = OfficialUDPRDataset("Dataset", section="3.2.1", stage=1)
sample = data[0]                  # 所有图像张量均为 [480, 640]
calibration = data.calibration() # 常规列向量/OpenCV K、R、t
lower, upper = data.fringe_order_bounds("left")
```

## 项目汇报说明

### 1. 项目题目

推荐中文题目：

> 基于序数条纹级推断与置信度物理约束的单帧双目相位恢复

英文题目：

> Confidence-Guided Stereo Phase Retrieval with Ordinal Fringe-Order
> Inference

### 2. 研究背景与问题定义

条纹投影轮廓术通过投影仪向物体表面投射正弦条纹，相机采集受物体
高度调制后的条纹，再从条纹中恢复绝对相位并重建三维形貌。传统高精度
方法通常需要采集多幅相移条纹和 Gray Code 图案，测量精度高，但不适合
运动物体和高速场景。

单帧条纹投影只需采集一幅图像，具有较高的测量速度，但同时面临两个
核心问题：

1. 单幅条纹计算得到的包裹相位精度有限；
2. 绝对相位存在 \(2\pi\) 周期模糊，需要准确判断每个像素的条纹级。

本项目研究如何利用双相机的相位和几何关系，在不使用目标场景相位真值
的情况下，从左右两幅单帧条纹中恢复准确的绝对相位。

### 3. Baseline 方法

本项目以论文 *Untrained deep learning-based phase retrieval for fringe
projection profilometry* 提出的 UDPR/UCNNet 为 baseline。该方法不依赖
大规模带标签数据预训练，而是针对每个待测场景，利用双目系统的物理
一致性优化网络。

第一阶段输入左右单幅条纹 \(I_L,I_R\) 以及 WFT 计算的粗包裹相位
\(\phi_L^c,\phi_R^c\)，预测左右条纹级 \(K_L,K_R\)，并计算粗绝对相位：

\[
\Phi_i^c=\phi_i^c+2\pi K_i.
\]

第二阶段输入左右条纹和粗绝对相位，进一步得到精细绝对相位
\(\hat\Phi_L,\hat\Phi_R\)。

Baseline 使用三类场景无关约束：

- **相位一致性**：同一三维点在左右相机中应具有相同的投影相位；
- **结构一致性**：左右对应图像区域应具有相似的局部结构；
- **三维一致性**：相机—投影仪与双目系统计算的几何对应应一致。

因此，Baseline 可以不使用当前场景的相位真值，通过物理一致性完成
单场景优化。

### 4. Baseline 的主要不足

#### 4.1 条纹级被作为普通连续量回归

条纹级 \(K\) 本质上是离散且有序的变量。连续回归没有显式描述各候选
条纹级的概率，也无法判断当前结果是否存在歧义。在遮挡、弱纹理和低
质量条纹区域，条纹级错误会直接造成约 \(2\pi\) 的相位跳变。

#### 4.2 不同质量像素使用近似固定的损失权重

暗区、饱和区、遮挡区和弱纹理区域的可靠程度明显不同。尤其在遮挡区域
不存在有效双目对应，在弱纹理区域使用 SSIM 寻找对应也容易产生歧义。
固定权重会使错误对应点干扰网络优化。

#### 4.3 每个场景都需要重新优化

Baseline 针对每个场景从随机参数开始优化。该方法不依赖训练集，但计算
成本较高，难以直接应用于实时三维测量。

#### 4.4 第二阶段直接预测大范围绝对相位

第二阶段的实际任务是修正粗绝对相位，而不是重新估计完整绝对相位。
直接回归大范围相位会增加网络的优化难度。

### 5. 项目总体方案

本项目保留 Baseline 的双目物理约束思想，重新设计条纹级表示、相位
细化方式和像素可靠性建模：

```text
左右单幅条纹
      ↓
WFT 粗包裹相位
      ↓
第一阶段：序数条纹级概率预测
      ↓
粗绝对相位
      ↓
第二阶段：有界相位残差预测
      ↓
置信度加权双目物理优化
      ↓
精细绝对相位与三维重建
```

### 6. 计划验证的创新点

> 以下内容是已经完成模型骨架、但仍需要训练和消融实验验证的方法设计，
> 不能作为已经得到实验支持的结论。

#### 创新点一：基于序数概率的条纹级推断

将所有候选条纹级表示为概率分布：

\[
p(k\mid I,\phi^c),\qquad k\in[K_{\min},K_{\max}],
\]

并通过 soft-argmax 得到可微条纹级：

\[
\hat K=\sum_k k\,p(k).
\]

该方法利用了条纹级的离散、有序属性，并避免无约束连续回归产生不合理
结果。进一步使用归一化概率熵表示条纹级不确定度：

\[
U_K=-\frac{1}{\log N}\sum_k p(k)\log p(k).
\]

概率集中时不确定度较低；多个候选条纹级概率接近时，不确定度较高。
因此，模型能够同时输出条纹级及其可信程度。

#### 创新点二：有界绝对相位残差细化

第二阶段不直接预测完整绝对相位，而是预测粗相位的残差：

\[
\hat\Phi=\Phi^c+\Delta\Phi,\qquad
\Delta\Phi=\pi\tanh(z).
\]

由此保证：

\[
\Delta\Phi\in[-\pi,\pi].
\]

其物理含义是第二阶段只负责在当前条纹周期内修正粗相位，不应任意跨越
多个周期。该设计能够缩小输出空间、降低优化难度，并减少异常的跨周期
相位跳变。

#### 创新点三：置信度引导的物理一致性优化

计划使用以下因素构造像素级置信度：

\[
C=C_{\mathrm{mask}}
C_{\mathrm{intensity}}
C_{\mathrm{texture}}
C_{\mathrm{order}}
C_{\mathrm{phase}}.
\]

其中：

- \(C_{\mathrm{mask}}\)：官方数据提供的有效区域；
- \(C_{\mathrm{intensity}}\)：降低暗区和饱和区的权重；
- \(C_{\mathrm{texture}}\)：降低弱纹理区域的匹配权重；
- \(C_{\mathrm{order}}\)：根据条纹级概率熵计算；
- \(C_{\mathrm{phase}}\)：根据第二阶段预测的相位方差计算。

相位、结构和三维一致性损失统一采用置信度加权：

\[
\mathcal L_{\mathrm{phy}}=
\frac{\sum_x C(x)\rho(r(x))}
{\sum_x C(x)+\epsilon},
\]

其中 \(\rho\) 可使用 Huber 或 Charbonnier 鲁棒函数。该设计用于降低
遮挡、弱纹理和异常曝光像素对优化过程的干扰，并直接针对 Baseline
论文中提到的弱纹理区域失效问题。

#### 创新点四：共享初始化与轻量测试时自适应

为减少 Baseline 逐场景随机初始化带来的计算开销，后续计划：

1. 学习跨场景共享的网络初始化；
2. 测试时冻结主要特征提取网络；
3. 只优化低秩 Adapter、归一化参数或相位残差头；
4. 根据物理一致性残差进行自适应提前停止。

加入共享初始化后，该方法不应再称为“完全无训练”，更准确的表述是：

> 无目标域标签的测试时自适应相位恢复。

### 7. 官方数据集

当前使用 Baseline 论文公开的约 7.5 GB 官方数据：

| 数据部分 | 场景数 | 主要用途 |
|---|---:|---|
| section 3.2.1 | 100 | 充足且相似场景实验 |
| section 3.2.2 | 15 | 有限场景实验 |
| section 3.2.3 | 75 | 树皮、叶片等跨类别实验 |

公开数据共包含 190 个实验场景，每个场景包含：

- 左右相机单幅条纹；
- 左右 WFT 粗包裹相位；
- 左右条纹级真值；
- 左右粗绝对相位；
- 左右绝对相位真值；
- 左右有效区域 mask；
- Baseline 两阶段输出；
- 双相机及相机—投影仪标定参数。

数据图像分辨率为 \(640\times480\)。公开文件主要对应论文实验子集，
不包含论文所述全部 675 个采集场景，因此后续不能声称使用了完整的
675 场景数据集。

### 8. 当前已完成工作

#### 8.1 官方数据加载

已经实现 classic MAT、MATLAB v7.3 和普通 HDF5 的统一读取，处理了
MATLAB/Python 数组方向差异，并完成左右视图、两阶段输入、真值、mask
和官方输出的严格配对。

#### 8.2 标定参数解析

已经能够解析左相机—右相机、右相机—左相机、左相机—投影仪和右相机—
投影仪四组标定，并将 MATLAB 行向量形式转换为常见 OpenCV 列向量形式。

#### 8.3 Baseline 指标复现

使用左右有效 mask 复算官方 Stage-2 输出，结果如下：

| 数据部分 | 样本数 | 双视图 MAE |
|---|---:|---:|
| section 3.2.1 | 100 | 0.1098 rad |
| section 3.2.2 | 15 | 0.0944 rad |
| section 3.2.3 | 75 | 0.0813 rad |

复算结果与论文报告的约 \(0.10/0.08/0.07\) rad 处于相同量级，说明文件
配对、图像方向和 mask 语义基本正确。数值差异可能来自论文采用逐场景
平均、特定视图或未公开后处理，当前不能声称已经逐位复现论文表格。

#### 8.4 两阶段模型骨架

目前已经实现：

- 左右视图共享网络参数；
- 56 个候选条纹级的概率输出；
- soft-argmax 条纹级计算；
- 条纹级熵不确定度；
- 像素级条纹级上下界约束；
- 限制在 \([-\pi,\pi]\) 内的相位残差；
- 相位方差和置信度输出。

当前单元测试、六个数据分区检查和模型前向测试均已通过。

### 9. 后续实验设计

#### 9.1 对比方法

- WFT/TPR；
- 原始 UCNNet；
- 普通连续条纹级回归网络；
- 序数条纹级网络；
- 完整置信度引导方法；
- 共享初始化和测试时 Adapter 方法。

#### 9.2 评价指标

- 相位 MAE、RMSE；
- Bad-0.1、Bad-0.2、Bad-0.5 错误像素比例；
- 条纹级准确率和 \(2\pi\) 跳变率；
- 概率熵与实际预测错误的相关性；
- 普通、边缘、弱纹理、暗区和饱和区分区域误差；
- 单场景迭代数、运行时间、参数量和显存占用；
- 标定三角测量后的三维 RMSE 和 Chamfer Distance。

#### 9.3 消融实验

| 消融设置 | 验证目标 |
|---|---|
| 连续回归 → 序数概率推断 | 验证条纹级建模 |
| 去掉条纹级熵 | 验证不确定度作用 |
| 去掉图像质量置信度 | 验证异常曝光处理 |
| 直接相位回归 → 有界残差 | 验证相位细化方式 |
| 固定损失权重 → 置信度加权 | 验证物理损失设计 |
| 随机初始化 → 共享初始化 | 验证收敛速度 |
| 全参数优化 → Adapter | 验证计算效率 |

### 10. 当前项目边界

1. 当前完成的是官方保存输出的指标复算，还没有完整重新训练原论文
   UCNNet；
2. 创新两阶段网络已经完成前向结构，置信度物理损失、训练入口和
   Adapter 测试时优化仍需实现；
3. 当前结果证明了数据和评估链路正确，但尚不能证明创新方法优于
   Baseline；
4. 最终方法有效性必须通过三个数据子集、分区域实验和消融实验验证；
5. 官方公开数据只有 190 个实验场景，监督训练和数据划分需要避免
   数据泄漏。

### 11. 一分钟汇报摘要

> 本项目研究双目单帧条纹投影中的绝对相位恢复。Baseline 利用左右相机
> 之间的相位、结构和三维一致性，对每个场景单独优化一个无训练网络，
> 因而具有较好的跨场景泛化能力。但它将离散条纹级作为连续量回归，
> 无法描述条纹级歧义；同时对不同质量像素采用近似固定的权重，在弱纹理、
> 遮挡、暗区和饱和区容易受到错误对应影响，而且逐场景随机初始化的计算
> 时间较长。
>
> 本项目采用两阶段改进方案。第一阶段将条纹级恢复改为序数概率推断，
> 通过 soft-argmax 得到条纹级，并利用概率熵估计不确定度；第二阶段不
> 直接预测完整绝对相位，而是在粗相位基础上预测限制在正负 \(\pi\) 内
> 的残差。后续将利用有效 mask、图像强度、局部纹理、条纹级熵和相位
> 方差构建像素置信度，对双目物理损失进行自适应加权，并通过共享初始化
> 和轻量 Adapter 减少测试时优化成本。
>
> 当前已经完成官方 190 个场景的数据读取、四组标定解析、Baseline 输出
> 指标复现以及两阶段模型骨架。三个子集的双视图绝对相位 MAE 分别约为
> 0.110、0.094 和 0.081 rad，与论文结果处于相同量级。下一阶段将实现
> 完整物理一致性损失、训练流程和消融实验。

本项目实现四步相移条纹损坏、经典修复、学习式修复和物理约束相位恢复。所有脚本使用固定随机种子；输出结果标明数据来源。当前仓库中的 `results/*_smoke.json` 是用于验证链路的**合成小样本 smoke test**，不是论文最终结果。

## 数据集边界

| 数据集 | 官方内容 | 对本任务的用途 |
|---|---|---|
| DL-SLP / GDD | 10,000+ 真实单幅变形条纹—高度对；Zenodo 包 13.8 GB | 单幅条纹/高度外部验证；它不是天然四步 PSP 数据 |
| SFNet SynthFringe | 两幅不同频率条纹输入、绝对相位标签；官方 Dropbox | fringe-to-phase 与跨频率泛化；不是四步等相移序列 |
| Middlebury 2003 | 9 个 RGB 立体视图、2 个结构光测得的视差图，450×375（quarter） | 几何 benchmark；结构光只用于制作 GT，不含投影条纹 |

因此，三者不能未经说明就直接充当 `{I1…I4}`。真正四步数据用 `convert_fourstep.py`；单幅/双幅数据应按各自协议训练，并作为跨数据集实验。若需要严格的四步真实实验，还应补充自采标定数据或公开四步 FPP 数据。

## 一键复现

```bash
cd /home/fq/paper
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/pip install -r requirements.txt

PYTHONPATH=. .venv/bin/python scripts/download_data.py middlebury --extract
PYTHONPATH=. .venv/bin/python scripts/inspect_data.py data/raw/middlebury
PYTHONPATH=. .venv/bin/python scripts/make_synthetic.py --samples 24
PYTHONPATH=. .venv/bin/python scripts/visualize.py
PYTHONPATH=. .venv/bin/python train.py
PYTHONPATH=. .venv/bin/python test.py --method physics --checkpoint runs/physics/last.pt
```

大型数据按需下载：

```bash
# 约 13.8 GB，解压还需额外空间
PYTHONPATH=. .venv/bin/python scripts/download_data.py dlslp --extract
# Dropbox 可能要求浏览器确认；脚本会检查下载物是否真的是 zip
PYTHONPATH=. .venv/bin/python scripts/download_data.py sfnet --extract
```

四步相移目录转换：

```bash
PYTHONPATH=. .venv/bin/python scripts/convert_fourstep.py \
  --root /path/to/data --names I1.png I2.png I3.png I4.png
```

NPZ 统一契约为 `fringe[4,H,W]`、`phase[H,W]`、`depth[H,W]`、`valid[H,W]`。真实深度评估必须填写相机—投影仪标定模型；代码中的归一化 phase-to-depth 仅是 smoke test 代理，不能作为毫米误差发表。

## 方法和消融

- `psp`：四步 Hariharan/等步长 PSP；
- `telea`, `ns`：逐帧 OpenCV 修复后 PSP；
- `unet`：4→4 条纹恢复；
- `gan`：Pix2Pix PatchGAN + 配对/物理损失；
- `phase`：4→sin/cos→wrapped phase；
- `physics`：条纹与相位双头网络。

```bash
PYTHONPATH=. .venv/bin/python train.py --model unet
PYTHONPATH=. .venv/bin/python train.py --model gan
PYTHONPATH=. .venv/bin/python train.py --ablate physics
PYTHONPATH=. .venv/bin/python train.py --ablate phase
PYTHONPATH=. .venv/bin/python train.py --ablate geometry
```

损坏比例在 `configs/default.yaml` 改为 0.05/0.10/0.20/0.30。完整论文实验至少运行 3 个种子并报告均值±标准差。

## 环境与硬件

本次验证环境是 Python 3.13.13、PyTorch 2.13.0+cpu、无 CUDA。论文训练建议 Python 3.11、PyTorch 2.4+、单卡 12 GB 以上显存；256²、batch 4 可按显存调整。`requirements.txt` 给出可安装下界，完整锁定应在目标 GPU 上导出 `pip freeze`。

详细实验论述和 LaTeX 表格见 [paper/experiment.tex](paper/experiment.tex)。
