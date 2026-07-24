# 面向相移结构光三维重建的物理约束条纹断裂修复：实验报告

## 1. 相关工作与研究假设

四步相移法从正交强度差恢复相位：
\[
\phi=\operatorname{atan2}(I_4-I_2,I_1-I_3).
\]
Feng 等的 fringe-pattern-analysis 工作证明 CNN 可从单帧条纹学习相位；Yin 等进一步把条纹成像模型写入学习约束，核心价值是减少仅靠数据拟合产生的非物理解。高动态范围研究则说明饱和并非普通随机缺失：它与反射率、曝光和投影强度相关。附件中的自监督相位展开、无训练相位恢复和单模型自恢复论文分别支持无标签约束、跨场景物理一致性及 fringe-to-fringe 辅助任务。

本项目的创新假设是：共享编码器同时预测四幅修复条纹与圆周相位，配合多帧重投影和断裂边界梯度约束，比“逐帧修图再 PSP”更能保存正交相移关系。相位头预测 sin/cos 后用 atan2，避免把 \(-\pi\) 和 \(\pi\) 当作相距 \(2\pi\)。

## 2. 公开数据核验

### DL-SLP / Gaussian Depth Disc

- 官方论文：`https://pmc.ncbi.nlm.nih.gov/articles/PMC11355059/`
- 官方归档：`https://zenodo.org/records/10404434`
- 归档大小：13.8 GB；本次已下载官方 README，未下载整个包。
- 论文报告 10,000+ 物理数据对，采集相机为 640×480、12-bit 灰度；系统内部用四步相移产生展开相位/高度，但发布学习对是变形条纹—高度图。
- 包含高度 ground truth；不能据此声称每个发布样本都暴露四幅原始相移帧，必须下载 HDF5 后以键名复核。

### SFNet / SynthFringe

- 官方项目：`https://wonhoe-kim.github.io/SFNet/`
- 官方数据：项目页指向 Dropbox；服务器报告压缩下载约 42.33 GB，本次未拉取。
- 输入为两幅不同频率条纹，目标为绝对相位；公开资料表明约 18,000 样本/500 个对象。
- 它不是四步等相移数据。项目页未承诺 metric depth 标签，故本实验只把它用于相位和跨数据集评估。

### Middlebury 2003

- 官方页：`https://vision.middlebury.edu/stereo/data/scenes2003/`
- 本次实际下载 `cones-png-2.zip`（786,747 bytes），解析得到 4 幅 450×375 PNG：两个 RGB 视图和两个 disparity GT。
- 无 phase GT、无投影条纹。它的 disparity 是用结构光测出的立体真值；适合几何 benchmark，不适合训练四步相移修复。

这一区分决定了论文实验必须分 native protocol 报告，不能把不同模态混成一张无说明的主表。

## 3. 环境、代码和训练

已验证环境：Python 3.13.13、PyTorch 2.13.0+cpu、31 GiB RAM、无 CUDA。模型训练建议 Python 3.11、PyTorch 2.4+ 和 12 GB 以上 GPU。代码入口：

- `scripts/download_data.py`：带来源 manifest 的下载；
- `scripts/inspect_data.py`：数量、分辨率、phase/depth 候选统计；
- `scripts/convert_fourstep.py`：真实四步数据统一成 NPZ；
- `fringe_repair/degradation.py`：5/10/20/30% 不规则断裂、饱和、块遮挡、高斯和椒盐噪声；
- `train.py`, `test.py`：训练、消融、指标与效率；
- `paper/experiment.tex`：可直接纳入论文。

默认权重为 \(\lambda_s=0.2,\lambda_g=0.1,\lambda_d=0.1,\lambda_f=0.2\)，AdamW 学习率 \(2\times10^{-4}\)，30 epochs，batch 4。正式实验应按对象划分 70/15/15%，运行三个种子，并在验证集固定超参数。

## 4. 已执行验证与结果边界

单元测试 2/2 通过：四步合成—PSP 相位往返误差小于 \(10^{-5}\)，损坏生成严格可复现。12 个 64×80 合成样本上完成 1 epoch CPU smoke training。

| Method | PSNR | SSIM | Phase RMSE (rad) | Depth proxy RMSE | Chamfer proxy |
|---|---:|---:|---:|---:|---:|
| damaged PSP | 11.604 | 0.741 | 0.819 | 0.436 | 0.355 |
| Telea–PSP | 16.404 | 0.857 | 0.663 | 0.437 | 0.357 |
| NS–PSP | 17.116 | 0.880 | 0.596 | 0.435 | 0.355 |
| Ours, 1 epoch smoke | 9.584 | -0.001 | 0.850 | 0.415 | 0.333 |

这些数值只证明程序链路可运行。1 epoch 的模型尚未收敛，且 depth/Chamfer 是未标定归一化代理，绝不能作为论文结论。CPU 上该模型为 473,286 参数、47.22 ms/帧、21.18 FPS（64×80 输入）；不同硬件/分辨率必须重新测量。

## 5. 消融矩阵

正式执行项为：去掉 shift physical loss；去掉 phase loss；去掉 depth geometry loss；四种损坏比例；DL-SLP、SynthFringe 与真实四步数据的 native/cross-domain 测试。每项记录 PSNR、SSIM、圆周 Phase RMSE、标定后的毫米 Depth RMSE、3-D Chamfer、参数、FLOPs、同步后的 GPU latency/FPS。

## 6. 论文级分析

普通修复可能 PSNR 高而三维误差不降，因为 PSNR 奖励逐像素强度接近，PSP 却依赖四帧正交差的比值；四帧间很小但相关的偏差就会明显旋转相位向量。重投影损失把预测相位重新生成四帧，直接约束跨帧关系，因此更接近测量目标。

物理项不能替代标定。真实 depth loss 应由相机射线、投影仪相位平面和三角测量得到，而不是相位线性缩放；当前通用实现只给出了可替换接口。跨数据集性能还会受 gamma、曝光、频率、12-bit 到 8-bit 量化和物体分布影响，建议加入辐射线性化、频率条件编码和不确定度头。

工业应用最有价值的场景是反光金属、深孔遮挡和在线检测，但部署前必须增加：相机/投影仪联合标定、HDR 多曝光真值、无效像素置信度、时序一致性、标定漂移测试和真实工件重复性/再现性（Gage R&R）。
