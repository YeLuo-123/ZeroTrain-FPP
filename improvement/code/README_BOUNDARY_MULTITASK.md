# f=2/f=64 多任务分界线网络

本实现复用 `traditional_unet_phase.py` 中老师给出的 `ConvBlock` 和经典三层 U-Net
编解码结构，将单幅相位回归改为分界线、K2、相位和置信度联合学习。

## 数据生成

```powershell
python baseline/code/generate_phase_boundary_dataset_v2.py --output-dir improvement/result/phase_boundary_dataset_v2 --num-samples 1000 --noise
```

## 训练

```powershell
python improvement/code/train_boundary_multitask.py --data-dir improvement/result/phase_boundary_dataset_v2 --output-dir improvement/result/boundary_multitask_outputs
```

训练程序默认自动启用 CUDA AMP；若需禁用，使用 `--no-amp`。快速链路验证：

```powershell
python improvement/code/train_boundary_multitask.py --data-dir improvement/result/phase_boundary_dataset_v2 --output-dir improvement/result/boundary_smoke --smoke
```

## 推理与物理展开

```powershell
python improvement/code/infer_boundary_multitask.py `
  --weights improvement/result/boundary_multitask_outputs/best.pth `
  --input improvement/result/phase_boundary_dataset_v2/samples/scene_00000.npz `
  --output-dir improvement/result/boundary_inference
```

固定协议为低频2、高频64、频率比32。网络不直接回归高频绝对相位，而是预测
低频 K2 和修复相位，再由双频物理公式恢复 K64 与高频绝对相位。
