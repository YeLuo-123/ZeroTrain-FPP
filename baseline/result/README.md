# Baseline 复现结果

本目录保存老师代码及原项目确定性方案的复现产物，不包含改进网络结果。

## 目录说明

- `teacher_dual_frequency_smoke/`：老师的 f=2/f=64 双频三步/十二步无噪声仿真。`report.json` 状态为 PASS，频率比为 32，K2/K64 错误像素均为 0。
- `teacher_boundary_dataset_smoke/`：V2 数据生成器的六类最小样本，每类各一张，用于检查 NPZ 字段、标签和可视化。
- `f2_boundary_vs_source_reproduced/`：原项目 f=2/f=48、每类30场景的传统方法、IPS、PCHIP补线、混合及 Oracle 对比结果。
- `traditional_unet_validation_64/`：老师 U-Net 在同训练尺度64×64的新随机样本推理结果。
- `traditional_unet_validation/`：同一权重直接跨到600×800的诊断结果。
- `traditional_unet_validation_report_zh.md`：训练配置、验证指标和泛化结论汇总。

其中第三项是 PDF 确定性基线的复现，频率比为24；老师神经网络数据方案的前两项使用 f=2/f=64，频率比为32。后续改进实验统一采用老师的32倍频率比。
