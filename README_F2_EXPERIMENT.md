# 基频2分界线补全对比实验

本压缩包用于复现“基频2分界线补全”与 `Exp3_Simulation.m` 内部相移叠加
方案的仿真对比。

## 文件

- `Exp3_Simulation.m`：被对比的原始 MATLAB 程序；
- `fringe_repair/f2_boundary.py`：二维场景、三步成像、内部叠加、边界检测、
  伪边界剔除、保形补全和高频展开；
- `scripts/compare_f2_boundary_source.py`：30×3 场景对比与可视化入口；
- `tests/test_f2_boundary.py`：投影次数、解调、边界补全和展开测试；
- `results/f2_boundary_vs_source/`：本次指标、图像及中文实验报告。

## 环境

建议使用 Python 3.10 或更高版本：

```bash
python -m pip install -r requirements-f2-boundary.txt
```

## 复现实验

在压缩包根目录运行：

```bash
PYTHONPATH=. python scripts/compare_f2_boundary_source.py \
  --output results/f2_boundary_vs_source \
  --scenes 30
```

默认协议：

- 图像大小：180×240；
- 低频：2；
- 高频：48；
- 每种退化：30 个随机二维场景；
- 固定随机种子：20260730；
- 退化类型：五次谐波、分界线缺失、组合退化。

## 单元测试

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q tests/test_f2_boundary.py
```

当前环境没有 MATLAB/Octave，因此源代码方案由 Python 数值等价复现：
保留12种内部相移、整数投影次数、总投影数120以及 `c2/c3` 解调补偿。

