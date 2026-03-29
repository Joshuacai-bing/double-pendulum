# 双摆模拟项目 (Double Pendulum Simulation)

这是一个关于双摆（或单摆）物理系统模拟与分析的 EPQ 项目。本项目包含了相关代码实现、文档报告以及进度日志。

## 项目目录结构

- **`code/`** - 包含项目的所有 Python 源代码。
  - `double_pendulum_simulation.py`: 双摆系统的核心模拟代码。
  - `double_pendulum_student.py`: 适用于学生/学习参考的双摆代码版本。
  - `pendulum_simulation.py`: 基础单摆系统的模拟代码。
  - `helloworld.py`: 基础测试脚本。
- **`doc/`** - 项目报告与活动日志。
  - `main/main.tex`: 项目的最终或核心论文/报告（LaTeX 格式）。
  - `logActivity/log.md`: 项目研究过程的活动与心得日志。
- **`timeline.md`** - 项目的时间线规划与进度甘特图/任务列表。

## 如何运行

本项目的代码主要由 Python 编写。推荐使用 Python 3.x 环境，并可能需要安装相关的科学计算和可视化依赖库（如 `numpy`, `scipy`, `matplotlib` 等）。

进入 `code` 目录并运行相关的模拟脚本：

```bash
cd code
python double_pendulum_simulation.py
```

## 研究内容

本项目旨在通过计算机数值计算（如龙格-库塔法等）和可视化技术，由于双摆系统对初始条件高度敏感的混沌特性（蝴蝶效应），来展示和分析该复杂力学系统的动力学行为。