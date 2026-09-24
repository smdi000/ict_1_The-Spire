# Motion Intent and Trajectory Modeling

[中文](#中文) | [English](#english)

<a id="中文"></a>

# 中文

## 项目概览

这是一个历史动作数据处理与模型训练工程。代码将 HDF5 中的动作序列整理为滑动窗口样本，并在 MindSpore/Ascend 训练入口中联合计算动作类别与未来轨迹回归损失。仓库包含实验脚本、日志和资料；现有材料不足以证明训练流程可从头复现或模型达到特定效果。

## 问题与流程

从代码接口看，工程尝试根据一段多通道动作序列预测动作类别，并回归后续轨迹。训练入口声明输入为 39 通道、50 帧，输出包括 5 类意图 logits 与 72 维轨迹向量；这些是代码配置，不是经验证的模型效果。

~~~mermaid
flowchart LR
    H5[HDF5 动作序列] --> PRE[字段整理 / NaN 处理 / 维度对齐]
    PRE --> WIN[50 帧窗口化]
    WIN --> SPLIT[训练 / 验证数据划分]
    SPLIT --> MODEL[MindSpore 模型训练入口]
    MODEL --> OUT[动作类别 + 72 维轨迹回归]
~~~

## 主要实现

- 从 HDF5 按动作类别与速度字段读取数据，并处理缺失值及不同输入维度。
- 以固定长度滑动窗口构造分类与轨迹学习样本。
- 训练脚本组合交叉熵与均方误差损失，进行多任务训练，并支持 checkpoint 恢复。
- 数据划分脚本按类别分层拆分样本，并从训练子集计算轨迹标准化统计量。

## 仓库结构

- hprocess.py、h5_readme.txt：数据字段与预处理说明。
- train_optimized.py、split_dataset.py：训练及数据拆分脚本。
- resume_*.log：历史运行记录。
- fusion_result.json：计算图融合统计材料。
- 1291242.pdf、3.21_ict.zip、masterdevice_core.zip：历史资料或归档文件，未作为可复现代码依赖。

## 当前状态与限制

训练入口引用的模型与数据集模块未能在仓库根目录结构中确认；公开日志记录了数据加载与训练启动，但随后出现运行中断。因此这里只说明代码声明的流程，不报告准确率、轨迹误差或竞赛结果。原始动作数据、压缩包与论文材料的授权及隐私状态尚未核实；本 README 不展示原始记录或样本。

## 技术栈

Python · NumPy · h5py · scikit-learn · MindSpore · Ascend

---

<a id="english"></a>

# English

## Project Overview

This historical repository contains motion-data preprocessing and model-training experiments. The code converts HDF5 motion sequences into sliding-window samples and defines a MindSpore/Ascend training entry point with joint losses for intent classification and trajectory regression. Scripts, logs, and reference files are present, but the repository does not establish a fully reproducible training setup or a verified model result.

## Problem and Workflow

The code is structured to predict an intent class from a multichannel motion window and regress a future trajectory. The training entry point configures 39 input channels over 50 frames, five intent logits, and a 72-dimensional trajectory output. These are code-level settings, not validated performance claims.

~~~mermaid
flowchart LR
    H5[HDF5 motion sequences] --> PRE[Field mapping / NaN handling / dimension alignment]
    PRE --> WIN[50-frame windows]
    WIN --> SPLIT[Train / validation split]
    SPLIT --> MODEL[MindSpore training entry point]
    MODEL --> OUT[Intent class + 72-d trajectory regression]
~~~

## Implementation

- Reads HDF5 data by exercise and speed fields, handling missing values and varying input dimensions.
- Builds fixed-length sliding windows for classification and trajectory learning.
- Combines cross-entropy and mean-squared-error losses in a multi-task training loop with checkpoint-resume support.
- Splits samples by class and computes trajectory normalization statistics from the training subset.

## Repository Structure

- hprocess.py, h5_readme.txt: data fields and preprocessing notes.
- train_optimized.py, split_dataset.py: training and dataset-splitting scripts.
- resume_*.log: historical run logs.
- fusion_result.json: graph-fusion statistics.
- 1291242.pdf, 3.21_ict.zip, masterdevice_core.zip: historical reference or archive files, not verified reproducible dependencies.

## Status and Limitations

The training entry point imports model and dataset modules that could not be confirmed in the repository root tree. A public log shows data loading and training startup followed by a runtime interruption. Accordingly, this README describes the declared code path and reports no accuracy, trajectory error, or competition result. Authorization and privacy status for the original motion data, archives, and paper materials have not been verified; no raw records or samples are reproduced here.

## Technology

Python · NumPy · h5py · scikit-learn · MindSpore · Ascend
