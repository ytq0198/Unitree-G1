# Course Project 实验完成报告

## 1. 当前状态

**阶段：Height baseline 已打通，正在提升导航完成度。**

本文档只记录实际完成的代码与实验。待执行方案见 `experiment_design.md`。

## 2. 受控环境

- 本地代码：`D:\暑期综合实践活动\暑期综合实践活动II\labs\course project\course_project\course_project`
- 服务器工作区：`/mnt/localDisk3/weizian/unitree-g1-work`
- Conda：`/mnt/localDisk3/weizian/conda_envs/summer`
- 依赖：Python 3.11.15、`mjlab==1.5.0`、PyTorch 2.13.0
- GPU：4 x NVIDIA RTX A6000；单进程稳定并行数实测为 64 env
- GitHub：`https://github.com/ytq0198/Unitree-G1`

严格遵守服务器约束：仅修改 `/mnt/localDisk3/weizian` 内的本项目工作区与 Conda 环境，不修改 Lab4、Lab7 或其他实验目录。

## 3. 已完成实现

### 作业规定接口

- 83 维 AMP state 按固定顺序拼接。
- 深度图裁剪到 `[0.1, 5.0]` 并映射至 `[0, 1]`。
- 使用 wxyz 四元数提取 yaw，将世界坐标 waypoint 逆旋转到机体系。
- 导航奖励实现 `4*progress + 0.5*reached + 5*success`。
- 平滑惩罚使用动作二阶差分绝对值均值。

### 训练与迁移工具

- 增加 smoke、训练、评估、视频和提交命令行入口。
- 增加训练曲线分析、checkpoint 结构检查和 Lab7 warm-start 转换工具。
- Lab7 actor/critic 输入从 286 维映射到导航任务 285 维，仅删除 yaw-rate command 列；隐藏层、动作头和 AMP 判别器保持兼容。
- 导航距离改善转换为每秒改善率，抵消 RewardManager 的 `dt=0.02` 积分，保持 `student.py` 公式不变。
- waypoint command 限幅为 `0.6 m/s` 局部速度语义，匹配 Lab7 的平面速度控制范围。
- 增加可配置 PPO 学习率、导航权重、速度跟踪权重、平滑权重和 AMP scale。
- 增加 waypoint-conditioned velocity tracking 辅助奖励，待 GPU 空闲后执行 H5 对照。

## 4. 验证记录

| ID | 配置 | 结果 |
|---|---|---|
| LOCAL-001 | CPU 单元测试与语法检查 | 10/10 通过 |
| SERVER-001 | Height smoke，32 env，16 steps | 29 维动作、83 维 AMP、32 次 reset，全部通过 |
| SCALE-001 | 128/256/512 env | 当前 MuJoCo-Warp 在完整 70 m 场景首次 reset 时出现 CUDA illegal memory access；确定使用 64 env |
| H0 | 随机初始化，64 env，1000 iter | 约 153.6 万步；训练稳定，但只学会站立，训练 route progress 约 0.29%，success 0 |
| H1 | Lab7 warm start，导航权重 10，20 iter | 导航信号提高，但学习率过大导致步态快速遗忘；独立评估 progress 0 |
| H2 | 修正 `dt` 缩放，20 iter | 导航奖励约为 H1 的 5 倍；确认时间尺度修正有效 |
| H3 | warm start，`lr=1e-4`，200 iter | 训练 waypoint progress 末值 1.03%，AMP 判别器稳定收敛 |
| H4 | 0.6 m/s 局部命令，`lr=1e-4`，500 iter | 训练 progress 峰值 1.47%，训练日志首次出现非零 success |

## 5. H4 Checkpoint 筛选

统一使用 3 个独立随机起点、每起点最多 1000 steps：

| Checkpoint | Route progress | Route success | Smoothness |
|---|---:|---:|---:|
| `model_250.pt` | 0.01785 | 0 | 0.04075 |
| `model_400.pt` | 0.01759 | 0 | 0.04674 |
| `model_499.pt` | 0.01565 | 0 | 0.04385 |

当前候选为 `model_250.pt`。评估会在首次摔倒时终止，因此约 1.8% 的进度反映稳定行走时间不足，不是 reset 后指标被清零。

## 6. 结论与下一实验

1. 随机初始化 H0 会利用 alive/upright/平滑奖励形成“站立不动”局部最优，不能继续盲目延长。
2. Lab7 warm start 有效，但必须同时对齐输入维度、命令语义和微调学习率。
3. 修正 `dt` 后导航梯度有效；0.6 m/s 命令限幅显著改善未微调策略的平滑度。
4. 当前主要瓶颈是平均约 1 秒后摔倒。H5 将恢复与 Lab7 技能直接对应的局部速度跟踪辅助奖励，验证能否延长存活并提高路线进度。
5. 2026-08-12 18:25，再次检查确认四张 GPU 仍属于其他用户的四卡训练，其中 GPU 1 还运行 31.9 GB 服务。连续采样显示 GPU 0 虽利用率为 0%，但仍是该分布式任务的一部分，因此不抢占。
6. 已部署一次性 H5 资源守候脚本：仅当某张卡显存占用低于 2 GB 且利用率低于 10% 时自动启动，并在本项目 `outputs/job_logs` 中记录状态与日志。

H5 完成后将筛选 checkpoint，执行正式 10 起点、5000 steps 评估，再生成视频、提交包并运行 grading toolkit。
