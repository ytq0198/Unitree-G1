# Course Project 实验完成报告

## 1. 当前状态

**Height 接口 baseline 已完成训练、评估和提交导出，但当前策略尚不能正常完成复杂地形导航；继续优化 locomotion/navigation 是首要任务。**

- 本地代码：`D:\暑期综合实践活动\暑期综合实践活动II\labs\course project\course_project\course_project`
- 服务器工作区：`/mnt/localDisk3/weizian/unitree-g1-work`
- Conda：`/mnt/localDisk3/weizian/conda_envs/summer`
- GPU：4 x NVIDIA RTX A6000；完整场景单进程稳定上限为 64 env
- GitHub：<https://github.com/ytq0198/Unitree-G1>

服务器端只修改 `/mnt/localDisk3/weizian` 内的项目和 Conda 环境，不修改 Lab4、Lab7 或其他实验目录。

## 2. 已完成实现

- 独立实现 83 维 AMP state、深度归一化、yaw 坐标变换、导航奖励和平滑惩罚。
- 构建顺序 waypoint command、随机路线、路线推进/成功指标和 29 维直接关节控制环境。
- 支持 Height/Depth、训练、评估、视频和严格三文件提交导出。
- 支持 Lab7 checkpoint 迁移、可配置网络宽度、低噪声 warm start、朝向课程、命令速度和步态保持约束。
- 评估额外报告存活步数、跌倒率和 timeout rate，防止仅按训练奖励挑选模型。
- 本地测试 10/10 通过，关键 Python 文件通过语法检查。
- Depth smoke 已通过：2 env、16 steps、29 维动作、83 维 AMP state，深度张量为 `[2,1,60,80]`。

## 3. 实验记录

| ID | 主要配置 | 结论 |
|---|---|---|
| H0 | 随机初始化，64 env，1000 iter | 学会站立但不导航，progress 约 0.29% |
| H1-H4 | Lab7 warm start、修正 dt、0.6 m/s 局部命令 | 最佳 3 起点 progress 约 1.8%，但约 1 秒后跌倒 |
| H5 | waypoint velocity tracking weight=2 | 速度跟踪过强，独立评估约 0.8%，不如 H4 |
| H6-H7 | `forward_yaw` 命令与不同学习率 | 预训练策略基本忽略 yaw，训练快速退化，停止该路线 |
| H8 | 大网络 PPO warm start、aligned heading | 原 checkpoint 探索 std 约 0.5，微调时破坏步态 |
| H9 | aligned heading，std=0.1，lr=1e-5 | 建立当前最佳导航模型；随机朝向 3 起点约 1.50% |
| H10 | 从 H9 扩大初始朝向范围 | progress 降低，说明朝向课程跨度过快 |
| H11 | 随机朝向、关闭训练推扰、std=0.05/0.1 | 最佳约 1.16%，未超过 H9 |
| H12 | 恢复兼容的 Lab7 物理约束，速度 0.3/0.4 | 存活最多提高到 352 步，但 3 起点最高约 1.37% |
| H13 | `5e-6` 学习率，初始朝向范围 `0.30-0.60 rad` | 50 次更新后仍退化，说明仅缩小课程跨度不能阻止步态遗忘 |
| H14 | 冻结 actor，仅训练第一层 waypoint 两列 | 保持网络结构兼容；随机朝向、导航权重 2 的正式 progress 为 1.740% |
| H15 | H9 与 H14 waypoint 权重插值 | 75% 适配量取得当前最佳 1.972%，并显著延长存活 |

## 4. 正式 10 起点评估

统一使用 10 个不重复随机起点，每次最多 5000 步。

| 候选模型 | Route progress | Success | 平均步数 | 跌倒率 | Smoothness |
|---|---:|---:|---:|---:|---:|
| H9 `model_199.pt` | **0.017691** | 0 | **554.2** | 1.0 | 0.23550 |
| H12 gait=1.0 `model_100.pt` | 0.016022 | 0 | 172.1 | 1.0 | 0.21476 |
| H12 gait=0.5 `model_150.pt` | 0.014884 | 0 | 156.2 | 1.0 | 0.22368 |
| H12 稳定候选 `model_50.pt` | 0.013601 | 0 | 239.3 | 1.0 | **0.18558** |
| H14 waypoint-only `model_100.pt` | 0.017398 | 0 | 165.2 | 1.0 | 0.26902 |
| H15 waypoint blend 75% | **0.019724** | 0 | **1297.7** | **0.8** | **0.19691** |

H15 的逐起点诊断显示，聚合均值不能理解为正常行走：8/10 起点在 63-289 步内跌倒，最大位移仅约 0.6-2.0 m；2/10 起点达到 5000 步 timeout，其中 seed 103 仅移动 0.22 m，属于站立而非导航。当前成功率仍为 0，策略不能视为完成复杂地形穿越。

当前最佳 checkpoint（H15）：

```text
/mnt/localDisk3/weizian/unitree-g1-work/outputs/warmstart/h15_blends/
h9_h14m199_a75.pt
```

## 5. 提交验证

已从 H9 最佳模型生成：

```text
outputs/submission/policy.pt
outputs/submission/model.py
outputs/submission/student.py
```

`policy.pt` 可加载，并对 `[2,285]` 输入产生全为有限数值的 `[2,29]` 输出。该行为与 Lab7 官方导出适配器一致；环境的 joint-position action term 负责后续动作缩放。导出时使用与 checkpoint 一致的 `512-256-128` 网络，已修复旧入口默认按小网络重建的问题。

旧版 300 帧视频使用自动 reset wrapper，摔倒后会在同一视频中开始新 episode，因此不能作为连续穿越证据。现已修复为首次终止即停止录像；后续报告只使用单 episode 视频，并同时报告路线推进和终止原因。

## 6. 已验证结论

1. Lab7 warm start 有效，但只有同时对齐输入布局、命令语义、网络结构、学习率和探索噪声才不会破坏步态。
2. aligned-heading 证明预训练策略能沿前向 waypoint 推进；随机朝向与侧向控制是当前主要瓶颈。
3. 强速度跟踪、直接改为 `forward_yaw`、过快扩大朝向范围均无效。
4. 恢复物理稳定约束可延长部分模型存活时间，但目前不足以提高正式路线分。
5. 最后 checkpoint 不一定最佳，必须采用多 checkpoint、独立随机起点评估。
6. 仅更新 waypoint 输入列可避免其他 actor 参数漂移；与原策略做 75% 插值后，正式 route progress 相对 H9 提高约 11.5%，且 20% 起点达到 5000 步 timeout。

课程 notebook 指向的 `grading_toolkit/grade.py` 当前既不在本地课程目录，也未在服务器 `/mnt/localDisk3/weizian` 下找到，因此暂时无法运行官方评分器；现有评估严格复用 notebook 规定的 10 个随机起点协议。

## 7. 下一步

- 暂停 Depth，先重训 Height 基础步态并通过 6 m 粗糙地形硬门槛。
- L0 使用 512 env、4 seeds 平地预训练；seed 4101/4103 的后期 checkpoint 已达到 20 s 无早停、速度误差约 0.016-0.020 m/s，平地 walking success 为 37.5%。
- 平地模型直接迁移完整粗糙地形仍约 1 s 跌倒，因此进入 L1 温和粗糙课程，不允许跳级。
- Height 导航只有在首 waypoint 到达、route success 非零且真实单 episode 视频验证后才视为达标。

## 8. 2026-08-13 算法审计与修复

在继续堆叠训练迭代前，对命令、奖励、终止、重置、AMP、迁移和评测链路进行了逐项审计。

确认并修复了三个会直接影响结论的问题：

1. 原导航稠密奖励比较相邻时刻“到当前 waypoint 的距离”。waypoint 更新后，旧航点近距离会与新航点远距离相减，产生错误的大负奖励。现改为沿当前折线路段投影的连续路径位置差。
2. waypoint 是 yaw-local 二维命令，但原速度跟踪使用包含 roll/pitch 的完整 body-frame 速度。现统一为 yaw-local 线速度，保证坡地上的坐标语义一致。
3. Lab7 粗糙阶段含世界 +X 推进奖励，同时初始 yaw 在 `[-pi, pi]` 随机，和 body-frame 前进命令冲突；评测也用世界 +X 位移却未固定 yaw。现训练推进奖励改为沿命令方向，固定穿越评测将初始 yaw 设为 0。

验证结果：

- 本地 12/12 单元测试通过；新增直线段投影与 90 度转弯连续性测试。
- 服务器 Height smoke 通过；实际 actor/critic 维度分别为 285/297，Lab7 为 286/298，warm-start 删除列正确。
- AMP terminal state、timeout bootstrap 和 50 Hz 相邻帧时序正确。seed 5103 的判别器损失由 7.456 收敛到 0.00250，梯度惩罚由 0.639 降至 0.0000369，暂无失衡证据。
- 修正初始朝向后，平地 seed 4101 `model_799.pt` 在 32 环境达到 100% traversal/walking success、20 s 无提前终止，线速度误差 0.0159 m/s。此前 37.5% 是评测坐标错误，不是步态失败。
- 同一模型直接迁移完整粗糙地形仅有 10.4% 平均推进、91.3% 提前终止，因此仍需温和粗糙地形课程训练。
