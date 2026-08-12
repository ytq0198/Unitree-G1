# Course Project 实验完成报告

## 1. 当前状态

**Height baseline 已完成训练、正式评估和提交导出；当前重点是路线推进优化与 Depth 扩展。**

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

## 4. 正式 10 起点评估

统一使用 10 个不重复随机起点，每次最多 5000 步。

| 候选模型 | Route progress | Success | 平均步数 | 跌倒率 | Smoothness |
|---|---:|---:|---:|---:|---:|
| H9 `model_199.pt` | **0.017691** | 0 | **554.2** | 1.0 | 0.23550 |
| H12 gait=1.0 `model_100.pt` | 0.016022 | 0 | 172.1 | 1.0 | 0.21476 |
| H12 gait=0.5 `model_150.pt` | 0.014884 | 0 | 156.2 | 1.0 | 0.22368 |
| H12 稳定候选 `model_50.pt` | 0.013601 | 0 | 239.3 | 1.0 | **0.18558** |

当前最佳 checkpoint：

```text
/mnt/localDisk3/weizian/unitree-g1-work/outputs/rsl_rl/
course_project_navigation_amp_height/
2026-08-13_01-38-03_h9_std010/model_199.pt
```

## 5. 提交验证

已从 H9 最佳模型生成：

```text
outputs/submission/policy.pt
outputs/submission/model.py
outputs/submission/student.py
```

`policy.pt` 可加载，并对 `[2,285]` 输入产生全为有限数值的 `[2,29]` 输出。该行为与 Lab7 官方导出适配器一致；环境的 joint-position action term 负责后续动作缩放。导出时使用与 checkpoint 一致的 `512-256-128` 网络，已修复旧入口默认按小网络重建的问题。

## 6. 已验证结论

1. Lab7 warm start 有效，但只有同时对齐输入布局、命令语义、网络结构、学习率和探索噪声才不会破坏步态。
2. aligned-heading 证明预训练策略能沿前向 waypoint 推进；随机朝向与侧向控制是当前主要瓶颈。
3. 强速度跟踪、直接改为 `forward_yaw`、过快扩大朝向范围均无效。
4. 恢复物理稳定约束可延长部分模型存活时间，但目前不足以提高正式路线分。
5. 最后 checkpoint 不一定最佳，必须采用多 checkpoint、独立随机起点评估。

## 7. 下一步

- 生成 H9 视频并运行 grading toolkit，锁定当前可交付 baseline。
- 设计分阶段 heading curriculum：窄角度稳定后逐步扩大，并在每阶段使用独立验证早停。
- 完成 Depth smoke、策略结构和迁移训练，优先取得 Depth 功能分。
- 生成 TensorBoard 曲线、消融表和路线可视化，用于最终展示与报告。
