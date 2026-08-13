# Course Project 实验设计

## 1. 目标与评分对齐

本项目训练 Unitree G1 在随机生成的 5x5 复杂地形中，根据机体坐标系下的下一个 waypoint，直接输出 29 维关节动作。优先完成可提交的 Height baseline，再提高路线推进率并完成 Depth 扩展。

| 评分项 | 分值 | 实现与验证 |
|---|---:|---|
| 提交格式与有限动作 | 20 | `policy.pt`、`model.py`、`student.py`；导出后检查形状与有限值 |
| Route progress | 30 | 机体坐标 waypoint、稠密推进奖励、10 个随机起点评估 |
| Route success | 10 | waypoint 到达奖励、路线完成奖励与终止条件 |
| AMP | 10 | 83 维 AMP state、判别器与风格奖励 |
| Depth | 10 | 60x80 深度图、[0.1, 5.0] 裁剪归一化与 CNN |
| Smoothness | 20 | 一阶 action-rate 与二阶动作差分约束 |

## 2. Lab7 与大作业的关系

Lab7 学习的是速度指令跟踪下的粗糙地形步态；大作业新增长路线、随机初始朝向、顺序 waypoint、路线指标以及二维局部导航命令。因此不能只重复 Lab7 方法，而要解决三项迁移问题：

1. 将世界坐标 waypoint 变换为机体坐标，并限制为局部期望速度。
2. 将 Lab7 的三维 `[vx, vy, yaw_rate]` actor 输入迁移为二维 `[vx, vy]`，保留其余观察和网络参数。
3. 采用低学习率、低探索噪声和朝向课程学习，避免导航梯度破坏已有步态。

## 3. 奖励与约束

作业规定的导航奖励保持不变：

```text
r_nav = 4 * progress + 0.5 * waypoint_reached + 5 * route_success
```

`RewardManager` 会乘以 `dt=0.02`，因此环境向 `student.navigation_reward()` 传入距离改善率 `(d[t-1]-d[t])/dt`，积分后仍得到真实距离改善。

总训练信号还包括 waypoint 速度跟踪、alive/upright、关节限位、自碰撞、AMP 风格奖励和动作平滑惩罚。可选 gait-preservation 模块恢复 Lab7 中与二维命令兼容的躯干角速度、角动量、足端净空和落脚冲击约束；它用于消融，不改变正式路线指标。

## 4. 训练路线

### 阶段 A：Height baseline

- 完成五个规定函数、smoke、单元测试和提交导出。
- 从 Lab7 PPO locomotion checkpoint 迁移大网络 `512-256-128`。
- 使用 aligned-heading 课程学习获得可行走导航策略，再转入随机朝向。
- 多 checkpoint 统一评估，不默认采用最后一次保存。

### 阶段 B：稳定性与导航优化

- 对比命令速度、速度跟踪权重、探索标准差和 gait-preservation scale。
- 每组先做 3 起点筛选，再对候选做 10 起点、5000 步正式评估。
- 主要指标为 route progress；同时报告存活步数、跌倒率和平滑度。

### 阶段 C：Depth 与高分扩展（暂停）

- 保持 Height actor 的本体感知和导航结构，引入 60x80 depth encoder。
- 优先采用 Height teacher 到 Depth student 的蒸馏/微调，降低从零训练成本。
- 对 Height/Depth 做相同随机起点评估，报告复杂地形上的增益与计算开销。

Depth 只有在 Height 同时通过以下硬门槛后才启动：粗糙地形基础步态能够连续前进至少 6 m；大作业 10 起点评估中能够稳定到达首个 waypoint；route success 非零且单 episode 视频能连续展示复杂地形行走。

### 阶段 D：最终交付

- 固定最佳 commit 和 checkpoint。
- 生成视频、训练曲线、路线进度与消融图表。
- 构建严格三文件提交包，并运行课程 grading toolkit。
- 报告 motion 数据来源、许可、哈希、随机种子和完整训练配置。

## 5. 实验规范

每次正式实验记录日期、Git commit、服务器路径、GPU、seed、环境数、迭代数、关键权重、运行目录、候选 checkpoint、独立评估指标与异常情况。大型 checkpoint 和 TensorBoard 日志保留在服务器；GitHub 只保存代码、轻量结果和 Markdown 文档。

## 6. 下一阶段优先级

1. 重训可靠的 Height locomotion，先通过平地固定速度，再通过 6 m 粗糙地形门槛。
2. 将合格步态迁移到 waypoint 导航，要求首 waypoint 到达和非零 route success。
3. 完成 AMP scale、平滑权重和导航课程消融，并生成真实单 episode 视频。
4. 仅在 Height 达标后启动 Depth smoke 和 Height-to-Depth 初始化。
