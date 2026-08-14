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

`RewardManager` 会乘以 `dt=0.02`，因此环境向 `student.navigation_reward()` 传入沿当前折线路段投影得到的有符号路径推进率 `(s[t]-s[t-1])/dt`，并把一次性的 waypoint/success 指示量转换为除以 `dt` 的事件脉冲。积分后仍分别得到真实路径推进距离、0.5 到达奖励和 5.0 成功奖励。路径定义在 waypoint 切换处连续，也不会把偏离路线但靠近航点的径向运动误计为沿路线推进。

总训练信号还包括 waypoint 速度跟踪、alive/upright、关节限位、自碰撞、AMP 风格奖励和动作平滑惩罚。可选 gait-preservation 模块恢复 Lab7 中与二维命令兼容的躯干角速度、角动量、足端净空和落脚冲击约束；它用于消融，不改变正式路线指标。

安全项使用独立 `fell_over` 终止标志，不能把 route success 或 timeout 一并当作失败惩罚。由于环境按 `dt=0.02` 积分，跌倒权重必须通过消融确定，避免过弱时容忍短程摔倒，也避免过强时退化为原地站立。

## 4. 训练路线

### 阶段 A：Height baseline

- 完成五个规定函数、smoke、单元测试和提交导出。
- 从 Lab7 PPO locomotion checkpoint 迁移大网络 `512-256-128`。
- 使用 aligned-heading 课程学习获得可行走导航策略，再转入随机朝向。
- 多 checkpoint 统一评估，不默认采用最后一次保存。

### 阶段 B：稳定性与导航优化

- 对比命令速度、速度跟踪权重、探索标准差和 gait-preservation scale。
- 对航点处 90 度转弯做 `min_turning_speed=0.1/0.0` 消融。原命令在目标位于侧方时仍强制至少 0.1 m/s 前进；机器人以最大 0.25 rad/s 转向约需 6 s，可能从半宽约 1 m 的台阶/环沟中心平台走落。允许原地转向的分支必须先验证旧 checkpoint 的零样本效果，再从 H49 做短程微调；训练、评测、视频与提交适配器必须使用同一参数。
- 速度课程同时满足路线时限可达性：正式评测最多 5000 步（100 s），随机路线总长约 40-80 m；因此 `0.35 m/s` 即使全程无误也无法完成长路线。先在最低地形难度筛选可稳定转向的速度，再把最佳速度迁移到障碍课程，避免把物理时限误判为导航失败。
- 课程微调固定学习率，禁止 adaptive 调度自动放大低学习率；每个难度仅短训练并按独立评测早停。
- 沟槽与台阶分别维护专家 checkpoint：沟槽采用适度 AMP，台阶采用更强 AMP 与 Lab7 步态保持；单项达标后再通过混合场景训练或 actor 插值整合。
- 每组先做 3 起点筛选，再对候选做 10 起点、5000 步正式评估。
- 正式障碍课程使用 `--training-scenes 4`：四张完整地图作为 terrain columns 均匀分配给并行环境，WaypointCommand 依据每个环境的 terrain type 选择对应路线，使同一次 PPO 更新包含跨地图经验。单地图训练仅用于技能诊断，不再作为泛化模型的主要训练方式。
- 主要指标为 route progress；同时报告存活步数、跌倒率和平滑度。
- 在 Course Project 自身三类地形上使用难度课程：逐步增加 pile 高度、platform gap 宽度和 pyramid stairs 高度；正式评测固定为完整难度 1.0。
- 使用训练专用的首段随机起点偏移，在同一批环境中混合中心起点和 tile 边界样本；偏移始终小于半个 tile，正式评测保持原始中心起点。
- `platform_gap` 使用四向可进入的中心平台环形沟槽，`pyramid_stairs` 使用四向同心台阶；地形几何必须与 route graph 的入口方向一致。专项课程从窄沟/低台阶逐步增加到正式的 0.25 m 沟宽和 0.08 m 阶高。

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

## 7. 算法正确性门槛

正式训练前必须同时通过以下检查：

1. actor/critic 观测布局与 warm-start 删除列一致：Lab7 为 286/298 维，大作业为 285/297 维，且只删除命令中的 yaw-rate。
2. waypoint 命令与速度跟踪均使用 yaw-local 坐标系；地形坡度导致的 roll/pitch 不得改变二维导航命令语义。
3. 路径推进使用折线段投影，waypoint 切换前后连续；到达和成功标志只奖励一次。
4. AMP policy transition 使用真实 terminal AMP state，50 Hz 专家相邻帧与环境 0.02 s 控制周期一致。
5. 固定前向穿越评测必须令初始 yaw 与世界 +X 对齐；通用步态训练中的推进奖励则沿命令方向计算，不能混用世界轴与机体系目标。
6. Course Project 评测同时报告路线百分比、沿路米数、最近航点距离、首航点到达率和完整成功率，避免不同路线总长度掩盖真实运动能力。
7. 从合格步态或导航 checkpoint 继续训练时，对 actor/critic 观测归一化做冻结消融；归一化统计更新不受 PPO 学习率限制，可能在地形分布变化时破坏已有策略。
