# Course Project 实验设计

## 1. 项目目标

首要目标是完成可提交、可复现的 Height baseline，使 Unitree G1 在随机生成的 5x5 复杂地形中，根据机体坐标系下的下一 waypoint，直接输出 29 维关节动作并完成路线。随后围绕评分项优化导航进度、路线成功率、AMP 运动质量、深度感知和平滑度。

## 2. 评分对齐

| 评分项 | 分值 | 实现与证据 |
|---|---:|---|
| 提交格式与 29 维有限动作 | 20 | `policy.pt`、`model.py`、`student.py`；smoke 与 grading toolkit |
| Route progress | 30 | body-frame waypoint、稠密 progress reward、10 个随机起点评估 |
| Route success | 10 | waypoint reached 与 route success 奖励、完整路线终止条件 |
| AMP | 10 | 83 维 AMP state、LSGAN 判别器、风格奖励 |
| Depth | 10 | `[0.1,5.0]` 裁剪归一化、60x80 深度图、CNN 融合 |
| Smoothness | 20 | 一阶 action-rate 与二阶动作差分约束、统一评估 |

## 3. 技术结构

### 3.1 可复用基础

- Lab 4：G1 29 维直接关节控制、PPO 训练与 checkpoint 工作流。
- Lab 7：AMP state、判别器与风格奖励、粗糙地形、Height/Depth 感知链路。
- 同学 Lab 7：仅作为结构、超参数和行为基线参考，不直接当作课程大作业结果。

### 3.2 大作业新增能力

1. 将世界坐标 waypoint 按机器人 yaw 变换到机体坐标系。
2. 将长路线分解为顺序局部 waypoint，并使用 progress、reached、success 三层奖励。
3. 对随机起点和随机路线训练，并以 10 个独立随机起点统一评估。
4. 在导航奖励之外保留安全约束、AMP 风格奖励和二阶平滑约束。

### 3.3 奖励层级

导航奖励：

\[
r_{nav}=4r_{progress}+0.5r_{reached}+5r_{success}
\]

环境奖励由导航项、安全约束和平滑惩罚组成；AmpPPO 再追加按控制步长缩放的 AMP 风格奖励。具体权重以代码配置和每次实验记录为准。

## 4. 分阶段计划

### 阶段 A：Height Baseline

- 完成 `student.py` 五个公式与纯函数测试。
- 在服务器 `summer` 环境完成 Height smoke。
- 使用小规模训练确认 reward、reset、checkpoint 和评估链路。
- 使用实测稳定的 64 environments、1000 iterations 完成首版 baseline，随后按需要增加 iterations，并完成 10 起点评估和提交打包。

完成标准：smoke 通过，训练无 NaN，能导出提交包，grading toolkit 可运行，并得到第一组 route progress、route success、smoothness 指标。

### 阶段 B：Baseline 诊断与稳定性

- 检查 TensorBoard 曲线、摔倒位置、waypoint 切换和动作幅度。
- 对多个 checkpoint 做统一评估，不默认选择最后一个。
- 使用至少 2 个训练 seed 验证主要结论的稳定性。

### 阶段 C：得分优化

- 导航：奖励权重、waypoint threshold、episode horizon、局部目标尺度或归一化。
- 稳定性：安全奖励、动作一阶与二阶约束、AMP scale。
- 感知：在 Height 主线稳定后接入 Depth，先 smoke 再扩大预算。
- 训练效率：当前完整 primitive 场景在课程服务器的 MuJoCo-Warp 下单进程稳定上限为 64 environments；使用多 GPU 独立进程并行 seed/配置，而不在单进程继续增加 nworld。

### 阶段 D：消融与最终提交

- 导航奖励分项消融。
- Smoothness 权重与 AMP scale 对照。
- Height 与 Depth 同协议比较。
- 固定代码 commit，运行 10 起点评估、视频、提交打包和 grading toolkit。

## 5. 实验规范

每次正式训练必须记录：日期、Git commit、服务器路径、GPU、模式、seed、环境数、iterations、steps per env、关键权重、运行目录、最佳 checkpoint、评估指标和异常情况。

大型 checkpoint、视频和 TensorBoard 日志默认保存在服务器，不直接提交 GitHub；GitHub 保存代码、轻量配置、指标 JSON 和两份 Markdown 文档。

## 6. 当前优先级

1. Height baseline 可运行、可评估、可提交。
2. 提高 route progress 和 route success。
3. 在不明显损失导航能力的前提下降低 smoothness。
4. 验证 AMP 权重。
5. 完成 Depth bonus。

## 7. 待验证假设

- H1：body-frame waypoint 可以使策略泛化到不同位置和初始朝向。
- H2：三层导航奖励比仅终点奖励更容易学习长路线。
- H3：适量二阶平滑惩罚能降低动作突变且不显著降低成功率。
- H4：AMP scale 存在导航完成度与自然步态之间的最优区间。
- H5：Depth 在困难地形上的收益能够覆盖其训练成本和优化难度。
