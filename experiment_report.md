# Course Project 实验完成报告

## 1. 当前状态

**阶段：A0，本地 baseline 公式实现与测试。**

本文档只记录已经完成或实际运行的工作。计划中的实验写在 `experiment_design.md`，未运行的结果不得写成已完成。

## 2. 环境与路径

- 本地代码：`D:\暑期综合实践活动\暑期综合实践活动II\labs\course project\course_project\course_project`
- 服务器代码：`/mnt/localDisk3/weizian/RL learning/course project`
- 服务器：`weizian@10.98.36.128:9961`
- 目标环境：Conda `summer`，`mjlab==1.5.0`
- GPU：4 x NVIDIA RTX A6000，通常可用 2 至 3 张
- GitHub：`https://github.com/ytq0198/Unitree-G1`

## 3. 已完成工作

### 2026-08-12：代码与资料审计

- 确认课程大作业代码根目录和 Notebook 评分合同。
- 确认 `student.py` 包含 5 个必做函数：AMP state、depth 归一化、body-frame waypoint、导航奖励、二阶平滑度。
- 确认正式评估使用 10 个不重复随机起点。
- 确认参考 Lab 7 压缩包包含 Height/Depth `model_599.pt` 和完整 AMP-PPO 结构。
- 确认 Lab 7 模型不能直接作为导航策略，因为课程大作业新增 waypoint 观测和不同任务奖励。

### 2026-08-12：Baseline 纯函数实现

- 已实现 83 维 AMP state 的固定顺序拼接。
- 已实现深度 `[0.1,5.0]` 裁剪并线性映射到 `[0,1]`。
- 已实现 wxyz 四元数 yaw 提取和世界坐标到机体坐标的二维逆旋转。
- 已实现 `4*progress + 0.5*reached + 5*success` 导航奖励。
- 已实现动作二阶差分绝对值均值。
- 已增加独立单元测试和命令行运行入口。

## 4. 验证记录

| ID | 日期 | 范围 | 配置 | 结果 | 状态 |
|---|---|---|---|---|---|
| LOCAL-001 | 2026-08-12 | `student.py` 纯函数 | CPU，7 项 PyTorch unittest + `py_compile` | 7/7 通过，语法检查通过 | 完成 |
| SERVER-UT-001 | 2026-08-12 | `student.py` 纯函数 | 服务器 `summer`，GPU 3 可见 | 7/7 通过 | 完成 |
| SERVER-001 | 2026-08-12 | Height smoke | 32 env，16 steps，GPU 3 | 通过：29 维动作、83 维 AMP state、奖励有限、32 次手动 reset | 完成 |
| TRAIN-SHORT-001 | 2026-08-12 | Height 短训练 | 64 env，10 iterations，24 steps/env，seed 23，GPU 3 | 通过：15,360 steps，约 2,000-2,200 steps/s，生成 `model_0.pt`、`model_9.pt`；AMP 判别器 loss 约 7.6 降至 5.3 | 完成 |
| TRAIN-SCALE-001 | 2026-08-12 | Height 规模基准 | 128/256/512 env，5/5/20 iterations，GPU 2/3 | 三种规模均在首次 reset 前触发 MuJoCo-Warp CUDA illegal memory access；进程退出后 GPU 正常释放 | 失败并归档 |
| EVAL-SHORT-001 | 2026-08-12 | 短训练 checkpoint 加载 | `model_9.pt`，1 start，300 steps，seed 101 | progress 0.0108，success 0，smoothness 0.0213；评估链路通过 | 完成 |
| TRAIN-H0 | 2026-08-12 | Height baseline | 64 env，1000 iterations，24 steps/env，seed 23，GPU 3 | 待运行 | 未开始 |
| TRAIN-H0 | 待运行 | Height baseline | 2048 env，1000 iterations，seed 23 | 待运行 | 未开始 |

## 5. Baseline 指标

尚无课程大作业训练结果。Lab 7 的指标只用于说明运动控制基础，不记入本表。

| Run | Checkpoint | Route progress | Route success | Smoothness | 10 starts |
|---|---|---:|---:|---:|---:|
| 待运行 | - | - | - | - | - |

## 6. 已知风险与观察

- 本地工作区根目录的 `.git` 已确认是空目录，不是有效仓库。需先确认远端仓库结构，再在真正的代码目录建立独立 checkout，避免把课程 PPT、视频和其他 Lab 混入仓库。
- `gh auth` 在受限执行环境中无法联网验证，推送前需在允许网络访问的环境再次确认。
- Height baseline 已通过物理 smoke，但尚未完成 PPO/AMP 短训练和正式训练。
- 首次物理 smoke 暴露出服务器 editable `mujoco_warp` 的 `source.*` 导入路径问题。项目现已通过 `direct_url.json` 自动定位并解码 `%20` 路径；修复只增加运行时只读搜索路径，不修改 Lab 4 依赖目录或用户配置。复测已通过。
- 参考 Lab 7 checkpoint 的 Actor 输入结构与大作业可能不兼容，迁移前必须检查 state dict 和观测维度。
- 完整 70 m primitive-geometry 场景在当前 MuJoCo-Warp 构建下，128 及以上并行 worlds 会在首次 reset 前触发 CUDA illegal memory access；64 worlds 已完成 smoke 和短训练。默认值已据此改为 64，多 GPU 用于独立实验而不是单进程扩容。

## 7. 下一步

1. 启动 64 env、1000 iterations 的正式 Height baseline。
2. 对多个 checkpoint 执行统一评估，不默认选择最后一个。
3. 根据首版学习曲线决定增加 iterations、课程学习或 warm start。
4. 生成视频和提交包，并运行 grading toolkit。
