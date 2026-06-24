# prefill_cen_dis_sim

用于比较 LLM `prefill` 场景下集中式和分布式部署形态的离散仿真项目。当前项目只覆盖 `prefill` 请求处理，不包含 decode、PD 分离、网络传输等 e2e 联动仿真。

项目当前重点回答的问题是：在给定实例数、用户数、地域数、到达间隔、Prefill 时长和 KV cache 命中近似配置下，不同部署形态的排队和服务时延表现有什么差异。

## 1. 项目简介

当前支持两类部署形态：

- 集中式：所有实例共享一个全局实例池，请求到达后由最早空闲的任意实例处理。
- 分布式：实例和用户按地域均匀切分，每个地域的请求只能由本地域实例处理。

当前主要统计指标：

- 完成请求数
- 排队请求数和排队比例
- 平均等待时间、P99 等待时间、最大等待时间
- 平均服务时长、P50/P90/P99 服务时长、最大服务时长
- 服务时长达标率

这里的核心口径是：

```python
wait_time = start_service_time - arrival_time
service_duration = wait_time + prefill_time
```

是否算“排队请求”由 `queue_wait_threshold_ms` 决定。只有 `wait_time > queue_wait_threshold_ms` 的请求才会被记为排队。

## 2. 当前能力与非目标

当前能力：

- 生成按用户独立泊松过程到达的 prefill 请求流
- 比较集中式和分布式 prefill 调度效果
- 模拟 Prefill 时间区间采样
- 用 `prefill_zero_prob_by_region` 近似模拟 KV cache 命中后 prefill 直接为 0 的情况
- 输出批量实验结果、单配置诊断图和请求明细

当前非目标：

- 不模拟 decode 阶段
- 不模拟 prefill 和 decode 分离部署
- 不模拟网络链路、带宽、跨节点传输
- 不模拟 session 级连续 decode token 生成

如果你要做的是 decode、PD 分离、网络协同，这个仓库当前只能作为 prefill 阶段的基线参考，不能直接代表完整 e2e 行为。

## 3. 仓库结构与核心文件

- `simulation.py`
  当前主版本仿真模型。包含请求生成、Prefill 时间采样、集中式/分布式调度和统计函数。

- `run_compare.py`
  固定实例数，扫描不同用户规模，并比较不同地域数/部署模式的批量实验入口。

- `run_compare_instances.py`
  固定地域数，比较多组实例数配置，并把结果对齐到同一个“用户数 / 实例数”横轴上。

- `inspect_waits.py`
  单配置诊断入口。用于查看哪些请求发生等待、等待时长分布和部分排队请求明细。

- `run_64_64_1.py`
  单场景最小示例，适合做快速 sanity check。

- `simulation_v1.py` / `run_compare_v1.py`
  旧版实现，保留作历史参考，不建议作为主入口。

## 4. 模型说明

### 4.1 请求到达模型

每个用户的请求到达过程都按泊松过程建模，代码里通过指数分布的到达间隔生成：

```python
intervals = np_rng.exponential(self.arrival_rate, size=est_count)
```

含义：

- `arrival_rate` 单位是毫秒 `ms`
- 它表示“单个用户平均两次请求之间的时间间隔”
- 它不是系统总 QPS，也不是全局请求率

例如：

- `arrival_rate = 100000` 表示单个用户平均每 100 秒发一个请求
- 当用户数固定时，`arrival_rate` 越小，到达越密集，系统压力越大

### 4.2 Prefill 时间模型

默认情况下，Prefill 时间在 `[prefill_time_min, prefill_time_max]` 区间内均匀采样。

项目也支持把部分请求的 prefill 时间直接置为 `0`，近似模拟 KV cache 命中：

- `prefill_zero_prob_by_region = 0.6`
  表示所有地域都有 60% 概率让该请求 prefill 时间为 0

- `prefill_zero_prob_by_region = [0.1, 0.6, 0.3]`
  表示不同地域命中率不同

逻辑顺序是：

1. 先按 `prefill_zero_prob_by_region` 判断是否命中
2. 命中则 `prefill_time = 0`
3. 未命中则在 `[prefill_time_min, prefill_time_max]` 之间采样

### 4.3 资源切分约束

当前分布式仿真假设资源和用户都按地域均匀切分：

```python
instances_per_region = num_instances // num_regions
users_per_region = num_users // num_regions
```

因此必须满足：

- `num_instances % num_regions == 0`
- `num_users % num_regions == 0`

批量脚本和诊断脚本都对这两个条件做了校验。

### 4.4 调度方式

当前主路径不是通用事件队列驱动框架，而是：

- 先批量生成所有请求到达时间
- 再用“实例空闲时间最小堆”调度请求

集中式：

- 所有请求共用一个实例空闲时间堆

分布式：

- 每个地域维护一个独立的实例空闲时间堆

## 5. 环境与依赖

建议环境：

- Python 3.9+
- `numpy`
- `matplotlib`

安装依赖：

```powershell
pip install numpy matplotlib
```

进入项目目录：

```powershell
cd C:\Project\prefill_cen_dis_sim
```

## 6. 快速开始

如果你只是想尽快对齐使用方式，建议按下面顺序：

1. 先跑 `run_64_64_1.py`
   用一个最小场景确认环境和绘图都正常。

2. 再跑 `inspect_waits.py`
   用单配置看清楚“排队”到底发生在什么请求上。

3. 再跑 `run_compare.py`
   比较同一实例规模下，不同用户数和地域数的表现。

4. 最后跑 `run_compare_instances.py`
   比较不同实例规模在统一“用户/实例”维度上的差异。

常用命令：

```powershell
python run_64_64_1.py
python inspect_waits.py
python run_compare.py
python run_compare_instances.py
```

## 7. 入口脚本说明

### 7.1 `run_compare.py`

用途：

- 固定总实例数
- 扫描一段用户规模
- 比较不同 `REGIONS` 配置下的集中式 / 分布式表现

适用场景：

- 你想回答“同一总资源规模下，地域切分之后排队比例会怎么变”
- 你想看用户数增长时系统从轻载到排队恶化的过程

主要配置改哪里：

- `NUM_INSTANCES`
  总实例数

- `USER_RANGE`
  用户扫描范围，例如 `range(120, 521, 10)` 表示从 120 到 520，步长 10

- `REGIONS`
  要比较的地域数列表。`1` 通常代表集中式基线，其它值代表对应地域数的分布式

- `ARRIVAL_RATE`
  单用户平均请求间隔，单位 `ms`

- `PREFILL_TIME_MIN` / `PREFILL_TIME_MAX`
  Prefill 计算时长范围，单位 `ms`

- `SIM_DURATION`
  仿真总时长，单位 `ms`

- `QUEUE_WAIT_THRESHOLD_MS`
  排队判定阈值，单位 `ms`

- `SERVICE_DURATION_TARGET_MS`
  服务时长达标阈值，单位 `ms`

- `PREFILL_ZERO_PROBS_BY_REGION_COUNT`
  不同地域数下的 prefill 为 0 概率配置

输出内容：

- 控制台逐点结果
- 控制台最终摘要
- 一个 CSV 结果文件
- 一个 Matplotlib 图窗口

当前注意点：

- 代码里 `plt.savefig(...)` 被注释掉了
- 也就是说当前脚本会 `plt.show()` 展示图，但默认不会把 PNG 落盘
- README 以下所有说明都以这个当前行为为准

### 7.2 `run_compare_instances.py`

用途：

- 固定地域数
- 比较多组实例数配置
- 用统一的“用户数 / 实例数”横轴看不同实例规模的排队表现

适用场景：

- 你想回答“不同绝对实例数下，按同样用户/实例负载比，排队曲线是否一致”
- 你想比较多个容量档位，而不是只比较一个固定实例数

关键配置：

- `NUM_REGIONS`
  整个脚本共享的地域数，不是每个 case 单独配置

- `CASES`
  每个 case 是一个字典，包含：

```python
{"num_instances": 10, "user_range": range(10 * 5, 10 * 70 + 1, 10)}
```

含义：

- `num_instances` 是该组实验的总实例数
- `user_range` 是该组实验扫描的总用户数范围
- `user_range` 通常按实例数等比例设置，这样不同 case 会落在同一条 `用户数 / 实例数` 横轴上，方便横向对比

其他配置与 `run_compare.py` 类似：

- `ARRIVAL_RATE`
- `PREFILL_TIME_MIN` / `PREFILL_TIME_MAX`
- `SIM_DURATION`
- `QUEUE_WAIT_THRESHOLD_MS`
- `SERVICE_DURATION_TARGET_MS`
- `PREFILL_ZERO_PROBS_BY_REGION_COUNT`

输出内容：

- 控制台逐点结果
- 控制台最终摘要
- 一个 CSV 结果文件
- 一个 Matplotlib 图窗口

CSV 组织方式：

- 第一列是 `用户/实例`
- 每组实例数会有一组自己的指标列
- 如果某个 `用户/实例` 比例在某组实例数下不存在，对应列会留空

当前注意点：

- 代码里同样把 `plt.savefig(...)` 注释掉了
- 当前会展示图，但默认不会保存 PNG

### 7.3 `inspect_waits.py`

用途：

- 单配置诊断，不是批量实验脚本
- 用于找出哪些请求发生了等待，以及等待分布长什么样

适用场景：

- 你已经发现某组参数下排队比例异常，想进一步看排队明细
- 你想确认等待是偶发长尾，还是大面积持续排队

命令行参数：

- `--mode`
  `centralized` 或 `distributed`

- `--num-instances`
  总实例数

- `--num-users`
  总用户数

- `--num-regions`
  地域数

- `--arrival-rate`
  单用户平均请求间隔，单位 `ms`

- `--prefill-min`
  Prefill 最小时长，单位 `ms`

- `--prefill-max`
  Prefill 最大时长，单位 `ms`

- `--sim-duration`
  仿真总时长，单位 `ms`

- `--seed`
  随机种子

- `--prefill-zero-probs`
  Prefill 为 0 的概率。支持单个值，例如 `0.6`，也支持按地域逗号分隔，例如 `0.1,0.6,0.3`

- `--preview-limit`
  控制台最多打印多少条排队请求。`-1` 表示打印全部

- `--min-wait-threshold`
  只有 `wait_time > threshold` 的请求才会被当作“排队请求”展示

常用示例：

```powershell
python inspect_waits.py
python inspect_waits.py --mode distributed --num-regions 10 --num-users 10000
python inspect_waits.py --num-instances 8 --num-users 480 --num-regions 1 --arrival-rate 80000
python inspect_waits.py --prefill-zero-probs 0.6
python inspect_waits.py --num-regions 3 --prefill-zero-probs 0.1,0.6,0.3 --preview-limit -1
```

输出内容：

- 控制台摘要
- 排队请求明细表
- 等待时长诊断图

### 7.4 `run_64_64_1.py`

用途：

- 最小示例
- 快速 sanity check

适用场景：

- 第一次拉起项目，先确认依赖、绘图、基础逻辑是否正常
- 改了少量参数后，快速看排队和服务时长分布有没有明显变化

输出内容：

- 控制台打印完成请求数、排队比例、平均等待时间、平均服务时长
- 弹出三张子图：
  - 到达时间 vs 排队时长
  - 到达时间 vs 服务时长
  - 请求到达时间分布

注意：

- 它不是主实验入口
- 它适合做快速验证，不适合做正式批量对比

## 8. 参数配置方法

这一节按“怎么改”和“改了会怎样”来说明。

### 8.1 `num_instances`

含义：

- 系统总实例数

单位：

- 个

怎么改：

- 在 `run_compare.py` 里改 `NUM_INSTANCES`
- 在 `run_compare_instances.py` 里改 `CASES[*]["num_instances"]`
- 在 `inspect_waits.py` 里通过 `--num-instances` 传入

改大 / 改小的效果：

- 改大：容量提升，通常等待时间和排队比例下降
- 改小：更容易排队

约束：

- 必须能被 `num_regions` 整除

### 8.2 `num_users`

含义：

- 总用户数

单位：

- 个

怎么改：

- 在 `run_compare.py` 里改 `USER_RANGE`
- 在 `run_compare_instances.py` 里改 `CASES[*]["user_range"]`
- 在 `inspect_waits.py` 里通过 `--num-users` 传入

改大 / 改小的效果：

- 改大：系统到达总请求数增加，更容易进入排队状态
- 改小：整体负载下降

约束：

- 必须能被 `num_regions` 整除

### 8.3 `num_regions`

含义：

- 地域数

单位：

- 个

怎么改：

- 在 `run_compare.py` 里改 `REGIONS`
- 在 `run_compare_instances.py` 里改 `NUM_REGIONS`
- 在 `inspect_waits.py` 里通过 `--num-regions` 传入

改大 / 改小的效果：

- 改大：同样总实例数下，每个地域可用实例数变少，局部排队更容易出现
- 改小：资源池更集中，通常更利于平滑负载

### 8.4 `arrival_rate`

含义：

- 单用户平均请求间隔

单位：

- `ms`

怎么改：

- 在批量脚本里改 `ARRIVAL_RATE`
- 在诊断脚本里用 `--arrival-rate`

改大 / 改小的效果：

- 改大：单用户请求更稀疏，系统负载下降
- 改小：单用户请求更密集，系统负载上升

关键提醒：

- 它不是 QPS
- 它描述的是“每个用户平均多久来一个请求”

### 8.5 `prefill_time_min` / `prefill_time_max`

含义：

- Prefill 计算时长区间

单位：

- `ms`

怎么改：

- 在批量脚本里改 `PREFILL_TIME_MIN` / `PREFILL_TIME_MAX`
- 在诊断脚本里用 `--prefill-min` / `--prefill-max`

改大 / 改小的效果：

- 整体调大：实例占用时间更长，排队更容易恶化
- 整体调小：吞吐能力提升
- 拉大区间：服务时长抖动更大

### 8.6 `sim_duration`

含义：

- 仿真总时长

单位：

- `ms`

怎么改：

- 在批量脚本里改 `SIM_DURATION`
- 在诊断脚本里用 `--sim-duration`

改大 / 改小的效果：

- 改大：统计样本更多，结果更稳定，但运行时间更长
- 改小：运行更快，但统计波动更大

### 8.7 `queue_wait_threshold_ms`

含义：

- 认定为“排队请求”的等待阈值

单位：

- `ms`

怎么改：

- 在批量脚本里改 `QUEUE_WAIT_THRESHOLD_MS`
- 在诊断脚本里用 `--min-wait-threshold`

改大 / 改小的效果：

- 改大：更多短等待会被忽略，排队比例下降
- 改小：统计口径更严格，排队比例上升

### 8.8 `SERVICE_DURATION_TARGET_MS`

含义：

- 服务时长达标阈值

单位：

- `ms`

怎么改：

- 在 `run_compare.py` 和 `run_compare_instances.py` 中修改

改大 / 改小的效果：

- 改大：达标率更高
- 改小：达标率更低

关键提醒：

- 它只影响统计和报表
- 它不影响请求调度行为

### 8.9 `prefill_zero_prob_by_region` / `PREFILL_ZERO_PROBS_BY_REGION_COUNT`

含义：

- Prefill 直接为 0 的概率配置，用来近似 KV cache 命中

单位：

- 概率，范围 `[0, 1]`

怎么改：

- 在 `simulation.py` 中，底层参数名是 `prefill_zero_prob_by_region`
- 在 `run_compare.py` 和 `run_compare_instances.py` 中，通过 `PREFILL_ZERO_PROBS_BY_REGION_COUNT` 按地域数配置
- 在 `inspect_waits.py` 中，用 `--prefill-zero-probs`

改大 / 改小的效果：

- 改大：更多请求的 prefill 时间为 0，整体等待和服务时长通常下降
- 改小：更接近纯 prefill 计算负载

示例：

```python
PREFILL_ZERO_PROBS_BY_REGION_COUNT = {
    1: [0.6],
    10: [0.2] * 10,
}
```

## 9. 输出结果说明

### 9.1 当前会产出的内容

`run_compare.py`：

- 控制台表格和最终摘要
- CSV 文件
- 图窗口展示
- 默认不保存 PNG

`run_compare_instances.py`：

- 控制台表格和最终摘要
- CSV 文件
- 图窗口展示
- 默认不保存 PNG

`inspect_waits.py`：

- 控制台摘要
- 排队请求明细
- 图窗口展示

`run_64_64_1.py`：

- 控制台摘要
- 图窗口展示

### 9.2 文件名编码规则

批量脚本输出文件名里常见字段含义如下：

- `i...`
  实例数

- `u...`
  用户范围

- `r...`
  地域配置

- `a...`
  到达间隔 `arrival_rate`

- `p...`
  Prefill 时长区间

- `qw...`
  排队阈值 `queue_wait_threshold_ms`

- `sd...`
  服务时长达标阈值 `SERVICE_DURATION_TARGET_MS`

- `t...`
  仿真时长 `sim_duration`

- `k...`
  Prefill 为 0 的概率配置

例如：

```text
cmp_i80_u2400-10400x200_r1-20_a100s_p350ms-450ms_qw50_sd800_t60m_k0.0.csv
```

可以读成：

- 总实例数 `80`
- 用户范围 `2400` 到 `10400`，步长 `200`
- 地域配置 `1` 到 `20`
- 单用户平均请求间隔 `100s`
- Prefill 时间 `350ms` 到 `450ms`
- 排队阈值 `50ms`
- 服务时长达标阈值 `800ms`
- 仿真时长 `60m`
- Prefill 为 0 概率是 `0.0`

### 9.3 表格里常见指标含义

- `queued_count`
  排队请求数

- `queued_ratio`
  排队请求数 / 总请求数

- `avg_wait`
  平均等待时间

- `max_wait`
  最大等待时间

- `avg_duration`
  平均服务时长

- `within_target_ratio`
  服务时长不超过 `SERVICE_DURATION_TARGET_MS` 的请求占比

## 10. 常见实验改法

### 10.1 看排队定义对结果的影响

把：

```python
QUEUE_WAIT_THRESHOLD_MS = 50.0
```

改成：

```python
QUEUE_WAIT_THRESHOLD_MS = 0.0
```

含义：

- 所有非零等待都算排队

### 10.2 模拟更重的 Prefill 负载

把：

```python
PREFILL_TIME_MIN = 350
PREFILL_TIME_MAX = 450
```

改成：

```python
PREFILL_TIME_MIN = 1450
PREFILL_TIME_MAX = 1750
```

含义：

- 每次 prefill 占用实例更久，更容易排队

### 10.3 模拟 KV cache 命中

例如让所有 10 个地域都有 60% 概率命中：

```python
PREFILL_ZERO_PROBS_BY_REGION_COUNT = {
    1: [0.6],
    10: [0.6] * 10,
}
```

### 10.4 扫更密或更稀的用户规模

例如：

```python
USER_RANGE = range(100, 1001, 50)
```

含义：

- 从 100 用户扫到 1000 用户
- 步长 50

步长越小：

- 曲线更平滑
- 运行时间更长

### 10.5 比较多个实例档位

在 `run_compare_instances.py` 里改 `CASES`，例如：

```python
CASES = [
    {"num_instances": 4, "user_range": range(4 * 5, 4 * 70 + 1, 4)},
    {"num_instances": 8, "user_range": range(8 * 5, 8 * 70 + 1, 8)},
    {"num_instances": 16, "user_range": range(16 * 5, 16 * 70 + 1, 16)},
]
```

## 11. 维护者速览

如果你要继续开发这个仓库，先记住下面几点：

- 核心逻辑在 `simulation.py`
- `LLMDeploymentSimulation` 负责请求生成、Prefill 时间采样和调度
- `compute_stats` 负责把请求结果汇总成报表指标
- 当前主路径是“批量生成到达流 + 实例空闲时间最小堆调度”
- `Event` 数据结构虽然在代码里定义了，但当前主流程并不是通用事件队列驱动框架

关键入口关系：

- `run_compare.py` 调 `simulation.py` 做固定实例数、多用户规模、多地域数对比
- `run_compare_instances.py` 调 `simulation.py` 做固定地域数、多实例档位对比
- `inspect_waits.py` 调 `simulation.py` 做单配置诊断
- `run_64_64_1.py` 调 `simulation.py` 做最小示例

如果你只是要跑实验，不需要先读源码；如果你要改模型行为，优先读：

1. `LLMDeploymentSimulation`
2. `_generate_arrivals_numpy`
3. `_generate_prefill_times`
4. `run_centralized`
5. `run_distributed`
6. `compute_stats`
