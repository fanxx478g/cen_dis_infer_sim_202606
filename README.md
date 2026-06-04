# cen_dis_infer_sim_202606

用于比较集中式和分布式 LLM Prefill 部署形态的离散事件仿真项目。项目重点观察不同实例数、用户数、地域数、排队口径和 Prefill 命中概率配置下的：

- 排队请求数与排队比例
- 平均/最大等待时间
- 平均/最大服务时长
- 服务时长达标率

## 目录说明

- `simulation.py`
  当前主版本仿真模型。包含请求生成、集中式/分布式调度、统计函数。

- `run_compare.py`
  当前主要批量实验脚本。遍历多组用户规模与地域配置，输出控制台摘要、CSV 和对比图。

- `inspect_waits.py`
  单配置诊断脚本。用于查看哪些请求发生等待、等待时长分布和部分请求明细。

- `run_64_64_1.py`
  单场景示例脚本，适合快速验证模型和绘图。

- `simulation_v1.py` / `run_compare_v1.py`
  旧版实现，保留作历史参考。

- `*.csv` / `*.png`
  历史实验结果。

- `*.docx`
  背景分析材料，不参与代码运行。

## 环境要求

- Python 3.9+
- `numpy`
- `matplotlib`

安装依赖：

```powershell
pip install numpy matplotlib
```

## 快速开始

进入项目目录：

```powershell
cd C:\Project\prefill_cen_dis_sim
```

### 1. 跑批量对比

```powershell
python run_compare.py
```

输出内容：

- 控制台逐点结果
- 运行完成摘要
- 结果表格 `*.csv`
- 排队比例对比图 `*.png`

### 2. 跑单配置等待诊断

```powershell
python inspect_waits.py
```

可选示例：

```powershell
python inspect_waits.py --mode distributed --num-regions 10 --num-users 10000
python inspect_waits.py --prefill-zero-probs 0.6
python inspect_waits.py --num-regions 3 --prefill-zero-probs 0.1,0.6,0.3
```

### 3. 跑最小示例

```powershell
python run_64_64_1.py
```

## 模型假设

### 1. 请求到达

每个用户的请求到达过程建模为泊松过程，代码里通过指数分布的到达间隔实现：

```python
intervals = np_rng.exponential(self.arrival_rate, size=est_count)
```

- `arrival_rate` 表示每用户平均请求到达间隔，单位 ms
- 例如 `100000 ms` 表示平均每 100 秒来一个请求

### 2. Prefill 计算时间

默认情况下，Prefill 时间在区间 `[prefill_time_min, prefill_time_max]` 内均匀采样。

项目还支持“Prefill 直接为 0”的概率配置，用于近似模拟 KV cache 命中：

- `prefill_zero_prob_by_region=0.6`
  表示所有地域都有 60% 的概率让本次 Prefill 时间为 0

- `prefill_zero_prob_by_region=[0.1, 0.6, 0.3]`
  表示每个地域使用不同概率

逻辑是：

1. 先判断本次请求是否命中“Prefill 为 0”的概率
2. 如果命中，则 `prefill_time = 0`
3. 如果未命中，则仍在区间内随机取值

### 3. 资源切分

地域内资源和用户均匀切分：

- `instances_per_region = num_instances // num_regions`
- `users_per_region = num_users // num_regions`

因此：

- `num_instances` 必须能整除 `num_regions`
- `num_users` 必须能整除 `num_regions`

当前脚本已对这两个约束做参数校验。

### 4. 排队定义

请求等待时间定义为：

```python
wait_time = start_service_time - arrival_time
```

排队统计口径由 `queue_wait_threshold_ms` 控制：

- 当 `wait_time > queue_wait_threshold_ms` 时，记为“排队请求”
- 默认阈值是 `0 ms`
- 例如设置为 `50 ms` 时，等待 50ms 以内的请求不算排队

## 当前主要可配置参数

### `simulation.py` / `simulation_v1.py`

- `num_instances`
- `num_users`
- `num_regions`
- `arrival_rate`
- `prefill_time_min`
- `prefill_time_max`
- `sim_duration`
- `seed`
- `queue_wait_threshold_ms`
- `prefill_zero_prob_by_region`

### `run_compare.py`

- `NUM_INSTANCES`
- `USER_RANGE`
- `REGIONS`
- `ARRIVAL_RATE`
- `PREFILL_TIME_MIN`
- `PREFILL_TIME_MAX`
- `SIM_DURATION`
- `QUEUE_WAIT_THRESHOLD_MS`
- `SERVICE_DURATION_TARGET_MS`
- `PREFILL_ZERO_PROBS_BY_REGION_COUNT`

其中：

- `SERVICE_DURATION_TARGET_MS`
  用于统计“服务时长 <= 阈值”的请求数和占比

- `PREFILL_ZERO_PROBS_BY_REGION_COUNT`
  用于按地域数设置每个 region 的 `prefill_zero_prob_by_region`

示例：

```python
PREFILL_ZERO_PROBS_BY_REGION_COUNT = {
    1: [0.6],
    10: [0.2] * 10,
}
```

## 结果表字段

`run_compare.py` 生成的 CSV 目前包含这些指标：

- 完成请求数
- 有排队请求数
- 排队比例
- 平均服务时长(ms)
- 最大服务时长(ms)
- 平均等待时间(ms)
- 最大等待时间(ms)
- 服务时长 `<= SERVICE_DURATION_TARGET_MS` 的请求数
- 服务时长 `<= SERVICE_DURATION_TARGET_MS` 的请求占比

其中：

- 服务时长 = 等待时间 + Prefill 计算时间
- 服务时长达标占比的分母是全部完成请求数

## 图像说明

`run_compare.py` 当前绘制的是：

- 横轴：`用户数 / 实例数`
- 纵轴：`排队请求数 / 总请求数`

也就是排队比例对比图。

`inspect_waits.py` 会绘制：

- 等待时长直方图
- 等待时长箱线图
- 等待时长 vs 到达时间散点图
- 等待时长累计分布图

## 常见修改场景

### 改排队口径

把：

```python
QUEUE_WAIT_THRESHOLD_MS = 50.0
```

改成：

```python
QUEUE_WAIT_THRESHOLD_MS = 0.0
```

或其他阈值。

### 改服务时长达标阈值

把：

```python
SERVICE_DURATION_TARGET_MS = 800.0
```

改成你关心的目标值，例如 `600.0` 或 `1000.0`。

### 模拟 KV cache 命中

例如设置所有 10 个地域都有 60% 的概率让 Prefill 为 0：

```python
PREFILL_ZERO_PROBS_BY_REGION_COUNT = {
    1: [0.6],
    10: [0.6] * 10,
}
```

## 备注

- 当前推荐优先使用 `simulation.py`、`run_compare.py`、`inspect_waits.py`
- 旧版 `simulation_v1.py` / `run_compare_v1.py` 不再作为主入口
