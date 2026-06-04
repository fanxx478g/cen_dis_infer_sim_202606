"""64实例 64用户 1地域 仿真"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

from simulation import LLMDeploymentSimulation, compute_stats

QUEUE_WAIT_THRESHOLD_MS = 0.0
PREFILL_ZERO_PROB_BY_REGION = [0.0]

sim = LLMDeploymentSimulation(
    num_instances=64,
    num_users=64,
    num_regions=1,
    arrival_rate=60000.0,
    prefill_time_min=450.0,
    prefill_time_max=550.0,
    sim_duration=3600000.0,
    seed=None,
    queue_wait_threshold_ms=QUEUE_WAIT_THRESHOLD_MS,
    prefill_zero_prob_by_region=PREFILL_ZERO_PROB_BY_REGION,
)

results = sim.run_centralized()
stats = compute_stats(results, QUEUE_WAIT_THRESHOLD_MS)

print(f"完成请求数:   {stats['count']}")
print(f"有排队请求数: {stats['queued_count']}")
print(f"排队请求比例: {stats['queued_ratio']:.2%}")
print(f"平均服务时长: {stats['avg_duration']:.2f} ms")
print(f"平均等待时间: {stats['avg_wait']:.2f} ms")

arrival_times = [r.arrival_time for r in results]
wait_times = [r.wait_time for r in results]
durations = [r.service_duration for r in results]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("64实例 64用户 1地域 仿真结果", fontsize=14)

ax = axes[0]
ax.scatter(arrival_times, wait_times, s=2, alpha=0.5, color="steelblue")
ax.set_xlabel("到达时间 (ms)")
ax.set_ylabel("排队时长 (ms)")
ax.set_title("所有请求的排队时长 vs 到达时间")

ax = axes[1]
ax.scatter(arrival_times, durations, s=2, alpha=0.5, color="coral")
ax.set_xlabel("到达时间 (ms)")
ax.set_ylabel("服务时长 (ms)")
ax.set_title("所有请求的服务时长 vs 到达时间")

ax = axes[2]
ax.hist(arrival_times, bins=60, color="seagreen", edgecolor="white", alpha=0.8)
ax.set_xlabel("到达时间 (ms)")
ax.set_ylabel("请求数")
ax.set_title("请求到达时间分布 (按分钟分桶)")

plt.tight_layout()
plt.show()
