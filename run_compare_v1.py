"""320实例，对比1地域 vs 32地域，用户数19200-48000"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

from simulation import LLMDeploymentSimulation, compute_stats

NUM_INSTANCES = 320
USER_RANGE = range(19200, 48001, 2000)
REGIONS = [1, 32]
SEED = 42
QUEUE_WAIT_THRESHOLD_MS = 0.0

results_data = {r: [] for r in REGIONS}
total_steps = len(REGIONS) * len(USER_RANGE)
current_step = 0

for num_regions in REGIONS:
    print(f"\n{'='*60}")
    print(f"地域数: {num_regions}, 每地域实例数: {NUM_INSTANCES // num_regions}")
    print(f"{'='*60}")
    print(f"{'用户数':>8} {'用户/实例':>10} {'完成请求数':>10} {'有排队请求数':>12} {'排队比例':>10} {'平均服务时长':>14} {'平均等待时间':>14}")
    print("-" * 80)

    for num_users in USER_RANGE:
        current_step += 1
        print(f"[{current_step}/{total_steps} ({current_step*100//total_steps}%)] ", end="")
        sim = LLMDeploymentSimulation(
            num_instances=NUM_INSTANCES,
            num_users=num_users,
            num_regions=num_regions,
            arrival_rate=60000.0,
            prefill_time_min=450.0,
            prefill_time_max=550.0,
            sim_duration=3600000.0,
            seed=SEED,
            queue_wait_threshold_ms=QUEUE_WAIT_THRESHOLD_MS,
        )

        if num_regions == 1:
            reqs = sim.run_centralized()
        else:
            reqs = sim.run_distributed()

        stats = compute_stats(reqs, QUEUE_WAIT_THRESHOLD_MS)
        users_per_instance = num_users / NUM_INSTANCES

        print(f"{num_users:>8} {users_per_instance:>10.1f} {stats['count']:>10} {stats['queued_count']:>12} "
              f"{stats['queued_ratio']:>10.2%} {stats['avg_duration']:>12.2f}ms {stats['avg_wait']:>12.2f}ms")

        results_data[num_regions].append({
            'num_users': num_users,
            'users_per_instance': users_per_instance,
            'queued_ratio': stats['queued_ratio'],
        })

# 画图
fig, ax = plt.subplots(figsize=(10, 6))

for num_regions in REGIONS:
    data = results_data[num_regions]
    x = [d['users_per_instance'] for d in data]
    y = [d['queued_ratio'] for d in data]
    label = f"{num_regions}地域 (每地域{NUM_INSTANCES // num_regions}实例)"
    marker = 'o' if num_regions == 1 else 's'
    ax.plot(x, y, marker=marker, markersize=4, label=label)

ax.set_xlabel("用户数/实例数")
ax.set_ylabel("有排队请求数 / 总请求数")
ax.set_title(f"{NUM_INSTANCES}实例: 1地域 vs 32地域 排队比例对比")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
