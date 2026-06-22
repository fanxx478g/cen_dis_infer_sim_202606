"""Batch compare centralized vs distributed simulation configs."""

import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from simulation import LLMDeploymentSimulation, compute_stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

NUM_INSTANCES = 2
# USER_RANGE = range(20 * 5 * 1, 20 * 70 * 1 + 1, 20)
USER_RANGE = range(2 * 60 * 1, 2 * 260 * 1 + 1, 2 * 5)
REGIONS = [1]
SEED = 42
PREFILL_ZERO_PROBS_BY_REGION_COUNT = {
    1: [0.0],
    2: [0.0] * 2,
}

ARRIVAL_RATE = 100 * 1000.0
PREFILL_TIME_MIN = 350
PREFILL_TIME_MAX = 450
# PREFILL_TIME_MIN = 1350
# PREFILL_TIME_MAX = 1750
SIM_DURATION = 60 * 60 * 1000.0
QUEUE_WAIT_THRESHOLD_MS = 50.0
SERVICE_DURATION_TARGET_MS = 800

METRICS = [
    ("count", "完成请求数", "int"),
    ("queued_count", "有排队请求数", "int"),
    ("queued_ratio", "排队比例", "ratio"),
    ("avg_duration", "平均服务时长(ms)", "float"),
    ("max_duration", "最大服务时长(ms)", "float"),
    ("avg_wait", "平均等待时间(ms)", "float"),
    ("max_wait", "最大等待时间(ms)", "float"),
    ("within_target_count", f"服务时长<={int(SERVICE_DURATION_TARGET_MS)}ms请求数", "int"),
    ("within_target_ratio", f"服务时长<={int(SERVICE_DURATION_TARGET_MS)}ms请求占比", "ratio"),
]


def validate_config() -> None:
    if NUM_INSTANCES <= 0:
        raise ValueError("NUM_INSTANCES must be positive.")
    if ARRIVAL_RATE <= 0:
        raise ValueError("ARRIVAL_RATE must be positive.")
    if PREFILL_TIME_MIN <= 0 or PREFILL_TIME_MAX <= 0:
        raise ValueError("PREFILL_TIME_MIN and PREFILL_TIME_MAX must be positive.")
    if PREFILL_TIME_MIN > PREFILL_TIME_MAX:
        raise ValueError("PREFILL_TIME_MIN must be <= PREFILL_TIME_MAX.")
    if SIM_DURATION <= 0:
        raise ValueError("SIM_DURATION must be positive.")
    if QUEUE_WAIT_THRESHOLD_MS < 0:
        raise ValueError("QUEUE_WAIT_THRESHOLD_MS must be >= 0.")
    if SERVICE_DURATION_TARGET_MS <= 0:
        raise ValueError("SERVICE_DURATION_TARGET_MS must be positive.")
    if not REGIONS:
        raise ValueError("REGIONS cannot be empty.")
    if USER_RANGE.step == 0:
        raise ValueError("USER_RANGE step cannot be 0.")

    for num_regions in REGIONS:
        if num_regions <= 0:
            raise ValueError(f"num_regions must be positive, got {num_regions}.")
        if NUM_INSTANCES % num_regions != 0:
            raise ValueError(
                f"NUM_INSTANCES={NUM_INSTANCES} cannot be evenly divided by num_regions={num_regions}."
            )

    for num_users in USER_RANGE:
        for num_regions in REGIONS:
            if num_users % num_regions != 0:
                raise ValueError(
                    f"num_users={num_users} cannot be evenly divided by num_regions={num_regions}."
                )

    for num_regions in REGIONS:
        probs = build_prefill_zero_probs(num_regions)
        if len(probs) != num_regions:
            raise ValueError(
                f"prefill zero probability config length must match num_regions={num_regions}."
            )
        if any(v < 0.0 or v > 1.0 for v in probs):
            raise ValueError("prefill zero probability values must be within [0, 1].")


def compact_ms(ms_value: float) -> str:
    if ms_value % 1000 == 0:
        seconds = ms_value / 1000
        if seconds % 60 == 0:
            return f"{int(seconds / 60)}m"
        return f"{int(seconds)}s"
    return f"{int(ms_value)}ms"


def build_output_stem() -> str:
    region_part = f"{REGIONS[0]}-{REGIONS[-1]}" if len(REGIONS) > 1 else str(REGIONS[0])
    user_part = f"{USER_RANGE.start}-{USER_RANGE.stop - 1}x{USER_RANGE.step}"
    return (
        f"cmp_i{NUM_INSTANCES}"
        f"_u{user_part}"
        f"_r{region_part}"
        f"_a{compact_ms(ARRIVAL_RATE)}"
        f"_p{compact_ms(PREFILL_TIME_MIN)}-{compact_ms(PREFILL_TIME_MAX)}"
        f"_qw{int(QUEUE_WAIT_THRESHOLD_MS)}"
        f"_sd{int(SERVICE_DURATION_TARGET_MS)}"
        f"_t{compact_ms(SIM_DURATION)}"
        f"_k{PREFILL_ZERO_PROBS_BY_REGION_COUNT[1][0]}"
    )


def build_prefill_zero_probs(num_regions: int):
    region_probs = PREFILL_ZERO_PROBS_BY_REGION_COUNT.get(num_regions)
    if region_probs is not None:
        return region_probs
    return [0.0] * num_regions


def config_name(num_regions: int) -> str:
    return f"集中式({num_regions}地域)" if num_regions == 1 else f"分布式({num_regions}地域)"


def short_config_name(num_regions: int) -> str:
    return "集中式" if num_regions == 1 else f"{num_regions}地域分布式"


def format_metric(value, kind: str) -> str:
    if kind == "int":
        return str(int(value))
    if kind == "ratio":
        return f"{value:.4%}"
    return f"{value:.4f}"


def first_crossing(num_regions: int, key: str, threshold: float):
    for num_users in USER_RANGE:
        if stats_by_config[num_regions][num_users][key] >= threshold:
            return num_users
    return None


def print_final_summary():
    print("\n" + "=" * 80)
    print("运行完成摘要")
    print("=" * 80)
    print(f"CSV结果: {CSV_PATH}")
    print(f"图片结果: {FIG_PATH}")
    print(
        f"参数: inst={NUM_INSTANCES}, users={USER_RANGE.start}-{USER_RANGE.stop - 1} step {USER_RANGE.step}, "
        f"regions={REGIONS}, arrival={int(ARRIVAL_RATE)}ms, "
        f"prefill={int(PREFILL_TIME_MIN)}-{int(PREFILL_TIME_MAX)}ms, duration={int(SIM_DURATION)}ms, "
        f"queue_wait_threshold={QUEUE_WAIT_THRESHOLD_MS:.1f}ms, "
        f"service_duration_target={SERVICE_DURATION_TARGET_MS:.1f}ms, "
        f"prefill_zero_probs={PREFILL_ZERO_PROBS_BY_REGION_COUNT}, seed={SEED}"
    )

    print("\n关键结论")
    for num_regions in REGIONS:
        ratios = [stats_by_config[num_regions][u]["queued_ratio"] for u in USER_RANGE]
        avg_durations = [stats_by_config[num_regions][u]["avg_duration"] for u in USER_RANGE]
        avg_waits = [stats_by_config[num_regions][u]["avg_wait"] for u in USER_RANGE]
        within_target_ratios = [stats_by_config[num_regions][u]["within_target_ratio"] for u in USER_RANGE]

        min_user = min(USER_RANGE, key=lambda u: stats_by_config[num_regions][u]["queued_ratio"])
        max_user = max(USER_RANGE, key=lambda u: stats_by_config[num_regions][u]["queued_ratio"])
        q10 = first_crossing(num_regions, "queued_ratio", 0.10)
        q50 = first_crossing(num_regions, "queued_ratio", 0.50)
        last_user = USER_RANGE.stop - 1
        last_stats = stats_by_config[num_regions][last_user]

        print(f"- {short_config_name(num_regions)}:")
        print(
            f"  平均排队比例 {sum(ratios) / len(ratios):.4%}，平均服务时长 {sum(avg_durations) / len(avg_durations):.2f}ms，"
            f"平均等待时间 {sum(avg_waits) / len(avg_waits):.2f}ms"
        )
        print(
            f"  平均服务时长<={SERVICE_DURATION_TARGET_MS:.0f}ms 请求占比 "
            f"{sum(within_target_ratios) / len(within_target_ratios):.4%}"
        )
        print(
            f"  最低排队比例出现在 users={min_user} ({stats_by_config[num_regions][min_user]['queued_ratio']:.4%})，"
            f"最高排队比例出现在 users={max_user} ({stats_by_config[num_regions][max_user]['queued_ratio']:.4%})"
        )
        print(
            f"  排队比例首次达到 10% 的用户数: {q10 if q10 is not None else '未达到'}，"
            f"首次达到 50% 的用户数: {q50 if q50 is not None else '未达到'}"
        )
        print(
            f"  最大用户点 users={last_user}: 排队比例 {last_stats['queued_ratio']:.4%}, "
            f"平均服务时长 {last_stats['avg_duration']:.2f}ms, 最大服务时长 {last_stats['max_duration']:.2f}ms, "
            f"平均等待时间 {last_stats['avg_wait']:.2f}ms, 最大等待时间 {last_stats['max_wait']:.2f}ms, "
            f"服务时长<={SERVICE_DURATION_TARGET_MS:.0f}ms 请求占比 {last_stats['within_target_ratio']:.4%}"
        )

    if 1 in REGIONS:
        baseline_region = 1
        for num_regions in REGIONS:
            if num_regions == baseline_region:
                continue
            first_user = USER_RANGE.start
            last_user = USER_RANGE.stop - 1
            first_delta = (
                stats_by_config[num_regions][first_user]["queued_ratio"]
                - stats_by_config[baseline_region][first_user]["queued_ratio"]
            )
            last_delta = (
                stats_by_config[num_regions][last_user]["queued_ratio"]
                - stats_by_config[baseline_region][last_user]["queued_ratio"]
            )
            last_wait_delta = (
                stats_by_config[num_regions][last_user]["avg_wait"]
                - stats_by_config[baseline_region][last_user]["avg_wait"]
            )
            last_within_target_delta = (
                stats_by_config[num_regions][last_user]["within_target_ratio"]
                - stats_by_config[baseline_region][last_user]["within_target_ratio"]
            )
            print(f"- 对比集中式 vs {short_config_name(num_regions)}:")
            print(f"  在最小用户点 users={first_user}，排队比例差值 {first_delta:+.4%}")
            print(
                f"  在最大用户点 users={last_user}，排队比例差值 {last_delta:+.4%}，"
                f"平均等待时间差值 {last_wait_delta:+.2f}ms，"
                f"服务时长<={SERVICE_DURATION_TARGET_MS:.0f}ms 请求占比差值 {last_within_target_delta:+.4%}"
            )


validate_config()
OUTPUT_STEM = build_output_stem()
CSV_PATH = Path(f"{OUTPUT_STEM}.csv")
FIG_PATH = Path(f"{OUTPUT_STEM}.png")

results_data = {r: [] for r in REGIONS}
stats_by_config = {r: {} for r in REGIONS}
total_steps = len(REGIONS) * len(USER_RANGE)
current_step = 0

for num_regions in REGIONS:
    print(f"\n{'=' * 60}")
    print(f"{config_name(num_regions)}, 每地域实例数: {NUM_INSTANCES // num_regions}")
    print(f"各地域prefill为0概率: {build_prefill_zero_probs(num_regions)}")
    print(f"{'=' * 60}")
    print(
        f"{'用户数':>8} {'用户/实例':>10} {'完成请求数':>10} {'有排队请求数':>12} "
        f"{'排队比例':>10} {'平均服务时长':>14} {'平均等待时间':>14} "
        f"{'最大服务时长':>14} {'最大等待时间':>14} "
        f"{f'<={int(SERVICE_DURATION_TARGET_MS)}ms数':>14} {f'<={int(SERVICE_DURATION_TARGET_MS)}ms占比':>16}"
    )
    print("-" * 160)

    for num_users in USER_RANGE:
        current_step += 1
        print(f"[{current_step}/{total_steps} ({current_step * 100 // total_steps}%)] ", end="")
        sim = LLMDeploymentSimulation(
            num_instances=NUM_INSTANCES,
            num_users=num_users,
            num_regions=num_regions,
            arrival_rate=ARRIVAL_RATE,
            prefill_time_min=PREFILL_TIME_MIN,
            prefill_time_max=PREFILL_TIME_MAX,
            sim_duration=SIM_DURATION,
            seed=SEED,
            queue_wait_threshold_ms=QUEUE_WAIT_THRESHOLD_MS,
            prefill_zero_prob_by_region=build_prefill_zero_probs(num_regions),
        )

        reqs = sim.run_centralized() if num_regions == 1 else sim.run_distributed()
        stats = compute_stats(reqs, QUEUE_WAIT_THRESHOLD_MS, SERVICE_DURATION_TARGET_MS)
        users_per_instance = num_users / NUM_INSTANCES

        print(
            f"{num_users:>8} {users_per_instance:>10.1f} {stats['count']:>10} {stats['queued_count']:>12} "
            f"{stats['queued_ratio']:>10.4%} {stats['avg_duration']:>14.4f}ms {stats['avg_wait']:>14.4f}ms "
            f"{stats['max_duration']:>14.4f}ms {stats['max_wait']:>14.4f}ms "
            f"{stats['within_target_count']:>14} {stats['within_target_ratio']:>16.4%}"
        )

        results_data[num_regions].append(
            {
                "num_users": num_users,
                "users_per_instance": users_per_instance,
                "queued_ratio": stats["queued_ratio"],
            }
        )
        stats_by_config[num_regions][num_users] = stats

fieldnames = ["用户数", "用户/实例"]
for num_regions in REGIONS:
    for _, label, _ in METRICS:
        fieldnames.append(f"{short_config_name(num_regions)}-{label}")

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for num_users in USER_RANGE:
        users_per_instance = round(num_users / NUM_INSTANCES, 1)
        row = {"用户数": num_users, "用户/实例": users_per_instance}
        for num_regions in REGIONS:
            stats = stats_by_config[num_regions][num_users]
            prefix = short_config_name(num_regions)
            for key, label, kind in METRICS:
                row[f"{prefix}-{label}"] = format_metric(stats[key], kind)
        writer.writerow(row)
print(f"\n表格已保存至 {CSV_PATH}")

markers = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]
fig, ax = plt.subplots(figsize=(10, 6))

for i, num_regions in enumerate(REGIONS):
    data = results_data[num_regions]
    x = [d["users_per_instance"] for d in data]
    y = [d["queued_ratio"] for d in data]
    label = f"{short_config_name(num_regions)} (每地域{NUM_INSTANCES // num_regions}实例)"
    ax.plot(x, y, marker=markers[i % len(markers)], markersize=4, label=label)

ax.set_xlabel("用户数 / 实例数")
ax.set_ylabel("排队请求数 / 总请求数")
ax.set_title(
    f"排队请求数 / 总请求数 对比 "
    f"(inst={NUM_INSTANCES}, regions={REGIONS}, arrival={int(ARRIVAL_RATE)}ms, "
    f"prefill={int(PREFILL_TIME_MIN)}-{int(PREFILL_TIME_MAX)}ms)"
)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig(FIG_PATH, dpi=200)
# print(f"图像已保存至 {FIG_PATH}")
print_final_summary()
plt.show()
