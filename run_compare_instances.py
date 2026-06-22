"""Batch compare multiple instance-count cases under a fixed region count."""

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

NUM_REGIONS = 1
CASES = [
    {"num_instances": 3, "user_range": range(3 * 5, 3 * 70 + 1, 3)},
    {"num_instances": 6, "user_range": range(6 * 5, 6 * 70 + 1, 6)},
    {"num_instances": 9, "user_range": range(9 * 5, 9 * 70 + 1, 9)},
    {"num_instances": 12, "user_range": range(12 * 5, 12 * 70 + 1, 12)},
    {"num_instances": 15, "user_range": range(15 * 5, 15 * 70 + 1, 15)},
    {"num_instances": 30, "user_range": range(30 * 5, 30 * 70 + 1, 30)},
    # {"num_instances": 12, "user_range": range(12 * 60, 12 * 260 + 1, 12 * 5)},
    # {"num_instances": 18, "user_range": range(18 * 60, 18 * 260 + 1, 18 * 5)},
    # {"num_instances": 42, "user_range": range(42 * 60, 42 * 260 + 1, 42 * 5)},
    # {"num_instances": 48, "user_range": range(48 * 60, 48 * 260 + 1, 48 * 5)},
]
SEED = 42
PREFILL_ZERO_PROBS_BY_REGION_COUNT = {
    1: [0.0],
    2: [0.0] * 2,
}

ARRIVAL_RATE = 100 * 1000.0
PREFILL_TIME_MIN = 1450
PREFILL_TIME_MAX = 1750
# PREFILL_TIME_MIN = 350
# PREFILL_TIME_MAX = 450
SIM_DURATION = 60 * 60 * 1000.0
QUEUE_WAIT_THRESHOLD_MS = 50.0
SERVICE_DURATION_TARGET_MS = 2000.0

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
    if NUM_REGIONS <= 0:
        raise ValueError("NUM_REGIONS must be positive.")
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
    if not CASES:
        raise ValueError("CASES cannot be empty.")

    probs = build_prefill_zero_probs(NUM_REGIONS)
    if len(probs) != NUM_REGIONS:
        raise ValueError(
            f"prefill zero probability config length must match num_regions={NUM_REGIONS}."
        )
    if any(v < 0.0 or v > 1.0 for v in probs):
        raise ValueError("prefill zero probability values must be within [0, 1].")

    seen_instances = set()
    for case in CASES:
        num_instances = case["num_instances"]
        user_range = case["user_range"]
        if num_instances <= 0:
            raise ValueError(f"num_instances must be positive, got {num_instances}.")
        if num_instances in seen_instances:
            raise ValueError(f"Duplicate num_instances detected: {num_instances}.")
        seen_instances.add(num_instances)
        if user_range.step == 0:
            raise ValueError(f"user_range step cannot be 0 for num_instances={num_instances}.")
        if num_instances % NUM_REGIONS != 0:
            raise ValueError(
                f"num_instances={num_instances} cannot be evenly divided by num_regions={NUM_REGIONS}."
            )
        for num_users in user_range:
            if num_users % NUM_REGIONS != 0:
                raise ValueError(
                    f"num_users={num_users} cannot be evenly divided by num_regions={NUM_REGIONS}."
                )


def compact_ms(ms_value: float) -> str:
    if ms_value % 1000 == 0:
        seconds = ms_value / 1000
        if seconds % 60 == 0:
            return f"{int(seconds / 60)}m"
        return f"{int(seconds)}s"
    return f"{int(ms_value)}ms"


def compact_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def build_output_stem() -> str:
    instance_part = "-".join(str(case["num_instances"]) for case in CASES)
    return (
        f"cmp_by_inst_r{NUM_REGIONS}"
        f"_i{instance_part}"
        f"_a{compact_ms(ARRIVAL_RATE)}"
        f"_p{compact_ms(PREFILL_TIME_MIN)}-{compact_ms(PREFILL_TIME_MAX)}"
        f"_qw{int(QUEUE_WAIT_THRESHOLD_MS)}"
        f"_sd{int(SERVICE_DURATION_TARGET_MS)}"
        f"_t{compact_ms(SIM_DURATION)}"
        f"_k{compact_float(build_prefill_zero_probs(NUM_REGIONS)[0])}"
    )


def build_prefill_zero_probs(num_regions: int):
    region_probs = PREFILL_ZERO_PROBS_BY_REGION_COUNT.get(num_regions)
    if region_probs is not None:
        return region_probs
    return [0.0] * num_regions


def instance_group_name(num_instances: int) -> str:
    return f"{num_instances}实例"


def format_metric(value, kind: str) -> str:
    if value == "":
        return ""
    if kind == "int":
        return str(int(value))
    if kind == "ratio":
        return f"{value:.4%}"
    return f"{value:.4f}"


def users_per_instance_values():
    values = set()
    for case in CASES:
        num_instances = case["num_instances"]
        for num_users in case["user_range"]:
            values.add(round(num_users / num_instances, 6))
    return sorted(values)


def first_crossing(num_instances: int, key: str, threshold: float):
    for num_users in CASE_STATS[num_instances]:
        if CASE_STATS[num_instances][num_users][key] >= threshold:
            return num_users
    return None


def print_final_summary():
    print("\n" + "=" * 80)
    print("运行完成摘要")
    print("=" * 80)
    print(f"CSV结果: {CSV_PATH}")
    print(f"图片结果: {FIG_PATH}")
    print(
        f"参数: regions={NUM_REGIONS}, instances={[case['num_instances'] for case in CASES]}, "
        f"arrival={int(ARRIVAL_RATE)}ms, prefill={int(PREFILL_TIME_MIN)}-{int(PREFILL_TIME_MAX)}ms, "
        f"duration={int(SIM_DURATION)}ms, queue_wait_threshold={QUEUE_WAIT_THRESHOLD_MS:.1f}ms, "
        f"service_duration_target={SERVICE_DURATION_TARGET_MS:.1f}ms, "
        f"prefill_zero_probs={build_prefill_zero_probs(NUM_REGIONS)}, seed={SEED}"
    )

    print("\n关键结论")
    for case in CASES:
        num_instances = case["num_instances"]
        user_range = case["user_range"]
        ratios = [CASE_STATS[num_instances][u]["queued_ratio"] for u in user_range]
        avg_durations = [CASE_STATS[num_instances][u]["avg_duration"] for u in user_range]
        avg_waits = [CASE_STATS[num_instances][u]["avg_wait"] for u in user_range]
        within_target_ratios = [CASE_STATS[num_instances][u]["within_target_ratio"] for u in user_range]

        min_user = min(user_range, key=lambda u: CASE_STATS[num_instances][u]["queued_ratio"])
        max_user = max(user_range, key=lambda u: CASE_STATS[num_instances][u]["queued_ratio"])
        q10 = first_crossing(num_instances, "queued_ratio", 0.10)
        q50 = first_crossing(num_instances, "queued_ratio", 0.50)
        last_user = user_range.stop - 1
        last_stats = CASE_STATS[num_instances][last_user]

        print(f"- {instance_group_name(num_instances)}:")
        print(
            f"  平均排队比例 {sum(ratios) / len(ratios):.4%}，平均服务时长 "
            f"{sum(avg_durations) / len(avg_durations):.2f}ms，平均等待时间 {sum(avg_waits) / len(avg_waits):.2f}ms"
        )
        print(
            f"  平均服务时长<={SERVICE_DURATION_TARGET_MS:.0f}ms 请求占比 "
            f"{sum(within_target_ratios) / len(within_target_ratios):.4%}"
        )
        print(
            f"  最低排队比例出现在 users={min_user} ({CASE_STATS[num_instances][min_user]['queued_ratio']:.4%})，"
            f"最高排队比例出现在 users={max_user} ({CASE_STATS[num_instances][max_user]['queued_ratio']:.4%})"
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


validate_config()
OUTPUT_STEM = build_output_stem()
CSV_PATH = Path(f"{OUTPUT_STEM}.csv")
FIG_PATH = Path(f"{OUTPUT_STEM}.png")

results_data = {case["num_instances"]: [] for case in CASES}
CASE_STATS = {case["num_instances"]: {} for case in CASES}
total_steps = sum(len(list(case["user_range"])) for case in CASES)
current_step = 0

for case in CASES:
    num_instances = case["num_instances"]
    user_range = case["user_range"]
    mode_name = "集中式" if NUM_REGIONS == 1 else f"{NUM_REGIONS}地域分布式"

    print(f"\n{'=' * 60}")
    print(f"{instance_group_name(num_instances)} | {mode_name} | 每地域实例数: {num_instances // NUM_REGIONS}")
    print(f"各地域prefill为0概率: {build_prefill_zero_probs(NUM_REGIONS)}")
    print(f"{'=' * 60}")
    print(
        f"{'用户数':>8} {'用户/实例':>10} {'完成请求数':>10} {'有排队请求数':>12} "
        f"{'排队比例':>10} {'平均服务时长':>14} {'平均等待时间':>14} "
        f"{'最大服务时长':>14} {'最大等待时间':>14} "
        f"{f'<={int(SERVICE_DURATION_TARGET_MS)}ms数':>14} {f'<={int(SERVICE_DURATION_TARGET_MS)}ms占比':>16}"
    )
    print("-" * 160)

    for num_users in user_range:
        current_step += 1
        print(f"[{current_step}/{total_steps} ({current_step * 100 // total_steps}%)] ", end="")
        sim = LLMDeploymentSimulation(
            num_instances=num_instances,
            num_users=num_users,
            num_regions=NUM_REGIONS,
            arrival_rate=ARRIVAL_RATE,
            prefill_time_min=PREFILL_TIME_MIN,
            prefill_time_max=PREFILL_TIME_MAX,
            sim_duration=SIM_DURATION,
            seed=SEED,
            queue_wait_threshold_ms=QUEUE_WAIT_THRESHOLD_MS,
            prefill_zero_prob_by_region=build_prefill_zero_probs(NUM_REGIONS),
        )

        reqs = sim.run_centralized() if NUM_REGIONS == 1 else sim.run_distributed()
        stats = compute_stats(reqs, QUEUE_WAIT_THRESHOLD_MS, SERVICE_DURATION_TARGET_MS)
        users_per_instance = num_users / num_instances

        print(
            f"{num_users:>8} {users_per_instance:>10.1f} {stats['count']:>10} {stats['queued_count']:>12} "
            f"{stats['queued_ratio']:>10.4%} {stats['avg_duration']:>14.4f}ms {stats['avg_wait']:>14.4f}ms "
            f"{stats['max_duration']:>14.4f}ms {stats['max_wait']:>14.4f}ms "
            f"{stats['within_target_count']:>14} {stats['within_target_ratio']:>16.4%}"
        )

        results_data[num_instances].append(
            {
                "num_users": num_users,
                "users_per_instance": users_per_instance,
                "queued_ratio": stats["queued_ratio"],
            }
        )
        CASE_STATS[num_instances][num_users] = stats

fieldnames = ["用户/实例"]
for case in CASES:
    num_instances = case["num_instances"]
    prefix = instance_group_name(num_instances)
    fieldnames.append(f"{prefix}-用户数")
    for _, label, _ in METRICS:
        fieldnames.append(f"{prefix}-{label}")

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for users_per_instance in users_per_instance_values():
        row = {"用户/实例": f"{users_per_instance:.1f}"}
        for case in CASES:
            num_instances = case["num_instances"]
            prefix = instance_group_name(num_instances)
            matched_users = None
            for num_users in case["user_range"]:
                if abs(num_users / num_instances - users_per_instance) < 1e-9:
                    matched_users = num_users
                    break
            if matched_users is None:
                row[f"{prefix}-用户数"] = ""
                for _, label, _ in METRICS:
                    row[f"{prefix}-{label}"] = ""
                continue

            stats = CASE_STATS[num_instances][matched_users]
            row[f"{prefix}-用户数"] = matched_users
            for key, label, kind in METRICS:
                row[f"{prefix}-{label}"] = format_metric(stats[key], kind)
        writer.writerow(row)
print(f"\n表格已保存至 {CSV_PATH}")

markers = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]
fig, ax = plt.subplots(figsize=(10, 6))

for i, case in enumerate(CASES):
    num_instances = case["num_instances"]
    data = results_data[num_instances]
    x = [d["users_per_instance"] for d in data]
    y = [d["queued_ratio"] for d in data]
    ax.plot(
        x,
        y,
        marker=markers[i % len(markers)],
        markersize=4,
        label=f"{instance_group_name(num_instances)}",
    )

ax.set_xlabel("用户数 / 实例数")
ax.set_ylabel("排队请求数 / 总请求数")
ax.set_title(
    f"排队请求数 / 总请求数 对比 "
    f"(regions={NUM_REGIONS}, instances={[case['num_instances'] for case in CASES]}, "
    f"arrival={int(ARRIVAL_RATE)}ms, prefill={int(PREFILL_TIME_MIN)}-{int(PREFILL_TIME_MAX)}ms)"
)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig(FIG_PATH, dpi=200)
# print(f"图像已保存至 {FIG_PATH}")
print_final_summary()
plt.show()
