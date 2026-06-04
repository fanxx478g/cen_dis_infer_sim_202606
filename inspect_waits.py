"""Inspect queued requests for one simulation configuration."""

import argparse
import statistics
import sys

import matplotlib
import matplotlib.pyplot as plt

from simulation import LLMDeploymentSimulation

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False


DEFAULT_NUM_INSTANCES = 4
DEFAULT_NUM_USERS = 4 * 60
DEFAULT_NUM_REGIONS = 1
DEFAULT_MODE = "centralized"
DEFAULT_ARRIVAL_RATE = 100 * 1000.0
DEFAULT_PREFILL_MIN = 350.0
DEFAULT_PREFILL_MAX = 450.0
DEFAULT_SIM_DURATION = 3600000.0
DEFAULT_SEED = 42
DEFAULT_PREVIEW_LIMIT = 50
DEFAULT_PREFILL_ZERO_PROBS = None


def percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect waiting requests for one simulation config.")
    parser.add_argument("--num-instances", type=int, default=DEFAULT_NUM_INSTANCES)
    parser.add_argument("--num-users", type=int, default=DEFAULT_NUM_USERS)
    parser.add_argument("--num-regions", type=int, default=DEFAULT_NUM_REGIONS)
    parser.add_argument(
        "--mode",
        choices=["centralized", "distributed"],
        default=DEFAULT_MODE,
        help="Simulation mode to inspect.",
    )
    parser.add_argument("--arrival-rate", type=float, default=DEFAULT_ARRIVAL_RATE, help="Mean inter-arrival time in ms.")
    parser.add_argument("--prefill-min", type=float, default=DEFAULT_PREFILL_MIN, help="Minimum prefill time in ms.")
    parser.add_argument("--prefill-max", type=float, default=DEFAULT_PREFILL_MAX, help="Maximum prefill time in ms.")
    parser.add_argument("--sim-duration", type=float, default=DEFAULT_SIM_DURATION, help="Simulation duration in ms.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--prefill-zero-probs",
        type=str,
        default=DEFAULT_PREFILL_ZERO_PROBS,
        help="Prefill zero probability. Use a single value like '0.6' or comma-separated per-region values like '0.1,0.2,0.3'.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=DEFAULT_PREVIEW_LIMIT,
        help="How many waiting requests to print. Use -1 to print all.",
    )
    parser.add_argument(
        "--min-wait-threshold",
        type=float,
        default=0.0,
        help="Only treat wait_time > threshold as waiting, in ms.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_instances <= 0:
        raise ValueError("num_instances must be positive.")
    if args.num_users <= 0:
        raise ValueError("num_users must be positive.")
    if args.num_regions <= 0:
        raise ValueError("num_regions must be positive.")
    if args.arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive.")
    if args.prefill_min <= 0 or args.prefill_max <= 0:
        raise ValueError("prefill times must be positive.")
    if args.prefill_min > args.prefill_max:
        raise ValueError("prefill_min must be <= prefill_max.")
    if args.sim_duration <= 0:
        raise ValueError("sim_duration must be positive.")
    if args.num_instances % args.num_regions != 0:
        raise ValueError("num_instances must be divisible by num_regions.")
    if args.num_users % args.num_regions != 0:
        raise ValueError("num_users must be divisible by num_regions.")
    if args.preview_limit < -1 or args.preview_limit == 0:
        raise ValueError("preview_limit must be -1 or a positive integer.")
    if args.min_wait_threshold < 0:
        raise ValueError("min_wait_threshold must be >= 0.")


def parse_prefill_zero_probs(raw_value, num_regions: int):
    if raw_value is None:
        return None
    if "," not in raw_value:
        value = float(raw_value)
        if value < 0.0 or value > 1.0:
            raise ValueError("prefill zero probability must be within [0, 1].")
        return value

    values = [float(v.strip()) for v in raw_value.split(",") if v.strip()]
    if len(values) != num_regions:
        raise ValueError("comma-separated prefill zero probabilities must match num_regions.")
    if any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError("prefill zero probability values must be within [0, 1].")
    return values


def run_simulation(args):
    sim = LLMDeploymentSimulation(
        num_instances=args.num_instances,
        num_users=args.num_users,
        num_regions=args.num_regions,
        arrival_rate=args.arrival_rate,
        prefill_time_min=args.prefill_min,
        prefill_time_max=args.prefill_max,
        sim_duration=args.sim_duration,
        seed=args.seed,
        prefill_zero_prob_by_region=parse_prefill_zero_probs(args.prefill_zero_probs, args.num_regions),
    )
    if args.mode == "centralized":
        return sim, sim.run_centralized()
    return sim, sim.run_distributed()


def print_summary(args, sim, requests, waiting_requests):
    total = len(requests)
    queued = len(waiting_requests)
    queued_ratio = queued / total if total else 0.0

    print("=" * 80)
    print("排队请求诊断")
    print("=" * 80)
    print(f"模式: {args.mode}")
    print(f"实例总数: {args.num_instances}")
    print(f"用户总数: {args.num_users}")
    print(f"地域数: {args.num_regions}")
    print(f"每地域实例数: {sim.instances_per_region}")
    print(f"每地域用户数: {sim.users_per_region}")
    print(f"平均到达间隔: {args.arrival_rate:.0f} ms")
    print(f"Prefill时间: {args.prefill_min:.0f}~{args.prefill_max:.0f} ms")
    print(f"仿真时长: {args.sim_duration:.0f} ms")
    print(f"随机种子: {args.seed}")
    print(f"排队阈值: wait_time > {args.min_wait_threshold:.3f} ms")
    print("-" * 80)
    print(f"总请求数: {total}")
    print(f"排队请求数: {queued}")
    print(f"排队比例: {queued_ratio:.4%}")

    if not waiting_requests:
        print("没有请求发生等待。")
        return

    wait_times = [req.wait_time for req in waiting_requests]
    print(f"平均等待时长: {statistics.mean(wait_times):.4f} ms")
    print(f"P50等待时长: {percentile(wait_times, 50):.4f} ms")
    print(f"P90等待时长: {percentile(wait_times, 90):.4f} ms")
    print(f"P99等待时长: {percentile(wait_times, 99):.4f} ms")
    print(f"最大等待时长: {max(wait_times):.4f} ms")


def print_waiting_requests(args, waiting_requests):
    if not waiting_requests:
        return

    if args.preview_limit == -1:
        rows = waiting_requests
        title = "全部排队请求"
    else:
        rows = waiting_requests[: args.preview_limit]
        title = f"前 {len(rows)} 条排队请求"

    print("\n" + title)
    print("-" * 80)
    print(f"{'req_id':>8} {'region':>8} {'arrival_ms':>14} {'wait_ms':>12} {'start_ms':>14} {'end_ms':>14}")
    for req in rows:
        print(
            f"{req.id:>8} {req.region:>8} {req.arrival_time:>14.3f} {req.wait_time:>12.3f} "
            f"{req.start_service_time:>14.3f} {req.end_service_time:>14.3f}"
        )

    if args.preview_limit != -1 and len(waiting_requests) > len(rows):
        print(f"... 其余 {len(waiting_requests) - len(rows)} 条未打印，可用 --preview-limit -1 查看全部。")


def plot_waiting_requests(args, waiting_requests):
    if not waiting_requests:
        print("没有排队请求，跳过绘图。")
        return

    wait_times = [req.wait_time for req in waiting_requests]
    arrival_times = [req.arrival_time for req in waiting_requests]
    regions = [req.region for req in waiting_requests]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"排队请求诊断 ({args.mode}, users={args.num_users}, instances={args.num_instances}, regions={args.num_regions})",
        fontsize=14,
    )

    ax = axes[0, 0]
    ax.hist(wait_times, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_title("等待时长分布直方图")
    ax.set_xlabel("等待时长 (ms)")
    ax.set_ylabel("排队请求数")

    ax = axes[0, 1]
    ax.boxplot(wait_times, vert=True, showfliers=False, patch_artist=True, boxprops={"facecolor": "coral", "alpha": 0.7})
    ax.set_title("等待时长箱线图")
    ax.set_ylabel("等待时长 (ms)")
    ax.set_xticks([1])
    ax.set_xticklabels(["wait"])

    ax = axes[1, 0]
    scatter = ax.scatter(arrival_times, wait_times, c=regions, cmap="tab10", s=10, alpha=0.7)
    ax.set_title("等待时长 vs 到达时间")
    ax.set_xlabel("到达时间 (ms)")
    ax.set_ylabel("等待时长 (ms)")
    if len(set(regions)) > 1:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("地域")

    ax = axes[1, 1]
    sorted_waits = sorted(wait_times)
    cdf_y = [(i + 1) / len(sorted_waits) for i in range(len(sorted_waits))]
    ax.plot(sorted_waits, cdf_y, color="seagreen")
    ax.set_title("等待时长累计分布")
    ax.set_xlabel("等待时长 (ms)")
    ax.set_ylabel("累计比例")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    validate_args(args)
    sim, requests = run_simulation(args)
    waiting_requests = [req for req in requests if req.wait_time > args.min_wait_threshold]
    print_summary(args, sim, requests, waiting_requests)
    print_waiting_requests(args, waiting_requests)
    plot_waiting_requests(args, waiting_requests)


if __name__ == "__main__":
    main()
