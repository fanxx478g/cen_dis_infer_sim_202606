"""
LLM部署方式仿真：集中式 vs 分布式
模拟不同部署方式下用户的请求处理延迟体验
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import heapq
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence, Union


@dataclass
class Request:
    """单个prefill请求"""
    id: int
    arrival_time: float  # 到达时间 (ms)
    region: int  # 所属地域
    start_service_time: float = 0.0  # 开始被实例处理的时间 (ms)
    end_service_time: float = 0.0  # 处理完成时间 (ms)
    service_duration: float = 0.0  # 服务时长 = 排队等待 + prefill计算 (ms)

    @property
    def wait_time(self) -> float:
        """排队等待时间"""
        return self.start_service_time - self.arrival_time


@dataclass(order=True)
class Event:
    """仿真事件"""
    time: float  # 事件发生时间 (ms)
    event_type: int = field(compare=False)  # 0=到达, 1=离开
    request_id: int = field(compare=False, default=0)
    region: int = field(compare=False, default=0)


ARRIVAL = 0
DEPARTURE = 1


class LLMDeploymentSimulation:
    """LLM prefill请求部署仿真"""

    def __init__(
        self,
        num_instances: int = 4,
        num_users: int = 100,
        num_regions: int = 2,
        arrival_rate: float = 60000.0,
        prefill_time_min: float = 450.0,
        prefill_time_max: float = 550.0,
        sim_duration: float = 3600000.0,
        seed: int = 42,
        queue_wait_threshold_ms: float = 0.0,
        prefill_zero_prob_by_region: Optional[Union[float, Sequence[float]]] = None,
    ):
        """
        Args:
            num_instances: 实例总数
            num_users: 用户总数
            num_regions: 地域数
            arrival_rate: 每个用户的请求平均到达间隔 (ms), 即泊松过程参数
            prefill_time_min: prefill计算时间下界 (ms)
            prefill_time_max: prefill计算时间上界 (ms)
            sim_duration: 仿真时长 (ms), 默认1小时
            seed: 随机种子
        """
        self.num_instances = num_instances
        self.num_users = num_users
        self.num_regions = num_regions
        self.arrival_rate = arrival_rate
        self.prefill_time_min = prefill_time_min
        self.prefill_time_max = prefill_time_max
        self.sim_duration = sim_duration
        self.seed = seed
        self.queue_wait_threshold_ms = queue_wait_threshold_ms
        self.prefill_zero_prob_by_region = self._resolve_prefill_zero_probs(prefill_zero_prob_by_region)
        self.rng = random.Random(seed)

        self.instances_per_region = num_instances // num_regions
        self.users_per_region = num_users // num_regions

    def _resolve_prefill_zero_probs(
        self,
        prefill_zero_prob_by_region: Optional[Union[float, Sequence[float]]],
    ) -> List[float]:
        if prefill_zero_prob_by_region is None:
            probs = [0.0] * self.num_regions
        elif isinstance(prefill_zero_prob_by_region, (int, float)):
            probs = [float(prefill_zero_prob_by_region)] * self.num_regions
        else:
            probs = [float(v) for v in prefill_zero_prob_by_region]
            if len(probs) != self.num_regions:
                raise ValueError(
                    "prefill_zero_prob_by_region length must match num_regions."
                )

        if any(v < 0.0 or v > 1.0 for v in probs):
            raise ValueError("prefill_zero_prob_by_region values must be within [0, 1].")
        return probs

    def _sample_prefill_time(self, region: int) -> float:
        if self.rng.random() < self.prefill_zero_prob_by_region[region]:
            return 0.0
        return self.rng.uniform(self.prefill_time_min, self.prefill_time_max)

    def _generate_arrivals(self) -> Tuple[List[Tuple[float, int, int]], List[Request]]:
        """生成仿真时长内所有请求的到达事件"""
        rng = self.rng
        arrivals = []
        requests = []
        request_id = 0

        for region in range(self.num_regions):
            for _ in range(self.users_per_region):
                t = rng.expovariate(1.0 / self.arrival_rate)
                while t < self.sim_duration:
                    arrivals.append((t, request_id, region))
                    requests.append(Request(id=request_id, arrival_time=t, region=region))
                    request_id += 1
                    t += rng.expovariate(1.0 / self.arrival_rate)

        return arrivals, requests

    def run_centralized(self) -> List[Request]:
        """
        集中式部署仿真
        所有请求维护一个队列，任何空闲实例都可服务队列中最早到达的请求
        优化：按到达时间排序遍历 + 实例空闲时间堆(大小=实例数)
        """
        arrivals, requests = self._generate_arrivals()
        arrivals.sort()

        # 实例空闲时间堆，初始全部空闲(free_time=0)
        free_heap = [0.0] * self.num_instances
        heapq.heapify(free_heap)

        for t, req_id, _ in arrivals:
            req = requests[req_id]
            earliest_free = heapq.heappop(free_heap)
            start = max(t, earliest_free)
            pf_time = self._sample_prefill_time(req.region)
            req.start_service_time = start
            req.end_service_time = start + pf_time
            req.service_duration = (start - t) + pf_time
            heapq.heappush(free_heap, start + pf_time)

        return [r for r in requests if r.service_duration > 0]

    def run_distributed(self) -> List[Request]:
        """
        分布式部署仿真
        每个地域维护独立队列，每个地域的实例只服务本地域的请求
        优化：按到达时间排序遍历 + 每地域实例空闲时间堆
        """
        arrivals, requests = self._generate_arrivals()
        arrivals.sort()

        # 每地域一个实例空闲时间堆
        free_heaps = [[0.0] * self.instances_per_region for _ in range(self.num_regions)]
        for h in free_heaps:
            heapq.heapify(h)

        for t, req_id, region in arrivals:
            req = requests[req_id]
            earliest_free = heapq.heappop(free_heaps[region])
            start = max(t, earliest_free)
            pf_time = self._sample_prefill_time(region)
            req.start_service_time = start
            req.end_service_time = start + pf_time
            req.service_duration = (start - t) + pf_time
            heapq.heappush(free_heaps[region], start + pf_time)

        return [r for r in requests if r.service_duration > 0]


def compute_stats(
    requests: List[Request],
    queue_wait_threshold_ms: float = 0.0,
    service_duration_target_ms: Optional[float] = None,
) -> Dict:
    """计算请求统计信息"""
    if not requests:
        return {"count": 0}

    durations = [r.service_duration for r in requests]
    wait_times = [r.wait_time for r in requests]

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

    queued_count = sum(1 for w in wait_times if w > queue_wait_threshold_ms)
    stats = {
        "count": len(durations),
        "queued_count": queued_count,
        "queued_ratio": queued_count / len(wait_times),
        "avg_duration": statistics.mean(durations),
        "p50_duration": percentile(durations, 50),
        "p90_duration": percentile(durations, 90),
        "p99_duration": percentile(durations, 99),
        "max_duration": max(durations),
        "avg_wait": statistics.mean(wait_times),
        "p99_wait": percentile(wait_times, 99),
        "max_wait": max(wait_times),
    }
    if service_duration_target_ms is not None:
        within_target_count = sum(1 for d in durations if d <= service_duration_target_ms)
        stats["within_target_count"] = within_target_count
        stats["within_target_ratio"] = within_target_count / len(durations)
    return stats


def compute_region_stats(
    requests: List[Request],
    num_regions: int,
    queue_wait_threshold_ms: float = 0.0,
    service_duration_target_ms: Optional[float] = None,
) -> List[Dict]:
    """按地域计算统计信息"""
    region_requests = {r: [] for r in range(num_regions)}
    for req in requests:
        region_requests[req.region].append(req)
    return [compute_stats(region_requests[r], queue_wait_threshold_ms, service_duration_target_ms) for r in range(num_regions)]


def print_comparison(sim: LLMDeploymentSimulation):
    """打印集中式 vs 分布式对比结果"""
    print("=" * 80)
    print("LLM部署方式仿真 - 集中式 vs 分布式")
    print("=" * 80)
    print(f"\n{'仿真参数':-^80}")
    print(f"  实例总数:       {sim.num_instances}")
    print(f"  用户总数:       {sim.num_users}")
    print(f"  地域数:         {sim.num_regions}")
    print(f"  每地域实例数:   {sim.instances_per_region}")
    print(f"  每地域用户数:   {sim.users_per_region}")
    print(f"  到达间隔:       {sim.arrival_rate:.0f} ms (每用户)")
    print(f"  Prefill时间:    {sim.prefill_time_min:.0f}~{sim.prefill_time_max:.0f} ms (均匀分布)")
    print(f"  仿真时长:       {sim.sim_duration:.0f} ms ({sim.sim_duration/3600000:.1f} 小时)")
    print(f"  随机种子:       {sim.seed}")

    total_arrival_rate = sim.num_users / sim.arrival_rate * 1000  # requests/s
    avg_prefill = (sim.prefill_time_min + sim.prefill_time_max) / 2
    total_capacity = sim.num_instances / avg_prefill * 1000  # requests/s
    print(f"\n  总到达率:       {total_arrival_rate:.2f} 请求/秒")
    print(f"  总处理能力:     {total_capacity:.2f} 请求/秒")
    print(f"  系统负载率:     {total_arrival_rate/total_capacity*100:.1f}%")

    print(f"\n{'运行仿真...':-^80}")
    cen_results = sim.run_centralized()
    dis_results = sim.run_distributed()

    cen_stats = compute_stats(cen_results, sim.queue_wait_threshold_ms)
    dis_stats = compute_stats(dis_results, sim.queue_wait_threshold_ms)

    print(f"\n{'整体对比':-^80}")
    print(f"{'指标':<20} {'集中式':>20} {'分布式':>20} {'差异':>20}")
    print("-" * 80)

    metrics = [
        ("完成请求数", "count", "{:.0f}", False),
        ("有排队请求数", "queued_count", "{:.0f}", False),
        ("排队请求比例", "queued_ratio", "{:.2%}", True),
        ("平均服务时长(ms)", "avg_duration", "{:.2f}", True),
        ("P50服务时长(ms)", "p50_duration", "{:.2f}", True),
        ("P90服务时长(ms)", "p90_duration", "{:.2f}", True),
        ("最大服务时长(ms)", "max_duration", "{:.2f}", True),
        ("平均等待时间(ms)", "avg_wait", "{:.2f}", True),
        ("最大等待时间(ms)", "max_wait", "{:.2f}", True),
    ]

    for label, key, fmt, show_diff in metrics:
        cv = cen_stats[key]
        dv = dis_stats[key]
        if show_diff:
            diff = dv - cv
            sign = "+" if diff >= 0 else ""
            print(f"{label:<20} {fmt.format(cv):>20} {fmt.format(dv):>20} {sign}{fmt.format(diff):>19}")
        else:
            print(f"{label:<20} {fmt.format(cv):>20} {fmt.format(dv):>20} {'':>20}")

    # 按地域分析
    print(f"\n{'分布式 - 按地域分析':-^80}")
    dis_region_stats = compute_region_stats(dis_results, sim.num_regions, sim.queue_wait_threshold_ms)
    cen_region_stats = compute_region_stats(cen_results, sim.num_regions, sim.queue_wait_threshold_ms)

    for region in range(sim.num_regions):
        rs = dis_region_stats[region]
        cs = cen_region_stats[region]
        print(f"\n  地域 {region}:")
        print(f"    {'指标':<20} {'集中式':>15} {'分布式':>15}")
        print(f"    {'-'*50}")
        for label, key, fmt in [
            ("完成请求数", "count", "{:.0f}"),
            ("有排队请求数", "queued_count", "{:.0f}"),
            ("排队请求比例", "queued_ratio", "{:.2%}"),
            ("平均服务时长(ms)", "avg_duration", "{:.2f}"),
            ("平均等待时间(ms)", "avg_wait", "{:.2f}"),
        ]:
            print(f"    {label:<20} {fmt.format(cs[key]):>15} {fmt.format(rs[key]):>15}")

    print(f"\n{'='*80}")
    return cen_results, dis_results


def plot_comparison(sim: LLMDeploymentSimulation, cen_results: List[Request], dis_results: List[Request]):
    """绘制对比图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("matplotlib未安装，跳过图表绘制。安装命令: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LLM部署方式仿真对比: 集中式 vs 分布式", fontsize=14)

    # 1. 服务时长分布直方图
    ax = axes[0, 0]
    cen_durations = [r.service_duration for r in cen_results]
    dis_durations = [r.service_duration for r in dis_results]
    max_dur = max(max(cen_durations), max(dis_durations))
    avg_prefill = (sim.prefill_time_min + sim.prefill_time_max) / 2
    bins = [avg_prefill + i * max(50, (max_dur - avg_prefill) / 50) for i in range(52)]
    ax.hist(cen_durations, bins=bins, alpha=0.6, label="集中式", color="steelblue", edgecolor="white")
    ax.hist(dis_durations, bins=bins, alpha=0.6, label="分布式", color="coral", edgecolor="white")
    ax.set_xlabel("服务时长 (ms)")
    ax.set_ylabel("请求数")
    ax.set_title("服务时长分布")
    ax.legend()
    ax.set_xlim(left=sim.prefill_time_min * 0.9)

    # 2. 等待时间分布直方图
    ax = axes[0, 1]
    cen_waits = [r.wait_time for r in cen_results if r.wait_time > sim.queue_wait_threshold_ms]
    dis_waits = [r.wait_time for r in dis_results if r.wait_time > sim.queue_wait_threshold_ms]
    if cen_waits or dis_waits:
        all_waits = cen_waits + dis_waits
        if all_waits:
            max_wait = max(all_waits)
            bins_w = [i * max(10, max_wait / 50) for i in range(52)]
            if cen_waits:
                ax.hist(cen_waits, bins=bins_w, alpha=0.6, label="集中式", color="steelblue", edgecolor="white")
            if dis_waits:
                ax.hist(dis_waits, bins=bins_w, alpha=0.6, label="分布式", color="coral", edgecolor="white")
    ax.set_xlabel("等待时间 (ms)")
    ax.set_ylabel("请求数")
    ax.set_title("排队等待时间分布 (排除零等待)")
    ax.legend()

    # 3. 分布式各地域服务时长箱线图
    ax = axes[1, 0]
    dis_by_region = {r: [] for r in range(sim.num_regions)}
    cen_by_region = {r: [] for r in range(sim.num_regions)}
    for req in dis_results:
        dis_by_region[req.region].append(req.service_duration)
    for req in cen_results:
        cen_by_region[req.region].append(req.service_duration)

    box_data = []
    box_labels = []
    for region in range(sim.num_regions):
        box_data.append(cen_by_region[region])
        box_labels.append(f"集中-地域{region}")
    for region in range(sim.num_regions):
        box_data.append(dis_by_region[region])
        box_labels.append(f"分布-地域{region}")

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
    colors = ["steelblue"] * sim.num_regions + ["coral"] * sim.num_regions
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("服务时长 (ms)")
    ax.set_title("各地域服务时长分布 (无异常值)")
    ax.tick_params(axis="x", rotation=45)

    # 4. 关键指标对比柱状图
    ax = axes[1, 1]
    cen_stats = compute_stats(cen_results, sim.queue_wait_threshold_ms)
    dis_stats = compute_stats(dis_results, sim.queue_wait_threshold_ms)
    x_labels = ["平均服务时长", "P90服务时长", "P99服务时长", "平均等待"]
    cen_vals = [cen_stats["avg_duration"], cen_stats["p90_duration"], cen_stats["p99_duration"], cen_stats["avg_wait"]]
    dis_vals = [dis_stats["avg_duration"], dis_stats["p90_duration"], dis_stats["p99_duration"], dis_stats["avg_wait"]]

    x = range(len(x_labels))
    w = 0.35
    ax.bar([i - w/2 for i in x], cen_vals, w, label="集中式", color="steelblue", alpha=0.8)
    ax.bar([i + w/2 for i in x], dis_vals, w, label="分布式", color="coral", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("ms")
    ax.set_title("关键延迟指标对比")
    ax.legend()

    plt.tight_layout()
    plt.show()
