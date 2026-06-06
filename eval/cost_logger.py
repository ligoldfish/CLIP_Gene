"""收敛/效率记录（命门图）：记 epoch/step/wall-clock 到达目标指标。

三 init（gene/random/scratch）各产一 CSV → compare 出"暖启动更快收敛"对比。
注：time.time 在脚本运行期可用（非 workflow 沙箱），此处直接用。
"""
import csv
import os
import time
from typing import List, Optional


class CostLogger:
    def __init__(self, run_name: str, init_mode: str, target_metric: str,
                 target_value: float, out_csv: str):
        self.run_name = run_name
        self.init_mode = init_mode
        self.target_metric = target_metric
        self.target_value = target_value
        self.out_csv = out_csv
        self.t0 = time.time()
        self.rows = []
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["run", "init", "epoch", "step", "wall_s", "metric"])

    def log(self, epoch: int, step: int, metric_value: float):
        wall = time.time() - self.t0
        self.rows.append((epoch, step, wall, metric_value))
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([self.run_name, self.init_mode, epoch, step,
                                    f"{wall:.1f}", f"{metric_value:.4f}"])

    def steps_to_target(self) -> Optional[dict]:
        for (epoch, step, wall, m) in self.rows:
            if m >= self.target_value:
                return {"epoch": epoch, "step": step, "wall_s": wall, "metric": m}
        return None

    @staticmethod
    def compare(csv_paths: List[str], target_value: float, metric_name: str = "metric") -> dict:
        out = {}
        for path in csv_paths:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            init = rows[0]["init"]
            hit = next((r for r in rows if float(r["metric"]) >= target_value), None)
            out[init] = None if hit is None else {
                "epoch": int(hit["epoch"]), "step": int(hit["step"]), "wall_s": float(hit["wall_s"])}
        return out
