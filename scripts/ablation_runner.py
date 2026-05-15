from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from typing import List


@dataclass(frozen=True)
class AblationConfig:
    gene_layers: int
    shallow_type: str
    shallow_layers: int
    use_tleg: bool
    distill_mode: str
    frozen: bool
    teacher_total_layers: int

    @property
    def tag(self) -> str:
        return (
            f"gene-last{self.gene_layers}_"
            f"shallow-{self.shallow_type}{self.shallow_layers}_"
            f"tleg-{'y' if self.use_tleg else 'n'}_"
            f"distill-{self.distill_mode}_"
            f"frozen-{'y' if self.frozen else 'n'}"
        )

    @property
    def gene_layer_ids(self) -> List[int]:
        g = max(0, min(int(self.gene_layers), int(self.teacher_total_layers)))
        return list(range(self.teacher_total_layers - g + 1, self.teacher_total_layers + 1))

    @property
    def distill_tap_layers(self) -> List[int]:
        first_gene = min(self.gene_layer_ids) if self.gene_layer_ids else self.teacher_total_layers + 1
        remaining = list(range(1, first_gene))
        if not remaining:
            return []
        r = max(remaining)
        taps = [max(1, min(r, math.floor(r * k / 3))) for k in range(1, 4)]
        return sorted(set(taps))


def build_grid(
    shallow_types: List[str],
    shallow_layers: int,
    gene_layer_values: List[int],
    tleg_values: List[str],
    distill_modes: List[str],
    frozen_values: List[str],
    teacher_total_layers: int,
) -> List[AblationConfig]:
    def _to_bool(x: str) -> bool:
        x = str(x).strip().lower()
        if x in {"y", "yes", "true", "1"}:
            return True
        if x in {"n", "no", "false", "0"}:
            return False
        raise ValueError(f"Unsupported boolean-like value: {x!r}")

    grid = []
    for gene_layers, shallow_type, use_tleg, distill_mode, frozen in itertools.product(
        gene_layer_values, shallow_types, tleg_values, distill_modes, frozen_values
    ):
        grid.append(
            AblationConfig(
                gene_layers=int(gene_layers),
                shallow_type=shallow_type,
                shallow_layers=int(shallow_layers),
                use_tleg=_to_bool(use_tleg),
                distill_mode=str(distill_mode),
                frozen=_to_bool(frozen),
                teacher_total_layers=int(teacher_total_layers),
            )
        )
    return grid


def slice_batch(configs: List[AblationConfig], num_batches: int, batch_index: int) -> List[AblationConfig]:
    if num_batches <= 1:
        return configs
    if batch_index < 0 or batch_index >= num_batches:
        raise ValueError(f"batch_index must be in [0, {num_batches}), got {batch_index}")

    out = []
    for idx, cfg in enumerate(configs):
        if idx % num_batches == batch_index:
            out.append(cfg)
    return out


def build_command(
    python_bin: str,
    entry: str,
    cfg: AblationConfig,
    base_args: List[str],
    out_dir_root: str,
) -> List[str]:
    cmd = [python_bin, "-m", entry]
    for frag in base_args:
        cmd.extend(shlex.split(frag))

    cmd.extend(["--shallow_layers", str(cfg.shallow_layers)])
    cmd.extend(["--shallow_type", cfg.shallow_type])
    cmd.extend(["--gene_layers", str(cfg.gene_layers)])
    cmd.extend(["--distill_mode", cfg.distill_mode])

    cmd.append("--no_clip_init")
    cmd.append("--use_tleg" if cfg.use_tleg else "--no_tleg")
    cmd.append("--frozen" if cfg.frozen else "--no_frozen")

    if out_dir_root:
        cmd.extend(["--out_dir", os.path.join(out_dir_root, cfg.tag)])
    return cmd


def main():
    p = argparse.ArgumentParser("Batch ablation command generator / runner")
    p.add_argument("--entry", type=str, default="scripts.pretrain_cc3m", help="python -m <entry>")
    p.add_argument("--python_bin", type=str, default="python")
    p.add_argument("--base_arg", action="append", default=[], help="Extra arg fragment appended to every run.")
    p.add_argument("--out_dir_root", type=str, default="outputs/ablations")

    p.add_argument("--shallow_types", type=str, nargs="+", default=["transformer", "cnn"])
    p.add_argument("--shallow_layers", type=int, default=3)
    p.add_argument("--gene_layer_values", type=int, nargs="+", default=[2, 3])
    p.add_argument("--tleg_values", type=str, nargs="+", default=["y", "n"])
    p.add_argument("--distill_modes", type=str, nargs="+", default=["none", "tap", "tap_logit"], choices=["none", "tap", "tap_logit"])
    p.add_argument("--frozen_values", type=str, nargs="+", default=["y", "n"])
    p.add_argument("--teacher_total_layers", type=int, default=12)

    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--batch_index", type=int, default=0)
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--run", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")

    args = p.parse_args()

    configs = build_grid(
        shallow_types=args.shallow_types,
        shallow_layers=args.shallow_layers,
        gene_layer_values=args.gene_layer_values,
        tleg_values=args.tleg_values,
        distill_modes=args.distill_modes,
        frozen_values=args.frozen_values,
        teacher_total_layers=args.teacher_total_layers,
    )
    configs = slice_batch(configs, num_batches=args.num_batches, batch_index=args.batch_index)

    manifest = []
    for cfg in configs:
        cmd = build_command(
            python_bin=args.python_bin,
            entry=args.entry,
            cfg=cfg,
            base_args=args.base_arg,
            out_dir_root=args.out_dir_root,
        )
        manifest.append(
            {
                **asdict(cfg),
                "tag": cfg.tag,
                "gene_layer_ids": cfg.gene_layer_ids,
                "distill_tap_layers": cfg.distill_tap_layers,
                "command": cmd,
                "command_str": shlex.join(cmd),
            }
        )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entry": args.entry,
                    "num_batches": args.num_batches,
                    "batch_index": args.batch_index,
                    "runs": manifest,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    for item in manifest:
        print(item["command_str"])

    if not args.run:
        return

    for item in manifest:
        ret = subprocess.run(item["command"], check=False)
        if ret.returncode != 0 and args.stop_on_error:
            raise SystemExit(ret.returncode)


if __name__ == "__main__":
    main()
