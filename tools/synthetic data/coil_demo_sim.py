"""
coil_demo_sim.py — Galaxy tool wrapper
---------------------------------------
Synthetic coil-command generator for MiniMak demo sequences.

Adapted from minimak_firmware/Synthetic Data/coil_demo_sim.py for use as a
Galaxy tool. Parameters are passed via CLI arguments rather than hardcoded
constants, and outputs (CSV + PNG) are written to explicit paths.

Demo sequences (match TimerAO_Tick logic in FormMain.vb):
  1  TF1     — ramp TF bank 1 (AO ch 0)
  2  TF2     — ramp TF bank 2 (AO ch 1)
  3  PF1     — ramp PF bank 1 (AO ch 2)
  4  PF2     — ramp PF bank 2 (AO ch 3)
  7  PF1&2R  — alternate ramp PF1 / PF2 (ring movement)
  8  TF1&PF  — alternate ramp TF1 / PF1

PC DAQ channel mapping (FormMain.vb:2640-2646):
  AO ch 0 → TF bank 1
  AO ch 1 → TF bank 2
  AO ch 2 → PF bank 1
  AO ch 3 → PF bank 2
"""

import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


DEMO_META = {
    1: {"label": "SQ1 TF1 DEMO",  "title": "SQ1 — TF1 Demo (TF Bank 1 ramp)",                    "channels": ["TF1"]},
    2: {"label": "SQ2 TF2 DEMO",  "title": "SQ2 — TF2 Demo (TF Bank 2 ramp)",                    "channels": ["TF2"]},
    3: {"label": "SQ3 PF DEMO1",  "title": "SQ3 — PF Demo 1 (PF Bank 1 ramp)",                   "channels": ["PF1"]},
    4: {"label": "SQ4 PF2",       "title": "SQ4 — PF Bank 2 ramp",                               "channels": ["PF2"]},
    7: {"label": "SQ7 PF DEMO2",  "title": "SQ7 — PF Demo 2 (Ramp PF1 then PF2, ring movement)", "channels": ["PF1", "PF2"]},
    8: {"label": "SQ8 TF1&PF",    "title": "SQ8 — Ramp TF1 then PF1",                            "channels": ["TF1", "PF1"]},
}

CHANNEL_NAMES = {
    "TF1": "TF bank 1",
    "TF2": "TF bank 2",
    "PF1": "PF bank 1",
    "PF2": "PF bank 2",
}

CHANNEL_COLORS = {
    "TF1": "steelblue",
    "TF2": "tomato",
    "PF1": "seagreen",
    "PF2": "darkorange",
}


def clamp(value: float, limit: float) -> float:
    return max(0.0, min(value, limit))


def simulate(demo: int, interval_ms: float, aoinc: float,
             v_max: float, v_min: float, alimit: float, n_ticks: int) -> list:
    """
    Reproduce the TimerAO_Tick ramp logic from FormMain.vb for the
    selected demo sequence. Returns a list of dicts, one per tick.
    """
    if demo not in DEMO_META:
        raise ValueError(f"Unknown demo {demo}. Valid values: {sorted(DEMO_META)}")

    records = []
    Aout0: float         = 0.0
    counter: int         = 0
    channel_switch: bool = False
    t_ms: float          = 0.0

    for _ in range(n_ticks):
        t_ms += interval_ms

        counter += 1
        Aout0 += counter * aoinc
        if Aout0 >= v_max:
            Aout0 = v_min
            counter = 0

        ch = [0.0, 0.0, 0.0, 0.0]

        if demo in (1, 2, 3, 4):
            channel_map = {1: 0, 2: 1, 3: 2, 4: 3}
            ch[channel_map[demo]] = clamp(Aout0, alimit)

        elif demo == 7:
            if Aout0 == v_min:
                channel_switch = not channel_switch
            if channel_switch:
                ch[2] = clamp((v_max - Aout0) / 2, alimit)
                ch[3] = 0.0
            else:
                ch[3] = clamp(Aout0 / 2, alimit)
                ch[2] = 0.0

        elif demo == 8:
            if Aout0 == v_min:
                channel_switch = not channel_switch
            if channel_switch:
                ch[0] = clamp(v_max - Aout0, alimit)
                ch[2] = clamp(Aout0 / 2, alimit)
            else:
                ch[0] = clamp(Aout0, alimit)
                ch[2] = clamp((v_max - Aout0) / 2, alimit)

        records.append({
            "t_ms": round(t_ms, 3),
            "t_s":  round(t_ms / 1000.0, 6),
            "TF1":  round(ch[0], 6),
            "TF2":  round(ch[1], 6),
            "PF1":  round(ch[2], 6),
            "PF2":  round(ch[3], 6),
        })

    return records


def write_csv(records: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Written: {path}  ({len(records)} rows)")


def write_png(records: list, demo: int, output_path: str) -> None:
    meta = DEMO_META[demo]
    t = [r["t_s"] for r in records]

    fig, ax = plt.subplots(figsize=(12, 4))
    for col in meta["channels"]:
        ax.step(t, [r[col] for r in records], where="post",
                label=CHANNEL_NAMES[col], color=CHANNEL_COLORS[col])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PSU setpoint voltage (V)")
    ax.set_title(meta["title"])
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, format="png")
    plt.close()
    print(f"Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MiniMak synthetic coil-command generator")
    parser.add_argument("--demo",        type=int,   required=True,
                        choices=sorted(DEMO_META),
                        help="Demo sequence to simulate (1,2,3,4,7,8)")
    parser.add_argument("--duration",     type=float, default=20.0,
                        help="Total simulation duration [s]")
    parser.add_argument("--timestep-ms", type=float, default=100.0,
                        help="Timer period / timestep [ms]")
    parser.add_argument("--aoinc",       type=float, default=0.025,
                        help="Voltage step multiplier per tick")
    parser.add_argument("--v-max",       type=float, default=5.0,
                        help="Ramp ceiling [V]")
    parser.add_argument("--v-min",       type=float, default=0.0,
                        help="Ramp floor [V]")
    parser.add_argument("--alimit",      type=float, default=5.0,
                        help="Hardware output clamp [V]")
    parser.add_argument("--output-csv",  required=True,
                        help="Output voltage/current time-series CSV path")
    parser.add_argument("--output-png",  required=True,
                        help="Output channel voltage plot PNG path")
    args = parser.parse_args()

    n_ticks = max(1, int(args.duration * 1000 / args.timestep_ms))
    records = simulate(
        demo=args.demo,
        interval_ms=args.timestep_ms,
        aoinc=args.aoinc,
        v_max=args.v_max,
        v_min=args.v_min,
        alimit=args.alimit,
        n_ticks=n_ticks,
    )

    write_csv(records, args.output_csv)
    write_png(records, args.demo, args.output_png)


if __name__ == "__main__":
    main()
