"""
Synthetic coil-command generator for all active MiniMak demo sequences.
Galaxy-compatible wrapper — all parameters come from argparse; output paths
are supplied by Galaxy's job runner.

Demo sequences:
  1  TF1 DEMO  — ramp TF bank 1 (AO ch 0)
  2  TF2 DEMO  — ramp TF bank 2 (AO ch 1)
  3  PF DEMO1  — ramp PF bank 1 (AO ch 0)
  4  PF2       — ramp PF bank 2 (AO ch 1)
  7  PF DEMO2  — alternate ramp PF1 / PF2 (ring movement)
  8  TF1&PF    — alternate ramp TF1 / PF1

Reproduces the exact TimerAO_Tick logic from FormMain.vb.

PC DAQ channel mapping (FormMain.vb:2640-2646):
  AO ch 0 → TF bank 1
  AO ch 1 → TF bank 2
  AO ch 2 → PF bank 1
  AO ch 3 → PF bank 2
"""

import argparse
import csv

# ── module-level parameter defaults (overwritten by parse_args at runtime) ───
interval_ms: float = 100.0   # timer period (ms)
aoinc: float       = 0.025   # voltage step multiplier per tick
v_max: float       = 5.0     # ramp ceiling (V)
v_min: float       = 0.0     # ramp floor (V)
alimit: float      = 5.0     # hardware output clamp (V)
n_ticks: int       = 200     # number of timer ticks to simulate

DEMO_META = {
    1: {"label": "TF1 DEMO",  "title": "TF1 Demo — TF Bank 1 ramp",                     "channels": ["V_ch0"]},
    2: {"label": "TF2 DEMO",  "title": "TF2 Demo — TF Bank 2 ramp",                     "channels": ["V_ch1"]},
    3: {"label": "PF DEMO1",  "title": "PF Demo 1 — PF Bank 1 ramp",                    "channels": ["V_ch0"]},
    4: {"label": "PF2",       "title": "Seq 4 — PF Bank 2 ramp",                        "channels": ["V_ch1"]},
    7: {"label": "PF DEMO2",  "title": "PF Demo 2 — Ramp PF1 then PF2 (ring movement)", "channels": ["V_ch0", "V_ch1"]},
    8: {"label": "TF1&PF",    "title": "Seq 8 — Ramp TF1 then PF1",                     "channels": ["V_ch1", "V_ch2"]},
}

CHANNEL_NAMES = {
    "V_ch0": "TF bank 1 (AO ch 0)",
    "V_ch1": "TF bank 2 (AO ch 1)",
    "V_ch2": "PF bank 1 (AO ch 2)",
    "V_ch3": "PF bank 2 (AO ch 3)",
}

CHANNEL_COLORS = {
    "V_ch0": "steelblue",
    "V_ch1": "tomato",
    "V_ch2": "seagreen",
    "V_ch3": "darkorange",
}


# ── simulation ────────────────────────────────────────────────────────────────

def clamp(value: float, limit: float) -> float:
    return max(0.0, min(value, limit))


def simulate(demo: int) -> list[dict]:
    """Reproduce TimerAO_Tick logic for the requested demo sequence."""
    if demo not in DEMO_META:
        raise ValueError(f"Unknown demo {demo}. Valid values: {sorted(DEMO_META)}")

    records = []
    Aout0: float         = 0.0
    counter: int         = 0
    channel_switch: bool = False
    t_ms: float          = 0.0

    for _ in range(n_ticks):
        t_ms += interval_ms

        # ramp accumulator — FormMain.vb:2650-2657
        counter += 1
        Aout0 += counter * aoinc
        if Aout0 >= v_max:
            Aout0 = v_min
            counter = 0

        ch = [0.0, 0.0, 0.0, 0.0]

        if demo in (1, 2, 3, 4):
            # single-channel sawtooth — FormMain.vb:2665-2671
            channel_map = {1: 0, 2: 1, 3: 0, 4: 1}
            ch[channel_map[demo]] = clamp(Aout0, alimit)

        elif demo == 7:
            # alternate PF1 / PF2 — FormMain.vb:2701-2740
            if Aout0 == v_min:
                channel_switch = not channel_switch
            if channel_switch:
                ch[0] = clamp(v_max - Aout0, alimit)
                ch[1] = 0.0
            else:
                ch[1] = clamp(Aout0, alimit)
                ch[0] = 0.0

        elif demo == 8:
            # alternate TF1 / PF1 — FormMain.vb:2742-2773
            if Aout0 == v_min:
                channel_switch = not channel_switch
            if channel_switch:
                ch[1] = clamp(v_max - Aout0, alimit)
                ch[2] = clamp(Aout0 / 2, alimit)
            else:
                ch[1] = clamp(Aout0 / 2, alimit)
                ch[2] = clamp(v_max - Aout0, alimit)

        records.append({
            "t_ms":           round(t_ms, 3),
            "t_s":            round(t_ms / 1000.0, 6),
            "Aout0":          round(Aout0, 6),
            "channel_switch": int(channel_switch),
            "V_ch0":          round(ch[0], 6),
            "V_ch1":          round(ch[1], 6),
            "V_ch2":          round(ch[2], 6),
            "V_ch3":          round(ch[3], 6),
        })

    return records


# ── output writers ────────────────────────────────────────────────────────────

def write_csv(records: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows → {path}")


def write_png(records: list[dict], demo: int, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend for headless Galaxy jobs
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
    except ImportError:
        print("matplotlib not available — skipping PNG output.")
        return

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
    plt.savefig(path, dpi=150, format="png")
    print(f"Saved plot → {path}")
    plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────────────────────

# maps the select value from the XML to the internal demo integer
DEMO_KEY_MAP = {
    "tf1_demo": 1,
    "tf2_demo": 2,
    "pf_demo1": 3,
    "pf2":      4,
    "pf_demo2": 7,
    "tf1_pf":   8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic coil voltage waveforms for MiniMak demo sequences."
    )
    p.add_argument(
        "--demo", choices=list(DEMO_KEY_MAP), required=True,
        help="Demo name: " + ", ".join(f"{k} (seq {v})" for k, v in DEMO_KEY_MAP.items()),
    )
    p.add_argument("--interval_ms", type=float, default=100.0,
                   help="Timer interval in ms (default: 100)")
    p.add_argument("--aoinc",       type=float, default=0.025,
                   help="Voltage step multiplier per tick (default: 0.025)")
    p.add_argument("--v_max",       type=float, default=5.0,
                   help="Ramp ceiling in V (default: 5.0)")
    p.add_argument("--v_min",       type=float, default=0.0,
                   help="Ramp floor in V (default: 0.0)")
    p.add_argument("--alimit",      type=float, default=5.0,
                   help="Hardware output clamp in V (default: 5.0)")
    p.add_argument("--n_ticks",     type=int,   default=200,
                   help="Number of timer ticks to simulate (default: 200)")
    p.add_argument("--output_csv",  required=True,
                   help="Output path for the CSV dataset")
    p.add_argument("--output_png",  default=None,
                   help="Output path for the PNG plot (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = DEMO_KEY_MAP[args.demo]

    # push CLI values into module globals so simulate() picks them up
    interval_ms = args.interval_ms
    aoinc       = args.aoinc
    v_max       = args.v_max
    v_min       = args.v_min
    alimit      = args.alimit
    n_ticks     = args.n_ticks

    records = simulate(demo)
    write_csv(records, args.output_csv)

    if args.output_png:
        write_png(records, demo, args.output_png)
