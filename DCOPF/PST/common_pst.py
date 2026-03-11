import pandas as pd
import math

N_STEPS     = 21
NEUTRAL_TAP = 10
ALPHA_STEP  = 3.0   # degrees per tap step
PHI_MAX_DEG = ALPHA_STEP * NEUTRAL_TAP    # 30°
PHI_MAX_RAD = math.radians(PHI_MAX_DEG)

def build_pst_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct the two DataFrames required by create_phase_tap_changers.

    ptc_df — one row per PST transformer
    ──────────────────────────────────────
      id                : matches the transformer id
      target_deadband   : regulation deadband (MW) — unused in OPF (manual mode)
      regulation_mode   : 'FIXED_TAP' for OPF-controlled PST
                          'ACTIVE_POWER_CONTROL' for closed-loop regulation
      low_tap           : index of the first step (0-based)
      tap               : current active tap (we start at neutral = 10)

    steps_df — one row per tap step, repeated id for each step of same PST
    ────────────────────────────────────────────────────────────────────────
      id    : transformer id (repeated for each step)
      alpha : phase shift angle in degrees for this step
      rho   : turns ratio (1.0 = ideal, no voltage magnitude change)
      r, x  : additional series impedance per step (pu, on transformer base)
               set to 0 for an ideal PST
      g, b  : additional shunt admittance per step (pu)
               set to 0 for an ideal PST

    Note on alpha sign convention in pypowsybl
    -------------------------------------------
      alpha > 0 : phase of winding-2 leads winding-1  → P flows from 1 to 2
      alpha < 0 : phase of winding-2 lags  winding-1  → P flows from 2 to 1
      Consistent with IEC 60076-3.
    """
    # ── ptc_df: one row for PST_T ─────────────────────────────────────────────
    ptc_df = pd.DataFrame.from_records(
        index="id",
        columns=["id", "target_deadband", "regulation_mode", "low_tap", "tap", "regulating"],
        data=[
            ("PST_T",
             0.0,              # no deadband — OPF sets tap explicitly
             "ACTIVE_POWER_CONTROL",      # manual mode; OPF writes the tap via update_phase_tap_changers
             0,                # low_tap: first step index
             NEUTRAL_TAP,
             False)      # initial active tap (0° shift)
        ]
    )

    # ── steps_df: one row per tap step ────────────────────────────────────────
    # alpha goes from −30° (tap 0) to +30° (tap 20) in 3° increments
    steps_records = []
    for tap_idx in range(N_STEPS):
        alpha_deg = (tap_idx - NEUTRAL_TAP) * ALPHA_STEP   # −30, −27, ..., 0, ..., +30
        steps_records.append((
            "PST_T",    # id
            0.0,        # b  (no additional shunt susceptance)
            0.0,        # g  (no additional shunt conductance)
            0.0,        # r  (no additional series resistance)
            0.0,        # x  (no additional series reactance — base x set on transformer)
            1.0,        # rho (ideal turns ratio)
            alpha_deg,  # alpha: phase shift in degrees
        ))

    steps_df = pd.DataFrame.from_records(
        index="id",
        columns=["id", "b", "g", "r", "x", "rho", "alpha"],
        data=steps_records,
    )

    return ptc_df, steps_df
