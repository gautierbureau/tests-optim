"""
DC OPF — PTDF + PSDF formulation for the bypass PST topology
=============================================================

Topology (from dc_opf_pst_focus.py)
-------------------------------------
         ┌──── L12a  (B1→B2b, bypass) ─────────────────┐
G1 ── B1─┤                                              ├── B2b ──L2b_3── B3 ── G2/Load
         └──── L1_2a (B1→B2a) ── B2a ──[PST_T φ]──────┘

Key difference vs theta formulation
-------------------------------------
  Theta : variables θ_B2a, θ_B2b, θ_B3 + 4 flow-definition constraints
  PTDF  : no angle variables — branch flows are explicit linear functions
          of nodal injections and the PST angle φ via PTDF/PSDF coefficients

PTDF derivation (analytical, B1=slack, φ=0)
---------------------------------------------
  Susceptances:
    b_a    = 500  MW/rad  (L12a  x=0.2 pu)
    b_in   = 250  MW/rad  (L1_2a x=0.4 pu)
    b_pst  = 1000 MW/rad  (PST_T x=0.1 pu)
    b_out  = 500  MW/rad  (L2b_3 x=0.2 pu)

  Reduced susceptance matrix B_red (buses B2a, B2b, B3 — slack B1 removed):

           B2a    B2b    B3
    B2a  [ 1250  -1000    0  ]
    B2b  [-1000   2000  -500 ]
    B3   [    0   -500   500 ]

  det(B_red) = 437 500 000

  B_red_inv = (1/437500000) · [ 750000  500000  500000 ]
                                [ 500000  625000  625000 ]
                                [ 500000  625000 1500000 ]

  Since B2a and B2b are transit buses (no gen/load), only B3 has non-zero
  net injection.  PTDF(l, B3) = -b_l · [B_red_inv · e_B3]_endpoint:

    PTDF(L12a,  B3) = -b_a   · θ_B2b(e_B3) = -500 · 625000/D = -5/7 ≈ −0.7143
    PTDF(L1_2a, B3) = -b_in  · θ_B2a(e_B3) = -250 · 500000/D = -2/7 ≈ −0.2857
    PTDF(PST,   B3) = -b_pst · (θ_B2a-θ_B2b)(e_B3)
                    = -1000  · (500000-625000)/D = +2/7 ≈ −0.2857  [same sign as L1_2a]
    PTDF(L2b3,  B3) = -b_out · (θ_B2b-θ_B3)(e_B3)
                    = -500   · (625000-1500000)/D = -1.0

PSDF derivation  ∂P_l/∂φ  (unit PST angle, no net injection)
--------------------------------------------------------------
  PST injects +b_pst·φ at B2a and withdraws -b_pst·φ at B2b.
  Δθ = B_red_inv · [+1000φ, -1000φ, 0]^T

    Δθ_B2a = (1000·750000 - 1000·500000)/D · φ = +4/7 · φ
    Δθ_B2b = (1000·500000 - 1000·625000)/D · φ = -2/7 · φ
    Δθ_B3  = (1000·500000 - 1000·625000)/D · φ = -2/7 · φ

  PSDF(L12a)  = -b_a  · Δθ_B2b/φ          = -500 · (-2/7) = +1000/7 ≈ +142.9 MW/rad
  PSDF(L1_2a) = -b_in · Δθ_B2a/φ          = -250 · (+4/7) = -1000/7 ≈ -142.9 MW/rad
  PSDF(PST)   =  b_pst·(Δθ_B2a/φ-Δθ_B2b/φ+1) = 1000·(4/7+2/7+1) = +13000/7 ≈ +1857 MW/rad
  PSDF(L2b3)  =  b_out·(Δθ_B2b/φ-Δθ_B3/φ)  = 500·(-2/7-(-2/7)) = 0

Key insight from PSDF(L2b3) = 0
---------------------------------
  The PST shifts flow between the bypass path (L12a) and the PST path
  (L1_2a → PST_T) but NEVER changes the total flow reaching B3 via L2b_3.

  This is because the PST drives a circulating current around the inner loop
  [B1→L1_2a→B2a→PST→B2b→L12a(rev)→B1] without affecting the outer flow
  to B3.  The OPF uses φ to protect the more-loaded parallel path.

Full PTDF natural flow equations
----------------------------------
  P̃_l = PTDF(l, B3)·(Pg_G2 − Pd_B3) + PSDF(l)·φ   ∀ l

With Pd_B3 = 250 MW and PTDF/PSDF as derived:

  P̃_L12a  = -5/7·(Pg_G2-250) + 1000/7·φ  = (1250 - 5·Pg_G2)/7  + (1000/7)·φ
  P̃_L1_2a = -2/7·(Pg_G2-250) - 1000/7·φ  = ( 500 - 2·Pg_G2)/7  - (1000/7)·φ
  P̃_PST   = -2/7·(Pg_G2-250) + 13000/7·φ
  P̃_L2b3  = -1.0·(Pg_G2-250)              = 250 - Pg_G2   (independent of φ!)
"""

import math
from typing import Any

import numpy as np
import pypowsybl.network as pn
import pypowsybl.loadflow as plf
import pypowsybl.sensitivity as psa
import pypowsybl as ppw
import pandas as pd
import pyoptinterface as poi
from pandas import DataFrame
from pyoptinterface import xpress
from dc_opf_pst_focus import build_network

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MVA    = 100.0
KV          = 400.0
Z_BASE      = KV**2 / BASE_MVA

B_L12A      = BASE_MVA / 0.2        # 500  MW/rad
B_L1_2A     = BASE_MVA / 0.4        # 250  MW/rad
B_PST       = BASE_MVA / 0.1        # 1000 MW/rad
B_L2B_3     = BASE_MVA / 0.2        # 500  MW/rad

PMAX_L12A   = 130.0
PMAX_L1_2A  = 150.0
PMAX_PST    = 150.0
PMAX_L2B_3  = 200.0

PHI_MAX_DEG = 30.0
PHI_MAX_RAD = math.radians(PHI_MAX_DEG)
C_PST       = 5.0

BRANCHES    = ["L12a", "L1_2a", "PST_T", "L2b_3"]
GENS        = ["G1", "G2"]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Analytical PTDF and PSDF
# ─────────────────────────────────────────────────────────────────────────────

def compute_sensitivities_analytical() -> dict:
    """
    Returns PTDF and PSDF derived from the B_red matrix inversion.

    PTDF(l, G2): sensitivity of flow on l to a unit injection at B3 (G2 bus)
                 — withdrawal at slack B1
    PSDF(l):     sensitivity of flow on l to a unit PST angle shift (rad)
    """
    D = 437_500_000.0   # det(B_red)

    # B_red_inv columns (each is the angle response to unit injection at that bus)
    # Column for B3 (3rd column, 0-indexed):
    inv_col_B3 = np.array([500_000, 625_000, 1_500_000]) / D
    # θ_B2a, θ_B2b, θ_B3 per unit injection at B3

    # Branch flow = b_branch * Δθ_across_branch
    ptdf = {
        "L12a" : -B_L12A  *  inv_col_B3[1],                    # -b_a  * θ_B2b
        "L1_2a": -B_L1_2A *  inv_col_B3[0],                    # -b_in * θ_B2a
        "PST_T": -B_PST   * (inv_col_B3[0] - inv_col_B3[1]),   # -b_pst*(θ_B2a-θ_B2b)
        "L2b_3": -B_L2B_3 * (inv_col_B3[1] - inv_col_B3[2]),   # -b_out*(θ_B2b-θ_B3)
    }

    # Angle response to unit PST shift (effective injections +b_pst at B2a, -b_pst at B2b)
    inv_col_B2a = np.array([750_000, 500_000, 500_000]) / D
    inv_col_B2b = np.array([500_000, 625_000, 625_000]) / D
    delta_theta_phi = B_PST * (inv_col_B2a - inv_col_B2b)
    # delta_theta_phi[i] = Δθ_i per radian of φ

    psdf = {
        "L12a" : -B_L12A  *  delta_theta_phi[1],
        "L1_2a": -B_L1_2A *  delta_theta_phi[0],
        "PST_T":  B_PST   * (delta_theta_phi[0] - delta_theta_phi[1] - 1.0),
        "L2b_3":  B_L2B_3 * (delta_theta_phi[1] - delta_theta_phi[2]),
    }

    return {"ptdf": ptdf, "psdf": psdf}


def verify_sensitivities_pypowsybl(network: pn.Network, sens: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Cross-check analytical PTDF/PSDF against pypowsybl sensitivity analysis.
    Uses distributed_slack=False so G1 absorbs all imbalance (pure slack).
    """
    print("── Sensitivity verification: analytical vs pypowsybl ───────────")

    sa = psa.create_dc_analysis()

    # PTDF: sensitivity of branch flows to generator injections
    sa.add_branch_flow_factor_matrix(
        branches_ids  = BRANCHES,
        variables_ids = ["G1", "G2", "D3"],
        matrix_id     = "PTDF",
    )

    # PSDF: sensitivity of branch flows to PST angle
    # pypowsybl uses the phase tap changer id as the variable
    sa.add_branch_flow_factor_matrix(
        branches_ids  = BRANCHES,
        variables_ids = ["PST_T"],
        matrix_id     = "PSDF",
    )

    lf_params = plf.Parameters(distributed_slack=True)
    result    = sa.run(network, parameters=lf_params)

    df_ptdf = result.get_sensitivity_matrix("PTDF")
    df_psdf = result.get_sensitivity_matrix("PSDF")

    print(f"  {'Branch':<8}  {'PTDF_G2 (ana)':>14}  {'PTDF_G2 (ppw)':>14}  "
          f"{'PSDF (ana)':>12}  {'PSDF (ppw)':>12}")
    print("  " + "─" * 70)

    # pypowsybl PSDF is in MW/rad but the sensitivity may be in MW/degree
    # — check units from the value magnitude
    for br in BRANCHES:
        ptdf_ana  = sens["ptdf"][br]
        psdf_ana  = sens["psdf"][br]
        ptdf_ppw  = float(df_ptdf.loc["G2", br])
        psdf_ppw  = -float(df_psdf.loc["PST_T", br]) * (180 / math.pi)

        match_ptdf = "✓" if abs(ptdf_ana - ptdf_ppw) < 0.01 else "✗"
        match_psdf = "✓" if abs(psdf_ana - psdf_ppw) < 1.0   else "~"
        print(f"  {br:<8}  {ptdf_ana:>14.6f}  {ptdf_ppw:>14.6f} {match_ptdf} "
              f"  {psdf_ana:>12.4f}  {psdf_ppw:>12.4f} {match_psdf}")
    print()

    # ptdf = {
    #     "L12a": df_ptdf.loc["G2", "L12a"],
    #     "L1_2a": df_ptdf.loc["G2", "L1_2a"],
    #     "PST_T": df_ptdf.loc["G2", "PST_T"],
    #     "L2b_3": df_ptdf.loc["G2", "L2b_3"],  # -b_out*(θ_B2b-θ_B3)
    # }

    ptdf = {gen: {br: float(result.get_sensitivity_matrix("PTDF").loc[gen, br]) for br in BRANCHES} for gen in ["G1", "G2", "D3"]}

    # psdf = {
    #     "L12a": df_psdf.loc["PST_T", "L12a"] * (180 / math.pi),
    #     "L1_2a": df_psdf.loc["PST_T", "L1_2a"] * (180 / math.pi),
    #     "PST_T": df_psdf.loc["PST_T", "PST_T"] * (180 / math.pi),
    #     "L2b_3": df_psdf.loc["PST_T", "L2b_3"] * (180 / math.pi),
    # }

    psdf = {br: float(df_psdf.loc["PST_T", br])
                * (180 / math.pi)  # convert MW/deg → MW/rad
            for br in BRANCHES}

    return ptdf, psdf

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Parameters
# ─────────────────────────────────────────────────────────────────────────────

def extract_parameters(network: pn.Network) -> dict:
    gens  = network.get_generators()
    loads = network.get_loads()
    return dict(
        Pg_min = {g: float(gens.loc[g, "min_p"]) for g in gens.index},
        Pg_max = {g: float(gens.loc[g, "max_p"]) for g in gens.index},
        Pd_B3  = float(loads.loc["D3", "p0"]),
        cost   = {"G1": 30.0, "G2": 45.0},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LP solver — PTDF formulation
# ─────────────────────────────────────────────────────────────────────────────

def solve_lp_ptdf(params: dict,
                  sens: dict,
                  pmax_l12a:  float = PMAX_L12A,
                  pmax_l1_2a: float = PMAX_L1_2A,
                  pmax_pst:   float = PMAX_PST,
                  pmax_l2b3:  float = PMAX_L2B_3,
                  phi_fixed:  float = None) -> dict:
    """
    PTDF-based DC-OPF.

    Variables vs theta formulation
    --------------------------------
    Theta (previous)    PTDF (this)
    ─────────────────   ───────────────────────────────
    Pg_G1, Pg_G2        Pg_G1, Pg_G2          (same)
    θ_B2a, θ_B2b, θ_B3 (eliminated)
    P_L12a  + flow def  P_L12a  (bounded, PTDF eq replaces flow def)
    P_L1_2a + flow def  P_L1_2a
    P_PST   + flow def  P_PST
    P_L2b3  + flow def  P_L2b3
    φ, ψ                φ, ψ                  (same)
    ─────────────────   ───────────────────────────────
    11 variables        9 variables  (−3 angles, same branch flows)
    8  equalities       6 equalities (−4 flow defs + 2 PTDF eqs... 
                                      actually same count differently)

    PTDF flow constraint per branch l:
      P_l = PTDF(l,G2)·(Pg_G2 − Pd_B3) + PSDF(l)·φ

    Since PSDF(L2b3) = 0, the L2b3 equation reduces to:
      P_L2b3 = -1·(Pg_G2 - 250) = 250 - Pg_G2
    which is equivalent to the B3 balance.  We keep both for explicitness
    and let the solver handle the redundancy.
    """
    Pg_min = params["Pg_min"]
    Pg_max = params["Pg_max"]
    Pd_B3  = params["Pd_B3"]
    cost   = params["cost"]
    ptdf   = sens["ptdf"]
    psdf   = sens["psdf"]

    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)
    obj = poi.ExprBuilder()

    # ── Generator outputs ─────────────────────────────────────────────────────
    Pg = {g: model.add_variable(lb=Pg_min[g], ub=Pg_max[g], name=f"Pg_{g}")
          for g in ["G1", "G2"]}

    # ── Branch flow variables with thermal limit bounds ───────────────────────
    pmax = {
        "L12a" : pmax_l12a,
        "L1_2a": pmax_l1_2a,
        "PST_T": pmax_pst,
        "L2b_3": pmax_l2b3,
    }
    P = {br: model.add_variable(lb=-pmax[br], ub=pmax[br], name=f"P_{br}")
         for br in BRANCHES}

    # ── PST angle ─────────────────────────────────────────────────────────────
    if phi_fixed is not None:
        phi = model.add_variable(lb=phi_fixed, ub=phi_fixed, name="phi")
        psi = model.add_variable(lb=abs(phi_fixed), ub=abs(phi_fixed), name="psi")
    else:
        phi = model.add_variable(lb=-PHI_MAX_RAD, ub=PHI_MAX_RAD, name="phi")
        psi = model.add_variable(lb=0.0, ub=PHI_MAX_RAD, name="psi")

    # ── PTDF flow constraints ─────────────────────────────────────────────────
    #
    # P_l = PTDF(l,G2)·(Pg_G2 − Pd_B3) + PSDF(l)·φ
    # Rearranged:
    #   P_l − PTDF(l,G2)·Pg_G2 − PSDF(l)·φ = PTDF(l,G2)·(−Pd_B3)
    #
    cons_ptdf = {}
    for br in BRANCHES:
        # rhs = ptdf[br] * (-Pd_B3)
        # cons_ptdf[br] = model.add_linear_constraint(
        #     P[br] - ptdf[br] * Pg["G2"] - psdf[br] * phi,
        #     poi.Eq, rhs,
        #     name=f"ptdf_{br}",
        # )
        rhs = ptdf["D3"][br] * (-Pd_B3)
        cons_ptdf[br] = model.add_linear_constraint(
            P[br] - ptdf["G1"][br]*Pg["G1"] - ptdf["G2"][br]*Pg["G2"] - psdf[br]*phi,
            poi.Eq, rhs, name=f"ptdf_{br}",
        )

    # ── Nodal balance ─────────────────────────────────────────────────────────
    #
    # Only two independent balances needed:
    #
    # B1 (slack source):  Pg_G1 − P_L12a − P_L1_2a = 0
    #   Note: B2a, B2b are transit (no gen/load) — their balances are
    #   implicitly enforced by the PTDF construction
    con_B1 = model.add_linear_constraint(
        Pg["G1"] - P["L12a"] - P["L1_2a"],
        poi.Eq, 0.0, name="bal_B1",
    )

    # B3 (load bus):  Pg_G2 + P_L2b_3 = Pd_B3
    #   This fixes the total system balance (Pg_G1+Pg_G2 = Pd_B3)
    con_B3 = model.add_linear_constraint(
        Pg["G2"] + P["L2b_3"],
        poi.Eq, Pd_B3, name="bal_B3",
    )

    # ── |φ| linearisation ────────────────────────────────────────────────────
    model.add_linear_constraint(psi - phi, poi.Geq, 0.0, name="psi_pos")
    model.add_linear_constraint(psi + phi, poi.Geq, 0.0, name="psi_neg")

    # ── Objective ─────────────────────────────────────────────────────────────
    for g in ["G1", "G2"]:
        obj += cost[g] * Pg[g]
    obj += C_PST * psi
    model.set_objective(obj, poi.ObjectiveSense.Minimize)

    # ── Solve ─────────────────────────────────────────────────────────────────
    model.optimize()
    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status != poi.TerminationStatusCode.OPTIMAL:
        raise RuntimeError(f"LP not optimal: {status}")

    def val(v):  return model.get_value(v)
    def dual(c): return model.get_constraint_attribute(
        c, poi.ConstraintAttribute.Dual)

    phi_val = val(phi)
    return {
        "status"      : str(status),
        "total_cost"  : model.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
        "Pg"          : {g: val(Pg[g]) for g in ["G1","G2"]},
        "P"           : {br: val(P[br]) for br in BRANCHES},
        "phi_deg"     : math.degrees(phi_val),
        "phi_rad"     : phi_val,
        "psi_rad"     : val(psi),
        "LMP"         : {
            "B1": dual(con_B1),
            "B3": dual(con_B3),
        },
        "ptdf_duals"  : {br: dual(cons_ptdf[br]) for br in BRANCHES},
        "pmax"        : pmax,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Validate with pypowsybl DC load-flow
# ─────────────────────────────────────────────────────────────────────────────

def validate(results: dict) -> None:
    net = build_network()
    for g, pval in results["Pg"].items():
        net.update_generators(id=g, target_p=pval)

    phi_deg = results["phi_deg"]
    tap     = max(0, min(20, 10 + round(phi_deg / 3.0)))
    net.update_phase_tap_changers(id="PST_T", tap=tap)

    lf_params = plf.Parameters(distributed_slack=False)
    lf = ppw.loadflow.run_dc(net, parameters=lf_params)

    lines  = net.get_lines(all_attributes=True)
    trafos = net.get_2_windings_transformers(all_attributes=True)
    gens   = net.get_generators(all_attributes=True)

    lf_flows = {
        "L12a" : lines.loc["L12a",  "p1"],
        "L1_2a": lines.loc["L1_2a", "p1"],
        "PST_T": trafos.loc["PST_T","p1"],
        "L2b_3": lines.loc["L2b_3", "p1"],
    }

    print("\n── pypowsybl validation ─────────────────────────────────────────")
    for c in lf:
        print(f"  Component {c.connected_component_num}: {c.status}")
    print(f"  PST tap: {tap}  (φ_opt={phi_deg:+.2f}°, tap≈{(tap-10)*3:+.0f}°)")
    print(f"  {'Branch':<8}  {'OPF':>8}  {'LF':>8}  {'Δ':>7}")
    for br in BRANCHES:
        opf = results["P"][br]
        lf_v = lf_flows[br]
        print(f"  {br:<8}  {opf:>+8.2f}  {lf_v:>+8.2f}  {opf-lf_v:>+7.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(r: dict, sens: dict, title: str = "") -> None:
    sep = "─" * 70
    ptdf = sens["ptdf"]
    psdf = sens["psdf"]
    Pd   = 250.0

    print(f"\n{'═'*70}")
    if title:
        print(f"  {title}")
        print(f"{'═'*70}")

    gen_cost = sum(r["Pg"][g]*c for g,c in zip(["G1","G2"],[30,45]))
    pst_cost = C_PST * r["psi_rad"]
    print(f"  Status  : {r['status']}")
    print(f"  Cost    : ${r['total_cost']:>10,.2f}/h  "
          f"(gen=${gen_cost:,.0f}  PST_wear=${pst_cost:.2f})")
    print(sep)
    for g, mn, mx, c in [("G1",0,300,30),("G2",0,150,45)]:
        pg  = r["Pg"][g]
        lim = " ◄" if (abs(pg-mn)<0.5 or abs(pg-mx)<0.5) else ""
        print(f"  {g}: {pg:>7.2f} MW  [{mn}..{mx}]  ${c}/MWh{lim}")
    print(sep)
    phi = r["phi_deg"]
    print(f"  PST φ = {phi:>+6.2f}°  (limit ±{PHI_MAX_DEG:.0f}°)  "
          f"wear=${pst_cost:.2f}/h")
    print(sep)

    # Show PTDF equation for each branch
    print(f"  {'Branch':<8} {'Flow':>8}  {'Pmax':>6}  "
          f"{'Load%':>6}  {'PTDF·Pnet':>10}  {'PSDF·φ':>8}  Status")
    print(sep)
    Pg_G2     = r["Pg"]["G2"]
    phi_rad   = r["phi_rad"]
    for br in BRANCHES:
        flow     = r["P"][br]
        pm       = r["pmax"][br]
        ptdf_term = ptdf["G2"][br] * (Pg_G2 - Pd)
        psdf_term = psdf[br] * phi_rad
        load_pct  = 100 * abs(flow) / pm
        flag      = " ◄ BINDING" if load_pct > 99.0 else ""
        print(f"  {br:<8} {flow:>+8.2f}  {pm:>6.0f}  "
              f"{load_pct:>5.1f}%  {ptdf_term:>+10.3f}  "
              f"{psdf_term:>+8.3f}{flag}")
    print(sep)
    print(f"  Note: PSDF(L2b_3)=0 → L2b_3 flow = {ptdf['G2']['L2b_3']*(Pg_G2-Pd):+.2f} MW"
          f"  (φ-independent, = 250 − Pg_G2 = {250-Pg_G2:.2f})")
    print(sep)
    print(f"  LMPs:  B1=${r['LMP']['B1']:>8.4f}/MWh   B3=${r['LMP']['B3']:>8.4f}/MWh")
    print(f"  Congestion rent B1→B3 = ${r['LMP']['B3']-r['LMP']['B1']:>+.4f}/MWh")
    print(f"{'═'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Scenario sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_pmax(params: dict, sens: dict) -> None:
    print(f"\n{'═'*95}")
    print("  Sweep: L12a thermal limit  (PTDF formulation)")
    print(f"{'═'*95}")
    hdr = (f"  {'Pmax_a':>7}  {'φ°':>7}  "
           f"{'P_L12a':>8}  {'P_L1_2a':>8}  {'P_PST':>8}  {'P_L2b3':>8}  "
           f"{'Pg1':>7}  {'Pg2':>7}  "
           f"{'LMP_B1':>8}  {'LMP_B3':>8}  {'Cost':>10}")
    print(hdr)
    print("  " + "─" * 92)

    for pmax_a in [250, 180, 130, 110, 95, 80, 65, 50]:
        try:
            r = solve_lp_ptdf(params, sens, pmax_l12a=float(pmax_a))
            print(
                f"  {pmax_a:>7}  {r['phi_deg']:>+7.2f}  "
                f"{r['P']['L12a']:>+8.2f}  {r['P']['L1_2a']:>+8.2f}  "
                f"{r['P']['PST_T']:>+8.2f}  {r['P']['L2b_3']:>+8.2f}  "
                f"{r['Pg']['G1']:>7.2f}  {r['Pg']['G2']:>7.2f}  "
                f"{r['LMP']['B1']:>8.4f}  {r['LMP']['B3']:>8.4f}  "
                f"{r['total_cost']:>10.2f}"
            )
        except RuntimeError as e:
            print(f"  {pmax_a:>7}  INFEASIBLE ({e})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  DC OPF — PTDF+PSDF formulation, bypass PST topology             ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  B1(S1)──L12a(bypass)──────────────────────── B2b(S2)           ║")
    print("║  B1(S1)──L1_2a── B2a(S2)──[PST_T φ]── B2b(S2)──L2b_3── B3(S3) ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # ── Compute and verify sensitivities ─────────────────────────────────────
    sens = compute_sensitivities_analytical()

    print("── Analytical PTDF and PSDF ────────────────────────────────────")
    print(f"  {'Branch':<8}  {'PTDF(G2)':>10}  {'PSDF (MW/rad)':>14}")
    print("  " + "─" * 38)
    for br in BRANCHES:
        print(f"  {br:<8}  {sens['ptdf'][br]:>+10.6f}  {sens['psdf'][br]:>+14.4f}")
    print()
    print(f"  Key: PSDF(L2b_3)={sens['psdf']['L2b_3']:.4f} → PST does not change"
          f" total flow to B3")
    print()

    net    = build_network()
    pypow_df_ptdf, pypow_df_psdf = verify_sensitivities_pypowsybl(net, sens)
    pypow_sens = {"ptdf": pypow_df_ptdf, "psdf": pypow_df_psdf}
    sens = pypow_sens
    params = extract_parameters(net)

    # ── Case 1: no congestion ─────────────────────────────────────────────────
    r1 = solve_lp_ptdf(params, sens, pmax_l12a=110.0)
    print_results(r1, sens, "Case 1 — No congestion (Pmax_a=250 MW)")
    validate(r1)

    # ── Case 2: L12a tight, PST activated ────────────────────────────────────
    r2 = solve_lp_ptdf(params, sens, pmax_l12a=110.0)
    print_results(r2, sens, "Case 2 — L12a tight (110 MW): PST activated")
    validate(r2)

    # ── Case 3: L12a very tight ───────────────────────────────────────────────
    r3 = solve_lp_ptdf(params, sens, pmax_l12a=65.0)
    print_results(r3, sens, "Case 3 — L12a very tight (65 MW): PST near limit")
    validate(r3)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    sweep_pmax(params, sens)
