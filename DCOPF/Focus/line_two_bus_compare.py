"""
Two-bus AC line: Theta vs PTDF comparison
==========================================

  G1(slack) ── B1 ──[L12, b=250]── B2 ── G2 / D2=200MW

Analytical PTDF (2-bus, G1 slack)
----------------------------------
  B_red = [b_12] = [250]  →  B_red_inv = 1/250
  PTDF(L12, G2) = -b_12 · (1/250) = -1.0
  (trivial: the only line carries all net injection at B2)
"""

import math
import pypowsybl.network as pn
import pypowsybl.loadflow as plf
import pypowsybl.sensitivity as psa
import pypowsybl as ppw
import pyoptinterface as poi
from pyoptinterface import xpress

BASE_MVA = 100.0
KV       = 400.0
B_L12    = 250.0        # MW/rad  (x = 0.4 pu)
PMAX_L12 = 120.0        # MW
PD_B2    = 250.0        # MW


def build_network() -> pn.Network:
    net = pn.create_empty("two_bus_line")
    net.create_substations(id=["S1","S2"], country=["FR","FR"])
    net.create_voltage_levels(
        id=["VL1","VL2"], substation_id=["S1","S2"],
        topology_kind=["BUS_BREAKER","BUS_BREAKER"], nominal_v=[KV, KV],
    )
    net.create_buses(id=["B1","B2"], voltage_level_id=["VL1","VL2"])
    net.create_lines(
        id=["L12"],
        voltage_level1_id=["VL1"], bus1_id=["B1"],
        voltage_level2_id=["VL2"], bus2_id=["B2"],
        r=[0.0], x=[BASE_MVA/B_L12],
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )
    net.create_generators(
        id=["G1","G2"], voltage_level_id=["VL1","VL2"], bus_id=["B1","B2"],
        energy_source=["OTHER","OTHER"], min_p=[0.0,0.0], max_p=[300.0,150.0],
        target_p=[150.0,50.0], target_q=[0,1], target_v=[KV,KV], voltage_regulator_on=[True,False],
    )
    net.create_loads(
        id=["D2"], voltage_level_id=["VL2"], bus_id=["B2"], p0=[PD_B2], q0=[0.0],
    )
    return net


def get_ptdf(net: pn.Network) -> dict[str, float]:
    sa = psa.create_dc_analysis()
    sa.add_branch_flow_factor_matrix(["L12"], ["G1", "G2"], "PTDF")
    res = sa.run(net, parameters=plf.Parameters(distributed_slack=False))
    df = res.get_sensitivity_matrix("PTDF")
    print(f"PTDF(L12, G1) = {df.loc['G1', 'L12']:.6f}  (analytical = 0)\n")
    print(f"PTDF(L12, G2) = {df.loc['G2', 'L12']:.6f}  (analytical = -1.0)\n")
    ptdf = {
        "G1": float(df.loc["G1", "L12"]),
        "G2": float(df.loc["G2", "L12"]),
    }
    return ptdf

def solve_theta() -> dict:
    """
    Variables: Pg1, Pg2, θ_B2, P_L12
    θ_B1 = 0 (slack reference)
    Flow: P_L12 = b·(θ_B1 - θ_B2) = -b·θ_B2
    """
    m = xpress.Model()
    m.set_model_attribute(poi.ModelAttribute.Silent, True)
    Pg1  = m.add_variable(lb=0, ub=300,      name="Pg1")
    Pg2  = m.add_variable(lb=0, ub=150,      name="Pg2")
    t2   = m.add_variable(lb=-math.pi, ub=math.pi, name="t_B2")
    P12  = m.add_variable(lb=-PMAX_L12, ub=PMAX_L12, name="P_L12")

    m.add_linear_constraint(P12 + B_L12*t2,  poi.Eq, 0,     name="flow")
    m.add_linear_constraint(Pg2 + P12,        poi.Eq, PD_B2, name="bal_B2")
    m.add_linear_constraint(Pg1 - P12,        poi.Eq, 0, name="bal_B1")

    obj = poi.ExprBuilder()
    obj += 30*Pg1 + 45*Pg2
    m.set_objective(obj, poi.ObjectiveSense.Minimize)
    m.optimize()

    def v(x): return m.get_value(x)
    return {
        "form": "theta", "Pg1": v(Pg1), "Pg2": v(Pg2),
        "P_L12": v(P12), "theta_B2_deg": math.degrees(v(t2)),
        "cost": m.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
    }


def solve_ptdf(ptdf) -> dict:
    """
    Variables: Pg1, Pg2, P_L12  (no angles)
    PTDF constraint: P_L12 = ptdf_val·(Pg2 - PD_B2)
    """
    m = xpress.Model()
    m.set_model_attribute(poi.ModelAttribute.Silent, True)
    Pg1 = m.add_variable(lb=0, ub=300,      name="Pg1")
    Pg2 = m.add_variable(lb=0, ub=150,      name="Pg2")
    P12 = m.add_variable(lb=-PMAX_L12, ub=PMAX_L12, name="P_L12")

    ptdf_G1 = ptdf["G1"]
    ptdf_G2 = ptdf["G2"]
    m.add_linear_constraint(P12 - ptdf_G1 * Pg1 - ptdf_G2 * Pg2, poi.Eq, ptdf_G2 * (-PD_B2), name="ptdf1")
    m.add_linear_constraint(Pg2 + P12,           poi.Eq, PD_B2,             name="bal_B2")
    m.add_linear_constraint(Pg1 - P12,           poi.Eq, 0,             name="bal_B1")

    obj = poi.ExprBuilder()
    obj += 30*Pg1 + 45*Pg2
    m.set_objective(obj, poi.ObjectiveSense.Minimize)
    m.optimize()

    def v(x): return m.get_value(x)
    return {
        "form": "ptdf", "Pg1": v(Pg1), "Pg2": v(Pg2),
        "P_L12": v(P12),
        "cost": m.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
    }


def compare(r_th: dict, r_pt: dict) -> None:
    keys = ["Pg1","Pg2","P_L12","cost"]
    print(f"{'─'*46}")
    print(f"  {'Variable':<10}  {'Theta':>9}  {'PTDF':>9}  {'Δ':>7}")
    print(f"{'─'*46}")
    for k in keys:
        th, pt = r_th[k], r_pt[k]
        print(f"  {k:<10}  {th:>9.4f}  {pt:>9.4f}  {abs(th-pt):>7.1e}")
    print(f"{'─'*46}")
    print(f"  Theta: θ_B2={r_th['theta_B2_deg']:+.4f}°  "
          f"b·Δθ={B_L12*math.radians(r_th['theta_B2_deg']):+.4f} MW  (= -P_L12)")


def validate(r: dict) -> None:
    net = build_network()
    net.update_generators(id="G1", target_p=r["Pg1"])
    net.update_generators(id="G2", target_p=r["Pg2"])
    lf    = ppw.loadflow.run_dc(net, parameters=plf.Parameters(distributed_slack=False))
    lines = net.get_lines(all_attributes=True)
    gens  = net.get_generators(all_attributes=True)
    print(f"  LF ({r['form']}): L12={lines.loc['L12','p1']:+.4f} MW  "
          f"[OPF={r['P_L12']:+.4f}]  status={lf[0].status}")
    print(f"  G1={gens.loc['G1','p']:+.4f} MW  G2={gens.loc['G2','p']:+.4f} MW")


if __name__ == "__main__":
    net      = build_network()
    ptdf_val = get_ptdf(net)
    r_th     = solve_theta()
    r_pt     = solve_ptdf(ptdf_val)
    compare(r_th, r_pt)
    validate(r_th)
    validate(r_pt)
