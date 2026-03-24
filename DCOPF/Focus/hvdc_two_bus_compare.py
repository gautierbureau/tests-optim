"""
Two-bus HVDC: Theta vs PTDF comparison
=======================================

  G1(slack) ── B1 ──[HVDC_eq]── B2 ── G2 / D2=200MW

  HVDC droop: P_hvdc = P0 + k·(θ_B1 − θ_B2)
  P0=30MW, k=100MW/rad, Pmax=±100MW

PTDF decomposition
------------------
  P0      → demand shift: Pd_B2_eff = 200 + 30 = 230MW
  k·Δθ    → HVDC_eq line with b=k=100, x=BASE_MVA/k=1.0pu

Analytical PTDF (2-bus, G1 slack)
----------------------------------
  B_red = [b_eq] = [100]  →  B_red_inv = 1/100
  PTDF(HVDC_eq, G2) = -b_eq · B_red_inv = -100 · (1/100) = -1.0
  (all G2 injection flows through the only branch)
"""

import math
import pypowsybl.network as pn
import pypowsybl.loadflow as plf
import pypowsybl.sensitivity as psa
import pypowsybl as ppw
import pyoptinterface as poi
from pyoptinterface import xpress
import pandas as pd

BASE_MVA  = 100.0
KV        = 400.0
HVDC_P0   = 30.0
HVDC_K    = 100.0
HVDC_PMAX = 100.0
PD_B2     = 200.0
THETA_MAX = math.pi / 3

def build_network() -> pn.Network:
    net = pn.create_empty("two_bus_hvdc")
    net.create_substations(id=["S1","S2"], country=["FR","FR"])
    net.create_voltage_levels(
        id=["VL1","VL2"], substation_id=["S1","S2"],
        topology_kind=["BUS_BREAKER","BUS_BREAKER"], nominal_v=[KV,KV],
    )
    net.create_buses(id=["B1","B2"], voltage_level_id=["VL1","VL2"])

    net.create_vsc_converter_stations(id='CS-1', voltage_level_id='VL1', bus_id='B1',
                                          loss_factor=0.1, voltage_regulator_on=True, target_v=KV)
    net.create_vsc_converter_stations(id='CS-2', voltage_level_id='VL2', bus_id='B2',
                                      loss_factor=0.1, voltage_regulator_on=True, target_v=KV)
    net.create_hvdc_lines(id='HVDC-1', converter_station1_id='CS-1', converter_station2_id='CS-2',
                              r=1.0, nominal_v=KV, converters_mode='SIDE_1_RECTIFIER_SIDE_2_INVERTER',
                              max_p=HVDC_PMAX, target_p=HVDC_P0)
    net.create_extensions('hvdcAngleDroopActivePowerControl', id='HVDC-1', p0=HVDC_P0, droop=HVDC_K, enabled=True)
    net.create_generators(
        id=["G1","G2"], voltage_level_id=["VL1","VL2"], bus_id=["B1","B2"],
        energy_source=["OTHER","OTHER"], min_p=[0.0,0.0], max_p=[300.0,150.0],
        target_p=[150.0,80.0], target_q=[0,1], target_v=[KV,KV], voltage_regulator_on=[True,False],
    )
    net.create_loads(
        id=["D2"], voltage_level_id=["VL2"], bus_id=["B2"], p0=[PD_B2], q0=[0.0],
    )
    return net

def build_network_equivalent() -> pn.Network:
    net = pn.create_empty("two_bus_hvdc")
    net.create_substations(id=["S1","S2"], country=["FR","FR"])
    net.create_voltage_levels(
        id=["VL1","VL2"], substation_id=["S1","S2"],
        topology_kind=["BUS_BREAKER","BUS_BREAKER"], nominal_v=[KV,KV],
    )
    net.create_buses(id=["B1","B2"], voltage_level_id=["VL1","VL2"])
    net.create_lines(
        id=["HVDC_eq"],
        voltage_level1_id=["VL1"], bus1_id=["B1"],
        voltage_level2_id=["VL2"], bus2_id=["B2"],
        r=[0.0], x=[BASE_MVA/HVDC_K],
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    net.create_generators(
        id=["G1","G2"], voltage_level_id=["VL1","VL2"], bus_id=["B1","B2"],
        energy_source=["OTHER","OTHER"], min_p=[0.0,0.0], max_p=[300.0,150.0],
        target_p=[150.0,80.0], target_q=[0,1], target_v=[KV,KV], voltage_regulator_on=[True,False],
    )
    net.create_loads(
        id=["D2", "P0-", "P0+"], voltage_level_id=["VL2", "VL1", "VL2"], bus_id=["B2", "B1", "B2"], p0=[PD_B2, HVDC_P0, -HVDC_P0], q0=[0.0, 0.,0.],
    )
    return net

def get_ptdf(net: pn.Network):
    # sa = psa.create_dc_analysis()
    # sa.add_branch_flow_factor_matrix(["HVDC_eq"], ["G2"], "PTDF")
    # res = sa.run(net, parameters=plf.Parameters(distributed_slack=False))
    # val = float(res.get_sensitivity_matrix("PTDF").loc["G2","HVDC_eq"])
    # print(f"PTDF(HVDC_eq, G2) = {val:.6f}  (analytical = -1.0)\n")
    # return val
    sa = psa.create_dc_analysis()
    sa.add_branch_flow_factor_matrix(["HVDC_eq"], ["G1", "G2"], "PTDF")
    res = sa.run(net, parameters=plf.Parameters(distributed_slack=False))
    df = res.get_sensitivity_matrix("PTDF")
    print(f"PTDF(L12, G1) = {df.loc['G1', 'HVDC_eq']:.6f}  (analytical = 0)")
    print(f"PTDF(L12, G2) = {df.loc['G2', 'HVDC_eq']:.6f}  (analytical = -1.0)")
    ptdf = {
        "G1": float(df.loc["G1", "HVDC_eq"]),
        "G2": float(df.loc["G2", "HVDC_eq"]),
    }
    return ptdf

def solve_theta() -> dict:
    """
    Variables: Pg1, Pg2, θ_B2, P_hvdc
    θ_B1 = 0 (slack reference)
    HVDC droop: P_hvdc - k·(0 - θ_B2) = P0  →  P_hvdc + k·θ_B2 = P0
    """
    m = xpress.Model()
    m.set_model_attribute(poi.ModelAttribute.Silent, True)
    Pg1   = m.add_variable(lb=0, ub=300, name="Pg1")
    Pg2   = m.add_variable(lb=0, ub=150, name="Pg2")
    t2    = m.add_variable(lb=-THETA_MAX, ub=THETA_MAX, name="t_B2")
    P_hvdc = m.add_variable(lb=-HVDC_PMAX, ub=HVDC_PMAX, name="P_hvdc")

    m.add_linear_constraint(P_hvdc + HVDC_K*t2, poi.Eq, HVDC_P0, name="droop")
    m.add_linear_constraint(Pg2 + P_hvdc,        poi.Eq, PD_B2,   name="bal_B2")
    m.add_linear_constraint(Pg1 - P_hvdc,        poi.Eq, 0,   name="bal_B1")

    obj = poi.ExprBuilder()
    obj += 30*Pg1 + 45*Pg2
    m.set_objective(obj, poi.ObjectiveSense.Minimize)
    m.optimize()

    def v(x): return m.get_value(x)
    return {
        "form": "theta", "Pg1": v(Pg1), "Pg2": v(Pg2),
        "P_hvdc": v(P_hvdc), "theta_B2_deg": math.degrees(v(t2)),
        "cost": m.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
    }

def solve_ptdf(ptdf) -> dict:
    """
    Variables: Pg1, Pg2, P_hvdc_eq  (no angles)
    P0 → demand shift: Pd_B2_eff = PD_B2 + P0 = 230MW
    P_hvdc_eq = k·Δθ only; bounds: [-Pmax-P0, +Pmax-P0]
    PTDF constraint: P_hvdc_eq = ptdf_val·(Pg2 - Pd_B2_eff)
    """
    # Pd_eff = PD_B2 + HVDC_P0

    m = xpress.Model()
    m.set_model_attribute(poi.ModelAttribute.Silent, True)
    Pg1  = m.add_variable(lb=0, ub=300, name="Pg1")
    Pg2  = m.add_variable(lb=0, ub=150, name="Pg2")
    P_eq = m.add_variable(lb=-HVDC_PMAX-HVDC_P0, ub=HVDC_PMAX-HVDC_P0, name="P_hvdc_eq")

    ptdf_G1 = ptdf["G1"]
    ptdf_G2 = ptdf["G2"]
    m.add_linear_constraint(P_eq - ptdf_G1 * Pg1 - ptdf_G2 * Pg2, poi.Eq, -ptdf_G1 * HVDC_P0 + ptdf_G2 * (-PD_B2+HVDC_P0), name="ptdf1")
    m.add_linear_constraint(Pg2 + P_eq, poi.Eq, PD_B2 - HVDC_P0, name="bal_B2")
    m.add_linear_constraint(Pg1 - P_eq, poi.Eq, HVDC_P0, name="bal_B1")
    #
    # m.add_linear_constraint(P_eq - ptdf_val*Pg2, poi.Eq, ptdf_val*(-Pd_eff), name="ptdf")
    # m.add_linear_constraint(Pg2 + P_eq,          poi.Eq, Pd_eff,             name="bal_B2")

    obj = poi.ExprBuilder()
    obj += 30*Pg1 + 45*Pg2
    m.set_objective(obj, poi.ObjectiveSense.Minimize)
    m.optimize()

    def v(x): return m.get_value(x)
    P_eq_val = v(P_eq)
    return {
        "form": "ptdf", "Pg1": v(Pg1), "Pg2": v(Pg2),
        "P_hvdc": HVDC_P0 + P_eq_val, "P_hvdc_eq": P_eq_val,
        "cost": m.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
    }

def compare(r_th: dict, r_pt: dict) -> None:
    keys = ["Pg1","Pg2","P_hvdc","cost"]
    print(f"{'─'*48}")
    print(f"  {'Variable':<12}  {'Theta':>9}  {'PTDF':>9}  {'Δ':>7}")
    print(f"{'─'*48}")
    for k in keys:
        th, pt = r_th[k], r_pt[k]
        print(f"  {k:<12}  {th:>9.4f}  {pt:>9.4f}  {abs(th-pt):>7.1e}")
    print(f"{'─'*48}")
    print(f"  Theta: θ_B2={r_th['theta_B2_deg']:+.4f}°  "
          f"k·Δθ={-HVDC_K*math.radians(r_th['theta_B2_deg']):+.4f} MW  "
          f"(= P_hvdc - P0 = {r_th['P_hvdc']-HVDC_P0:+.4f} MW)")
    print(f"  PTDF:  P_hvdc_eq={r_pt['P_hvdc_eq']:+.4f} MW  "
          f"P_total=P0+P_eq={HVDC_P0:+.1f}+{r_pt['P_hvdc_eq']:.4f}={r_pt['P_hvdc']:+.4f} MW")


def validate(r: dict) -> None:
    net = build_network()
    net.update_generators(id="G1", target_p=r["Pg1"])
    net.update_generators(id="G2", target_p=r["Pg2"])
    lf    = ppw.loadflow.run_dc(net, parameters=plf.Parameters(distributed_slack=False))
    vscs  = net.get_vsc_converter_stations(all_attributes=True)
    lf_eq = vscs.loc["CS-1","p"]
    if r['form'] == "theta":
        print(f"  LF ({r['form']}): HVDC_eq={lf_eq:+.4f} MW  "
              f"(=P0 + k·Δθ, total={HVDC_P0+HVDC_K*(0-math.radians(r['theta_B2_deg'])):+.4f} MW)  status={lf[0].status}")

if __name__ == "__main__":
    net      = build_network()
    net_equivalent = build_network_equivalent()
    net_equivalent.save("hvdc_ptdf_line_equivalent.xiidm")
    ptdf = get_ptdf(net_equivalent)
    r_th     = solve_theta()
    r_pt     = solve_ptdf(ptdf)
    compare(r_th, r_pt)
    validate(r_th)
    validate(r_pt)
