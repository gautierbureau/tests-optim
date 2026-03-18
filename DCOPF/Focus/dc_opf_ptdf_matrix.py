"""
DC OPF — PTDF via pypowsybl + LP via pyoptinterface (HiGHS)

Line flow constraint:  -f_max ≤ M p - δ ≤ f_max
where  M = PTDF · Φ,  δ = PTDF · d,  M p - δ = P_flow
"""

import numpy as np
import pypowsybl as pp
import pypowsybl.sensitivity as sens
import pyoptinterface as poi
from pyoptinterface import highs

#  This is claude code without any review or run just to have a example to implement ptdf_formulation.py

# ── 1. Build Network with pypowsybl ───────────────────────────────────────────

n = pp.network.create_empty("3bus")

# Substations & voltage levels
for sub in ["S0", "S1", "S2"]:
    n.create_substations(id=sub)
for vl, sub in [("VL0","S0"), ("VL1","S1"), ("VL2","S2")]:
    n.create_voltage_levels(id=vl, substation_id=sub,
                            nominal_v=100., topology_kind="BUS_BREAKER")
for bus, vl in [("B0","VL0"), ("B1","VL1"), ("B2","VL2")]:
    n.create_buses(id=bus, voltage_level_id=vl)

# Lines (x in pu on 100 MVA base)
n.create_lines(
    id=["L01","L02","L12"],
    voltage_level1_id=["VL0","VL0","VL1"],  bus1_id=["B0","B0","B1"],
    voltage_level2_id=["VL1","VL2","VL2"],  bus2_id=["B1","B2","B2"],
    r=[0.,0.,0.], x=[0.10,0.20,0.15], g1=[0.,0.,0.], g2=[0.,0.,0.],
    b1=[0.,0.,0.], b2=[0.,0.,0.]
)

# Generators (bus 0 = slack)
n.create_generators(
    id=["G0","G1","G2"],
    voltage_level_id=["VL0","VL1","VL2"], bus_id=["B0","B1","B2"],
    target_p=[150.,0.,0.], min_p=[0.,0.,0.], max_p=[200.,150.,100.],
    target_v=[100.,100.,100.], voltage_regulator_on=[True,True,True]
)

# Loads
n.create_loads(
    id=["D0","D1","D2"],
    voltage_level_id=["VL0","VL1","VL2"], bus_id=["B0","B1","B2"],
    p0=[50.,120.,80.], q0=[0.,0.,0.]
)

# ── 2. Compute PTDF via pypowsybl sensitivity analysis ────────────────────────

lines    = ["L01","L02","L12"]
buses    = ["B0", "B1", "B2"]
gens     = ["G0", "G1", "G2"]
gen_bus  = {"G0":"B0", "G1":"B1", "G2":"B2"}
f_max    = np.array([100., 80., 60.])          # MW thermal limits
d        = np.array([50., 120., 80.])          # loads per bus
c        = np.array([20., 40., 30.])           # marginal costs
p_min    = np.array([0.,  0.,  0.])
p_max    = np.array([200., 150., 100.])

sa = sens.create_dc_analysis()
# Inject each bus as a "variable" injection, lines as "functions"
sa.add_branch_flow_factor_matrix(
    branches_ids=lines, variables_ids=buses,
    matrix_id="PTDF"
)
res  = sa.run(n)
PTDF = res.get_sensitivity_matrix("PTDF").to_numpy()   # (L x N)

# ── 3. Precompute M and δ ─────────────────────────────────────────────────────

L, N, G = len(lines), len(buses), len(gens)

# Φ (N x G): generator-to-bus map
bus_idx = {b: i for i, b in enumerate(buses)}
Phi = np.zeros((N, G))
for g, gid in enumerate(gens):
    Phi[bus_idx[gen_bus[gid]], g] = 1.

M     = PTDF @ Phi      # (L x G)
delta = PTDF @ d        # (L,)

# ── 4. Build & Solve LP ────────────────────────────────────────────────────────

model = highs.Model()

p = np.array([
    model.add_variable(lb=p_min[g], ub=p_max[g], name=gens[g])
    for g in range(G)
])

model.set_objective(
    poi.quicksum(c[g] * p[g] for g in range(G)),
    poi.ObjectiveSense.Minimize
)

# Power balance: 1ᵀ p = 1ᵀ d
model.add_linear_constraint(
    poi.quicksum(p[g] for g in range(G)),
    poi.Eq, d.sum()
)

# Line limits: -f_max ≤ M p - δ ≤ f_max
for l in range(L):
    P_flow = poi.quicksum(M[l,g] * p[g] for g in range(G)) - delta[l]
    model.add_linear_constraint(P_flow, poi.Geq, -f_max[l])
    model.add_linear_constraint(P_flow, poi.Leq,  f_max[l])

model.optimize()

# ── 5. Results ─────────────────────────────────────────────────────────────────

p_sol  = np.array([model.get_value(p[g]) for g in range(G)])
P_flow = M @ p_sol - delta

print("=== DC OPF Results ===")
print(f"Status : {model.get_model_attribute(poi.ModelAttribute.TerminationStatus)}")
print(f"Cost   : {model.get_model_attribute(poi.ModelAttribute.ObjectiveValue):.2f} $/h\n")
print("Generator dispatch:")
for g, gid in enumerate(gens):
    print(f"  {gid} = {p_sol[g]:7.2f} MW")
print("\nLine flows:")
for l, lid in enumerate(lines):
    print(f"  {lid} = {P_flow[l]:7.2f} MW  (limit ±{f_max[l]:.0f} MW)")
