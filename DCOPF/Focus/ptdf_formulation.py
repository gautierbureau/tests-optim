import pypowsybl as pp
import math
import pyoptinterface as poi
from pyoptinterface import xpress
import pypowsybl.loadflow as lf
import pypowsybl.sensitivity as sens
import pandas as pd
import numpy as np
from scipy import sparse

PHI_MAX   = math.radians(30.0)

def ptdf_formulation():
    network = pp.network.load("pst1.xiidm")

    generators = network.get_generators(all_attributes=True)
    loads = network.get_loads(all_attributes=True)
    buses = network.get_buses(all_attributes=True)
    lines = network.get_lines(all_attributes=True)
    tfos = network.get_2_windings_transformers(all_attributes=True)
    psts = network.get_phase_tap_changers(all_attributes=True)
    vls = network.get_voltage_levels(all_attributes=True)

    line_max_p = {
        'L12a': 110.0,
        'L1_2a': 150,
        'L2b_3': 200
    }

    tfos_max_p = {
        'PST_T': 150
    }

    lines['max_p'] = lines.index.map(line_max_p)
    tfos['max_p'] = tfos.index.map(tfos_max_p)

    df_branches = pd.concat([lines, tfos], sort=False)
    branches = df_branches.index.to_list()

    generators_cost = {
        'G1': 30,
        'G2': 45
    }

    generators['cost'] = generators.index.map(generators_cost)

    sa = sens.create_dc_analysis()
    # Inject each bus as a "variable" injection, lines as "functions"
    all_injections = pd.concat([generators['bus_id'], loads['bus_id']])
    one_injection_by_bus = all_injections.drop_duplicates()
    injection_buses = sorted(list(set(generators['bus_id']) | set(loads['bus_id'])))
    injection_variables = one_injection_by_bus.index
    sa.add_branch_flow_factor_matrix(branches, injection_variables, "PTDF")
    sa.add_branch_flow_factor_matrix(branches, psts.index, "PSDF")
    res = sa.run(network, parameters=lf.Parameters(distributed_slack=False))
    ptdf = {comp: {br: float(res.get_sensitivity_matrix("PTDF").loc[comp, br]) for br in branches} for comp in injection_variables}
    psdf = {br: float(res.get_sensitivity_matrix("PSDF").loc["PST_T", br]) * (180 / math.pi) for br in branches}

    print(ptdf)

    ptdf_matrix = pd.DataFrame(ptdf).values
    psdf_matrix = pd.DataFrame(psdf, index=[0]).values.T

    print(ptdf_matrix)
    print(psdf_matrix)

    bus_idx = {b: i for i, b in enumerate(injection_buses)}
    gen_to_bus = generators['bus_id'].to_dict()
    N, G = len(injection_buses), len(generators)
    Phi = np.zeros((N, G))
    for g, gid in enumerate(generators.index):
        Phi[bus_idx[gen_to_bus[gid]], g] = 1.

    print(Phi)

    all_gens = generators.index.tolist()
    phi = (pd.get_dummies(generators['bus_id'], dtype=int)
           .reindex(columns=injection_buses, fill_value=0)  # Ensure all injection buses are present
           .T  # Flip to (Buses x Generators)
           .reindex(columns=all_gens))  # Ensure columns match your generator list order

    Phi_matrix = phi.values
    Phi_sparse = sparse.csr_matrix(phi.values)
    print(Phi_matrix)
    print(Phi_sparse)

    M = ptdf_matrix @ Phi

    print(M)

    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)

    PGen = {
        g: model.add_variable(
            lb=generators.at[g, 'min_p'],
            ub=generators.at[g, 'max_p'],
            name=f"P_{g}"
        )
        for g in generators.index
    }

    PLine = {
        l: model.add_variable(
            lb=-lines.at[l, 'max_p'],
            ub=lines.at[l, 'max_p'],
            name=f"P_{l}"
        )
        for l in lines.index
    }

    PLine.update({
        t: model.add_variable(
            lb=-tfos.at[t, 'max_p'],
            ub=tfos.at[t, 'max_p'],
            name=f"P_{t}"
        )
        for t in tfos.index
    })

    phi = {
        l: model.add_variable(lb=-PHI_MAX, ub=PHI_MAX, name=f"phi_{l}")
        for l in psts.index
    }

    psi = {
        l: model.add_variable(lb=0, ub=PHI_MAX, name=f"psi_{l}")
        for l in psts.index
    }

    con_psi_lb, con_psi_ub = {}, {}

    for pstId, (phiVar, psiVar) in zip(phi.keys(), zip(phi.values(), psi.values())):
        con_psi_lb[pstId] = model.add_linear_constraint(psiVar - phiVar, poi.Geq, 0)
        con_psi_ub[pstId] = model.add_linear_constraint(psiVar + phiVar, poi.Geq, 0)

    # Power balance
    con_balance = model.add_linear_constraint(
        poi.quicksum(Pg for Pg in PGen.values()),
        poi.Eq, loads['p0'].sum()
    )

    print("number_of_variables: " + str(model.number_of_variables()))
    print("number_of_constraints: " + str(model.number_of_constraints(poi.ConstraintType.Linear)))

    # Loads data
    bus_load_series = loads.groupby('bus_id')['p0'].sum()
    nodal_load_vector = bus_load_series.reindex(injection_buses, fill_value=0.0)
    P_l = nodal_load_vector.values
    delta = ptdf_matrix @ P_l

    con_lb, con_ub = {}, {}

    for l, branchId in enumerate(branches):
        P_flow = poi.quicksum(M[l, g] * Pg for g, Pg in enumerate(PGen.values())) - delta[l] + poi.quicksum(psdf_matrix[l, p] * phiVar for p, phiVar in enumerate(phi.values()))
        con_lb[branchId] = model.add_linear_constraint(P_flow, poi.Geq, -df_branches['max_p'][branchId])
        con_ub[branchId] = model.add_linear_constraint(P_flow, poi.Leq, df_branches['max_p'][branchId])

    print("number_of_variables: " + str(model.number_of_variables()))
    print("number_of_constraints: " + str(model.number_of_constraints(poi.ConstraintType.Linear)))

    C_PST = 5
    obj = poi.ExprBuilder()
    for g, Pg in PGen.items():
        obj += generators['cost'][g] * Pg
    for psiVar in psi.values():
        obj += C_PST * psiVar

    model.set_objective(obj, poi.ObjectiveSense.Minimize)
    model.optimize()

    def v(x):
        return model.get_value(x)

    results = {}
    for g, Pg in PGen.items():
        results[g] = v(Pg)
    for phiId, phiVar in phi.items():
        results['phi_' + phiId] = math.degrees(v(phiVar))
    results['cost'] = model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)

    print(results)
    p_sol = np.array([model.get_value(Pg) for Pg in PGen.values()])
    phi_sol = np.array([model.get_value(phiVar) for phiVar in phi.values()])

    flows = M @ p_sol + psdf_matrix @ phi_sol - delta

    for l, branchId in enumerate(branches):
        print(f"{branchId}: {flows[l]:.2f} MW  (limit ±{df_branches['max_p'][branchId]:.0f} MW)")

    print("Active line constraints:")
    for branchId in branches:
        mu_lb = model.get_constraint_attribute(con_lb[branchId], poi.ConstraintAttribute.Dual)
        mu_ub = model.get_constraint_attribute(con_ub[branchId], poi.ConstraintAttribute.Dual)
        if abs(mu_lb) > 1e-6:
            print(f"  {branchId} lower bound ACTIVE  μ⁻ = {mu_lb:.4f}")
        if abs(mu_ub) > 1e-6:
            print(f"  {branchId} upper bound ACTIVE  μ⁺ = {mu_ub:.4f}")

    # Power balance dual = system marginal price λ₀
    lambda0 = model.get_constraint_attribute(con_balance, poi.ConstraintAttribute.Dual)
    print(f"System marginal price λ₀ = {lambda0:.4f} $/MWh")

    # PST absolute value constraints:  ψ ≥ |φ|  i.e.  ψ - φ ≥ 0  and  ψ + φ ≥ 0
    print("\nActive PST constraints:")
    for pstId in phi.keys():
        mu_lb = model.get_constraint_attribute(con_psi_lb[pstId], poi.ConstraintAttribute.Dual)
        mu_ub = model.get_constraint_attribute(con_psi_ub[pstId], poi.ConstraintAttribute.Dual)
        if abs(mu_lb) > 1e-6:
            print(f"  {pstId}  ψ - φ ≥ 0  ACTIVE  μ = {mu_lb:.4f}  (φ = ψ, positive shift binding)")
        if abs(mu_ub) > 1e-6:
            print(f"  {pstId}  ψ + φ ≥ 0  ACTIVE  μ = {mu_ub:.4f}  (φ = -ψ, negative shift binding)")

if __name__ == "__main__":
    ptdf_formulation()