"""
DC OPF — Incremental PTDF Constraint Generation
1. Solve power balance only
2. Run DC loadflow in pypowsybl → find violated lines
3. Fetch PTDF rows for violated lines → add constraints
4. Repeat until no violations
"""

import numpy as np
import math
import pypowsybl as pp
import pypowsybl.sensitivity as sens
import pypowsybl.loadflow as lf
import pyoptinterface as poi
from pyoptinterface import xpress
import pandas as pd

VIOL_TOL  = 1.0    # MW — violation threshold
MAX_ITER  = 20
PHI_MAX   = math.radians(30.0)

con_lb, con_ub = {}, {}
added = set()      # track which branches already have constraints

def get_ptdf_row(network, branch_ids):
    generators = network.get_generators(all_attributes=True)
    loads = network.get_loads(all_attributes=True)
    psts = network.get_phase_tap_changers(all_attributes=True)

    """Fetch PTDF and PSDF rows for a subset of branches from pypowsybl."""
    sa = sens.create_dc_analysis()

    all_injections = pd.concat([generators['bus_id'], loads['bus_id']])
    one_injection_by_bus = all_injections.drop_duplicates()
    injection_variables = one_injection_by_bus.index

    sa.add_branch_flow_factor_matrix(branch_ids, injection_variables, "PTDF")
    sa.add_branch_flow_factor_matrix(branch_ids, psts.index, "PSDF")
    res  = sa.run(network)

    ptdf = {comp: {br: float(res.get_sensitivity_matrix("PTDF").loc[comp, br]) for br in branch_ids} for comp in injection_variables}
    psdf = {comp: {br: float(res.get_sensitivity_matrix("PSDF").loc[comp, br]) * (180 / math.pi) for br in branch_ids} for comp in psts.index}

    ptdf_matrix = pd.DataFrame(ptdf).values
    psdf_matrix = pd.DataFrame(psdf).values

    return ptdf_matrix, psdf_matrix

def add_flow_constraint(model, df_branches, branchId, ptdf_row, psdf_row, Phi, P_l, PGen, phi):
    """Add upper/lower flow constraints for one branch."""
    l_ptdf = ptdf_row @ Phi          # (G,)
    l_psdf = psdf_row                # (K,)
    # l_hvdc = ptdf_row @ Phi_hvdc     # (H,)
    d_l    = ptdf_row @ P_l            # scalar delta for this line

    P_flow = (poi.quicksum(l_ptdf[g]  * Pg     for g,  Pg     in enumerate(PGen.values()))
            # + poi.quicksum(l_hvdc[h]  * P0var  for h,  P0var  in enumerate(P0.values()))
            + poi.quicksum(l_psdf[p]  * phiVar for p,  phiVar in enumerate(phi.values()))
            - d_l)

    f = df_branches['max_p'][branchId]
    con_lb[branchId] = model.add_linear_constraint(P_flow, poi.Geq, -f)
    con_ub[branchId] = model.add_linear_constraint(P_flow, poi.Leq,  f)
    added.add(branchId)

def inject_solution_into_network(network, PGen, phi, model):
    """Push optimal dispatch into pypowsybl for DC loadflow."""
    network.update_generators(
        id=list(PGen.keys()),
        target_p=[model.get_value(Pg) for Pg in PGen.values()]
    )
    psts_steps = network.get_phase_tap_changer_steps(all_attributes=True)
    psts = network.get_phase_tap_changers(all_attributes=True)

    psts_taps = {}
    for pstId, phiVar in phi.items():
        pst_steps = psts_steps.loc[pstId]
        zero_alpha_step = pst_steps[pst_steps['alpha'] == 0]
        phi_deg = math.degrees(model.get_value(phiVar))
        zero_alpha_step_index = zero_alpha_step.index[0]

        next_step_index = zero_alpha_step_index + 1
        alpha_next = pst_steps.loc[next_step_index, 'alpha']

        tap_step = zero_alpha_step_index + round(phi_deg / alpha_next)
        tap_step = max(psts['low_tap'][pstId], min(psts['high_tap'][pstId], tap_step))
        psts_taps[pstId] = tap_step

    network.update_phase_tap_changers(
        id=list(psts_taps.keys()),
        tap=[tap for tap in psts_taps.values()]
    )

def get_violated_branches(network, df_branches):
    """Run DC loadflow, return branches violating thermal limits."""
    lf.run_dc(network)
    flows = network.get_lines()[['p1']].rename(columns={'p1': 'flow'})
    flows['max_p'] = df_branches['max_p']
    violated = flows[flows['flow'].abs() > flows['max_p'] + VIOL_TOL]
    return violated.index.tolist(), flows

def ptdf_iterative(network):
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

    injection_buses = sorted(list(set(generators['bus_id']) | set(loads['bus_id'])))
    bus_idx = {b: i for i, b in enumerate(injection_buses)}
    gen_to_bus = generators['bus_id'].to_dict()
    N, G = len(injection_buses), len(generators)
    Phi = np.zeros((N, G))
    for g, gid in enumerate(generators.index):
        Phi[bus_idx[gen_to_bus[gid]], g] = 1.

    # Loads data
    bus_load_series = loads.groupby('bus_id')['p0'].sum()
    nodal_load_vector = bus_load_series.reindex(injection_buses, fill_value=0.0)
    P_l = nodal_load_vector.values

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

    C_PST = 5
    obj = poi.ExprBuilder()
    for g, Pg in PGen.items():
        obj += generators['cost'][g] * Pg
    for psiVar in psi.values():
        obj += C_PST * psiVar
    model.set_objective(obj, poi.ObjectiveSense.Minimize)

    for iteration in range(MAX_ITER):
        model.optimize()
        status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
        cost = model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)
        print(f"\n── Iteration {iteration} ── status={status}  cost={cost:.2f} $/h")

        inject_solution_into_network(network, PGen, phi, model)
        violated, flows = get_violated_branches(network, df_branches)

        for g, Pg in PGen.items():
            print(f"  {g} = {model.get_value(Pg):.2f} MW")
        for phiId, phiVar in phi.items():
            print(f'  phi_{phiId} = {math.degrees(model.get_value(phiVar))}')
        print(f"  {len(violated)} violated line(s): {violated}")
        for lineId in violated:
            print(f"    {lineId}: flow={flows.loc[lineId, 'flow']:.1f} MW, limit={flows.loc[lineId, 'max_p']:.1f} MW")

        # Only fetch PTDF for lines not yet constrained
        new_branches = [b for b in violated if b not in added]
        if not new_branches:
            print("  All violated lines already constrained — infeasible or numerical issue")
            return

        ptdf_rows, psdf_rows = get_ptdf_row(network, new_branches)
        for i, branchId in enumerate(new_branches):
            add_flow_constraint(model, df_branches, branchId, ptdf_rows[i], psdf_rows[i], Phi, P_l, PGen, phi)
            print(f"  + constraint added for {branchId}  "
                  f"(flow={flows.loc[branchId, 'flow']:.1f} MW, "
                  f"limit=±{df_branches['max_p'][branchId]:.0f} MW)")

    print(f"\n=== Final dispatch  ({len(added)} active line constraints) ===")
    for gId, Pg in PGen.items():
        print(f"  {gId} = {model.get_value(Pg):.2f} MW")

if __name__ == "__main__":
    network = pp.network.load("pst1.xiidm")
    ptdf_iterative(network)