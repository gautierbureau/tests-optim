import pypowsybl as pp
import math
import pyoptinterface as poi
from pyoptinterface import xpress

THETA_MAX   = math.pi / 3
PHI_MAX   = math.radians(30.0)
BASE_MVA = 100

def theta_formulation():
    network = pp.network.load("pst1.xiidm")

    generators = network.get_generators(all_attributes=True)
    loads = network.get_loads(all_attributes=True)
    buses = network.get_buses(all_attributes=True)
    lines = network.get_lines(all_attributes=True)
    tfos = network.get_2_windings_transformers(all_attributes=True)
    psts = network.get_phase_tap_changers(all_attributes=True)
    vls = network.get_voltage_levels(all_attributes=True)

    lines['nominal_v_1'] = lines['voltage_level1_id'].map(vls['nominal_v'])
    lines['nominal_v_2'] = lines['voltage_level2_id'].map(vls['nominal_v'])

    tfos['nominal_v_1'] = tfos['voltage_level1_id'].map(vls['nominal_v'])
    tfos['nominal_v_2'] = tfos['voltage_level2_id'].map(vls['nominal_v'])

    bus_map = {bus: {k: [] for k in ['PGen', 'PLoad', 'PLineIn', 'PLineOut']} for bus in buses.index}

    gen_groups = generators.index.to_series().groupby(generators['bus_id']).apply(list).to_dict()
    load_groups = loads.index.to_series().groupby(loads['bus_id']).apply(list).to_dict()

    # For lines, we handle Side 2 (In) and Side 1 (Out) separately
    line_in_groups = lines.index.to_series().groupby(lines['bus2_id']).apply(list).to_dict()
    line_out_groups = lines.index.to_series().groupby(lines['bus1_id']).apply(list).to_dict()

    tfos_in_groups = tfos.index.to_series().groupby(tfos['bus2_id']).apply(list).to_dict()
    tfos_out_groups = tfos.index.to_series().groupby(tfos['bus1_id']).apply(list).to_dict()

    # 3. Merge the results into the master map
    for bus_id in buses.index:
        bus_map[bus_id]['PGen'] = gen_groups.get(bus_id, [])
        bus_map[bus_id]['PLoad'] = load_groups.get(bus_id, [])
        bus_map[bus_id]['PLineIn'] = line_in_groups.get(bus_id, [])
        bus_map[bus_id]['PLineOut'] = line_out_groups.get(bus_id, [])
        bus_map[bus_id]['PLineIn'].extend(tfos_in_groups.get(bus_id, []))
        bus_map[bus_id]['PLineOut'].extend(tfos_out_groups.get(bus_id, []))

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

    theta = {
        b: model.add_variable(lb=-THETA_MAX, ub=THETA_MAX, name=f"theta_{b}")
        for b in buses.index[1:]
    }

    # Angle reference
    theta[buses.index[0]] = 0.0

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

    generators_cost = {
        'G1': 30,
        'G2': 45
    }

    generators['cost'] = generators.index.map(generators_cost)

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

    print(PLine)

    phi = {
        l: model.add_variable(lb=-PHI_MAX, ub=PHI_MAX, name=f"phi_{l}")
        for l in psts.index
    }

    psi = {
        l: model.add_variable(lb=0, ub=PHI_MAX, name=f"psi_{l}")
        for l in psts.index
    }

    print(phi)
    print(psi)

    for phiVar, psiVar in zip(phi.values(), psi.values()):
        model.add_linear_constraint(psiVar - phiVar, poi.Geq, 0)
        model.add_linear_constraint(psiVar + phiVar, poi.Geq, 0)

    for lidx, l in lines.iterrows():
        x = l['x']
        Z_BASE = l['nominal_v_2']**2 / BASE_MVA
        h = BASE_MVA / x * Z_BASE
        b1 = l['bus1_id']
        b2 = l['bus2_id']
        model.add_linear_constraint(PLine[lidx] - h * (theta[b1] - theta[b2]), poi.Eq, 0, name=f"flow_{lidx}")

    for pstidx, pst in psts.iterrows():
        x = tfos['x_at_current_tap'][pstidx]
        Z_BASE = tfos['nominal_v_2'][pstidx]**2 / BASE_MVA
        h = BASE_MVA / x * Z_BASE
        b1 = tfos['bus1_id'][pstidx]
        b2 = tfos['bus2_id'][pstidx]
        model.add_linear_constraint(PLine[pstidx] - h * (theta[b1] - theta[b2] + phi[pstidx]), poi.Eq, 0, name=f"flow_{pstidx}")

    # Power balance: generation - load - outgoing + incoming = 0
    for busId, busComponents in bus_map.items():
        expr = poi.ExprBuilder()
        for g in busComponents['PGen']:
            expr += PGen[g]
        for l in busComponents['PLoad']:
            expr -= loads['p0'][l]
        for l_in in busComponents['PLineIn']:
            expr += PLine[l_in]
        for l_out in busComponents['PLineOut']:
            expr -= PLine[l_out]
        model.add_linear_constraint(expr, poi.Eq, 0, name=f"bal_{busId}")

    C_PST = 5
    obj = poi.ExprBuilder()
    for g, Pg in PGen.items():
        obj += generators['cost'][g] * Pg
    for psiVar in psi.values():
        obj += C_PST * psiVar

    model.set_objective(obj, poi.ObjectiveSense.Minimize)
    print("number_of_variables: " + str(model.number_of_variables()))
    print("number_of_constraints: " + str(model.number_of_constraints(poi.ConstraintType.Linear)))
    model.optimize()

    def v(x): return model.get_value(x)
    results = {}
    for g, Pg in PGen.items():
        results[g] = v(Pg)
    for l, Pl in PLine.items():
        results[l] = v(Pl)
    for phiId, phiVar in phi.items():
        results['phi_' + phiId] = math.degrees(v(phiVar))
    results['cost'] = model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)

    print(results)

    # rhs = params["Pd"].get(bus, 0.0) + p0_shift
    # expr = poi.ExprBuilder()
    # for g in gens:
    #     expr += ctx["Pg"][g]
    #     if g in ctx.get("curt", {}):
    #         expr -= ctx["curt"][g]
    # for br in lines_in:
    #     expr += ctx["P"][br]
    # for br in lines_out:
    #     expr -= ctx["P"][br]
    # if bus in ctx.get("shed", {}):
    #     expr += ctx["shed"][bus]  # shed reduces net demand: move to lhs
    # m.add_linear_constraint(expr, poi.Eq, rhs, name=f"bal_{bus}")

if __name__ == "__main__":
    theta_formulation()