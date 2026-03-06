"""
Network topology
----------------
        Gen1 (slack)          Gen2
            |                   |
          Bus1 ---[Line12]--- Bus2
                                |
                              Load2
"""
import pypowsybl.network as pn
import pypowsybl.loadflow as lf
import pypowsybl as pp

def build_two_bus_network() -> pn.Network:
    """
    Create a minimal two-bus, one-line test network.

    Ratings
    -------
    Base MVA  : 100 MVA
    Voltage   : 400 kV (both buses)
    Gen 1     : 50–200 MW,  cost 30 $/MWh  (cheap, slack bus)
    Gen 2     : 0–150 MW,   cost 45 $/MWh  (expensive)
    Load      : 250 MW at Bus 2
    Line 1-2  : x = 0.1 pu  →  b = 10 pu,  P_max = 120 MW
    """
    network = pn.create_empty("two_bus_dc_opf")

    network.create_substations(id=["S1", "S2"], country=["FR", "FR"])

    network.create_voltage_levels(
        id            = ["VL1", "VL2"],
        substation_id = ["S1",  "S2"],
        topology_kind = ["BUS_BREAKER", "BUS_BREAKER"],
        nominal_v     = [400.0, 400.0],
    )

    network.create_buses(id=["B1", "B2"], voltage_level_id=["VL1", "VL2"])

    network.create_lines(
        id                   = ["L12"],
        voltage_level1_id    = ["VL1"],
        bus1_id              = ["B1"],
        voltage_level2_id    = ["VL2"],
        bus2_id              = ["B2"],
        r                    = [0.0],    # DC: resistance ignored
        x                    = [0.1],   # pu  →  b = 1/x = 10 pu
        g1=[0.0], b1=[0.0],
        g2=[0.0], b2=[0.0],
    )

    network.create_generators(
        id               = ["G1",   "G2"],
        voltage_level_id = ["VL1",  "VL2"],
        bus_id           = ["B1",   "B2"],
        energy_source    = ["OTHER","OTHER"],
        min_p            = [50.0,    0.0],   # MW
        max_p            = [200.0,  150.0],  # MW
        target_p         = [150.0,  100.0],  # initial dispatch (MW)
        target_q         = [0.0,  10.0],  # initial dispatch (MW)
        target_v         = [400.0,  400.0],  # kV (nominal)
        voltage_regulator_on = [True, False],
    )

    network.create_loads(
        id               = ["D2"],
        voltage_level_id = ["VL2"],
        bus_id           = ["B2"],
        p0               = [250.0],  # MW
        q0               = [0.0],
    )

    report_node_dc = pp.report.ReportNode()
    lf.run_dc(network, report_node=report_node_dc)
    print(report_node_dc)

    network.save("two_buses.xiidm")

    return network

if __name__ == "__main__":
    network = build_two_bus_network()
