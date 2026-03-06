import math
import pypowsybl.network as pn
import pypowsybl.sensitivity as psa
import pypowsybl as ppw
import pandas as pd
import pyoptinterface as poi
from pyoptinterface import xpress
from dc_opf_ptdf import compute_ptdf
from dc_opf_two_bus import extract_parameters

from dc_opf_automaton import solve_automaton_milp, print_results, validate_with_dc_lf

if __name__ == "__main__":
    network = pn.load("two_buses.xiidm")

    change_slack = False
    ptdf = compute_ptdf(network, change_slack)

    # ── Case 2: Line over limit — automaton trips it ──────────────────────────
    print("\n" + "═" * 58)
    print("  CASE 2 — Pmax = 80 MW  (natural flow ≈ 100 MW → TRIP) - Fix With Pmin G1 = 0 instead of 50 and Pmax G2 = 250 instead of 200. We are even forced to put 0 as line limits to force the automaton to trip.")
    print("═" * 58)
    params2  = extract_parameters(network)
    params2["ptdf"] = ptdf
    params2["P12_max"] = 0
    params2["Pg_min"]["G1"] = 0
    params2["Pg_max"]["G2"] = 250
    params2["M"] = sum(params2["Pg_max"].values()) * 2  # 200 + 150 = 350 M
    results2 = solve_automaton_milp(params2, False, False)
    print_results(params2, results2)
    validate_with_dc_lf(pn.load("two_buses.xiidm"), results2)

    # ── Case 3: Line over limit — automaton trips it ──────────────────────────
    print("\n" + "═" * 58)
    print(
        "  CASE 3 — Pmax = 80 MW  (natural flow ≈ 100 MW → TRIP) - Fix with load shedding and Pmin G1 = 0  if not the generator still needs to send power but as nowhere to go. Or we could put some curtailment as bus 1")
    print("═" * 58)
    params3 = extract_parameters(network)
    params3["ptdf"] = ptdf
    params3["P12_max"] = 80
    params3["M"] = sum(params3["Pg_max"].values()) # 200 + 150 = 350 M
    params3["Pg_min"]["G1"] = 0
    results3 = solve_automaton_milp(params3, True, False)
    print_results(params3, results3)
    validate_with_dc_lf(pn.load("two_buses.xiidm"), results3)

    # ── Case 4: Line over limit — automaton trips it ──────────────────────────
    print("\n" + "═" * 58)
    print(
        "  CASE 4 — Pmax = 80 MW  (natural flow ≈ 100 MW → TRIP) - Fix with load shedding and remove Pmin G1 = 0 with curtailment variable")
    print("═" * 58)
    params4 = extract_parameters(network)
    params4["ptdf"] = ptdf
    params4["P12_max"] = 80
    params4["M"] = sum(params4["Pg_max"].values())  # 200 + 150 = 350 M
    results4 = solve_automaton_milp(params4, True, True)
    print_results(params4, results4)
    validate_with_dc_lf(pn.load("two_buses.xiidm"), results4)
