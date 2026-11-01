"""Fast checks for the N2 VQE solution (no IBM, no hardware).

Run from this folder:

    python test_vqe_algo.py
"""
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

import vqe_algo as v

HERE = Path(__file__).resolve().parent

# --- exact energies -------------------------------------------------------
H6 = v.get_n2_hamiltonian_6q()
H8 = v.get_n2_hamiltonian_8q()
assert abs(v.exact_energy(H6) - v.REFERENCE["exact_6q"]) < 1e-9
assert abs(v.exact_energy(H8) - v.REFERENCE["exact_8q"]) < 1e-9
assert abs(v.hf_energy(H8) - v.REFERENCE["hf_8q"]) < 1e-9

# --- submitted 6-layer brickwall QASM -------------------------------------
qasm_path = HERE / "n2_8q_best.qasm"
qc = v.from_qasm(qasm_path.read_text(encoding="utf-8"))
assert qc.num_qubits == 8
assert qc.depth() == 13
assert qc.count_ops().get("swap", 0) == 0
energy = v.statevector_energy(qc, H8)
assert abs(energy - v.REFERENCE["brickwall_6l_energy"]) < 1e-8
parts = v.score_parts(8, abs(energy - v.exact_energy(H8)), qc.depth())
assert parts["score"] == v.REFERENCE["brickwall_6l_score"], parts

# --- 2-CX Givens circuit (hardware-optimal) -------------------------------
givens = v.solve_givens(H8)
assert givens["metrics"]["depth"] == 4
assert givens["metrics"]["swaps"] == 0
assert abs(givens["energy"] - v.REFERENCE["givens_energy"]) < 1e-8
assert givens["score"] == v.REFERENCE["givens_score"], givens["score"]

# every CX is nearest-neighbour on a line -> zero SWAPs after transpiling
isa = v.transpile_linear(givens["circuit"])
assert isa.count_ops().get("swap", 0) == 0
assert isa.depth() <= 2 * isa.num_qubits

# --- 6-qubit version also works -------------------------------------------
givens6 = v.solve_givens(H6)
assert givens6["error"] < 2e-3, givens6["error"]
assert givens6["score"] > 2900, givens6["score"]

# --- QASM round-trip preserves the state ----------------------------------
round_trip = v.from_qasm(v.to_qasm(givens["circuit"]))
assert abs(v.statevector_energy(round_trip, H8) - givens["energy"]) < 1e-10

# --- score formula sanity --------------------------------------------------
assert v.score_parts(8, 0.0, 16)["score"] == 4400
assert v.score_parts(8, 0.5, 16)["score"] == 0
assert v.score_parts(6, 0.0, 12)["score"] == 3000

print("OK: exact energies, QASM round-trip, zero-SWAP line, scores")
print(f"  brickwall 6 layers : {v.REFERENCE['brickwall_6l_score']:,}")
print(f"  givens 2 CX        : {v.REFERENCE['givens_score']:,}")
print(f"  givens 6 qubits    : {givens6['score']:,}")
