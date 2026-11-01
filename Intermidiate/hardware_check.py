"""Re-run the IBM hardware measurements for the N2 VQE circuits.

Nothing is sent to IBM unless you pass ``--run``.  Without it the script only
prints the transpiled circuit shape and the recorded reference numbers.

Examples (from this folder):

    python hardware_check.py --circuit givens
    python hardware_check.py --circuit givens --run
    python hardware_check.py --circuit 2layer --run
    python hardware_check.py --circuit best   --run
    python hardware_check.py --circuit givens --run --zne

Keys are read from the gitignored root ``.env`` (fallbacks: ``keys.txt``)
and are never printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import vqe_algo as v

HERE = Path(__file__).resolve().parent

BACKEND = "ibm_marrakesh"
QUBIT_LINE = [149, 148, 147, 137, 127, 126, 125, 124]

CIRCUITS = {
    "best": "n2_8q_best.qasm",
    "2layer": "n2_8q_2layer.qasm",
    "givens": "n2_8q_givens.qasm",
}

# Recorded on ibm_marrakesh, 4096 shots, resilience_level=1.
RECORDED = {
    "best": dict(gates="21 CZ", energy=-111.76266940363858, score=251,
                 job="dbji6b1ffo6c73b9nbmg"),
    "2layer": dict(gates="7 CZ", energy=-111.9994127667, score=2233,
                   job="dbjj61pkdqps7390mvd0"),
    "givens": dict(gates="2 CZ", energy=-112.1285148049, score=4145,
                   job="dbjj71hffo6c73b9p360"),
}


def build(circuit_name: str, optimization_level: int = 3, seed: int = 42):
    """Load the QASM and transpile it onto the fixed 8-qubit line."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    keys = v.load_keys()
    if not keys:
        raise SystemExit("no keys file found (.env or keys.txt)")

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=keys["IBM_Key"],
        instance=keys["IBM_Crn"],
    )
    backend = service.backend(BACKEND)

    qc = v.from_qasm((HERE / CIRCUITS[circuit_name]).read_text(encoding="utf-8"))
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        initial_layout=QUBIT_LINE,
        seed_transpiler=seed,
    )
    isa = pm.run(qc)
    return service, backend, qc, isa


def verify_layout(qc, isa, hamiltonian) -> float:
    """Replay the device gates on 8 qubits and check the noiseless energy."""
    from qiskit import QuantumCircuit

    p2i = {p: i for i, p in enumerate(QUBIT_LINE)}
    small = QuantumCircuit(qc.num_qubits)
    for inst in isa.data:
        qs = [isa.find_bit(q).index for q in inst.qubits]
        if all(q in p2i for q in qs):
            small.append(inst.operation, [p2i[q] for q in qs])
    return v.statevector_energy(small, hamiltonian)


def report(name: str, energy: float, depth: int, swaps: int, exact: float) -> int:
    error = abs(energy - exact)
    accuracy = max(0.0, 1.0 - error / v.BASELINE_ERROR)
    score = v.score_parts(8, error, depth, swaps)["score"]
    print()
    print(f"  circuit        : {name}")
    print(f"  hardware energy: {energy:.10f} Ha")
    print(f"  exact          : {exact:.10f} Ha")
    print(f"  error          : {error:.6e} Ha")
    print(f"  accuracy A     : {accuracy:.8f}  (A^2 = {accuracy ** 2:.8f})")
    print(f"  hardware score : {score:,}")
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", choices=sorted(CIRCUITS), default="givens")
    parser.add_argument("--run", action="store_true", help="actually submit to IBM")
    parser.add_argument("--zne", action="store_true",
                        help="resilience_level=2 + ZNE (1,3,5)")
    parser.add_argument("--shots", type=int, default=4096)
    args = parser.parse_args()

    hamiltonian = v.get_n2_hamiltonian_8q()
    exact = v.exact_energy(hamiltonian)

    print(f"backend      : {BACKEND}")
    print(f"qubit line   : {QUBIT_LINE}")
    print(f"circuit file : {CIRCUITS[args.circuit]}")
    print(f"exact energy : {exact:.10f} Ha")

    ref = RECORDED[args.circuit]
    print(f"recorded     : {ref['energy']:.10f} Ha ({ref['gates']}) "
          f"-> {ref['score']:,} points, job {ref['job']}")

    if not args.run:
        print("\nDRY RUN - nothing sent to IBM.  Add --run to execute.")
        print("Run `python hardware_check.py --circuit givens --run` for the "
              "best hardware circuit.")
        return 0

    service, backend, qc, isa = build(args.circuit)
    h_isa = hamiltonian.apply_layout(isa.layout)

    swaps = isa.count_ops().get("swap", 0)
    print(f"ISA depth    : {isa.depth()}   ops: {dict(isa.count_ops())}   "
          f"swaps: {swaps}")

    replay = verify_layout(qc, isa, hamiltonian)
    print(f"layout check : replayed statevector energy = {replay:.10f} Ha "
          f"(delta {abs(replay - v.statevector_energy(qc, hamiltonian)):.2e})")

    from qiskit_ibm_runtime import EstimatorV2 as Estimator

    estimator = Estimator(mode=backend)
    estimator.options.default_shots = args.shots
    estimator.options.resilience_level = 2 if args.zne else 1
    if args.zne:
        estimator.options.resilience.zne_mitigation = True
        estimator.options.resilience.zne.noise_factors = (1, 3, 5)
        estimator.options.resilience.zne.extrapolator = "exponential"

    job = estimator.run([(isa, h_isa)])
    print(f"job id       : {job.job_id()}")
    status = None
    for _ in range(90):
        try:
            status = job.status()
            print(f"status       : {status}")
            if str(status) in ("DONE", "ERROR", "CANCELLED"):
                break
        except Exception as exc:
            print(f"status error : {type(exc).__name__}: {exc}")
        time.sleep(10)
    if str(status) != "DONE":
        raise SystemExit(f"job ended with {status}")

    result = job.result()
    data = result[0].data
    energy = float(np.asarray(data.evs).reshape(-1)[0])

    if args.zne:
        # `evs_noise_factors` holds the energy at each noise factor (1, 3, 5);
        # `evs_extrapolated` holds one value per extrapolator tried.
        print("\n  ZNE raw points :", np.asarray(data.evs_noise_factors))
        print("  ZNE extrapolated:", np.asarray(data.evs_extrapolated))
        if np.isnan(energy):
            print("  exponential fit returned NaN - use the raw points above "
                  "for a linear fit.")

    score = report(args.circuit, energy, qc.depth(), swaps, exact)
    print(f"\n  compare with recorded score: {ref['score']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
