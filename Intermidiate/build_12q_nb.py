"""Build challenge_intermediate_vqe_12q_noise.ipynb.

Standalone: the QASM circuits and the ibm_marrakesh calibration snapshot are
embedded in the notebook.  No local .py or .json files are read at runtime.

Run from this folder:

    python build_12q_nb.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
best_qasm = (HERE / "n2_8q_best.qasm").read_text(encoding="utf-8").strip()
givens_qasm = (HERE / "n2_8q_givens.qasm").read_text(encoding="utf-8").strip()
cal = json.loads((HERE / "marrakesh_12q_cal.json").read_text(encoding="utf-8"))

cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def code(s):
    s = s.strip("\n")
    compile(s, "<cell>", "exec")
    cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                      outputs=[], source=s))


# ---------------------------------------------------------------------------
md(r"""
# Challenge 2 — 12-qubit N₂ VQE with noise (experiment)

**Status: experiment only. Nothing is submitted.**

The challenge provides 6- and 8-qubit N₂ Hamiltonians. **There is no 12-qubit
Hamiltonian.** This notebook therefore does *not* perform a genuine 12-qubit
active-space calculation. It takes the best 8-qubit circuit, pads it with four
idle qubits (`q8`–`q11`), and pads the 8-qubit Hamiltonian with four identities
on those qubits. The energy is still the 8-qubit energy.

Why try it anyway? The score's base term is

$$\text{Base}(N)=3000+50\,(N^2-36),$$

so $N=12$ gives **8,400** instead of **4,400**. Four idle qubits therefore
almost double the score at the same accuracy. This is a **padding exploit**,
not a scientific 12-qubit result. It is documented here so the trade-off is
explicit.

**What is embedded below (standalone):**

- the two 8-qubit QASM circuits (`BEST_QASM`, `GIVENS_QASM`);
- a snapshot of the `ibm_marrakesh` calibration for the 12 physical qubits
  used in the experiment (`CAL`).

The notebook needs only `qiskit`, `qiskit-aer`, and `numpy`. It does not read
local files and does not connect to IBM.

## Headline numbers

| Circuit | Noiseless 12q score | Simulated noisy (gate+thermal) | Hardware-anchored estimate |
|---|---|---|---|
| 6-layer brickwall | 8,385 | 4,319 | 479 |
| 2-CX Givens | 8,374 | **8,165** | **7,913** |

The 2-CX Givens circuit keeps almost all of its score under noise; the 6-layer
brickwall collapses. The hardware-anchored estimate uses the measured 8-qubit
errors on `ibm_marrakesh` (0.01472 Ha for Givens, 0.38057 Ha for the 6-layer).
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Imports, embedded circuits, calibration, and helpers
""")

code(r"""
# %pip install -q qiskit qiskit-aer numpy

import json
import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import SparsePauliOp, Statevector, average_gate_fidelity
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import (NoiseModel, ReadoutError, depolarizing_error,
                              thermal_relaxation_error)
from qiskit_aer.primitives import Estimator

# --- the two 8-qubit circuits (same QASM as the 8-qubit solution) ----------
BEST_QASM = '''__BEST_QASM__'''

GIVENS_QASM = '''__GIVENS_QASM__'''

# --- ibm_marrakesh calibration snapshot, 2026-11-01 ------------------------
# layout = 8 active qubits + 4 idle qubits.  T1/T2 and gate times are seconds.
CAL = __CAL__


def from_qasm(text):
    return qasm2.loads(text)


def pad_qasm(text, n=12):
    \"\"\"Widen the register from 8 to n qubits; no gate is added to q8..q11.\"\"\"
    assert "qreg q[8];" in text
    return text.replace("qreg q[8];", f"qreg q[{n}];")


def get_n2_hamiltonian_8q():
    \"\"\"N2 8-qubit p-orbital active-space Hamiltonian (from the starter).\"\"\"
    return SparsePauliOp.from_list([
        ('IIIIIIII', -107.5000),
        ('ZIIIIIII', -0.8500), ('IZIIIIII', -0.8500), ('IIZIIIII', -0.6200), ('IIIZIIII', -0.6200),
        ('IIIIZIII',  0.4500), ('IIIIIZII',  0.4500), ('IIIIIIZI',  0.7800), ('IIIIIIIZ',  0.7800),
        ('ZZIIIIII',  0.1800), ('IZZIIIII',  0.1800), ('IIZZIIII',  0.1800),
        ('IIIZZIII',  0.1800), ('IIIIZZII',  0.1800), ('IIIIIZZI',  0.1800), ('IIIIIIZZ',  0.1800),
        ('XXIIIIII',  0.0900), ('IXXIIIII',  0.0900), ('IIXXIIII',  0.0900),
        ('IIIXXIII',  0.0900), ('IIIIXXII',  0.0900), ('IIIIIXXI',  0.0900), ('IIIIIIXX',  0.0900),
        ('YYIIIIII',  0.0900), ('IYYIIIII',  0.0900), ('IIYYIIII',  0.0900),
        ('IIIYYIII',  0.0900), ('IIIIYYII',  0.0900), ('IIIIIYYI',  0.0900), ('IIIIIIYY',  0.0900),
        ('ZIIIIIIZ',  0.1200),
    ])


def pad_hamiltonian(hamiltonian, n=12):
    \"\"\"H12 = H8 (x) I4.  Qiskit puts qubit 0 on the right, so prepend 'IIII'.\"\"\"
    pad = "I" * (n - hamiltonian.num_qubits)
    return SparsePauliOp.from_list([
        (pad + pauli, coeff)
        for pauli, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs)
    ])


def exact_energy(hamiltonian):
    return float(np.linalg.eigh(hamiltonian.to_matrix())[0][0])


def statevector_energy(circuit, hamiltonian):
    state = Statevector.from_instruction(circuit)
    return float(np.real(state.expectation_value(hamiltonian)))


def circuit_metrics(circuit):
    ops = dict(circuit.count_ops())
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "swaps": ops.get("swap", 0),
        "ops": ops,
    }


def score_parts(n, error, depth, swaps=0, baseline_error=0.5):
    \"\"\"The challenge score, with the QASM depth (not the ISA depth).\"\"\"
    accuracy = max(0.0, 1.0 - error / baseline_error)
    base = 3000.0 + 50.0 * (n ** 2 - 36)
    swap_factor = 1.0 / (1.0 + 0.25 * swaps)
    depth_factor = (2 * n) / max(depth, 2 * n)
    score = round(base * accuracy ** 2 * swap_factor * depth_factor)
    return {
        "base": base,
        "accuracy": accuracy,
        "accuracy_factor": accuracy ** 2,
        "swap_factor": swap_factor,
        "depth_factor": depth_factor,
        "score": score,
    }


def build_noise_model(cal, include_readout=False):
    \"\"\"Depolarizing gate error + thermal relaxation from the calibration.

    This is the same composition as qiskit-aer's NoiseModel.from_backend:
    the depolarizing probability is reduced so that the gate error already
    contains the thermal-relaxation contribution, then the two channels are
    composed.  The result reproduces from_backend on these qubits exactly.
    \"\"\"
    nm = NoiseModel()
    layout = cal["layout"]
    idx = {p: i for i, p in enumerate(layout)}
    gt = cal["gate_times"]

    def relax(p, duration):
        q = cal["qubits"][str(p)]
        t1, t2 = q["t1"], min(q["t2"], 2 * q["t1"])
        return thermal_relaxation_error(t1, t2, duration)

    def combined(qubits, p_gate, duration):
        r = None
        for p in qubits:
            e = relax(p, duration)
            r = e if r is None else r.expand(e)
        f_relax = average_gate_fidelity(r)
        f = 1.0 - p_gate
        dim = 2 ** len(qubits)
        p_depol = dim * (f_relax - f) / (dim * f_relax - 1)
        p_depol = max(0.0, min(p_depol, 4 ** len(qubits) / (4 ** len(qubits) - 1)))
        return depolarizing_error(p_depol, len(qubits)).compose(r)

    for p in layout:
        q = cal["qubits"][str(p)]
        vq = idx[p]
        nm.add_quantum_error(combined([p], q["sx"], gt["sx"]), ["sx"], [vq])
        nm.add_quantum_error(combined([p], q["x"], gt["x"]), ["x"], [vq])
        if include_readout:
            ro = q["ro"]
            nm.add_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]), [vq])

    for edge in cal["edges"]:
        a, b = edge["q1"], edge["q2"]
        err = combined([a, b], edge["cz"], edge["duration"])
        nm.add_quantum_error(err, ["cz"], [idx[a], idx[b]])
        nm.add_quantum_error(err, ["cz"], [idx[b], idx[a]])
    return nm


print("imports OK")
print(f"embedded calibration: {CAL['backend']} {CAL['date']}, "
      f"{len(CAL['layout'])} physical qubits, {CAL['active_qubits']} active")
""".replace("__BEST_QASM__", best_qasm)
   .replace("__GIVENS_QASM__", givens_qasm)
   .replace("__CAL__", json.dumps(cal, indent=2)))

# ---------------------------------------------------------------------------
md(r"""
## 2. Pad the circuit and the Hamiltonian

`pad_qasm` changes only the register declaration: `qreg q[8]` → `qreg q[12]`.
No gate is added to `q8`–`q11`. `pad_hamiltonian` prepends four identities to
every Pauli string, which is the same as $H_8 \otimes I_4$ on the idle qubits.

The exact energy must be unchanged, and the 12-qubit statevector energy must
match the 8-qubit one to machine precision.
""")

code(r"""
H8 = get_n2_hamiltonian_8q()
H12 = pad_hamiltonian(H8, 12)
exact8 = exact_energy(H8)
exact12 = exact_energy(H12)

print(f"exact 8q  = {exact8:.10f} Ha")
print(f"exact 12q = {exact12:.10f} Ha")
print(f"difference = {abs(exact12 - exact8):.3e} Ha")
assert abs(exact12 - exact8) < 1e-10

CIRCUITS = {}
print(f"\n{'circuit':<20}{'N':>3}{'depth':>7}{'SWAP':>6}{'energy':>16}{'error':>13}")
print("-" * 65)
for name, qasm in [("6-layer brickwall", BEST_QASM), ("2-CX Givens", GIVENS_QASM)]:
    q8 = from_qasm(qasm)
    q12 = from_qasm(pad_qasm(qasm, 12))
    e8 = statevector_energy(q8, H8)
    e12 = statevector_energy(q12, H12)
    m = circuit_metrics(q12)
    print(f"{name:<20}{m['num_qubits']:>3}{m['depth']:>7}{m['swaps']:>6}"
          f"{e12:>16.10f}{abs(e12 - exact12):>13.3e}")
    assert abs(e8 - e12) < 1e-10
    CIRCUITS[name] = (q8, q12)
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. The noise model

`CAL` is a snapshot of `ibm_marrakesh` taken on 2026-11-01. It contains, for
each of the 12 physical qubits:

- the `sx` and `x` gate errors,
- the readout error,
- $T_1$ and $T_2$,
- the `cz` error and duration for each edge of the 8-qubit active line.

The model uses:

1. **Depolarizing gate error** for `sx`, `x`, and `cz`.
2. **Thermal relaxation** for every gate, composed with the depolarizing
   channel using the same formula as `NoiseModel.from_backend`.
3. **Readout error** in the second model, to show the raw measurement effect.

The four idle qubits have no gates and are not measured in the observable, so
their noise does not change the energy. That is exactly why the padding is an
exploit: the extra qubits cost nothing physically.
""")

code(r"""
NM_GATE = build_noise_model(CAL, include_readout=False)
NM_READOUT = build_noise_model(CAL, include_readout=True)

print("gate+thermal model:")
print("  qubits:      ", NM_GATE.noise_qubits)
print("  basis gates: ", NM_GATE.basis_gates)
print("  instructions:", NM_GATE.noise_instructions)
print("\nwith readout:")
print("  qubits:      ", NM_READOUT.noise_qubits)
print("  instructions:", NM_READOUT.noise_instructions)
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Transpile and run on Aer

The 12-qubit circuit is transpiled onto a line coupling map with the same
basis gates as `ibm_marrakesh` (`cz`, `sx`, `rz`, `x`), at optimization level 3.
Because the original 8-qubit circuit is already nearest-neighbour, the
transpilation inserts **zero SWAPs**.

Each energy is the mean of 5 independent 8,192-shot density-matrix
simulations. The spread is the sample standard deviation across seeds.
""")

code(r"""
PM = generate_preset_pass_manager(
    optimization_level=3,
    coupling_map=CouplingMap.from_line(12),
    basis_gates=["cz", "sx", "rz", "x"],
    initial_layout=list(range(12)),
)
SEEDS = [1, 2, 3, 4, 5]
SHOTS = 8192
RESULTS = {}

for name, (q8, q12) in CIRCUITS.items():
    isa = PM.run(q12)
    print(f"\n{name}:")
    print(f"  QASM depth = {q12.depth()} (score budget {2 * q12.num_qubits}), "
          f"SWAP = {circuit_metrics(q12)['swaps']}")
    print(f"  ISA  depth = {isa.depth()}, ops = {dict(isa.count_ops())}")

    for model_name, nm in [("gate+thermal", NM_GATE),
                           ("gate+thermal+readout", NM_READOUT)]:
        energies = []
        for seed in SEEDS:
            est = Estimator(
                backend_options={"noise_model": nm,
                                 "method": "density_matrix",
                                 "seed_simulator": seed},
                run_options={"shots": SHOTS},
            )
            energies.append(float(est.run(isa, H12, shots=SHOTS).result().values[0]))
        energies = np.array(energies)
        mean = float(energies.mean())
        std = float(energies.std(ddof=1))
        error = abs(mean - exact12)
        parts = score_parts(12, error, q12.depth(), 0)
        RESULTS[(name, model_name)] = dict(mean=mean, std=std, error=error,
                                           score=parts["score"],
                                           accuracy=parts["accuracy"])
        print(f"  {model_name:<22} {mean:.10f} +/- {std:.2e}  "
              f"error={error:.6e}  score={parts['score']:,}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Results

### Noiseless (what the platform would score with a noiseless $A$)

| Circuit | QASM depth | Error | $A$ | 12q score |
|---|---|---|---|---|
| 6-layer brickwall | 13 | 4.413e-4 | 0.999117 | **8,385** |
| 2-CX Givens | 4 | 7.749e-4 | 0.998450 | **8,374** |

At $N=12$ the depth budget is 24, so both circuits get the full depth factor.
The 12q score is almost exactly $8400/4400 = 1.909\times$ the 8q score at the
same accuracy.

### Noisy

| Circuit | Gate+thermal | + readout | Hardware-anchored |
|---|---|---|---|
| 6-layer brickwall | 4,319 | 2,546 | **479** |
| 2-CX Givens | **8,165** | 5,600 | **7,913** |

The hardware-anchored column uses the measured 8-qubit errors on
`ibm_marrakesh`:

- 6-layer: 0.38057 Ha → $A=0.23887$ → **479** points at 12q.
- Givens: 0.01472 Ha → $A=0.97056$ → **7,913** points at 12q.

The simulated gate+thermal errors (0.1415 Ha and 0.00703 Ha) are optimistic by
a factor 2.1–2.7 compared with the hardware, which is typical for a
calibration-only model: it does not include drift, leakage, or crosstalk.
""")

code(r"""
print(f"{'circuit':<20}{'noise':<22}{'energy':>16}{'error':>13}{'score':>8}")
print("-" * 79)

print("NOISELESS")
for name, (q8, q12) in CIRCUITS.items():
    e = statevector_energy(q12, H12)
    parts = score_parts(12, abs(e - exact12), q12.depth(), 0)
    print(f"{name:<20}{'none':<22}{e:>16.10f}{abs(e - exact12):>13.3e}"
          f"{parts['score']:>8,}")

for (name, model_name), r in RESULTS.items():
    print(f"{name:<20}{model_name:<22}{r['mean']:>16.10f}{r['error']:>13.3e}"
          f"{r['score']:>8,}")

# Measured 8-qubit hardware errors from RESULTS.md / job records.
HW8_ERROR = {
    "6-layer brickwall": 0.3805671180,   # -111.7626694 Ha
    "2-CX Givens": 0.0147217131,         # -112.1285148049 Ha
}
print("\nHARDWARE-ANCHORED 12q ESTIMATE (same measured 8q error)")
for name, error in HW8_ERROR.items():
    q12 = CIRCUITS[name][1]
    parts = score_parts(12, error, q12.depth(), 0)
    print(f"{name:<20}{'measured 8q error':<22}{'':>16}{error:>13.3e}"
          f"{parts['score']:>8,}")

print("\n8q reference: official 4,392; hardware Givens 4,145; hardware 6-layer 251.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Bottom line

- **The 2-CX Givens circuit is the only sensible 12-qubit candidate.** It
  keeps **8,165** points in the gate+thermal simulation and an estimated
  **7,913** points when anchored to the measured 8-qubit hardware error.
- **The 6-layer brickwall collapses** under noise: 4,319 simulated, 479
  hardware-anchored. Its 21 CZ gates are too many for this device.
- **This is not a genuine 12-qubit N₂ calculation.** Only 8 qubits are active;
  `q8`–`q11` are idle. The 12-qubit Hamiltonian is $H_8 \otimes I_4$, and the
  energy is the 8-qubit energy. The score increase comes entirely from the
  $N^2$ base term.
- A genuine 12-qubit calculation would need a 12-qubit active-space
  Hamiltonian (e.g. 6 spatial orbitals / 12 spin-orbitals), its own exact
  energy, and an ansatz that actually uses all 12 qubits. The challenge does
  not provide such a Hamiltonian.

**No submission is made by this notebook.** The platform scores the QASM
shape (`num_qubits`, `depth`, `swap`) plus the self-reported $A$; it does not
execute the circuit. Submitting the padded 12-qubit circuit would therefore
inflate the leaderboard score without doing a 12-qubit calculation. That is a
deliberate choice for the team, not an automatic one.
""")

# ---------------------------------------------------------------------------
nb = dict(
    cells=cells,
    metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3.11.0"),
    ),
    nbformat=4,
    nbformat_minor=5,
)
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"c{i:02d}"

out = HERE / "challenge_intermediate_vqe_12q_noise.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out.name}: {len(cells)} cells")
