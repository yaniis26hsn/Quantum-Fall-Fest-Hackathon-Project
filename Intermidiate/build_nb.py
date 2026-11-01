"""Build challenge_intermediate_vqe_OPTIMISE.ipynb from vqe_algo.py.

Run from this folder:

    python build_nb.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
algo = (HERE / "vqe_algo.py").read_text(encoding="utf-8")
best_qasm = (HERE / "n2_8q_best.qasm").read_text(encoding="utf-8").strip()
two_layer_qasm = (HERE / "n2_8q_2layer.qasm").read_text(encoding="utf-8").strip()

cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def code(s):
    s = s.strip("\n")
    compile(s, "<cell>", "exec")  # syntax-check every cell
    cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                      outputs=[], source=s))


# ---------------------------------------------------------------------------
md(r"""
# Challenge 2 — N₂ ground state with VQE, optimized

**Goal.** Maximize

$$\text{Score} = \text{round}\Big(\text{Base}(N)\times A^2\times \frac{1}{1+0.25\,N_{\text{swap}}}\times \frac{2N}{\max(D,2N)}\Big)$$

with $N \ge 6$, where $\text{Base}(N)=3000+50(N^2-36)$ and
$A=\max(0,1-|E_{\text{vqe}}-E_{\text{exact}}|/0.5)$.

**What the starter notebook does.** `real_amplitudes(reps=1, entanglement='linear')`
on 6 qubits with COBYLA(200): **2,641 points**. Changing only `num_qubits = 8`
gives **2,796** — worse than preparing the Hartree–Fock state alone (**4,001**).

**What the ground state actually is.** The 8-qubit exact state is

$$|\psi_0\rangle \approx 0.9914\,|11110000\rangle + 0.1280\,|11101000\rangle + \dots$$

i.e. **98.29% Hartree–Fock** plus **1.64% of a single q3→q4 excitation**.
The entanglement entropy across the middle cut is only **0.125 bits**. This is a
weakly correlated state, not a strongly correlated one.

**Consequences.**

1. A deep hardware-efficient ansatz is unnecessary — and actively harmful on hardware.
2. A 6-layer nearest-neighbour brickwall reaches **4,392 / 4,400** noiselessly.
3. A **2-CX Givens rotation** reaches **4,386** noiselessly and **4,145** on
   `ibm_marrakesh` — the best noisy score by a wide margin.

| Circuit | N | 2q gates | Noiseless error | Noiseless score | Hardware score |
|---|---|---|---|---|---|
| HF only | 8 | 0 | 2.32e-2 | 4,001 | — |
| 6-layer brickwall | 8 | 21 CX | 4.41e-4 | **4,392** | 251 |
| 2-layer brickwall | 8 | 7 CX | 7.75e-4 | 4,386 | 2,233 |
| 2-CX Givens | 8 | 2 CX | 7.75e-4 | 4,386 | **4,145** |
""")

md(r"""
## How the score works

- **Base** grows quadratically with $N$: 6 → 3,000; 8 → 4,400; 16 → 14,000.
- **Accuracy** is squared. At the challenge's reference error of 0.5 Ha,
  chemical accuracy (1.6 mHa) gives $A^2=0.9936$; 0.44 mHa gives $A^2=0.9982$.
- **SWAP** costs 25% per gate. Nearest-neighbour CXs on a line → zero SWAPs.
- **Depth** is free up to $D=2N$. At $N=8$ the budget is 16; the brickwall uses 13.

The starter leaves most of the depth budget unused *and* the ansatz cannot
express the tiny correlation, so COBYLA just drifts off HF and returns.
""")

code(r"""
# Run once if needed:
# %pip install -q qiskit qiskit-aer matplotlib requests scipy numpy

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

print("imports OK")
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Setup — the algorithm module

The algorithm module is embedded below so this notebook is self-contained.

- `brickwall_ansatz(n, layers)` — RY layer + alternating **even/odd** CX layers.
  Unlike `real_amplitudes(entanglement='linear')`, the CXs are parallel, so the
  depth is $1+2\,\text{layers}$ instead of a sequential ladder of length $N-1$.
- `givens_ansatz(n)` — HF preparation + `CX(a,b) · RY(θ,a) · CX(a,b)`, a
  particle-number-conserving rotation between the highest occupied and lowest
  unoccupied orbital.
- `solve_brickwall`, `solve_givens`, `score_parts`, `preview_score`,
  `transpile_linear`, `to_qasm` / `from_qasm`.
""")

code(algo)

# ---------------------------------------------------------------------------
md(r"""
## 2. The two N₂ Hamiltonians and their exact energies

The 6-qubit and 8-qubit Hamiltonians are the ones shipped with the challenge.
`exact_energy` diagonalises the full $2^N \times 2^N$ matrix — fine for $N \le 12$.
""")

code(r"""
H6 = get_n2_hamiltonian_6q()
H8 = get_n2_hamiltonian_8q()

for name, H in (("6q", H6), ("8q", H8)):
    exact = exact_energy(H)
    ehf = hf_energy(H)
    print(f"{name}: exact = {exact:.10f} Ha   HF = {ehf:.10f} Ha   "
          f"HF error = {ehf - exact:.3e} Ha")

exact8 = exact_energy(H8)
print(f"\nN=8 Hartree-Fock alone scores "
      f"{score_parts(8, hf_energy(H8) - exact8, 1)['score']:,} points")
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. The submitted circuit — 6-layer brickwall, 4,392 points

The QASM embedded in the next cell is exactly what was sent to
`challenge_intermediate`. It has 8 qubits, 56 RY + 21 CX, depth 13, zero SWAPs.
""")

code(r"""
BEST_QASM = __BEST_QASM__
submitted = from_qasm(BEST_QASM)
submitted_energy = statevector_energy(submitted, H8)
submitted_error = abs(submitted_energy - exact8)
submitted_metrics = circuit_metrics(submitted)
submitted_parts = score_parts(8, submitted_error, submitted_metrics["depth"],
                              submitted_metrics["swaps"])

print(f"energy = {submitted_energy:.10f} Ha")
print(f"exact  = {exact8:.10f} Ha")
print(f"error  = {submitted_error:.6e} Ha")
print(f"depth  = {submitted_metrics['depth']} (budget 16), "
      f"swaps = {submitted_metrics['swaps']}, ops = {submitted_metrics['ops']}")
preview_score(submitted, submitted_energy, exact8)
""".replace("__BEST_QASM__", '"""' + best_qasm + '"""'))

md(r"""
### Rebuild it from scratch (optional, ~3 minutes)

Set `REOPTIMIZE = True` to rerun the 56-parameter COBYLA optimization.
The result should match the QASM above to the last printed digit.
""")

code(r"""
REOPTIMIZE = False

if REOPTIMIZE:
    rebuilt = solve_brickwall(H8, layers=6, maxiter=2500)
    print(f"rebuilt energy = {rebuilt['energy']:.10f} Ha")
    print(f"rebuilt error  = {rebuilt['error']:.6e} Ha")
    print(f"rebuilt score  = {rebuilt['score']:,}")
else:
    print("REOPTIMIZE=False — using the embedded QASM (4,392).")
    print("Set REOPTIMIZE=True to rerun the 56-parameter optimization.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. The hardware-optimal circuit — one Givens rotation, 4,386 noiseless

The exact state is HF + a single q3→q4 excitation. A single
`CX(3,4) · RY(θ,3) · CX(3,4)` prepares the optimal 2D subspace state:

- 4 X (HF preparation) + 2 CX + 1 RY, **depth 4**
- error **7.75e-4 Ha**, $A^2 = 0.9969$
- score **4,386** — only 6 points below the 6-layer circuit, with 10× fewer
  two-qubit gates.

On hardware this is the difference between **251** and **4,145**.
""")

code(r"""
givens = solve_givens(H8)
print(f"theta  = {givens['params'][0]:.8f}")
print(f"energy = {givens['energy']:.10f} Ha")
print(f"error  = {givens['error']:.6e} Ha")
print(f"depth  = {givens['metrics']['depth']}, swaps = {givens['metrics']['swaps']}, "
      f"ops = {givens['metrics']['ops']}")
preview_score(givens["circuit"], givens["energy"], exact8)
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Side-by-side comparison

All circuits are evaluated on the same 8-qubit Hamiltonian. The HF row shows
that the starter's optimizer is not even beating the trivial reference state.
""")

code(r"""
def table_row(name, circuit):
    e = statevector_energy(circuit, H8)
    m = circuit_metrics(circuit)
    p = score_parts(8, abs(e - exact8), m["depth"], m["swaps"])
    return name, m["num_qubits"], m["depth"], m["swaps"], abs(e - exact8), p["score"]


hf_circuit = QuantumCircuit(8)
hf_circuit.x(range(4))
TWO_LAYER_QASM = __TWO_LAYER_QASM__
two_layer = from_qasm(TWO_LAYER_QASM)

rows = [
    table_row("HF only", hf_circuit),
    table_row("brickwall 2 layers", two_layer),
    table_row("brickwall 6 layers", submitted),
    table_row("Givens 2 CX", givens["circuit"]),
]

print(f"{'circuit':<22}{'N':>3}{'depth':>7}{'SWAP':>6}{'error':>12}{'score':>8}")
print("-" * 58)
for name, n, d, sw, err, sc in rows:
    print(f"{name:<22}{n:>3}{d:>7}{sw:>6}{err:>12.3e}{sc:>8,}")
""".replace("__TWO_LAYER_QASM__", '"""' + two_layer_qasm + '"""'))

# ---------------------------------------------------------------------------
md(r"""
## 6. With noise — the same circuits on `ibm_marrakesh`

Recorded with `EstimatorV2`, 4,096 shots, readout mitigation
(`resilience_level=1`), on the 8-qubit line
`[149, 148, 147, 137, 127, 126, 125, 124]` (zero SWAPs after transpiling).

| Circuit | 2q gates | Hardware energy | Error | Hardware score |
|---|---|---|---|---|
| 6-layer brickwall | 21 CZ | −111.7626694 | 0.3806 Ha | **251** |
| 2-layer brickwall | 7 CZ | −111.9994128 | 0.1438 Ha | **2,233** |
| **2-CX Givens** | **2 CZ** | **−112.1285148** | **0.0147 Ha** | **4,145** |

The Hamiltonian's large $Z$ coefficients (~0.85) amplify every bit flip, so
the noiseless-optimal circuit is the worst one on hardware. The 2-CX circuit
keeps 94.2% of its noiseless score.

ZNE (`resilience_level=2`, noise factors 1/3/5) returned `NaN` from the
exponential fit; its raw points were −112.12298, −112.11255, −112.08032,
and a linear extrapolation gives ~4,233. More shots would tighten that.

**Job IDs:** `dbji6b1ffo6c73b9nbmg` (6-layer),
`dbjj61pkdqps7390mvd0` (2-layer), `dbjj71hffo6c73b9p360` (Givens),
`dbjj8k1ffo6c73b9p5a0` (ZNE).
""")

code(r"""
RUN_HARDWARE = False  # set True to spend IBM Quantum time

BACKEND = "ibm_marrakesh"
QUBIT_LINE = [149, 148, 147, 137, 127, 126, 125, 124]

if RUN_HARDWARE:
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    keys = load_keys()
    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=keys["IBM_Key"], instance=keys["IBM_Crn"])
    backend = service.backend(BACKEND)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      initial_layout=QUBIT_LINE, seed_transpiler=42)
    isa = pm.run(givens["circuit"])
    H_isa = H8.apply_layout(isa.layout)

    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 4096
    estimator.options.resilience_level = 1
    job = estimator.run([(isa, H_isa)])
    print("job id:", job.job_id())

    e_noisy = float(np.asarray(job.result()[0].data.evs).reshape(-1)[0])
    err = abs(e_noisy - exact8)
    A = max(0.0, 1.0 - err / 0.5)
    print(f"hardware energy = {e_noisy:.10f} Ha")
    print(f"error           = {err:.6e} Ha")
    print(f"hardware score  = {round(4400 * A * A):,}")
else:
    print("RUN_HARDWARE=False — recorded result: 4,145 points.")
    print("Set RUN_HARDWARE=True to rerun the 2-CX circuit on ibm_marrakesh.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Submission

The official submission was made with the embedded 6-layer QASM and
`obtainedResult = 0.99911744`:

- **Submission ID:** `5571d09f-3e27-45b7-be28-35df1c7d1137`
- **Official score:** **4,392 points**
- The platform scores the QASM shape (N, depth, SWAPs) plus the reported $A$;
  it does not execute the circuit. That is why the official number is the
  noiseless one, while the same circuit scores 251 on real hardware.
""")

code(r"""
SUBMIT = False  # set True to send a new submission (10-second rate limit)

if SUBMIT:
    import requests

    keys = load_keys()
    payload = {
        "qasm": to_qasm(submitted),
        "challengeId": "challenge_intermediate",
        "obtainedResult": float(submitted_parts["accuracy"]),
    }
    response = requests.post(keys["Submission_URL"], json=payload,
                             headers={"X-API-Key": keys["Team_Key"]}, timeout=60)
    print(response.status_code, response.json())
else:
    print("SUBMIT=False — official result on record: 4,392 points.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Self-contained notebook

The algorithm module and both QASM circuits are embedded in the cells above,
so this notebook runs on its own. The only external file is the optional,
gitignored `.env` used by the hardware/submission cells.

**Bottom line.** The provided Hamiltonian's ground state is 98.3% HF. The
starter's `real_amplitudes` ansatz cannot express the small correlation, and
COBYLA(200) returns an energy worse than HF. A 6-layer even/odd brickwall
reaches **4,392/4,400** noiselessly; a 2-CX Givens rotation reaches **4,386**
and is the only circuit that survives real hardware (**4,145**). The official
platform score is **4,392**.
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

out = HERE / "challenge_intermediate_vqe_OPTIMISE.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out.name}: {len(cells)} cells")
