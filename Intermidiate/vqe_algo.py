"""N2 VQE — Challenge 2 (intermediate), optimized.

What this module is for
-----------------------
The starter notebook uses `real_amplitudes(reps=1, entanglement='linear')`
on 6 qubits and COBYLA(200).  That scores 2,641.  Simply changing N to 8
scores 2,796 -- worse than preparing the Hartree-Fock state alone (4,001).

The exact 8-qubit ground state is not strongly correlated:

    P(|11110000>) = 0.9829          (Hartree-Fock determinant)
    P(|11101000>) = 0.0164          (single excitation q3 -> q4)
    entanglement entropy ~ 0.125 bits

So the right circuits are:

* `brickwall_ansatz(8, 6)` -- nearest-neighbour even/odd CX layers.
  Depth 13 <= 2N, zero SWAPs, error 4.41e-4 Ha -> 4,392 / 4,400.
* `givens_ansatz(8)` -- a single particle-hole rotation between q3 and q4.
  Depth 4, 2 CX, error 7.75e-4 Ha -> 4,386 noiseless, 4,145 on
  ibm_marrakesh (the 6-layer circuit scores only 251 on hardware).

The scoring formula implemented here is the one from the challenge:

    Base(N) = 3000 + 50 * (N^2 - 36)
    A       = max(0, 1 - |E_vqe - E_exact| / 0.5)
    S_swap  = 1 / (1 + 0.25 * N_swap)
    D_depth = 2N / max(depth, 2N)
    Score   = round(Base * A^2 * S_swap * D_depth)
"""

import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap

BASELINE_ERROR = 0.5  # Hartree, the challenge's reference error
KEY_NAMES = ("IBM_Key", "IBM_Crn", "Team_Key", "Submission_URL")


# ---------------------------------------------------------------------------
# Hamiltonians
# ---------------------------------------------------------------------------

def get_n2_hamiltonian_6q() -> SparsePauliOp:
    """N2 6-qubit valence active-space Hamiltonian at equilibrium."""
    return SparsePauliOp.from_list([
        ('IIIIII', -107.5000),
        ('ZIIIII', -0.8500), ('IZIIII', -0.8500), ('IIZIII', -0.6200),
        ('IIIZII',  0.4500), ('IIIIZI',  0.4500), ('IIIIIZ',  0.7800),
        ('ZZIIII',  0.1800), ('IZZIII',  0.1800), ('IIZZII',  0.1800),
        ('IIIZZI',  0.1800), ('IIIIZZ',  0.1800),
        ('XXIIII',  0.0900), ('IXXIII',  0.0900), ('IIXXII',  0.0900),
        ('IIIXXI',  0.0900), ('IIIIXX',  0.0900),
        ('YYIIII',  0.0900), ('IYYIII',  0.0900), ('IIYYII',  0.0900),
        ('IIIYYI',  0.0900), ('IIIIYY',  0.0900),
        ('ZIIIIZ',  0.1200),
    ])


def get_n2_hamiltonian_8q() -> SparsePauliOp:
    """N2 8-qubit full p-orbital active-space Hamiltonian."""
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


# ---------------------------------------------------------------------------
# Energies and reference state
# ---------------------------------------------------------------------------

def exact_energy(hamiltonian: SparsePauliOp) -> float:
    """Lowest eigenvalue of the Hamiltonian (full diagonalisation)."""
    return float(np.linalg.eigh(hamiltonian.to_matrix())[0][0])


def hf_energy(hamiltonian: SparsePauliOp) -> float:
    """Energy of the Hartree-Fock determinant: first N/2 qubits occupied."""
    n = hamiltonian.num_qubits
    index = sum(1 << q for q in range(n // 2))
    state = np.zeros(2 ** n)
    state[index] = 1.0
    return float(np.real(state.conj() @ hamiltonian.to_matrix() @ state))


def hf_initial_params(nparams: int, n: int, seed: int = 42, noise: float = 0.05) -> np.ndarray:
    """First N/2 RY angles at pi (occupied), the rest near zero, plus jitter."""
    params = np.zeros(nparams)
    params[: n // 2] = np.pi
    rng = np.random.default_rng(seed)
    return params + rng.uniform(-noise, noise, nparams)


def statevector_energy(circuit: QuantumCircuit, hamiltonian: SparsePauliOp,
                       params=None) -> float:
    """Exact <psi|H|psi> for a (possibly parameterised) circuit."""
    bound = circuit.assign_parameters(params) if params is not None else circuit
    state = Statevector.from_instruction(bound)
    return float(np.real(state.expectation_value(hamiltonian)))


# ---------------------------------------------------------------------------
# Ansätze
# ---------------------------------------------------------------------------

def brickwall_ansatz(n: int, layers: int):
    """RY layer + alternating even/odd nearest-neighbour CX layers.

    This is the shallow linear topology the starter notebook *wanted*:
    every CX is between adjacent qubits, so it transpiles to a line with
    zero SWAPs.  Unlike `real_amplitudes(entanglement='linear')`, the CX
    layers are parallel (depth 2 per layer instead of a sequential ladder).
    """
    qc = QuantumCircuit(n)
    pv = ParameterVector("t", n * (layers + 1))
    k = 0
    for q in range(n):
        qc.ry(pv[k], q)
        k += 1
    for layer in range(layers):
        for q in range(layer % 2, n - 1, 2):
            qc.cx(q, q + 1)
        for q in range(n):
            qc.ry(pv[k], q)
            k += 1
    return qc, pv


def givens_ansatz(n: int, pair=None, occupied=None):
    """Hartree-Fock preparation + one particle-hole (Givens) rotation.

    The rotation is  CX(a,b) . RY(theta,a) . CX(a,b)  which mixes
    |1_a 0_b> and |0_a 1_b> without changing the particle number.
    For N2 the dominant correction to HF is exactly q3 -> q4.
    """
    if pair is None:
        pair = (n // 2 - 1, n // 2)
    if occupied is None:
        occupied = range(n // 2)
    a, b = pair
    qc = QuantumCircuit(n)
    qc.x(list(occupied))
    qc.cx(a, b)
    theta = Parameter("theta")
    qc.ry(theta, a)
    qc.cx(a, b)
    return qc, theta


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def optimize(circuit: QuantumCircuit, hamiltonian: SparsePauliOp,
             initial_params, maxiter: int = 2500, rhobeg: float = 0.1):
    """COBYLA on the exact statevector energy."""
    result = minimize(
        lambda p: statevector_energy(circuit, hamiltonian, p),
        initial_params,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": rhobeg},
    )
    return result


def _solve(circuit: QuantumCircuit, hamiltonian: SparsePauliOp, params) -> dict:
    exact = exact_energy(hamiltonian)
    energy = statevector_energy(circuit, hamiltonian, params)
    bound = circuit.assign_parameters(params)
    metrics = circuit_metrics(bound)
    parts = score_parts(metrics["num_qubits"], abs(energy - exact),
                        metrics["depth"], metrics["swaps"])
    return {
        "circuit": bound,
        "params": np.asarray(params, dtype=float),
        "energy": energy,
        "exact": exact,
        "error": abs(energy - exact),
        "metrics": metrics,
        "parts": parts,
        "score": parts["score"],
    }


def solve_brickwall(hamiltonian: SparsePauliOp, layers: int = 6,
                    maxiter: int = 2500, seed: int = 42) -> dict:
    """Optimise the even/odd brickwall.  6 layers -> 4,392 at N=8."""
    n = hamiltonian.num_qubits
    circuit, pv = brickwall_ansatz(n, layers)
    result = optimize(circuit, hamiltonian,
                      hf_initial_params(len(pv), n, seed=seed),
                      maxiter=maxiter)
    return _solve(circuit, hamiltonian, result.x)


def solve_givens(hamiltonian: SparsePauliOp, pair=None, seed: int = 42) -> dict:
    """Optimise the single Givens angle (1 parameter).  4,386 at N=8."""
    n = hamiltonian.num_qubits
    circuit, theta = givens_ansatz(n, pair=pair)
    grid = np.linspace(0.0, 2.0 * np.pi, 2001)
    energies = [statevector_energy(circuit, hamiltonian, [t]) for t in grid]
    start = float(grid[int(np.argmin(energies))])
    result = minimize(
        lambda p: statevector_energy(circuit, hamiltonian, p),
        [start],
        method="Nelder-Mead",
    )
    return _solve(circuit, hamiltonian, result.x)


# ---------------------------------------------------------------------------
# Scoring and circuit utilities
# ---------------------------------------------------------------------------

def score_parts(n: int, error: float, depth: int, swaps: int = 0,
                baseline_error: float = BASELINE_ERROR) -> dict:
    """Breakdown of the challenge score."""
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


def linear_coupling_map(n: int) -> CouplingMap:
    """Bidirectional line 0 - 1 - ... - (n-1)."""
    edges = []
    for i in range(n - 1):
        edges.extend([(i, i + 1), (i + 1, i)])
    return CouplingMap(edges)


def transpile_linear(circuit: QuantumCircuit, n: int | None = None,
                     optimization_level: int = 3, seed: int = 42) -> QuantumCircuit:
    """Transpile onto a linear coupling map (the hardware the score assumes)."""
    n = n or circuit.num_qubits
    return transpile(circuit, coupling_map=linear_coupling_map(n),
                     optimization_level=optimization_level,
                     seed_transpiler=seed)


def circuit_metrics(circuit: QuantumCircuit) -> dict:
    ops = dict(circuit.count_ops())
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "swaps": ops.get("swap", 0),
        "ops": ops,
    }


def to_qasm(circuit: QuantumCircuit) -> str:
    return qasm2.dumps(circuit)


def from_qasm(text: str) -> QuantumCircuit:
    return qasm2.loads(text)


def preview_score(circuit: QuantumCircuit, vqe_energy: float, exact: float,
                  baseline_error: float = BASELINE_ERROR) -> int:
    """Print the same breakdown as the challenge's local previewer."""
    metrics = circuit_metrics(circuit)
    parts = score_parts(metrics["num_qubits"], abs(vqe_energy - exact),
                        metrics["depth"], metrics["swaps"], baseline_error)
    print("=" * 56)
    print(f"SCORE BREAKDOWN — N2 ({metrics['num_qubits']} qubits)")
    print("=" * 56)
    print(f"  Base potential:        {parts['base']:,.0f} pts")
    print(f"  Absolute energy error: {abs(vqe_energy - exact):.6e} Ha")
    print(f"  Accuracy A:            {parts['accuracy']:.6f} "
          f"(A^2 = {parts['accuracy_factor']:.6f})")
    print(f"  SWAP gates:            {metrics['swaps']} "
          f"(factor {parts['swap_factor']:.3f})")
    print(f"  Depth:                 {metrics['depth']} / budget {2 * metrics['num_qubits']} "
          f"(factor {parts['depth_factor']:.3f})")
    print("-" * 56)
    print(f"  PROJECTED SCORE:       {parts['score']:,} POINTS")
    print("=" * 56)
    return parts["score"]


def load_keys(path: str | Path | None = None) -> dict:
    """Load ``KEY=VALUE`` secrets from a gitignored file.

    With no argument, search order is:

    1. ``$KEYS_FILE``
    2. ``<repo>/.env``            (the canonical, gitignored file)
    3. ``<repo>/keys.txt``
    4. ``<repo>/Begginer/keys.txt``
    5. ``<repo>/Intermidiate/keys.txt``
    6. ``./.env`` and ``./keys.txt``

    Real environment variables override values read from the file, so CI can
    inject secrets without creating a file at all.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = []
        env_path = os.environ.get("KEYS_FILE")
        if env_path:
            candidates.append(Path(env_path))
        try:
            here = Path(__file__).resolve().parent
        except NameError:  # embedded in a notebook: no __file__
            here = Path.cwd().resolve()
        repo = here.parent if here.name in ("Begginer", "Intermidiate") else here
        candidates += [
            repo / ".env",
            repo / "keys.txt",
            repo / "Begginer" / "keys.txt",
            repo / "Intermidiate" / "keys.txt",
            Path.cwd() / ".env",
            Path.cwd() / "keys.txt",
        ]

    keys = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                keys[key] = value
        break

    for key in set(keys) | set(KEY_NAMES):
        if os.environ.get(key):
            keys[key] = os.environ[key]
    return keys


# ---------------------------------------------------------------------------
# Reference numbers (used by tests and the notebook)
# ---------------------------------------------------------------------------

REFERENCE = {
    "exact_6q": -111.1031118833,
    "exact_8q": -112.1432365180,
    "hf_8q": -112.1200000000,
    "brickwall_6l_energy": -112.1427952383,
    "brickwall_6l_score": 4392,
    "brickwall_2l_energy": -112.1424620442,
    "brickwall_2l_score": 4386,
    "givens_energy": -112.1424616031,
    "givens_score": 4386,
}
