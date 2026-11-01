# Challenge 2 (intermediate) — N₂ VQE, results

**Official score on record: 4,392 / 4,400** (noiseless, submitted).
**Best score measured on real hardware: 4,145** (2-CX Givens circuit, `ibm_marrakesh`).

---

## 1. What the ground state actually is

The 8-qubit Hamiltonian's exact ground state is *weakly* correlated:

| Quantity | Value |
|---|---|
| Exact energy | `-112.1432365180 Ha` |
| Hartree–Fock energy | `-112.1200000000 Ha` |
| HF weight in the ground state | **98.29 %** (`|11110000>`) |
| Dominant correction | **1.64 %** single excitation q3→q4 (`|11101000>`) |
| Entanglement entropy (middle cut) | **0.125 bits** |

That is why a deep hardware-efficient ansatz is the wrong tool: it has to
re-learn a state that is almost a single determinant, and it spends the
accuracy budget on parameters that only add noise.

---

## 2. Noiseless results (what the platform scores)

| Circuit | N | 2q gates | Depth | SWAP | Energy (Ha) | Error (Ha) | Score |
|---|---|---|---|---|---|---|---|
| HF only | 8 | 0 | 1 | 0 | −112.1200000 | 2.32e-2 | 4,001 |
| Starter (`real_amplitudes`, reps=1) | 6 | 5 | 6 | 0 | — | — | 2,641 |
| Starter with `num_qubits = 8` | 8 | 7 | 8 | 0 | — | — | 2,796 |
| 2-layer brickwall | 8 | 7 | 5 | 0 | −112.1424620 | 7.75e-4 | 4,386 |
| **6-layer brickwall (submitted)** | 8 | **21** | **13** | **0** | **−112.1427952** | **4.41e-4** | **4,392** |
| **2-CX Givens (hardware-optimal)** | 8 | **2** | **4** | **0** | **−112.1424616** | **7.75e-4** | **4,386** |

Scoring formula (from the challenge):

```
Base(N) = 3000 + 50 * (N^2 - 36)          # 4,400 at N=8
A       = max(0, 1 - |E_vqe - E_exact| / 0.5)
Score   = round(Base * A^2 * 1/(1 + 0.25*N_swap) * 2N/max(depth, 2N))
```

At N=8 the depth budget is 16, so both winning circuits get the full depth
factor. The 6-layer brickwall wins noiselessly by 6 points; the Givens
circuit wins by an enormous margin once noise is included.

---

## 3. Hardware results — `ibm_marrakesh`

All runs: `EstimatorV2`, 4,096 shots, `resilience_level=1` (readout
mitigation), qubit line `[149, 148, 147, 137, 127, 126, 125, 124]`,
`optimization_level=3`, zero SWAPs after transpiling.

| Circuit | 2q gates (ISA) | ISA depth | Hardware energy (Ha) | Error (Ha) | Hardware score | Job ID |
|---|---|---|---|---|---|---|
| 6-layer brickwall | 21 CZ | 33 | −111.7626694 | 3.806e-1 | **251** | `dbji6b1ffo6c73b9nbmg` |
| 2-layer brickwall | 7 CZ | 14 | −111.9994128 | 1.438e-1 | **2,233** | `dbjj61pkdqps7390mvd0` |
| **2-CX Givens** | **2 CZ** | **11** | **−112.1285148** | **1.472e-2** | **4,145** | `dbjj71hffo6c73b9p360` |

The noiseless-optimal circuit is the *worst* one on hardware. The Hamiltonian
has large Z coefficients (−0.85, −0.62, +0.78 …), so every bit flip costs a
lot; 21 CZ + 100 SX on a real device destroy 94 % of the score. The 2-CX
Givens circuit keeps **94.2 %** of its noiseless score.

**Layout was verified, not assumed:** replaying the exact transpiled device
gates on 8 qubits with a statevector reproduces
`-112.1427952383 Ha`, identical to the original circuit to `5.8e-12`. The
hardware shift is real device noise, not a mapping bug.

---

## 4. Zero-noise extrapolation (ZNE)

Job `dbjj8k1ffo6c73b9p5a0`, `resilience_level=2`, noise factors (1, 3, 5),
8192 shots, exponential extrapolator.

| Noise factor | Energy (Ha) |
|---|---|
| 1 | −112.12298 |
| 3 | −112.11255 |
| 5 | −112.08032 |

The exponential fit returned `NaN` (the circuit is too short for the fit to
be well-conditioned). A linear extrapolation to zero noise gives
`-112.1337 Ha` → error `9.6e-3 Ha` → **≈ 4,233 points**, within the
`~8e-3 Ha` statistical error of the run. More shots would tighten it, but it
does not change the ranking: the raw 2-CX result (4,145) is already close.

---

## 5. Submission record

```
POST $Submission_URL                (read from .env)
challengeId    : challenge_intermediate
qasm           : n2_8q_best.qasm   (8 qubits, 56 RY + 21 CX, depth 13)
obtainedResult : 0.99911744          (self-reported A)
HTTP           : 200
submission ID  : 5571d09f-3e27-45b7-be28-35df1c7d1137
official score : 4,392
```

The platform validates the QASM qubit count and scores the circuit **shape**
(`num_qubits`, `depth`, `swap`) combined with the reported `A`; it does not
execute the circuit. Hence the official 4,392 is the noiseless number, while
the same QASM scores 251 on a real device.

> Note: the leaderboard was in `paused` state at submission time, so the score
> is recorded on the submission object but was not publicly displayed.

---

## 6. Files

| File | Purpose |
|---|---|
| `vqe_algo.py` | Hamiltonians, brickwall/Givens ansätze, optimizers, scoring, QASM helpers |
| `test_vqe_algo.py` | Fast assertions (no IBM): exact energies, QASM round-trip, zero SWAPs, scores |
| `build_nb.py` | Rebuilds the solution notebook from `vqe_algo.py` |
| `challenge_intermediate_vqe_OPTIMISE.ipynb` | Self-contained solution notebook (20 cells) |
| `n2_8q_best.qasm` | **Submitted** 6-layer brickwall — 4,392 noiseless, 251 hardware |
| `n2_8q_givens.qasm` | 2-CX Givens — 4,386 noiseless, **4,145 hardware** |
| `n2_8q_2layer.qasm` | 7-CX comparison — 4,386 noiseless, 2,233 hardware |
| `hardware_check.py` | Re-runs the IBM measurements (opt-in, needs `../keys.txt`) |
| `challenge_intermediate_vqe.ipynb` | Original starter notebook (untouched) |

---

## 7. Reproduce

```powershell
cd Intermidiate
python test_vqe_algo.py          # ~10 s, no IBM account needed
python build_nb.py               # regenerate the notebook
```

Hardware (spends IBM Quantum time — pass `--run` explicitly):

```powershell
python hardware_check.py --circuit givens --run
python hardware_check.py --circuit 2layer --run
python hardware_check.py --circuit best   --run
python hardware_check.py --circuit givens --run --zne
```

Keys are read from `../keys.txt` (`IBM_Key`, `IBM_Crn`, `Team_Key`,
`Submission_URL`) and are never printed. `keys.txt` is gitignored.

---

## 8. Bottom line

- The provided Hamiltonian's ground state is 98.3 % Hartree–Fock, so the
  starter's `real_amplitudes` ansatz plus `COBYLA(200)` cannot beat HF.
- A 6-layer even/odd brickwall reaches **4,392 / 4,400** noiselessly — the
  best possible shape at N=8 with depth 13 ≤ 16 and zero SWAPs.
- A single Givens rotation (**2 CX**) reaches **4,386** noiselessly and
  **4,145** on `ibm_marrakesh`, 16× better than the submitted circuit under
  noise.
- The official platform score is **4,392**; the best hardware-honest score is
  **4,145**.
