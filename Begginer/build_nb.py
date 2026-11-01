import json

algo = open("ghz_algo.py", encoding="utf-8").read()

cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def code(s):
    s = s.strip("\n")
    compile(s, "<cell>", "exec")  # verifie la syntaxe de chaque cellule
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))


md(r"""
# Challenge 1 — GHZ optimisé pour une vraie puce IBM

**Idée :** les deux algorithmes donnés (cascade linéaire et arbre binaire) ignorent la puce.
La cascade est trop profonde, et l'arbre relie des qubits éloignés, ce qui oblige à ajouter des SWAP.
Ici, on construit un **arbre parallèle qui ne suit que les vrais câbles de la puce** :

| | Cascade | Arbre binaire | **Notre version** |
|---|---|---|---|
| Profondeur | ≈ N | log₂N (idéal) | **≈ log₂N + 1 ou 2** |
| SWAP sur heavy-hex | 0 | beaucoup | **0** |
| Choix des qubits | au hasard | au hasard | **les moins bruités du jour** |
| Choix de N | fixé | fixé | **N qui maximise le score** |

Étapes :
1. lire la calibration du jour de la puce IBM (erreur de chaque lien et de chaque lecture) ;
2. faire grandir le GHZ couche par couche, depuis chaque qubit de départ possible, en gardant le meilleur ;
3. prédire le score pour chaque N, puis lancer les 3 meilleurs N sur la vraie machine ;
4. corriger les erreurs de lecture, calculer le score et soumettre le meilleur.
""")

code(r"""
# %pip install -q qiskit qiskit-ibm-runtime matplotlib requests
import math
import os
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# Load KEY=VALUE secrets from the gitignored root .env (or a keys.txt fallback).
def load_keys(path=None):
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = []
        env_path = os.environ.get("KEYS_FILE")
        if env_path:
            candidates.append(Path(env_path))
        here = Path.cwd().resolve()
        roots = (here, *here.parents)
        candidates += [root / ".env" for root in roots]
        candidates += [path for root in roots for path in
                       (root / "keys.txt", root / "Begginer" / "keys.txt",
                        root / "Intermidiate" / "keys.txt")]
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
    for key in set(keys) | {"IBM_Key", "IBM_Crn", "Team_Key", "Submission_URL"}:
        if os.environ.get(key):
            keys[key] = os.environ[key]
    return keys
""")

md(r"""
## 1. Connexion à IBM Quantum
Les clés sont lues depuis le fichier `.env` à la racine du dépôt (ignoré par git).
Renseigne `IBM_Key` et `IBM_Crn` ; si tu as déjà enregistré un compte avec
`save_account`, laisse `TOKEN = None`.
""")

code(r"""
keys = load_keys()
TOKEN = keys.get("IBM_Key")       # cle API IBM Quantum (ne partage pas ce fichier)
INSTANCE = keys.get("IBM_Crn")    # CRN de ton instance si tu en as plusieurs
BACKEND_NAME = None               # ex: "ibm_fez" ; None = la machine la moins chargee

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN, instance=INSTANCE) if TOKEN \
    else QiskitRuntimeService()
backend = service.backend(BACKEND_NAME) if BACKEND_NAME else \
    service.least_busy(operational=True, simulator=False, min_num_qubits=100)
print("Machine :", backend.name, "-", backend.num_qubits, "qubits")
""")

md(r"""
## 2. L'algorithme
- **`hardware_info`** lit la calibration : l'erreur de chaque lien CNOT, l'erreur de lecture et le T1 de chaque qubit. Les liens et qubits cassés sont exclus.
- **`grow_tree`** : à chaque couche, *tous* les qubits déjà intriqués font en parallèle une CNOT vers un voisin libre. C'est l'arbre binaire, mais seulement sur les câbles de la puce, donc **0 SWAP**. Les liens les moins bruités passent en premier.
- **`estimate_correlation`** prédit C. Astuce : le score ne regarde que les comptages `00…0`/`11…1`, donc **seules les erreurs qui retournent un bit (X/Y) comptent**. Les erreurs de phase (Z) sont invisibles.
- **`search`** essaie chaque qubit comme point de départ, avec plusieurs réglages, et garde le meilleur plan pour chaque N.
""")

code(algo)

md(r"""
## 3. Recherche du meilleur N sur la calibration du jour
Le score de base monte en N², mais la corrélation C baisse à chaque qubit ajouté : il existe un **N optimal**. La recherche prend environ 1 minute.
""")

code(r"""
info = hardware_info(backend)
print(f"{len(info['edges'])} liens utilisables, {len(info['bad'])} qubits exclus")
best = search(info)

ns = sorted(best)
plt.figure(figsize=(8, 3.5))
plt.plot(ns, [best[n]["score"] for n in ns])
plt.xlabel("N (qubits)"); plt.ylabel("score prédit"); plt.title(f"Score prédit sur {backend.name}")
plt.grid(alpha=.3); plt.show()

candidates = sorted(ns, key=lambda n: -best[n]["score"])[:1]
# on teste aussi autour de l'optimum : la prediction n'est qu'une estimation
for n in (candidates[0] - 6, candidates[0] + 8):
    if n in best:
        candidates.append(n)
for n in candidates:
    b = best[n]
    print(f"N={n:3d}  profondeur={b['depth']} (min {1 + math.ceil(math.log2(n))})  C prédit={b['corr']:.3f}  score prédit={b['score']:,}")
""")

md(r"""
## 4. Comparaison avec les algorithmes donnés
Pour le même N, on transpile la cascade et l'arbre du notebook sur la vraie carte de la puce.
Le calcul suppose **zéro bruit (C = 1)** : il ne garde que les pénalités SWAP et profondeur.
""")

code(r"""
def build_ghz_linear(n):
    qc = QuantumCircuit(n, n); qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n)); return qc

def build_ghz_tree(n):
    qc = QuantumCircuit(n, n); qc.h(0); step = 1
    while step < n:
        for i in range(0, n - step, 2 * step):
            qc.cx(i, i + step)
        step *= 2
    qc.measure(range(n), range(n)); return qc

N0 = candidates[0]
ours = build_ghz(best[N0]["ent"], best[N0]["layers"])
for name, qc in [("cascade", build_ghz_linear(N0)), ("arbre binaire", build_ghz_tree(N0)), ("NOTRE GHZ", ours)]:
    if name == "NOTRE GHZ":
        t = transpile(qc, coupling_map=backend.coupling_map, initial_layout=best[N0]["ent"],
                      optimization_level=3, seed_transpiler=42)
    else:
        t = transpile(qc, coupling_map=backend.coupling_map, optimization_level=3, seed_transpiler=42)
    sw, d = t.count_ops().get("swap", 0), t.depth() - 1
    print(f"{name:14s} SWAP={sw:3d}  profondeur={d:3d}  score max (si C=1) = {challenge_score(N0, 1.0, d, sw):,}")
""")

md(r"""
## 5. Exécution sur la vraie machine
- Chaque circuit est posé **exactement** sur les qubits choisis (`initial_layout`), donc le transpileur n'ajoute aucun SWAP.
- **Twirling de mesure** : il rend l'erreur de lecture symétrique (0→1 aussi probable que 1→0), ce qui est plus simple à corriger.
- **Découplage dynamique (XY4)** : il protège les qubits qui attendent.
- Pour chaque candidat, on ajoute 2 petits circuits de **calibration de lecture**, tous à 0 et tous à 1, sur les mêmes qubits.
""")

code(r"""
SHOTS = 4000
pubs, tags = [], []
for n in candidates:
    ent, layers = best[n]["ent"], best[n]["layers"]
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=ent, seed_transpiler=42)
    cal0 = QuantumCircuit(n, n); cal0.measure(range(n), range(n))
    cal1 = QuantumCircuit(n, n); cal1.x(range(n)); cal1.measure(range(n), range(n))
    for kind, qc in (("ghz", build_ghz(ent, layers)), ("cal0", cal0), ("cal1", cal1)):
        isa = pm.run(qc)
        if kind == "ghz":
            n2q = sum(1 for ins in isa.data if ins.operation.num_qubits == 2)
            assert n2q == n - 1, f"le transpileur a ajoute des portes ({n2q} portes 2q au lieu de {n - 1})"
        pubs.append(isa); tags.append((n, kind))

sampler = SamplerV2(mode=backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XY4"
sampler.options.twirling.enable_measure = True
job = sampler.run(pubs)
print("Job envoyé :", job.job_id(), "- en attente de la machine...")
res = job.result()
counts = {tag: r.data.c.get_counts() for tag, r in zip(tags, res)}
""")

md(r"""
## 6. Corrélation brute et corrigée
**Correction de lecture :** on mesure, qubit par qubit, la probabilité de mal lire un 0 (circuit `cal0`) ou un 1 (circuit `cal1`).
La probabilité que les N lectures soient justes est le produit de ces probabilités. On divise donc la part observée de `00…0` par ce produit, et pareil pour `11…1`.
C'est le terme principal de la méthode M3 recommandée par le challenge.
""")

code(r"""
def flip_rates(cnt, n, prepared):
    # proba, pour chaque qubit, de lire l'inverse de ce qui a ete prepare
    tot = sum(cnt.values()); wrong = [0] * n
    for bits, c in cnt.items():
        for i, b in enumerate(reversed(bits)):   # qiskit : bit 0 a droite
            if b != prepared:
                wrong[i] += c
    return [w / tot for w in wrong]

USE_MITIGATION = True
results = {}
for n in candidates:
    g = counts[(n, "ghz")]; tot = sum(g.values())
    c0, c1 = g.get("0" * n, 0) / tot, g.get("1" * n, 0) / tot
    p10, p01 = flip_rates(counts[(n, "cal0")], n, "0"), flip_rates(counts[(n, "cal1")], n, "1")
    keep0, keep1 = math.prod(1 - p for p in p10), math.prod(1 - p for p in p01)
    raw, mit = c0 + c1, min(1.0, c0 / keep0 + c1 / keep1)
    corr = mit if USE_MITIGATION else raw
    d = best[n]["depth"]
    results[n] = dict(raw=raw, mit=mit, corr=corr, score=challenge_score(n, corr, d))
    print(f"N={n:3d}  C brute={raw:.3f}  C corrigée={mit:.3f}  -> score={results[n]['score']:,}")

N_BEST = max(results, key=lambda n: results[n]["score"])
print("\nMeilleur :", N_BEST, "qubits")
""")

md("## 7. Prévisualisation officielle du score (fonction du notebook d'origine)")

code(r"""
def preview_challenge_score(circuit: QuantumCircuit, correlation: float):
    num_qubits = circuit.num_qubits
    ops = dict(circuit.count_ops())
    measured_ops = ops.get('measure', 0)
    gate_depth = circuit.depth() - (1 if measured_ops > 0 else 0)
    swap_count = ops.get('swap', 0)
    random_threshold = 1.0 / (2.0 ** (num_qubits - 1))
    base_score = 1000.0 + 50.0 * (num_qubits ** 2 - 4)
    if correlation <= random_threshold:
        print("Correlation sous le bruit aleatoire : score = 0"); return 0
    norm_fidelity = (correlation - random_threshold) / (1.0 - random_threshold)
    fidelity_factor = norm_fidelity ** 2
    swap_penalty = 1.0 / (1.0 + 0.25 * swap_count)
    min_theoretical_depth = 1 + math.ceil(math.log2(num_qubits))
    depth_efficiency = min_theoretical_depth / max(gate_depth, min_theoretical_depth)
    total_score = round(base_score * fidelity_factor * swap_penalty * depth_efficiency)
    print(f"N={num_qubits} | base={base_score:,.0f} | C={correlation:.2%} | fidelite={fidelity_factor:.3f} | "
          f"SWAP={swap_count} | profondeur={gate_depth} (min {min_theoretical_depth}, eff {depth_efficiency:.3f})")
    print(f"SCORE PROJETE : {total_score:,} points")
    return total_score

submission_circuit = build_ghz(best[N_BEST]["ent"], best[N_BEST]["layers"])
submission_correlation = results[N_BEST]["corr"]
preview_challenge_score(submission_circuit, submission_correlation)
display(submission_circuit.draw("mpl", fold=-1))
""")

md("## 8. Soumission")

code(r"""
keys = load_keys()
API_KEY = keys.get("Team_Key")
SUBMISSION_URL = keys.get("Submission_URL")
CHALLENGE_ID = "challenge_beginner"

if not API_KEY or not SUBMISSION_URL:
    raise RuntimeError("Team_Key / Submission_URL missing from the keys file")

payload = {"qasm": qasm2.dumps(submission_circuit), "challengeId": CHALLENGE_ID,
           "obtainedResult": float(submission_correlation)}
print(f"Soumission d'un GHZ a {submission_circuit.num_qubits} qubits (job IBM {job.job_id()})...")
try:
    r = requests.post(SUBMISSION_URL, json=payload, headers={"X-API-Key": API_KEY}, timeout=20)
    if r.ok:
        s = r.json()["submission"]; print(f"[OK] id={s['id']}  score officiel={s['score']:,}")
    else:
        print(f"[X] {r.status_code}:", r.json().get("error", r.text))
except Exception as e:
    print("[!] Erreur reseau :", e)
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
json.dump(nb, open("challenge_scalable_ghz_OPTIMISE.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("notebook ecrit,", len(cells), "cellules")
