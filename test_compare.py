"""Compare cascade / arbre / notre GHZ sur la meme puce heavy-hex synthetique (logique de la cellule 4)."""
import random
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
import ghz_algo as g

cm = CouplingMap.from_heavy_hex(9)
rng = random.Random(3)
edges = {(min(a, b), max(a, b)): rng.uniform(0.002, 0.008) for a, b in cm.get_edges()}
n = cm.size()
nbrs = {q: [] for q in range(n)}
for a, b in edges:
    nbrs[a].append(b); nbrs[b].append(a)
info = dict(n=n, edges=edges, nbrs=nbrs, ro={q: rng.uniform(0.005, 0.03) for q in range(n)},
            t1={q: rng.uniform(100e-6, 300e-6) for q in range(n)}, bad=set(), layer_time=150e-9)
best = g.search(info, n_min=38, n_max=40)
N0 = 39
b = best[N0]

def lin(k):
    qc = QuantumCircuit(k, k); qc.h(0)
    for i in range(k - 1): qc.cx(i, i + 1)
    qc.measure(range(k), range(k)); return qc

def tree(k):
    qc = QuantumCircuit(k, k); qc.h(0); s = 1
    while s < k:
        for i in range(0, k - s, 2 * s): qc.cx(i, i + s)
        s *= 2
    qc.measure(range(k), range(k)); return qc

for name, qc, lay in [("cascade", lin(N0), None), ("arbre", tree(N0), None), ("notre", g.build_ghz(b["ent"], b["layers"]), b["ent"])]:
    t = transpile(qc, coupling_map=cm, initial_layout=lay, optimization_level=3, seed_transpiler=42)
    sw, d = t.count_ops().get("swap", 0), t.depth() - 1
    print(f"{name:8s} SWAP={sw:3d} profondeur={d:3d} score max si C=1: {g.challenge_score(N0, 1.0, d, sw):,}  | avec C~{b['corr']:.2f}: {g.challenge_score(N0, b['corr'], d, sw):,}")
