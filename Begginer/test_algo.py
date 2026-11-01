"""Verifie la logique sur une puce heavy-hex synthetique (sans Aer, sans IBM)."""
import time
import random

from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import ghz_algo as g

cm = CouplingMap.from_heavy_hex(9)  # 193 qubits, meme topologie que les puces IBM
rng = random.Random(3)
edges = {}
for a, b in cm.get_edges():
    edges[(min(a, b), max(a, b))] = rng.uniform(0.002, 0.008)
n = cm.size()
nbrs = {q: [] for q in range(n)}
for a, b in edges:
    nbrs[a].append(b)
    nbrs[b].append(a)
info = dict(n=n, edges=edges, nbrs=nbrs, ro={q: rng.uniform(0.005, 0.03) for q in range(n)},
            t1={q: rng.uniform(100e-6, 300e-6) for q in range(n)}, bad=set(), layer_time=150e-9)

t0 = time.time()
best = g.search(info, n_max=156)
print(f"recherche: {time.time() - t0:.1f}s")
for k in (10, 20, 30, 40, 60, 80, 100, 156):
    b = best[k]
    print(f"N={k:3d} profondeur={b['depth']:2d} (min {1 + (k - 1).bit_length()}) C~{b['corr']:.3f} score~{b['score']:,}")
top = max(best.values(), key=lambda b: b["score"])
print("MEILLEUR N:", len(top["ent"]), top)

# 1) chaque CNOT est sur un lien physique  2) chaque couche n'utilise un qubit qu'une fois
for nn in (12, 156):
    b = best[nn]
    for layer in b["layers"]:
        qs = [q for e in layer for q in e]
        assert len(qs) == len(set(qs))
        for u, v in layer:
            assert (min(u, v), max(u, v)) in edges
    qc = g.build_ghz(b["ent"], b["layers"])
    assert qc.depth() - 1 == b["depth"], (qc.depth(), b["depth"])
    assert len(set(b["ent"])) == nn

# 3) l'etat produit est bien un GHZ (verifie exactement sur 12 qubits)
b = best[12]
sv = Statevector(g.build_ghz(b["ent"], b["layers"], measure=False))
p = sv.probabilities_dict()
assert abs(p["0" * 12] - 0.5) < 1e-9 and abs(p["1" * 12] - 0.5) < 1e-9, p
print("OK: CNOT sur liens physiques, couches paralleles valides, etat GHZ exact")
