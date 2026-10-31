import math
import random

from qiskit import QuantumCircuit


def hardware_info(backend):
    """Lit la calibration du jour : erreur de chaque lien, erreur de lecture, T1 de chaque qubit."""
    t = backend.target
    n = t.num_qubits
    two_q = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)

    def props(op, qargs):
        return t[op].get(qargs) if op in t.operation_names else None

    sx, ro, t1 = {}, {}, {}
    for q in range(n):
        p = props("sx", (q,))
        sx[q] = p.error if (p and p.error is not None) else 3e-4
        p = props("measure", (q,))
        ro[q] = p.error if (p and p.error is not None) else 2e-2
        qp = t.qubit_properties[q] if t.qubit_properties else None
        t1[q] = qp.t1 if (qp is not None and qp.t1) else 150e-6

    edges, layer_time = {}, None
    for (a, b), p in t[two_q].items():
        if p is None or p.error is None:
            continue
        e = p.error + 2 * sx[b]          # CX = H . CZ . H  -> erreur 2 qubits + 2 portes sx
        if e >= 0.1:                      # lien casse ou mal calibre : on ne l'utilise jamais
            continue
        key = (min(a, b), max(a, b))
        edges[key] = min(edges.get(key, 1.0), e)
        if layer_time is None and p.duration:
            layer_time = p.duration + 2 * 40e-9
    nbrs = {q: [] for q in range(n)}
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    bad = {q for q in range(n) if ro[q] > 0.1}  # qubits a la lecture trop mauvaise : exclus
    return dict(n=n, edges=edges, nbrs=nbrs, ro=ro, t1=t1, bad=bad, layer_time=layer_time or 150e-9)


def edge_err(info, a, b):
    return info["edges"][(min(a, b), max(a, b))]


def grow_tree(info, root, w_future=0.003, rng=None):
    """Fait grandir le GHZ depuis `root`, couche par couche, jusqu'a ne plus pouvoir.

    A chaque couche, TOUS les qubits deja intriques font en parallele une CNOT vers un
    voisin libre (comme l'arbre binaire), mais seulement vers un voisin physique (0 SWAP).
    Les liens les moins bruites passent en premier ; `w_future` favorise les voisins qui
    ont encore des voisins libres, pour garder un front large et donc une faible profondeur.
    """
    nbrs, ro, bad = info["nbrs"], info["ro"], info["bad"]
    ent_set = {root}
    layers = []
    while True:
        cands = []
        for u in ent_set:
            for v in nbrs[u]:
                if v in ent_set or v in bad:
                    continue
                free = sum(1 for w in nbrs[v] if w not in ent_set and w != u and w not in bad)
                cost = -math.log(1 - edge_err(info, u, v)) - math.log(1 - ro[v]) - w_future * free
                if rng is not None:
                    cost += rng.uniform(0, 0.003)
                cands.append((cost, u, v))
        if not cands:
            return layers
        cands.sort()
        used, layer = set(), []
        for cost, u, v in cands:
            if u not in used and v not in used:
                layer.append((u, v))  # deja trie du moins cher au plus cher
                used.update((u, v))
        ent_set.update(v for _, v in layer)
        layers.append(layer)


def truncate(root, layers, n):
    """Garde les n premiers qubits : couches completes puis les liens les moins chers de la derniere."""
    ent, out = [root], []
    for layer in layers:
        room = n - len(ent)
        if room <= 0:
            break
        part = layer[:room]
        out.append(part)
        ent.extend(v for _, v in part)
    return (ent, out) if len(ent) == n else None


def estimate_correlation(info, ent, layers):
    """Estimation de C = P(00..0) + P(11..1).

    Seules les erreurs qui retournent un bit (X ou Y) comptent : C ne regarde que les
    comptages, donc une erreur de phase (Z) est invisible pour le score.
    """
    logc = 0.0
    seen = {ent[0]}
    for layer in layers:
        busy = set()
        for u, v in layer:
            logc += math.log(1 - edge_err(info, u, v) * 12 / 15)  # 12 des 15 Paulis a 2 qubits contiennent X/Y
            busy.update((u, v))
        for q in seen - busy:  # qubit deja intrique qui attend : il peut se relaxer 1 -> 0 (T1)
            logc += math.log(1 - 0.5 * (1 - math.exp(-info["layer_time"] / info["t1"][q])))
        seen.update(v for _, v in layer)
    for q in ent:
        logc += math.log(1 - info["ro"][q])
    return math.exp(logc)


def challenge_score(n, corr, depth, swaps=0, exponent=2.0):
    """Formule du challenge (exposant 2 dans preview_challenge_score, 1.5 dans l'enonce)."""
    rand = 1.0 / 2 ** (n - 1)
    if corr <= rand:
        return 0
    f = ((corr - rand) / (1 - rand)) ** exponent
    dmin = 1 + math.ceil(math.log2(n))
    return round((1000 + 50 * (n * n - 4)) * f / (1 + 0.25 * swaps) * dmin / max(depth, dmin))


def search(info, n_min=8, n_max=None, seed=0):
    """Essaie chaque qubit comme racine et plusieurs reglages ; garde le meilleur plan pour chaque N."""
    rng = random.Random(seed)
    n_max = n_max or info["n"]
    roots = [q for q in range(info["n"]) if q not in info["bad"] and len(info["nbrs"][q]) >= 2]
    best = {}
    for root in roots:
        for w, r in ((0.0, None), (0.003, None), (0.01, None), (0.003, rng)):
            layers = grow_tree(info, root, w, r)
            total = 1 + sum(len(l) for l in layers)
            for n in range(n_min, min(n_max, total) + 1):
                ent, lay = truncate(root, layers, n)
                c = estimate_correlation(info, ent, lay)
                s = challenge_score(n, c, 1 + len(lay))
                if n not in best or s > best[n]["score"]:
                    best[n] = dict(score=s, corr=c, depth=1 + len(lay), ent=ent, layers=lay)
    return best


def build_ghz(ent, layers, measure=True):
    """Circuit a N qubits : le qubit i du circuit est pose sur le qubit physique ent[i]."""
    idx = {p: i for i, p in enumerate(ent)}
    qc = QuantumCircuit(len(ent), len(ent) if measure else 0, name=f"GHZ{len(ent)}")
    qc.h(0)
    for layer in layers:
        for u, v in layer:
            qc.cx(idx[u], idx[v])
    if measure:
        qc.measure(range(len(ent)), range(len(ent)))
    return qc
