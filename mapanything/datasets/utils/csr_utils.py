import os
import numpy as np


def _load_covis_graph(scene_root: str, scene_meta: dict):
    sm = scene_meta.get("scene_modalities", {})

    def _load_csr_mmap_npy(dir_path: str):
        indptr = np.load(os.path.join(dir_path, "indptr.npy"), mmap_mode="r")
        indices = np.load(os.path.join(dir_path, "indices.npy"), mmap_mode="r")
        data = np.load(os.path.join(dir_path, "data.npy"), mmap_mode="r")
        shape = np.load(os.path.join(dir_path, "shape.npy"))
        shape = tuple(shape.tolist())
        return {
            "format": "csr",
            "indptr": indptr,      # memmap
            "indices": indices,    # memmap
            "data": data,          # memmap
            "shape": shape,
        }

    key_view = sm.get("covis_graph_view_csr")
    if key_view is None:
        raise KeyError("scene_meta.scene_modalities missing 'covis_graph_view_csr'")

    rel = key_view.get("scene_key", None)
    fmt = key_view.get("format", "")

    if rel is None:
        raise KeyError("covis_graph_view_csr missing 'scene_key'")

    abs_path = os.path.join(scene_root, rel)

    if fmt in ("csr_mmap_npy", "csr_mmap") or os.path.isdir(abs_path):
        return _load_csr_mmap_npy(abs_path)
    raise FileNotFoundError(f"Cannot load covis graph: format={fmt}, path={abs_path}")

# 修改下面的函数缩进为4个空格
def _is_csr_graph(x):
    """Check if input is a CSR graph dictionary."""
    return isinstance(x, dict) and x.get("format", None) in ["csr", "csr_npz"]
    
def _csr_row(g, i: int):
    """Return neighbors and weights for row i from CSR graph."""
    indptr = g["indptr"]
    indices = g["indices"]
    data = g["data"]
    s = int(indptr[i]); e = int(indptr[i + 1])
    return indices[s:e], data[s:e]
    
def _keep_w_in_range(w: np.ndarray, w_min: float = None, w_max: float = None) -> np.ndarray:
    """Return boolean mask where w in [w_min, w_max]."""
    keep = np.ones_like(w, dtype=bool)
    if w_min is not None:
        keep &= (w >= float(w_min))
    if w_max is not None:
        keep &= (w <= float(w_max))
    return keep
    
def _csr_edge(g, i: int, j: int) -> float:
    """Return edge weight from i to j, 0 if edge doesn't exist."""
    nbrs, w = _csr_row(g, i)
    if nbrs.size == 0:
        return 0.0
    for n, ww in zip(nbrs, w):
        if int(n) == int(j):
                return float(ww)
    return 0.0
    
def _edge_w(g, u: int, v: int, bidirectional: bool = True) -> float:
    """Return edge weight between u and v (optionally bidirectional)."""
    w = float(_csr_edge(g, u, v))
    if bidirectional:
        w = max(w, float(_csr_edge(g, v, u)))
    return w
    
def _weighted_choice(nbrs, w, temperature=1.0):
    """Sample neighbor proportional to softmax(w/temperature)."""
    if len(nbrs) == 1:
        return int(nbrs[0])
    w = np.asarray(w, dtype=np.float32)
    t = max(float(temperature), 1e-8)
    z = (w / t) - (w / t).max()
    p = np.exp(z)
    p = p / (p.sum() + 1e-12)
    return int(np.random.choice(nbrs, p=p))


def _random_walk_sampling_csr(
    g,
    num_of_samples: int,
    max_retries: int,
    min_covis: float,
    max_covis: float,
    restart_prob: float,
    temperature: float,
    topk_step: int,
    rng: np.random.Generator,
):
    """
    Random walk sampling on CSR graph.
        
    Args:
        g: CSR graph dictionary.
        num_of_samples: Number of samples to collect.
        max_retries: Maximum number of restart attempts.
        min_covis: Minimum edge weight threshold.
        max_covis: Maximum edge weight threshold.
        restart_prob: Probability of restarting walk from start node.
        temperature: Temperature for softmax sampling.
        topk_step: Number of top neighbors to consider at each step.
            
    Returns:
        Array of sampled node indices.
    """
    N = int(g["shape"][0])
    excluded_nodes = set()
    best_walk = []

    for _ in range(max_retries):
        visited = set()
        walk = []
        stack = []

        all_nodes = set(range(N))
        available_nodes = list(all_nodes - excluded_nodes)
        if not available_nodes:
            break

        start = int(rng.choice(available_nodes))
        walk.append(start)
        visited.add(start)
        stack.append(start)

        while len(walk) < num_of_samples and stack:
            cur = stack[-1]

            if rng.random() < restart_prob:
                cur = start

            nbrs, w = _csr_row(g, cur)
            if nbrs.size == 0:
                stack.pop()
                continue

            keep = _keep_w_in_range(w, min_covis, max_covis)
            nbrs = nbrs[keep]
            w = w[keep]
            if nbrs.size == 0:
                stack.pop()
                continue

            if topk_step is not None and nbrs.size > int(topk_step):
                k = int(topk_step)
                part = np.argpartition(w, -k)[-k:]
                nbrs = nbrs[part]
                w = w[part]

            mask = np.array([int(n) not in visited for n in nbrs], dtype=bool)
            if mask.any():
                nbrs2 = nbrs[mask]
                w2 = w[mask]
                nxt = _weighted_choice(nbrs2, w2, temperature=temperature)
                walk.append(nxt)
                visited.add(nxt)
                stack.append(nxt)
            else:
                stack.pop()

        if len(walk) > len(best_walk):
            best_walk = walk
        if len(walk) >= num_of_samples:
            return np.array(walk, dtype=np.int64)

        excluded_nodes.update(visited)
    return np.array(best_walk, dtype=np.int64)

def _greedy_chain_sampling_csr_once(
    g,
    num_of_samples: int,
    min_covis: float,
    max_covis: float,
    topk_step: int,
    start: int,
    bidirectional_edge: bool,
    enforce_global_max: bool,
    rng: np.random.Generator,
):
    """
    Greedy chain sampling: at each step choose highest-weight neighbor.
        
    Args:
            g: CSR graph dictionary.
            num_of_samples: Number of samples to collect.
            min_covis: Minimum edge weight threshold.
            max_covis: Maximum edge weight threshold.
            topk_step: Number of top neighbors to consider at each step.
            start: Starting node index.
            bidirectional_edge: Whether to consider bidirectional edge weights.
            enforce_global_max: Whether to enforce global max constraint on all historical nodes.
            
    Returns:
        Array of sampled node indices.
    """
    N = int(g["shape"][0])
    if N <= 0:
        return np.array([], dtype=np.int64)

    if start is None:
        start = int(rng.integers(0, N))

    eps = 1e-12

    walk = [start]
    visited = set([start])
    cur = start

    while len(walk) < num_of_samples:
        nbrs, w_dir = _csr_row(g, cur)
        if nbrs.size == 0:
            break

        if bidirectional_edge:
            w_eff = np.empty_like(w_dir, dtype=np.float32)
            for k, (n, wd) in enumerate(zip(nbrs, w_dir)):
                n = int(n)
                w_eff[k] = max(float(wd), float(_csr_edge(g, n, cur)))
        else:
            w_eff = w_dir.astype(np.float32, copy=False)

        keep = np.ones_like(w_eff, dtype=bool)
        if min_covis is not None:
            keep &= (w_eff >= float(min_covis))
        if max_covis is not None:
            keep &= (w_eff < float(max_covis) - eps)

        nbrs = nbrs[keep]
        w_eff = w_eff[keep]
        if nbrs.size == 0:
            break

        if topk_step is not None and nbrs.size > int(topk_step):
            k = int(topk_step)
            part = np.argpartition(w_eff, -k)[-k:]
            nbrs = nbrs[part]
            w_eff = w_eff[part]

        order = np.argsort(-w_eff)
        chosen = None
        for j in order:
            cand = int(nbrs[j])
            if cand in visited:
                continue

            if enforce_global_max and (max_covis is not None) and (len(walk) >= 2):
                ok = True
                for u in walk:
                    if u == cand:
                        ok = False
                        break
                    if _edge_w(g, int(u), cand, bidirectional=bidirectional_edge) >= float(max_covis) - eps:
                        ok = False
                        break
                if not ok:
                    continue

            chosen = cand
            break

        if chosen is None:
            break

        walk.append(chosen)
        visited.add(chosen)
        cur = chosen
    return np.array(walk, dtype=np.int64)

def _greedy_chain_sampling_csr(
    g,
    num_of_samples: int,
    min_covis: float,
    max_covis: float,
    topk_step: int,
    max_retries: int,
    bidirectional_edge: bool,
    rng: np.random.Generator,
):
    """
    Greedy chain sampling with multiple retries.
        
    Args:
            g: CSR graph dictionary.
            num_of_samples: Number of samples to collect.
            min_covis: Minimum edge weight threshold.
            max_covis: Maximum edge weight threshold.
            topk_step: Number of top neighbors to consider at each step.
            max_retries: Maximum number of restart attempts.
            bidirectional_edge: Whether to consider bidirectional edge weights.
            
    Returns:
        Array of sampled node indices.
    """
    N = int(g["shape"][0])
    best_walk = np.array([], dtype=np.int64)

    candidates = list(range(N))
    if not candidates:
        return best_walk

    for _ in range(max_retries):
        start = int(rng.choice(candidates))
        walk = _greedy_chain_sampling_csr_once(
            g,
            num_of_samples,
            min_covis=min_covis,
            max_covis=max_covis,
            topk_step=topk_step,
            start=start,
            bidirectional_edge=bidirectional_edge,
            enforce_global_max=True,
            rng=rng,
        )
        if len(walk) > len(best_walk):
            best_walk = walk
        if len(walk) >= num_of_samples:
            return walk
    return best_walk

def _csr_sampling(
    view_graph,
    num_of_samples,
    rng: np.random.Generator,
    max_retries=4,
    sampling_mode="random_walk",
    use_bidirectional_covis=True,
    covisibility_thres=0.05,
    covisibility_thres_max=1.0,
    topk_step=50,
    walk_restart_prob=0.10,
    walk_temperature=1.0,
):
    """
    Walk-based sampling on view graph.
        
    Args:
            view_graph: View-level CSR graph.
            num_of_samples: Number of views to sample.
            max_retries: Maximum number of restart attempts.
            sampling_mode: Sampling mode, "random_walk" or "greedy_chain".
            use_bidirectional_covis: Whether to use bidirectional edge weights.
            covisibility_thres: Minimum edge weight threshold.
            covisibility_thres_max: Maximum edge weight threshold.
            topk_step: Number of top neighbors to consider at each step.
            walk_restart_prob: Restart probability for random walk.
            walk_temperature: Temperature parameter for random walk.
            
    Returns:
        Array of sampled view indices.
    """
    if sampling_mode == "greedy_chain":
        return _greedy_chain_sampling_csr(
            view_graph,
            num_of_samples,
            min_covis=covisibility_thres,
            max_covis=covisibility_thres_max,
            topk_step=topk_step,
            max_retries=max_retries,
            bidirectional_edge=use_bidirectional_covis,
            rng=rng,
        )
    else:  # "random_walk"
        return _random_walk_sampling_csr(
            view_graph,
            num_of_samples,
            max_retries=max_retries,
            min_covis=covisibility_thres,
            max_covis=covisibility_thres_max,
            restart_prob=walk_restart_prob,
            temperature=walk_temperature,
            topk_step=topk_step,
            rng=rng,
        )

