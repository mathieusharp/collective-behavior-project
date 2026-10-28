"""Prey school attacked by one predator, after Klamser & Romanczuk (2021).

Python reimplementation of the 2D model of https://github.com/PaPeK/PredatorPrey
(src/agents_dynamics.cpp, src/agents_interact.cpp, src/swarmdyn.cpp), with the
default options of that code: open space, Voronoi interactions, initial school
ordered on concentric circles, predator following the prey it can catch.

Each agent is a position and a unit heading vector; speeds are constant.
"""
import numpy as np

from .geometry import (cluster_labels, delaunay_edges, largest_cluster, norm, perp,
                       ray_to_alpha_shape, turn, unit)
from .params import Params


def init_school(p, rng):
    """Prey on concentric circles with a small random shift, all heading the same way (C++: IC=3)."""
    theta = 2 * np.pi * rng.random()
    order = rng.permutation(p.N)
    pos = np.zeros((p.N, 2))
    circle, n_circle, n_cap, k = 1, 3, 3, 0
    for i in range(p.N):
        if i == n_cap:
            circle += 1
            n_circle = int(2 * np.pi * circle)
            n_cap += n_circle
            k = 0
        angle = 2 * np.pi * k / n_circle
        pos[order[i]] = p.rep_range * circle * np.array([np.cos(angle), np.sin(angle)]) \
            + 0.8 * p.rep_range * rng.random(2)
        k += 1
    u = np.tile([np.cos(theta), np.sin(theta)], (p.N, 1))
    return pos, u


def alignment_strengths(p, rng):
    """Alignment strength of each prey, around p.alg_strength (C++: heteroPopulation)."""
    mean, sig = p.alg_strength, p.alg_sigma
    if sig == 0:
        return np.full(p.N, mean)
    if mean == 0:
        mean = 0.05 * sig
    alg = rng.normal(mean, sig, p.N)
    if (alg < 0).any():
        alg[alg < 0] = 0
        if (alg.mean() - mean) / sig > 0.1:  # truncation shifted the mean too much
            alg = rng.wald(mean, mean**3 / sig**2, p.N)
    rounded, factor = np.round(alg, 1), 2
    while rounded.sum() == 0:  # keep the population heterogeneous
        rounded = np.round(alg * factor, 1)
        factor += 1
    return rounded


def move(pos, u, force, speed, dt, noise=0.0):
    """Turns the heading towards the force (only its perpendicular part counts), then moves."""
    dphi = (np.sum(force * perp(u), axis=-1) * dt + noise) / speed
    u = turn(u, dphi)
    return pos + speed * u * dt, u


def prey_step(pos, u, alg, p, rng, pred_pos=None):
    """One Euler step for all prey. With a predator, its Voronoi neighbours flee."""
    n = len(pos)
    vel = p.speed0 * u
    points = pos if pred_pos is None else np.vstack([pos, pred_pos])
    edges = delaunay_edges(points)
    i, j = edges[edges[:, 1] < n].T

    r = pos[j] - pos[i]
    d = norm(r)
    # distance regulation: repulsion below rep_range, attraction above (C++: SFM_DRA)
    f_dist = -np.tanh(p.rep_steepness * (d - p.rep_range))[:, None] * r / d[:, None]
    f_alg = vel[j] - vel[i]
    sum_dist = np.zeros((n, 2))
    sum_alg = np.zeros((n, 2))
    np.add.at(sum_dist, i, f_dist)
    np.add.at(sum_dist, j, -f_dist)
    np.add.at(sum_alg, i, f_alg)
    np.add.at(sum_alg, j, -f_alg)

    # The C++ code divides by a neighbour counter that is incremented for only one
    # end of each Voronoi edge, chosen by CGAL's edge order. We pick that end at random.
    owner = np.where(rng.random(len(i)) < 0.5, i, j)
    count = np.bincount(owner, minlength=n)[:, None]
    force = np.where(count > 0,
                     (p.rep_strength * sum_dist + alg[:, None] * sum_alg) / np.maximum(count, 1),
                     0.0)

    if pred_pos is not None:
        fleeing = edges[edges[:, 1] == n, 0]
        force[fleeing] += p.flee_strength * unit(pos[fleeing] - pred_pos)

    noise = np.sqrt(2 * p.Dphi * p.dt) * rng.standard_normal(n)
    return move(pos, u, force, p.speed0, p.dt, noise)


def create_predator(pos, cluster, heading, p, rng):
    """Places the predator pred_radius away from the school, at a random angle (C++: CreatePredator)."""
    direction = np.arctan2(heading[1], heading[0])
    side = 1 if rng.uniform(-p.pred_angle_noise, p.pred_angle_noise) >= 0 else -1
    angle = direction + side * (p.pred_angle + rng.uniform(-p.pred_angle_noise, p.pred_angle_noise))
    u = -np.array([np.cos(angle), np.sin(angle)])  # pointing towards the school
    school = pos[cluster]
    com = school.mean(axis=0)
    x = com - u * (np.abs((school - com) @ u).max() + 1)  # surely behind the school
    dist = ray_to_alpha_shape(school, x, u, p.cludist / 2 + 0.001)
    if np.isfinite(dist):
        x = x + (dist - p.pred_radius) * u
    return x, u


def polarization(u):
    """Norm of the mean heading: 1 for a fully aligned school, about 0 for a disordered one."""
    return norm(u.mean(axis=0))


def simulate(p=None, seed=None):
    """Runs one simulation: equilibration, then attack. Returns a dict of results."""
    p = p or Params()
    rng = np.random.default_rng(seed)
    n_memory = round((p.pred_time - p.pred_memory) / p.dt)
    n_pred = round(p.pred_time / p.dt)
    n_attack = round(p.attack_time / p.dt)
    every = round(p.output / p.dt)

    # EQUILIBRATION, restarted if the school has split (C++: MinCluster)
    alg = alignment_strengths(p, rng)
    best = None
    for _ in range(p.max_restarts):
        pos, u = init_school(p, rng)
        heading = np.zeros(2)
        for s in range(n_pred):
            if s == n_memory:
                cluster = largest_cluster(pos, p.cludist)
            if s >= n_memory:  # the predator observes the school heading before attacking
                heading += u[cluster].mean(axis=0)
            pos, u = prey_step(pos, u, alg, p, rng)
        if best is None or len(cluster) > len(best[2]):
            best = (pos, u, cluster, heading)
        if len(cluster) >= p.min_cluster * p.N:
            break
    pos, u, cluster, heading = best
    polarization_start = polarization(u[cluster])

    # ATTACK, only on the largest cluster (C++: split_notInCluster)
    active = np.zeros(p.N, dtype=bool)
    active[cluster] = True
    pred_x, pred_u = create_predator(pos, cluster, heading, p, rng)
    hunting = False
    pred_cluster = cluster
    kills = []
    frames = {"t": [], "prey_pos": [], "prey_u": [], "pred_pos": [], "pred_u": []}

    for s in range(n_attack):
        idx = np.flatnonzero(active)
        if s % every == 0:
            frames["t"].append(s * p.dt)
            frames["prey_pos"].append(np.where(active[:, None], pos, np.nan))
            frames["prey_u"].append(np.where(active[:, None], u, np.nan))
            frames["pred_pos"].append(pred_x)
            frames["pred_u"].append(pred_u)

        P, U = prey_step(pos[idx], u[idx], alg[idx], p, rng, pred_x)
        pos[idx], u[idx] = P, U

        # the predator only sees its Voronoi neighbours in front of it
        edges = delaunay_edges(np.vstack([P, pred_x]))
        nn = edges[edges[:, 1] == len(idx), 0]
        nn = nn[(P[nn] - pred_x) @ pred_u > 0]
        d = norm(P[nn] - pred_x)

        # catch probability (C++: Fitness)
        in_reach = d < p.kill_range
        p_catch = np.where(in_reach, (p.kill_range - d) / p.kill_range, 0.0) \
            / max(in_reach.sum(), 1) * p.kill_rate * p.dt
        caught = rng.random(len(nn)) < p_catch

        # predator: approach a cluster, then hunt the prey it can catch (C++: MovePredator)
        close = d <= p.pred_circle_radius
        if not hunting and close.any():
            hunting = True
            labels = cluster_labels(P, p.cludist)
            nearest = nn[close][np.argmin(d[close])]
            pred_cluster = idx[labels == labels[nearest]]
        elif hunting and not close.any():
            hunting = False
            pred_cluster = idx[largest_cluster(P, p.cludist)]
        if hunting:
            weights = p_catch[close]
            if weights.sum() == 0:
                weights = 1 / d[close]
            target = weights @ P[nn[close]] / weights.sum()
        else:
            members = pred_cluster[active[pred_cluster]]
            target = pos[members if len(members) else idx].mean(axis=0)
        force = p.pred_strength * unit(target - pred_x)
        pred_x, pred_u = move(pred_x, pred_u, force, p.pred_speed, p.dt)

        for k in idx[nn[caught]]:
            kills.append((s * p.dt, k))
            active[k] = False

    result = {key: np.array(val) for key, val in frames.items()}
    result.update(kills=np.array(kills).reshape(-1, 2), alg=alg, cluster_size=len(cluster),
                  polarization_start=polarization_start, seed=seed)
    return result
