"""Tests of the predprey package. Run from the repository root: python3 -m pytest"""
import numpy as np

from predprey.geometry import (cluster_labels, delaunay_edges, largest_cluster, norm,
                               ray_to_alpha_shape, turn)
from predprey.params import Params
from predprey.simulation import alignment_strengths, init_school, prey_step, simulate


def test_delaunay_edges_square_with_centre():
    points = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]], dtype=float)
    edges = delaunay_edges(points)
    assert (edges[:, 0] < edges[:, 1]).all()
    assert len(np.unique(edges, axis=0)) == len(edges)
    # the centre is a Voronoi neighbour of the four corners
    assert {tuple(e) for e in edges if 4 in e} == {(0, 4), (1, 4), (2, 4), (3, 4)}


def test_turn_rotates_and_keeps_unit_length():
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    rotated = turn(u, [np.pi / 2, np.pi])
    assert np.allclose(rotated, [[0, 1], [0, -1]])
    assert np.allclose(norm(rotated), 1)


def test_clusters_two_groups():
    rng = np.random.default_rng(0)
    big = rng.random((30, 2))
    small = rng.random((10, 2)) + 100
    points = np.vstack([big, small])
    labels = cluster_labels(points, cludist=6)
    assert len(np.unique(labels)) == 2
    assert set(largest_cluster(points, cludist=6)) == set(range(30))


def test_ray_to_alpha_shape():
    # a square block of prey from x = 10 to x = 14
    xs, ys = np.meshgrid(np.arange(10, 15), np.arange(-2, 3))
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    origin = np.array([0.0, 0.0])
    assert np.isclose(ray_to_alpha_shape(points, origin, np.array([1.0, 0.0]), 3), 10)
    assert ray_to_alpha_shape(points, origin, np.array([-1.0, 0.0]), 3) == np.inf


def test_init_school():
    p = Params(N=100)
    pos, u = init_school(p, np.random.default_rng(0))
    assert pos.shape == (100, 2)
    assert np.allclose(u, u[0])  # everyone heads the same way
    assert np.isclose(norm(u[0]), 1)


def test_alignment_strengths():
    p = Params(N=1000, alg_strength=2, alg_sigma=0.1)
    alg = alignment_strengths(p, np.random.default_rng(0))
    assert (alg >= 0).all()
    assert np.allclose(alg, np.round(alg, 1))
    assert abs(alg.mean() - 2) < 0.02


def test_close_prey_repel():
    p = Params(Dphi=0)  # no noise
    # Two prey closer than rep_range. As in the C++ code, only one end of each
    # Voronoi edge counts it, so only one of them turns away: distance still grows.
    pos = np.array([[0.0, 0.0], [0.0, 0.5]])
    u = np.tile([1.0, 0.0], (2, 1))
    new_pos, new_u = prey_step(pos, u, np.full(2, 2.0), p, np.random.default_rng(0))
    assert norm(new_pos[1] - new_pos[0]) > 0.5
    assert np.allclose(norm(new_u), 1)


def test_prey_flee_from_predator():
    p = Params(Dphi=0)
    pos = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.5]])
    u = np.tile([1.0, 0.0], (3, 1))
    pred = np.array([-2.0, 0.0])  # behind and slightly below the school
    _, new_u = prey_step(pos, u, np.zeros(3), p, np.random.default_rng(0), pred)
    assert new_u[0, 1] > 0  # the closest prey turns away from the predator


def test_simulate_is_reproducible():
    p = Params(N=50, pred_time=2, attack_time=2)
    a = simulate(p, seed=3)
    b = simulate(p, seed=3)
    assert np.allclose(a["pred_pos"], b["pred_pos"])
    assert np.array_equal(a["kills"], b["kills"])
    assert a["prey_pos"].shape == (10, 50, 2)  # 2 time units, one frame every 0.2
