"""Neighbours, clusters and other geometric helpers.

Positions are arrays of shape (n, 2); directions are unit vectors, never angles,
so that the code can later be extended to 3D.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError, cKDTree


def delaunay_edges(points):
    """Pairs (i, j), i < j, of Voronoi neighbours (= edges of the Delaunay triangulation)."""
    n = len(points)
    if n < 3:
        return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int).reshape(-1, 2)
    try:
        tri = Delaunay(points)
    except QhullError:  # degenerate configuration
        try:
            tri = Delaunay(points, qhull_options="QJ")
        except QhullError:  # all points aligned: neighbours are consecutive along the line
            direction = points[-1] - points[0]
            order = np.argsort(points @ direction)
            return np.sort(np.column_stack([order[:-1], order[1:]]), axis=1)
    s = tri.simplices
    edges = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def norm(v):
    return np.linalg.norm(v, axis=-1)


def unit(v):
    return v / norm(v)[..., None]


def perp(u):
    """u rotated by +90 degrees."""
    return np.stack([-u[..., 1], u[..., 0]], axis=-1)


def turn(u, dphi):
    """Rotates the unit vectors u by the angles dphi."""
    dphi = np.asarray(dphi)[..., None]
    return unit(u * np.cos(dphi) + perp(u) * np.sin(dphi))


def cluster_labels(points, cludist):
    """Label of the cluster of each point (chains of points closer than cludist)."""
    n = len(points)
    pairs = cKDTree(points).query_pairs(cludist, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def largest_cluster(points, cludist):
    """Indices of the points in the largest cluster."""
    labels = cluster_labels(points, cludist)
    return np.flatnonzero(labels == np.bincount(labels).argmax())


def ray_to_alpha_shape(points, origin, u, alpha_radius):
    """Distance from origin, along u, to the boundary of the alpha shape of points.

    C++: DistP2AlphaShape (CGAL alpha shape + ray casting). Here the alpha shape
    is approximated by the Delaunay edges shorter than 2 * alpha_radius; starting
    from outside the school, the first edge hit is on its boundary.
    Returns inf if the ray misses the school.
    """
    edges = delaunay_edges(points)
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    keep = norm(b - a) <= 2 * alpha_radius
    a, b = a[keep], b[keep]
    # solve origin + t * u = a + s * (b - a), with t > 0 and 0 <= s <= 1
    ab = b - a
    denom = u[0] * ab[:, 1] - u[1] * ab[:, 0]
    ao = a - origin
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ao[:, 0] * ab[:, 1] - ao[:, 1] * ab[:, 0]) / denom
        s = (ao[:, 0] * u[1] - ao[:, 1] * u[0]) / denom
    hit = (denom != 0) & (t > 0) & (s >= 0) & (s <= 1)
    return t[hit].min() if hit.any() else np.inf
