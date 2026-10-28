"""Model parameters.

Default values are those of the original code (SwarmDynByPy.get_base_params
in https://github.com/PaPeK/PredatorPrey). The C++ name is given when it differs.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    # simulation
    N: int = 400
    dt: float = 0.02
    output: float = 0.2              # time between two recorded frames
    pred_time: float = 200.0         # equilibration before the predator appears
    attack_time: float = 60.0        # C++: record_time
    min_cluster: float = 0.9         # restart if the largest cluster is smaller (fraction of N)
    max_restarts: int = 3            # C++: maxiter

    # prey
    speed0: float = 1.0
    Dphi: float = 0.5                # angular noise
    rep_range: float = 1.0
    rep_strength: float = 2.0
    rep_steepness: float = -2.0
    alg_strength: float = 2.0        # mean alignment strength of the population
    alg_sigma: float = 0.075         # C++: pava_sig, spread of alg_strength between prey
    flee_strength: float = 4.0

    # predator
    pred_speed: float = 2.0          # C++: pred_speed0
    pred_strength: float = 2.0
    pred_angle: float = np.pi        # attack angle relative to the school heading
    pred_angle_noise: float = 3.0    # C++ passes pi through "%d", so the real value is 3
    pred_radius: float = 4.5         # initial distance to the school
    pred_memory: float = 1.0         # C++: pred_memo_t, time over which the school heading is averaged
    kill_range: float = 3.0
    kill_rate: float = 1.0

    # clusters
    cludist: float = 6.0             # two prey closer than this are in the same cluster

    @property
    def pred_circle_radius(self):
        # C++: predcirclerad, the predator switches to hunting when prey are this close
        return 3 * self.kill_range
