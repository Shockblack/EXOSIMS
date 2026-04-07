import numpy as np
from typing import List, Callable
import astropy.units as u

StabilityFn = Callable[[float, List, List, List, List, List], bool]

def petrovich_stability_fn(M_s, M_p, a, e, i, planinds):
    """
    Stability criterion from Petrovich, C. 2015, ApJ, 808, 120.
    https://doi.org/10.1088/0004-637X/808/2/120

    r_ap  >  Y + 1.15, where r_ap = a_out * (1 - e_out) / [a_in * (1 + e_in)]

    and Y = 2.4 * [max(mu_in, mu_out)]^(1/3) * sqrt(a_out / a_in)

    Valid for:
        - Planet-to-star mass ratios  mu_in, mu_out  in  [10^-4, 10^-2]
        - Integration times up to 10^8 inner orbits
        - Mutual inclinations  < 40 degrees

    Args:
        M_s (float):
            Mass of the star in solar masses
        M_p (list of floats):
            List of planet masses in solar masses
        a (list of floats):
            List of planet semi-major axes in AU
        e (list of floats):
            List of planet eccentricities
        i (list of floats):
            List of planet inclinations in degrees
        planinds (list of ints):
            List of indices corresponding to the planets being considered for placement in this system. These indices map to the global planet pool.

    Returns:
        bool:
            True if system is stable according to Petrovich 2015 criterion, False otherwise
    """
    if len(planinds) < 2:
        return True

    planinds_sorted = sorted(planinds, key=lambda k: a[k])

    for p1, p2 in zip(planinds_sorted[:-1], planinds_sorted[1:]):
        m1 = M_p[p1]
        m2 = M_p[p2]
        a1, a2 = a[p1], a[p2]
        e1, e2 = e[p1], e[p2]

        mu_in  = m1 / M_s    # inner planet-to-star mass ratio
        mu_out = m2 / M_s    # outer planet-to-star mass ratio

        # Pericenter / apocenter proximity ratio (Petrovich 2015, Eq. 1)
        r_ap = a2 * (1.0 - e2) / (a1 * (1.0 + e1))

        # RHS stability threshold (Petrovich 2015, Eq. 3)
        Y = 2.4 * max(mu_in, mu_out) ** (1.0 / 3.0) * np.sqrt(a2 / a1)

        if r_ap <= Y + 1.15:
            return False

    return True


def pack_one_system(M_s, M_p, a, e, i, candidates: List[int], stability_fn: StabilityFn) -> tuple[List[int], List[int]]:
    """
    Packs stars with planets drawn from a list of candidates.

    Because all orbital elements are pre-assigned, placement is purely a
    binary accept/reject decision: does adding this planet (at its fixed `a`,
    `e`, ...) keep the system stable?

    Args:
        M_s (float):
            Mass of the star in solar masses
        M_p (list of floats):
            List of planet masses in solar masses
        a (list of floats):
            List of planet semi-major axes in AU
        e (list of floats):
            List of planet eccentricities
        i (list of floats):
            List of planet inclinations in degrees
        candidates (list of ints):
            List of indices corresponding to the planets being considered for placement in this system. These indices map to the global planet pool.
        stability_fn (function):
            Function that takes in the star mass, planet masses, and orbital elements of a candidate system and returns True if the system is stable and False otherwise.

    Returns:
        system (PlanetarySystem object):
            PlanetarySystem object containing the star and the planets that were successfully placed in a stable configuration
        leftover (list of Planet objects):
            List of Planet objects that were not placed in the system (either because they were rejected for instability or because max_passes was reached)
    """

    stable_plan_inds = []

    for i, candidate_idx in enumerate(candidates):
        if stability_fn(M_s, M_p, a, e, i, stable_plan_inds + [candidate_idx]):
            stable_plan_inds.append(candidate_idx)

    # Only keep candidates that were not placed this pass for the next pass
    candidates = [idx for idx in candidates if idx not in stable_plan_inds]

    return stable_plan_inds, candidates

def pack_all_systems(M_s, M_p, a, e, i, stability_fn, pre_sort = "semi_major_axis"):
    """
    Distribute a pre-parametrised planet pool across all stars.

    Stars are processed in the order given.  Each star is packed until no
    remaining planet can be stably added; unused planets roll over to the next
    star.

    Args:
        M_s (array-like or astropy Quantity):
            Array of star masses in solar masses
        M_p (array-like or astropy Quantity):
            Array of planet masses in solar masses
        a (array-like or astropy Quantity):
            Array of planet semi-major axes in AU
        e (array-like):
            Array of planet eccentricities
        i (array-like or astropy Quantity):
            Array of planet inclinations in degrees
        stability_fn (function):
            Function that takes in the star mass, planet masses, and orbital elements of a candidate system and returns True if the system is stable and False otherwise.
        pre_sort (str, optional):
            Method for pre-sorting the planet candidates before packing. Options are "sma" (default), "mass", or "none".
    
    Returns:
        sInds (list of ints):
            List of star indices
        plan2star (list of ints):
            List mapping from planet index to star index for planets that were successfully placed in systems
        leftover (list of ints):
            List of planet indices that were not placed in any system
    """
    # Check units and convert to arrays if in astropy Quantity format, if not convert to numpy arrays.
    M_s = M_s.to(u.Msun).value if isinstance(M_s, u.Quantity) else np.asarray(M_s)
    M_p = M_p.to(u.Msun).value if isinstance(M_p, u.Quantity) else (M_p*u.Mearth).to(u.Msun).value
    a = a.to(u.AU).value if isinstance(a, u.Quantity) else np.asarray(a)
    e = e.value if isinstance(e, u.Quantity) else np.asarray(e) # should be unitless but just in case
    i = i.to(u.deg).value if isinstance(i, u.Quantity) else np.asarray(i)

    # List of indexes of planets that have not yet been placed in a system
    remaining = list(range(len(M_p)))
    sInds = list(range(len(M_s)))

    if pre_sort == "sma":
        remaining.sort(key=lambda idx: a[idx])
    elif pre_sort == "mass":
        remaining.sort(key=lambda idx: M_p[idx], reverse=True)

    plan2star = np.full(len(M_p), -1, dtype=int)

    for star_idx in sInds:
        if not remaining:
            break

        stable_plan_inds, remaining = pack_one_system(M_s[star_idx], M_p, a, e, i, remaining, stability_fn)
        
        plan2star[stable_plan_inds] = star_idx

    return sInds, plan2star, remaining