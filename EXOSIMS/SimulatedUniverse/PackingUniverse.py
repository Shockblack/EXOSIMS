import astropy.units as u
import numpy as np
from EXOSIMS.util.stability import petrovich_stability_fn, pack_all_systems

from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse


class PackingUniverse(SimulatedUniverse):
    """
    Simulated universe implementation that packs planets in stable configurations based on SAG13 Planet Population module.

    Args:
        specs:
            user specified values

    Notes:
        The occurrence rate in these universes is set entirely by the radius
        distribution.

    """

    def __init__(self, earthPF=False, **specs):
        self.earthPF = earthPF
        SimulatedUniverse.__init__(self, **specs)
        self._outspec["earthPF"] = self.earthPF

    def gen_physical_properties(self, **specs):
        """Generating universe based on SAG13 planet radius and period sampling.

        All parameters except for albedo and mass are sampled, while those are
        calculated via the physical model.

        """

        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList

        # treat eta as the rate parameter of a Poisson distribution
        targetSystems = np.random.poisson(lam=PPop.eta, size=TL.nStars)
        self.nPlans = np.sum(targetSystems)

        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(
            self.nPlans,
            commonSystemPlane=self.commonSystemPlane,
            commonSystemPlaneParams=self.commonSystemPlaneParams,
        )
        self.setup_system_planes()
        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)

        self.Mp = PPMod.calc_mass_from_radius(self.Rp)  # mass

        # pack the orbits with planets
        sInds, plan2star, leftover = pack_all_systems(
            M_s = TL.MsTrue,
            M_p = self.Mp,
            a = self.a,
            e = self.e,
            i = self.I,
            stability_fn = petrovich_stability_fn,
            pre_sort = "sma"
        )
        self.sInds = np.array(sInds)
        self.plan2star = np.array(plan2star)
        if len(leftover) > 0:
            self.vprint(f"Warning: {len(leftover)} planets were not placed in any system due to instability. These planets will be dropped from the simulation.")
        self.nPlans = len(self.plan2star)

        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])
        self.gen_M0()  # initial mean anomaly

        # Use Earth Phase Function
        if self.earthPF:
            self.phiIndex = (
                np.ones(self.nPlans, dtype=int) * 2
            )  # Used to switch select specific phase function for each planet
        else:
            self.phiIndex = np.asarray(
                []
            )  # Used to switch select specific phase function for each planet
        ZL = self.ZodiacalLight
        if self.fixed_nEZ_val is not None:
            self.nEZ = np.ones((self.nPlans,)) * self.fixed_nEZ_val
        elif self.commonSystemnEZ:
            # Assign the same nEZ to all planets in the system
            self.nEZ = ZL.gen_systemnEZ(TL.nStars)[self.plan2star]
        else:
            # Assign a unique nEZ to each planet
            self.nEZ = ZL.gen_systemnEZ(self.nPlans)
