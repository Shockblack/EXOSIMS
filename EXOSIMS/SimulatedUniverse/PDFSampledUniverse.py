import numpy as np

from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse


class PDFSampledUniverse(SimulatedUniverse):
    """Simulated Universe module based on SAG13 Planet Population module.
    """

    def __init__(self, earthPF=False, **specs):
        self.earthPF = earthPF
        SimulatedUniverse.__init__(self, **specs)
        self._outspec["earthPF"] = self.earthPF

    def gen_physical_properties(self, **specs):
        """Generating universe that includes both real exoplanets from the NASA Exoplanet Archive (NEA) and synthetic planets
        sampled from a modified density function to match the SAG13 occurrence rates.

        All parameters except for albedo and mass are sampled, while those are
        calculated via the physical model.

        """

        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList


        # 1. First check how many host stars acually have planets
        # 2. Generate a the target system sample with the given eta
        # 3. Randomly remove Nplanets that are known from the sample list
        # i.e. randomly select N_tot - N_NEA of the N_tot sampled planets

        targetSystems = np.random.poisson(lam=PPop.eta, size=TL.nStars)
        planinds = []
        starinds = []
        planmask = []
        for j, n in enumerate(targetSystems):
            # Check if the star has known planets
            tmp = np.where(PPop.hostname == TL.Name[j])[0]
            planinds = np.hstack((planinds, tmp))
            if len(tmp) >= n and len(tmp) > 0:
                starinds = np.hstack((starinds, [j] * len(tmp)))
                planmask = np.hstack((planmask, [True] * len(tmp)))
            else:
                starinds = np.hstack((starinds, [j] * n))
                planmask = np.hstack((planmask, [True] * len(tmp) + [False] * (n - len(tmp))))
        
        planinds = planinds.astype(int)
        starinds = starinds.astype(int)
        planmask = planmask.astype(bool)
        
        self.plan2star = starinds
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(starinds)
        N_NEA = len(planinds)

        # This masks the parameter sampling to replace places where mask = True with real planet data
        PPop.planmask = planmask 
        # This gives the indices of the NEA planet hosts that appear in the target list
        PPop.planinds = planinds

        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(
            self.nPlans,
            commonSystemPlane=self.commonSystemPlane,
            commonSystemPlaneParams=self.commonSystemPlaneParams,
        )
        self.setup_system_planes()
        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)
        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])
        self.gen_M0()  # initial mean anomaly
        self.Mp = PPMod.calc_mass_from_radius(self.Rp)  # mass

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
        if self.commonSystemnEZ:
            # Assign the same nEZ to all planets in the system
            self.nEZ = ZL.gen_systemnEZ(TL.nStars)[self.plan2star]
        else:
            # Assign a unique nEZ to each planet
            self.nEZ = ZL.gen_systemnEZ(self.nPlans)
