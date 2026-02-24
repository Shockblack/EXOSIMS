from EXOSIMS.PlanetPopulation.SAG13 import SAG13
from EXOSIMS.util import getExoplanetArchive as nea
from EXOSIMS.util._numpy_compat import copy_if_needed
from EXOSIMS.util.InverseTransformSampler import InverseTransformSampler
from scipy.stats import gaussian_kde
import numpy as np
import astropy.units as u
import astropy.constants as const

class KnownPlanetsSAG13(SAG13):
    """
    Planet population that combines both confirmed exoplanets from the NASA Exoplanet Archive (NEA)
    and synthetic planets. The final distribution of confirmed and synthetic planets follows the
    SAG13 occurence rates from Kepler.

    This class inherits from the SAG13 class and provides methods to approximate the probability density
    functions for the NEA data. In general, the final PDF we wish to sample from is:
        f_sample = (N_tot / N_sampled) * f_target - (N_nea / N_sampled) * f_nea

    """

    def __init__(self, **specs):
        """
        Initialize the planetSample class.

        Parameters
        ----------
        specs : dict
            Dictionary of specifications for the planet population.
        """
        super().__init__(**specs)
        # Additional initialization code can go here if needed.
        
        SAG13coeffs=[[0.38, -0.19, 0.26, 0.0], [0.73, -1.18, 0.59, 3.4]]
        SAG13starMass=1.0 * u.Msun
        
        mu = const.G * SAG13starMass
        SAG13coeffs = np.array(SAG13coeffs, dtype=float)
        SAG13coeffs = SAG13coeffs.T
        SAG13coeffs = SAG13coeffs[:, np.argsort(SAG13coeffs[3, :])]

        m = mu.to("AU3/year2").value


        # Create a dictionary of ranges for the parameters
        self.NEA_ranges = {
            'pl_orbsmax': self.arange,
            'pl_rade':self.Rprange,
            }
        
        self.archive = self.generate_NEA_planets()
    
        self.N_nea = len(self.archive['pl_orbsmax'])
        self.hostname = self.archive['hostname']

        # Placeholder
        self.planmask = None
        self.planinds = None
        # Wanted to add a method that enables us to resample around the same NEA
        # population without breaking the code, nplans enables us to do so by
        # letting my append the planetmask with more synthetic planets
        self.nPlans = None
        

        # Semi-major axis distributions given radius for two cases
        self.f_sma_given_Rp1 = lambda a, beta=self.beta[0], m=m, C=self.Ca[0], \
            smaknee=self.smaknee: self.dist_sma_given_radius(a, beta, m, C, smaknee)
        self.f_sma_given_Rp2 = lambda a, beta=self.beta[1], m=m, C=self.Ca[1], \
            smaknee=self.smaknee: self.dist_sma_given_radius(a, beta, m, C, smaknee)
            
        
    def generate_NEA_planets(self):
        """Helper function to generate the NEA planets within the specified ranges.

        Returns
        -------
        pandas.DataFrame
        A pandas DataFrame containing the exoplanet data from the NEA, filtered by the specified ranges.
        """


        archive = nea.getExoplanetArchivePSCP()

        # Limit the data to the ranges defined in NEA_ranges
        for key, value in self.NEA_ranges.items():
            archive = archive[(archive[key] >= value[0]) & (archive[key] <= value[1])]

        return archive.reset_index(drop=True)


    def infer_PDF(self, array, array_range, data):
        """Given an array of values, a range for those values, and data,
        this function approximates the probability density function (PDF) for the data
        within the specified range. It uses a Gaussian kernel density estimation (KDE) to
        compute the PDF values for the input data.

        Parameters
        ----------
        array : array_like
            An array of values for which the PDF is to be computed.
        array_range : list or tuple
            A range (min, max) within which the PDF is computed.
        data : array_like
            The data from which the PDF is estimated.

        Returns
        -------
        f : numpy.ndarray
            An array containing the estimated PDF values.
        """

        array = np.array(array, ndmin=1)
        
        f = np.zeros(array.shape)

        mask = np.array((array >= array_range[0]) & (array <= array_range[1]), ndmin=1)

        kde = gaussian_kde(data)

        f[mask] = kde.evaluate(array[mask])

        return f
    
    
    def gen_radius_sma(self, n):
        N_sampled = n - self.N_nea

        dist_Rp = lambda Rp: (n / N_sampled) * self.dist_radius(Rp) - (self.N_nea / N_sampled) * self.dist_rad_nea(Rp)

        # Sample radius
        Rp = InverseTransformSampler(dist_Rp, self.Rprange[0].value, self.Rprange[1].value)(N_sampled)
        a = np.zeros_like(Rp)

        # Get modified density function for semi-major axis
        dist_a_given_Rp1 = lambda a: (n / len(Rp[Rp < self.Rplim[1]])) * self.f_sma_given_Rp1(a) - \
              (self.N_nea / len(Rp[Rp < self.Rplim[1]])) * self.dist_sma_nea(a)

        # Sample semi-major axis given radius
        a[Rp < self.Rplim[1]] = InverseTransformSampler(dist_a_given_Rp1, self.arange[0].value, self.arange[1].value)(len(Rp[Rp < self.Rplim[1]]))

        if len(Rp[Rp>= self.Rplim[1]]) > 0:
            dist_a_given_Rp2 = lambda a: (n / len(Rp[Rp >= self.Rplim[1]])) * self.f_sma_given_Rp2(a) - \
                (self.N_nea / len(Rp[Rp >= self.Rplim[1]])) * self.dist_sma_nea(a)
            a[Rp >= self.Rplim[1]] = InverseTransformSampler(dist_a_given_Rp2, self.arange[0].value, self.arange[1].value)(len(Rp[Rp >= self.Rplim[1]]))
            

        # Stack the data
        # a = np.concatenate((self.archive['pl_orbsmax'], a))
        # Rp = np.concatenate((self.archive['pl_rade'], Rp))

        # Create a and Rp arrays that include the NEA data
        # Should be planetary parameters where planmask = True and sampled parameters where planmask = False
        a_tmp = np.ones(n)
        Rp_tmp = np.ones(n)

        # This is not good practice but I want a menthod that lets us resample 
        # without losing the original SimulatedUniverse
        planmask = self.planmask
        if n > self.nPlans and self.planmask is not None:
            # If we are generating more planets than the original number of planets, we need to append the planmask with False values for the new planets
            planmask = np.hstack((self.planmask, [False] * (n - self.nPlans)))
        elif n < self.nPlans:
            error_msg = "Cannot generate fewer planets ({}) than the original number of planets ({}). \nThis is due to the way we handle indexing the NEA planets.".format(n, self.nPlans)
            raise ValueError(error_msg)
        
        a_tmp[planmask] = self.archive['pl_orbsmax']
        a_tmp[~planmask] = a
        Rp_tmp[planmask] = self.archive['pl_rade']
        Rp_tmp[~planmask] = Rp

        return Rp_tmp * u.earthRad, a_tmp * u.AU
        
    
    def gen_plan_params(self, n):

        # By this point, planinds should've been set in SimulatedUniverse
        # If so, limit archive/make sure it is already limited
        if self.planinds is not None and (len(self.planinds) != len(self.archive)):
            self.archive = self.archive.iloc[self.planinds].reset_index(drop=True)
            self.N_nea = len(self.planinds)
        
        # Get inferred distributions from NEA data
        self.dist_rad_nea = lambda Rp: self.infer_PDF(Rp, self.Rprange.value, self.archive['pl_rade'])
        self.dist_sma_nea = lambda a: self.infer_PDF(a, self.arange.value, self.archive['pl_orbsmax'])

        n = self.gen_input_check(n)
        
        Rp, a = self.gen_radius_sma(n)

        C1 = np.exp(-self.erange[0] ** 2 / (2.0 * self.esigma**2))
        ar = self.arange.to("AU").value
        if self.constrainOrbits:
            # restrict semi-major axis limits
            arcon = np.array(
                [ar[0] / (1.0 - self.erange[0]), ar[1] / (1.0 + self.erange[0])]
            )
            # clip sma values to sma range
            sma = np.clip(a.to("AU").value, arcon[0], arcon[1])
            # upper limit for eccentricity given sma
            elim = np.zeros(len(sma))
            amean = np.mean(ar)
            elim[sma <= amean] = 1.0 - ar[0] / sma[sma <= amean]
            elim[sma > amean] = ar[1] / sma[sma > amean] - 1.0
            elim[elim > self.erange[1]] = self.erange[1]
            elim[elim < self.erange[0]] = self.erange[0]
            # additional constant
            C2 = C1 - np.exp(-(elim**2) / (2.0 * self.esigma**2))
            a = sma * u.AU
        else:
            C2 = self.enorm
        
        e = self.esigma * np.sqrt(-2.0 * np.log(C1 - C2 * np.random.uniform(size=n)))

        # generate albedo from semi-major axis
        p = self.PlanetPhysicalModel.calc_albedo_from_sma(a, self.prange)

        return a, e, p, Rp