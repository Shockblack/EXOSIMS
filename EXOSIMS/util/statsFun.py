import numpy as np
from scipy.optimize import fmin_l_bfgs_b, brentq
from scipy.stats import gaussian_kde, iqr
from scipy.fftpack import dct, idct

"""
Collection of useful statistics routines
"""


def simpSample(f, numTest, xMin, xMax, M=None, verb=False):
    """
    Use the rejection sampling method to generate random samples

    Args:
        f (callable):
            Function definition encoding PDF to sample from
        numTest (int):
            Number of samples to generate
        xMin (float):
            Lower bound
        xMax (float):
            Upper bound
        M (float, optional):
            Maximum value over interval.  If None (default) this will be estimated
            directly from the function definition.
        verb (bool):
            If True, print number of iterations required to produce sample.
            Defaults False.

    Returns:
        ~numpy.ndarray(float):
            Random samples.  Has size of numTest.


    .. note::

        If xMin==xMax, returns an array where all values are equal to this value.

    """

    if xMin == xMax:
        return np.zeros(numTest) + xMin

    # find max value if not provided
    if M is None:
        M = calcM(f, xMin, xMax)

    # initialize
    n = 0
    X = np.zeros(numTest)
    numIter = 0
    maxIter = 1000

    nSamp = max(2 * numTest, 1000 * 1000)
    while n < numTest and numIter < maxIter:
        xd = np.random.uniform(low=xMin, high=xMax, size=nSamp)
        yd = np.random.uniform(low=0, high=M, size=nSamp)
        pd = f(xd)

        xd = xd[yd < pd]
        X[n : min(n + len(xd), numTest)] = xd[: min(len(xd), numTest - n)]
        n += len(xd)
        numIter += 1

    if numIter == maxIter:
        raise Exception("Failed to converge.")

    if verb:
        print("Finished in " + repr(numIter) + " iterations.")

    return X


def calcM(f, xMin, xMax):
    """Compute maximum value of a function over an interval

    Args:
        f (callable):
            Function definition
        xMin (float):
            Lower bound
        xMax (float):
            Upper bound

    Returns:
        float:
            Maximum value in bound

    """

    # first do a coarse grid to get ic
    dx = np.linspace(xMin, xMax, 1000 * 1000)
    ic = np.argmax(f(dx))

    # now optimize
    g = lambda x: -f(x)
    M = fmin_l_bfgs_b(g, [dx[ic]], approx_grad=True, bounds=[(xMin, xMax)])
    M = f(M[0])

    return M


def eqLogSample(f, numTest, xMin, xMax, bins=10):
    """
    Generate samples (via rejection sampling) of a given probability density function
    in equally spaced logarithmic bins over a provided range.

    Args:
        f (callable):
            Function definition encoding PDF to sample from
        numTest (int):
            Number of samples to generate
        xMin (float):
            Lower bound
        xMax (float):
            Upper bound
        bins (int):
            Number of bins to use.  Defaults to 10.

    Returns:
        ~numpy.ndarray(float):
            Random samples.  Has size of numTest.

    """

    out = np.array([])
    bounds = np.logspace(np.log10(xMin), np.log10(xMax), bins + 1)
    for j in np.arange(1, bins + 1):
        out = np.concatenate(
            (out, simpSample(f, numTest // bins, bounds[j - 1], bounds[j]))
        )

    return out

def KDE_func(data, bw, array_range):
    """Given an array of values, a range for those values, and data,
    this function approximates the probability density function (PDF) for the data
    within the specified range. It uses a Gaussian kernel density estimation (KDE) to
    compute the PDF values for the input data.

    Parameters
    ----------
    data : array_like
        The data from which the PDF is estimated.
    bw : float
        The bandwidth for the Gaussian kernel density estimation.
    array_range : list or tuple
        A range (min, max) within which the PDF is computed.

    Returns
    -------
    func : callable
        A function that takes an array of values and returns the estimated PDF values.
    """

    def func(array):
        """Inner function to compute the PDF using Gaussian KDE."""
        kde = gaussian_kde(data)
        
        array = np.array(array, ndmin=1)
    
        f = np.zeros(array.shape)

        mask = np.array((array >= array_range[0]) & (array <= array_range[1]), ndmin=1)

        kde = gaussian_kde(data, bw_method=bw)

        f[mask] = kde.evaluate(array[mask])

        return f

    return func

def scotts_bw(data):
    """Calculate the Scott's method bandwidth for Gaussian KDE. Assumes data is one-dimensional.

    Args:
        data (array_like): Data for which to calculate the bandwidth.

    Returns:
        float: The calculated bandwidth.
    """
    n = len(data)
    return np.std(data) * n ** (-1 / 5.0)

def silverman_bw(data):
    """Calculate the Silverman's method bandwidth for Gaussian KDE. Assumes data is one-dimensional.

    Args:
        data (array_like): Data for which to calculate the bandwidth.

    Returns:
        float: The calculated bandwidth.
    """
    n = len(data)
    iqr_value = iqr(data)
    std_iqr = min(np.std(data), iqr_value / 1.34)
    return 0.9 * std_iqr * n ** (-1 / 5.0)

def ISJ_bw(data):
    """Improved Sheather-Jones bandwidth selection from Botev et al. (2010).
    Adapted from the implementation by Daniel B. Smith;
    https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py

    Parameters
    ----------
    data : array_like
        Data for which to calculate the bandwidth.

    Returns
    -------
    float
        The calculated bandwidth.
    """

    # Parameters to set up the mesh on which to calculate
    N = 2**14

    minimum = min(data)
    maximum = max(data)
    Range = maximum - minimum
    MIN = minimum - Range/10
    MAX = maximum + Range/10

    # Range of the data
    R = MAX-MIN

    # Histogram the data to get a crude first approximation of the density
    M = len(data)

    DataHist, _ = np.histogram(data, bins=N, range=(MIN,MAX))
    DataHist = DataHist/M
    DCTData = dct(DataHist, norm=None)

    I = [iN*iN for iN in np.arange(1, N)]
    SqDCTData = (DCTData[1:]/2)**2

    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    t_star = brentq(fixed_point, 0, guess, args=(M, I, SqDCTData))

    bandwidth = np.sqrt(t_star)*R
    
    return bandwidth

def fixed_point(t, M, I, a2):
    """Fix point function from Botev et al. (2010).
    Adapted from the implementation by Daniel B. Smith; 
    https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py

    Parameters
    ----------
    t : float
        Initial guess
    M : int
        Number of data points
    I : array_like
        Array of integers from 1 to N-1, squared
    a2 : array_like
        DCT of the data halved and squared

    Returns
    -------
    _type_
        _description_
    """

    l=7

    I = np.float64(I)
    M = np.float64(M)
    a2 = np.float64(a2)

    f = 2*np.pi**(2*l)*np.sum(I**l*a2*np.exp(-I*np.pi**2*t))

    for s in range(l, 1, -1):
        K0 = np.prod(np.arange(1, 2*s+1, 2))/np.sqrt(2*np.pi)
        const = (1 + (1/2)**(s + 1/2))/3
        time=(2*const*K0/M/f)**(2/(3+2*s))
        f=2*np.pi**(2*s)*np.sum(I**s*a2*np.exp(-I*np.pi**2*time))
    return t-(2*M*np.sqrt(np.pi)*f)**(-2/5)