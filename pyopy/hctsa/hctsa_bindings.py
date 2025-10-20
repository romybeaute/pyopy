# coding=utf-8
import numpy as np
from numpy import array, float64
from pyopy.base import MatlabSequence
from pyopy.hctsa.hctsa_bindings_gen import HCTSASuper, HCTSAOperation


class CO_AddNoise(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Adds Gaussian-distributed noise to the time series with increasing standard
    % deviation, eta, across the range eta = 0, 0.1, ..., 2, and measures the
    % mutual information at each point
    % Can be measured using histograms with extraParam bins (implemented using
    % CO_HistogramAMI), or using the Information Dynamics Toolkit.
    %
    % The output is a set of statistics on the resulting set of automutual
    % information estimates, including a fit to an exponential decay, since the
    % automutual information decreases with the added white noise.
    %
    % Can calculate these statistics for time delays 'tau', and for a number 'extraParam'
    % bins.
    %
    % This algorithm is quite different, but was based on the idea of 'noise
    % titration' presented in: "Titration of chaos with added noise", Chi-Sang Poon
    % and Mauricio Barahona P. Natl. Acad. Sci. USA, 98(13) 7107 (2001)
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % tau, the time delay for computing AMI
    %
    % amiMethod, the method for computing AMI:
    %      * one of 'std1','std2','quantiles','even' for histogram-based estimation,
    %      * one of 'gaussian','kernel','kraskov1','kraskov2' for estimation using JIDT
    %
    % extraParam, e.g., the number of bins input to CO_HistogramAMI, or parameter
    %             for IN_AutoMutualInfo
    %
    % randomSeed: settings for resetting the random seed for reproducible results
    %               (using BF_ResetSeed)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21, 22, 24, 26, 21, 22, 23, 25, 27, 22)

    TAGS = ('AMI', 'correlation', 'entropy')

    def __init__(self, tau=1.0, amiMethod='quantiles', extraParam=10.0, randomSeed='default'):
        super(CO_AddNoise, self).__init__()
        self.tau = tau
        self.amiMethod = amiMethod
        self.extraParam = extraParam
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_AddNoise', x, )
        elif self.amiMethod is None:
            return eng.run_function(1, 'CO_AddNoise', x, self.tau)
        elif self.extraParam is None:
            return eng.run_function(1, 'CO_AddNoise', x, self.tau, self.amiMethod)
        elif self.randomSeed is None:
            return eng.run_function(1, 'CO_AddNoise', x, self.tau, self.amiMethod, self.extraParam)
        return eng.run_function(1, 'CO_AddNoise', x, self.tau, self.amiMethod, self.extraParam, self.randomSeed)


class CO_AutoCorr(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, a scalar time series column vector.
    % tau, the time-delay. If tau is a scalar, returns autocorrelation for y at that
    %       lag. If tau is a vector, returns autocorrelations for y at that set of
    %       lags.
    % whatMethod, the method of computing the autocorrelation: 'Fourier',
    %             'TimeDomainStat', or 'TimeDomain'.
    %
    %---OUTPUT: the autocorrelation at the given time-lag.
    %
    %---NOTES:
    % Specifying whatMethod = 'TimeDomain' can tolerate NaN values in the time
    % series.
    %
    % Computing mean/std across the full time series makes a significant difference
    % for short time series, but can produce values outside [-1,+1]. The
    % filtering-based method used by Matlab's autocorr, is probably the best for
    % short time series, and is implemented here by specifying: whatMethod =
    % 'Fourier'.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 4)

    TAGS = ('autocorrelation', 'correlation')

    def __init__(self, tau=1.0, whatMethod='Fourier'):
        super(CO_AutoCorr, self).__init__()
        self.tau = tau
        self.whatMethod = whatMethod

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_AutoCorr', x, )
        elif self.whatMethod is None:
            return eng.run_function(1, 'CO_AutoCorr', x, self.tau)
        return eng.run_function(1, 'CO_AutoCorr', x, self.tau, self.whatMethod)


class CO_AutoCorrShape(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Outputs include the number of peaks, and autocorrelation in the
    % autocorrelation function (ACF) itself.
    %
    %---INPUTS:
    % y, the input time series
    % stopWhen, the criterion for the maximum lag to measure the ACF up to.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (22,)

    TAGS = ('correlation',)

    def __init__(self, stopWhen='drown'):
        super(CO_AutoCorrShape, self).__init__()
        self.stopWhen = stopWhen

    def _eval_hook(self, eng, x):
        if self.stopWhen is None:
            return eng.run_function(1, 'CO_AutoCorrShape', x, )
        return eng.run_function(1, 'CO_AutoCorrShape', x, self.stopWhen)


class CO_CompareMinAMI(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Finds the first minimum of the automutual information by various different
    % estimation methods, and sees how this varies over different coarse-grainings
    % of the time series.
    %
    % The function returns a set of statistics on the set of first minimums of the
    % automutual information function obtained over a range of the number of bins
    % used in the histogram estimation, when specifying 'numBins' as a vector
    %
    %---INPUTS:
    % y, the input time series
    %
    % binMethod, the method for estimating mutual information (input to CO_HistogramAMI)
    %
    % numBins, the number of bins for the AMI estimation to compare over (can be a
    %           scalar or vector)
    %
    % Outputs include the minimum, maximum, range, number of unique values, and the
    % position and periodicity of peaks in the set of automutual information
    % minimums.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (26, 31, 26, 26)

    TAGS = ('AMI', 'correlation')

    def __init__(self, binMethod='std1', numBins=array(( 2,  3,  4,  5,
                           6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
       70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80))):
        super(CO_CompareMinAMI, self).__init__()
        self.binMethod = binMethod
        self.numBins = numBins

    def _eval_hook(self, eng, x):
        if self.binMethod is None:
            return eng.run_function(1, 'CO_CompareMinAMI', x, )
        elif self.numBins is None:
            return eng.run_function(1, 'CO_CompareMinAMI', x, self.binMethod)
        return eng.run_function(1, 'CO_CompareMinAMI', x, self.binMethod, self.numBins)


class CO_Embed2(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Embeds the z-scored time series in a two-dimensional time-delay
    % embedding space with a given time-delay, tau, and outputs a set of
    % statistics about the structure in this space, including angular
    % distribution, etc.
    %
    %---INPUTS:
    % y, the column-vector time series
    % tau, the time-delay (can be 'tau' for first zero-crossing of ACF)
    %
    %---OUTPUTS: include the distribution of angles between successive points in the
    % space, stationarity of this angular distribution, euclidean distances from the
    % origin, and statistics on outliers.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13,)

    TAGS = ('correlation', 'embedding')

    def __init__(self, tau='tau'):
        super(CO_Embed2, self).__init__()
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_Embed2', x, )
        return eng.run_function(1, 'CO_Embed2', x, self.tau)


class CO_Embed2_AngleTau(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Investigates how the autocorrelation of angles between successive points in
    % the two-dimensional time-series embedding change as tau varies from
    % tau = 1, 2, ..., maxTau.
    %
    %---INPUTS:
    % y, a column vector time series
    % maxTau, the maximum time lag to consider
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21,)

    TAGS = ('correlation', 'embedding')

    def __init__(self, maxTau=50.0):
        super(CO_Embed2_AngleTau, self).__init__()
        self.maxTau = maxTau

    def _eval_hook(self, eng, x):
        if self.maxTau is None:
            return eng.run_function(1, 'CO_Embed2_AngleTau', x, )
        return eng.run_function(1, 'CO_Embed2_AngleTau', x, self.maxTau)


class CO_Embed2_Basic(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Computes a set of point density measures in a plot of y_i against y_{i-tau}.
    %
    % INPUTS:
    % y, the input time series
    %
    % tau, the time lag (can be set to 'tau' to set the time lag the first zero
    %                       crossing of the autocorrelation function)
    %
    % Outputs include the number of points near the diagonal, and similarly, the
    % number of points that are close to certain geometric shapes in the y_{i-tau},
    % y_{tau} plot, including parabolas, rings, and circles.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (17, 19)

    TAGS = ('correlation',)

    def __init__(self, tau=1.0):
        super(CO_Embed2_Basic, self).__init__()
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_Embed2_Basic', x, )
        return eng.run_function(1, 'CO_Embed2_Basic', x, self.tau)


class CO_Embed2_Dist(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Returns statistics on the sequence of successive Euclidean distances between
    % points in a two-dimensional time-delay embedding space with a given
    % time-delay, tau.
    %
    % Outputs include the autocorrelation of distances, the mean distance, the
    % spread of distances, and statistics from an exponential fit to the
    % distribution of distances.
    %
    %---INPUTS:
    % y, a z-scored column vector representing the input time series.
    % tau, the time delay.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (18,)

    TAGS = ('correlation', 'embedding')

    def __init__(self, tau='tau'):
        super(CO_Embed2_Dist, self).__init__()
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_Embed2_Dist', x, )
        return eng.run_function(1, 'CO_Embed2_Dist', x, self.tau)


class CO_Embed2_Shapes(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Takes a shape and places it on each point in the two-dimensional time-delay
    % embedding space sequentially. This function counts the points inside this shape
    % as a function of time, and returns statistics on this extracted time series.
    %
    %---INPUTS:
    % y, the input time-series as a (z-scored) column vector
    % tau, the time-delay
    % shape, has to be 'circle' for now...
    % r, the radius of the circle
    %
    %---OUTPUTS:
    % The constructed time series of the number of nearby points, and
    % include the autocorrelation, maximum, median, mode, a Poisson fit to the
    % distribution, histogram entropy, and stationarity over fifths of the time
    % series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (30, 29)

    TAGS = ('correlation', 'embedding')

    def __init__(self, tau='tau', shape='circle', r=0.1):
        super(CO_Embed2_Shapes, self).__init__()
        self.tau = tau
        self.shape = shape
        self.r = r

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_Embed2_Shapes', x, )
        elif self.shape is None:
            return eng.run_function(1, 'CO_Embed2_Shapes', x, self.tau)
        elif self.r is None:
            return eng.run_function(1, 'CO_Embed2_Shapes', x, self.tau, self.shape)
        return eng.run_function(1, 'CO_Embed2_Shapes', x, self.tau, self.shape, self.r)


class CO_FirstMin(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, the input time series
    % minWhat, the type of correlation to minimize: either 'ac' for autocorrelation,
    %           or 'mi' for automutual information. By default, 'mi' specifies the
    %           'gaussian' method from the Information Dynamics Toolkit. Other
    %           options can also be implemented as 'mi-kernel', 'mi-kraskov1',
    %           'mi-kraskov2' (all from Information Dynamics Toolkit implementations),
    %           or 'mi-hist' (histogram-based method).
    %
    % Note that selecting 'ac' is unusual operation: standard operations are the
    % first zero-crossing of the autocorrelation (as in CO_FirstZero), or the first
    % minimum of the mutual information function ('mi').
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (14, 23, 22, 21, 21, 25, 25)

    TAGS = ('AMI', 'autocorrelation', 'correlation', 'tau')

    def __init__(self, minWhat='ac', extraParam=None):
        super(CO_FirstMin, self).__init__()
        self.minWhat = minWhat
        self.extraParam = extraParam

    def _eval_hook(self, eng, x):
        if self.minWhat is None:
            return eng.run_function(1, 'CO_FirstMin', x, )
        elif self.extraParam is None:
            return eng.run_function(1, 'CO_FirstMin', x, self.minWhat)
        return eng.run_function(1, 'CO_FirstMin', x, self.minWhat, self.extraParam)


class CO_FirstZero(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    %
    % y, the input time series
    % corrFun, the self-correlation function to measure:
    %         (i) 'ac': normal linear autocorrelation function. Uses CO_AutoCorr to
    %                   calculate autocorrelations.
    % maxTau, a maximum time-delay to search up to.
    %
    % In future, could add an option to return the point at which the function
    % crosses the axis, rather than the first integer lag at which it has already
    % crossed (what is currently implemented).
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (15,)

    TAGS = ('autocorrelation', 'correlation', 'tau')

    def __init__(self, corrFun='ac', maxTau=None):
        super(CO_FirstZero, self).__init__()
        self.corrFun = corrFun
        self.maxTau = maxTau

    def _eval_hook(self, eng, x):
        if self.corrFun is None:
            return eng.run_function(1, 'CO_FirstZero', x, )
        elif self.maxTau is None:
            return eng.run_function(1, 'CO_FirstZero', x, self.corrFun)
        return eng.run_function(1, 'CO_FirstZero', x, self.corrFun, self.maxTau)


class CO_HistogramAMI(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The approach used to bin the data is provided.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % tau, the time-lag (1 by default)
    %
    % meth, the method of computing automutual information:
    %           (i) 'even': evenly-spaced bins through the range of the time series,
    %           (ii) 'std1', 'std2': bins that extend only up to a multiple of the
    %                                standard deviation from the mean of the time
    %                                series to exclude outliers,
    %           (iii) 'quantiles': equiprobable bins chosen using quantiles.
    %
    % numBins, the number of bins, required by some methods, meth (see above)
    %
    %---OUTPUT: the automutual information calculated in this way.
    
    % Uses the hist2 function (renamed NK_hist2.m here) by Nedialko Krouchev, obtained
    % from Matlab Central,
    % http://www.mathworks.com/matlabcentral/fileexchange/12346-hist2-for-the-people
    % [[hist2 for the people by Nedialko Krouchev, 20 Sep 2006 (Updated 21 Sep 2006)]]
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (23, 22, 22, 28, 27, 27, 23, 22, 22, 23, 22, 22)

    TAGS = ('AMI', 'correlation', 'information')

    def __init__(self, tau=MatlabSequence('1:5'), meth='std1', numBins=2.0):
        super(CO_HistogramAMI, self).__init__()
        self.tau = tau
        self.meth = meth
        self.numBins = numBins

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_HistogramAMI', x, )
        elif self.meth is None:
            return eng.run_function(1, 'CO_HistogramAMI', x, self.tau)
        elif self.numBins is None:
            return eng.run_function(1, 'CO_HistogramAMI', x, self.tau, self.meth)
        return eng.run_function(1, 'CO_HistogramAMI', x, self.tau, self.meth, self.numBins)


class CO_NonlinearAutocorr(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Nonlinear autocorrelations are of the form:
    % <x_i x_{i-\tau_1} x{i-\tau_2}...>
    % The usual two-point autocorrelations are
    % <x_i.x_{i-\tau}>
    %
    % Assumes that all the taus are much less than the length of the time
    % series, N, so that the means can be approximated as the sample means and the
    % standard deviations approximated as the sample standard deviations and so
    % the z-scored time series can simply be used straight-up.
    %
    %---INPUTS:
    % y  -- should be the z-scored time series (Nx1 vector)
    % taus -- should be a vector of the time delays as above (mx1 vector)
    %   e.g., [2] computes <x_i x_{i-2}>
    %   e.g., [1,2] computes <x_i x_{i-1} x{i-2}>
    %   e.g., [1,1,3] computes <x_i x_{i-1}^2 x{i-3}>
    % doAbs [opt] -- a boolean (0,1) -- if one, takes an absolute value before
    %                taking the final mean -- useful for an odd number of
    %                contributions to the sum. Default is to do this for odd
    %                numbers anyway, if not specified.
    %
    %---NOTES:
    % (*) For odd numbers of regressions (i.e., even number length
    %         taus vectors) the result will be near zero due to fluctuations
    %         below the mean; even for highly-correlated signals. (doAbs)
    %
    % (*) doAbs = 1 is really a different operation that can't be compared with
    %         the values obtained from taking doAbs = 0 (i.e., for odd lengths
    %         of taus)
    % (*) It can be helpful to look at nlac at each iteration.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 9, 8, 9, 9, 10, 11, 12, 13, 14, 9, 8, 9, 9, 9, 10, 8, 8, 9, 9, 8, 9, 10, 8, 8, 8, 8)

    TAGS = ('autocorrelation', 'correlation', 'nonlinearautocorr')

    def __init__(self, taus=array((1, 2)), doAbs=None):
        super(CO_NonlinearAutocorr, self).__init__()
        self.taus = taus
        self.doAbs = doAbs

    def _eval_hook(self, eng, x):
        if self.taus is None:
            return eng.run_function(1, 'CO_NonlinearAutocorr', x, )
        elif self.doAbs is None:
            return eng.run_function(1, 'CO_NonlinearAutocorr', x, self.taus)
        return eng.run_function(1, 'CO_NonlinearAutocorr', x, self.taus, self.doAbs)


class CO_RM_AMInformation(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Wrapper for Rudy Moddemeijer's information code to calculate automutual
    % information.
    %
    %---INPUTS:
    % y, the input time series
    % tau, the time lag at which to calculate the automutual information
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21, 21, 22, 21, 21, 21, 21, 21, 21, 21, 21)

    TAGS = ('AMI', 'correlation', 'information')

    def __init__(self, tau=0.0):
        super(CO_RM_AMInformation, self).__init__()
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_RM_AMInformation', x, )
        return eng.run_function(1, 'CO_RM_AMInformation', x, self.tau)


class CO_StickAngles(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Line-of-sight angles between time-series points treat each time-series value
    % as a stick protruding from an opaque baseline level.
    % Statistics are returned on the raw time series, where sticks protrude
    % from the zero-level, and the z-scored time series, where sticks
    % protrude from the mean level of the time series.
    %
    %---INPUTS:
    % y, the input time series
    %
    %---OUTPUTS: are returned on the obtained sequence of angles, theta, reflecting the
    % maximum deviation a stick can rotate before hitting a stick representing
    % another time point. Statistics include the mean and spread of theta,
    % the different between positive and negative angles, measures of symmetry of
    % the angles, stationarity, autocorrelation, and measures of the distribution of
    % these stick angles.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (16,)

    TAGS = ('correlation',)

    def __init__(self):
        super(CO_StickAngles, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'CO_StickAngles', x, )


class CO_TSTL_amutual(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses amutual code from TSTOOL, which uses a histogram method with n bins to
    % estimate the mutual information of a time series across a range of
    % time-delays, tau.
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    %
    % y, the time series
    %
    % maxTau, the maximum lag for which to calculate the auto mutual information
    %
    % numBins, the number of bins for histogram calculation
    %
    % versionTwo, uses amutual2 instead of amutual (from the TSTOOL package)
    %
    %---OUTPUTS:
    % A number of statistics of the function over the range of tau, including the
    % mean mutual information, its standard deviation, first minimum, proportion of
    % extrema, and measures of periodicity in the positions of local maxima.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, maxTau=None, numBins=None, versionTwo=None):
        super(CO_TSTL_amutual, self).__init__()
        self.maxTau = maxTau
        self.numBins = numBins
        self.versionTwo = versionTwo

    def _eval_hook(self, eng, x):
        if self.maxTau is None:
            return eng.run_function(1, 'CO_TSTL_amutual', x, )
        elif self.numBins is None:
            return eng.run_function(1, 'CO_TSTL_amutual', x, self.maxTau)
        elif self.versionTwo is None:
            return eng.run_function(1, 'CO_TSTL_amutual', x, self.maxTau, self.numBins)
        return eng.run_function(1, 'CO_TSTL_amutual', x, self.maxTau, self.numBins, self.versionTwo)


class CO_TranslateShape(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    % geometric shapes moved across the time series.
    %
    % Inputs specify a shape and its size, and a method for moving this shape
    % through the time domain.
    %
    % This is usually more informative in an embedding space (CO_Embed2_...), but
    % here we do it just in the temporal domain (_t_).
    %
    % In the future, could perform a similar analysis with a soft boundary, some
    % decaying force function V(r), or perhaps truncated...?
    %
    %---INPUTS:
    % y, the input time series
    % shape, the shape to move about the time-domain ('circle')
    % d, a parameter specifying the size of the shape (e.g., d = 2)
    % howToMove, a method specifying how to move the shape about, e.g., 'pts'
    %               places the shape on each point in the time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (31, 31, 31, 33)

    TAGS = ('correlation',)

    def __init__(self, shape='circle', d=1.5, howToMove='pts'):
        super(CO_TranslateShape, self).__init__()
        self.shape = shape
        self.d = d
        self.howToMove = howToMove

    def _eval_hook(self, eng, x):
        if self.shape is None:
            return eng.run_function(1, 'CO_TranslateShape', x, )
        elif self.d is None:
            return eng.run_function(1, 'CO_TranslateShape', x, self.shape)
        elif self.howToMove is None:
            return eng.run_function(1, 'CO_TranslateShape', x, self.shape, self.d)
        return eng.run_function(1, 'CO_TranslateShape', x, self.shape, self.d, self.howToMove)


class CO_f1ecac(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds where autocorrelation function first crosses 1/e
    %
    %---INPUT:
    % y, the input time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (9,)

    TAGS = ('autocorrelation', 'correlation', 'tau')

    def __init__(self):
        super(CO_f1ecac, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'CO_f1ecac', x, )


class CO_fzcglscf(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Returns the first zero-crossing of the generalized self-correlation function
    % introduced in Duarte Queiros and Moyano in Physica A, Vol. 383, pp. 10--15
    % (2007) in the paper "Yet on statistical properties of traded volume:
    % Correlation and mutual information at different value magnitudes"
    % Uses CO_glscf to calculate the generalized self-correlations.
    % Keeps calculating until the function finds a minimum, and returns this lag.
    %
    %---INPUTS:
    % y, the input time series.
    % alpha, the parameter alpha.
    % beta, the parameter beta.
    % maxtau [opt], a maximum time delay to search up to (default is the time-series
    %                length).
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (17, 15, 16, 15, 15, 15, 15, 15, 15, 15, 15, 16, 15, 15, 16, 15)

    TAGS = ('correlation', 'glscf', 'tau')

    def __init__(self, alpha=1.0, beta=1.0, maxtau=None):
        super(CO_fzcglscf, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.maxtau = maxtau

    def _eval_hook(self, eng, x):
        if self.alpha is None:
            return eng.run_function(1, 'CO_fzcglscf', x, )
        elif self.beta is None:
            return eng.run_function(1, 'CO_fzcglscf', x, self.alpha)
        elif self.maxtau is None:
            return eng.run_function(1, 'CO_fzcglscf', x, self.alpha, self.beta)
        return eng.run_function(1, 'CO_fzcglscf', x, self.alpha, self.beta, self.maxtau)


class CO_glscf(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % This function was introduced in Queiros and Moyano in Physica A, Vol. 383, pp.
    % 10--15 (2007) in the paper "Yet on statistical properties of traded volume:
    % Correlation and mutual information at different value magnitudes"
    %
    % The function considers magnitude correlations.
    %
    %---INPUTS:
    % y, the input time series
    % alpha and beta are real and nonzero parameters
    % tau is the time-delay (can also be 'tau' to set to first zero-crossing of the ACF)
    %
    % When alpha = beta estimates how values of the same order of magnitude are
    % related in time
    % When alpha ~= beta, estimates correlations between different magnitudes of the
    % time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (15, 15, 15, 15, 15, 17, 14, 14, 14, 14, 14, 16, 14, 14, 14, 14, 14, 16, 14, 14, 14, 14, 14, 16, 14, 14, 14, 14, 14, 16, 14, 14, 14, 14, 14, 16)

    TAGS = ('correlation', 'glscf')

    def __init__(self, alpha=1.0, beta=1.0, tau=1.0):
        super(CO_glscf, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.alpha is None:
            return eng.run_function(1, 'CO_glscf', x, )
        elif self.beta is None:
            return eng.run_function(1, 'CO_glscf', x, self.alpha)
        elif self.tau is None:
            return eng.run_function(1, 'CO_glscf', x, self.alpha, self.beta)
        return eng.run_function(1, 'CO_glscf', x, self.alpha, self.beta, self.tau)


class CO_tc3(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Computes the tc3 function, a normalized nonlinear autocorrelation, at a
    % given time-delay, tau.
    %
    %---INPUTS:
    % y, input time series
    % tau, time lag
    %
    %---OUTPUTS:
    % The raw tc3 expression, its magnitude, the numerator and its magnitude, and
    % the denominator.
    %
    % See documentation of the TSTOOL package (http://www.physik3.gwdg.de/tstool/)
    % for further details about this function.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (8, 8, 8, 9, 9)

    TAGS = ('autocorrelation', 'correlation', 'nonlinear')

    def __init__(self, tau=1.0):
        super(CO_tc3, self).__init__()
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_tc3', x, )
        return eng.run_function(1, 'CO_tc3', x, self.tau)


class CO_trev(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Calculates the trev function, a normalized nonlinear autocorrelation,
    % mentioned in the documentation of the TSTOOL nonlinear time-series analysis
    % package (available here: http://www.physik3.gwdg.de/tstool/).
    %
    % The quantity is often used as a nonlinearity statistic in surrogate data
    % analysis, cf. "Surrogate time series", T. Schreiber and A. Schmitz, Physica D,
    % 142(3-4) 346 (2000).
    %
    %---INPUTS:
    %
    % y, time series
    %
    % tau, time lag (can be 'ac' or 'mi' to set as the first zero-crossing of the
    %       autocorrelation function, or the first minimum of the automutual
    %       information function, respectively)
    %
    %---OUTPUTS:
    % the raw trev expression, its magnitude, the numerator and its magnitude, and
    % the denominator.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (9, 9, 9, 10, 10)

    TAGS = ('autocorrelation', 'correlation', 'nonlinear')

    def __init__(self, tau=1.0):
        super(CO_trev, self).__init__()
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'CO_trev', x, )
        return eng.run_function(1, 'CO_trev', x, self.tau)


class CP_ML_StepDetect(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Gives information about discrete steps in the signal, using the function
    % l1pwc from Max A. Little's step detection toolkit.
    %
    % cf.,
    % "Sparse Bayesian Step-Filtering for High-Throughput Analysis of Molecular
    % Machine Dynamics", Max A. Little, and Nick S. Jones, Proc. ICASSP (2010)
    %
    % "Steps and bumps: precision extraction of discrete states of molecular machines"
    % M. A. Little, B. C. Steel, F. Bai, Y. Sowa, T. Bilyard, D. M. Mueller,
    % R. M. Berry, N. S. Jones. Biophysical Journal, 101(2):477-485 (2011)
    %
    % Software available at: http://www.maxlittle.net/software/index.php
    %
    %---INPUTS:
    % y, the input time series
    %
    % method, the step-detection method:
    %           (i) 'kv': Kalafut-Visscher
    %                 cf. The algorithm described in:
    %                 Kalafut, Visscher, "An objective, model-independent method for
    %                 detection of non-uniform steps in noisy signals", Comp. Phys.
    %                 Comm., 179(2008), 716-723.
    %
    %           (ii) 'l1pwc': L1 method
    %                 This code is based on code originally written by Kim et al.:
    %                 "l_1 Trend Filtering", S.-J. Kim et al., SIAM Review 51, 339
    %                 (2009).
    %
    % params, the parameters for the given method used:
    %           (i) 'kv': (no parameters required)
    %           (ii) 'l1pwc': params = lambda
    %
    %---OUTPUTS:
    % Statistics on the output of the step-detection method, including the intervals
    % between change points, the proportion of constant segments, the reduction in
    % variance from removing the piece-wise constants, and stationarity in the
    % occurrence of change points.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (26, 25, 25)

    TAGS = ('l1pwc', 'lengthdep', 'stepdetection')

    def __init__(self, method='l1pwc', params=10.0):
        super(CP_ML_StepDetect, self).__init__()
        self.method = method
        self.params = params

    def _eval_hook(self, eng, x):
        if self.method is None:
            return eng.run_function(1, 'CP_ML_StepDetect', x, )
        elif self.params is None:
            return eng.run_function(1, 'CP_ML_StepDetect', x, self.method)
        return eng.run_function(1, 'CP_ML_StepDetect', x, self.method, self.params)


class CP_l1pwc_sweep_lambda(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Gives information about discrete steps in the signal across a range of
    % regularization parameters lambda, using the function l1pwc from Max Little's
    % step detection toolkit.
    %
    % cf.,
    % "Sparse Bayesian Step-Filtering for High-Throughput Analysis of Molecular
    % Machine Dynamics", Max A. Little, and Nick S. Jones, Proc. ICASSP (2010)
    %
    %---INPUTS:
    % y, the input time series
    %
    % lambdar, a vector specifying the lambda parameters to use
    %
    %---OUTPUTS:
    % At each iteration, the CP_ML_StepDetect code was run with a given
    % lambda, and the number of segments, and reduction in root mean square error
    % from removing the piecewise constants was recorded. Outputs summarize how the
    % these quantities vary with lambda.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (31,)

    TAGS = ('l1pwc', 'stepdetection')

    def __init__(self, lambdar=MatlabSequence('0:0.05:0.95')):
        super(CP_l1pwc_sweep_lambda, self).__init__()
        self.lambdar = lambdar

    def _eval_hook(self, eng, x):
        if self.lambdar is None:
            return eng.run_function(1, 'CP_l1pwc_sweep_lambda', x, )
        return eng.run_function(1, 'CP_l1pwc_sweep_lambda', x, self.lambdar)


class CP_wavelet_varchg(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Finds variance change points using functions from Matlab's Wavelet Toolbox,
    % including the primary function wvarchg, which estimates the change points in
    % the time series.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    %
    % level, the level of wavelet decomposition
    %
    % maxnchpts, the maximum number of change points
    %
    % minDelay, the minimum delay between consecutive change points (can be
    %           specified as a proportion of the time-series length, e.g., 0.02
    %           ensures that change points are separated by at least 2% of the
    %           time-series length)
    %
    %
    %---OUTPUT:
    % The optimal number of change points.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (30, 30, 30, 30, 31, 31, 31, 31)

    TAGS = ('varchg', 'wavelet', 'waveletTB')

    def __init__(self, wname='db3', level=2.0, maxnchpts=10.0, minDelay=0.01):
        super(CP_wavelet_varchg, self).__init__()
        self.wname = wname
        self.level = level
        self.maxnchpts = maxnchpts
        self.minDelay = minDelay

    def _eval_hook(self, eng, x):
        if self.wname is None:
            return eng.run_function(1, 'CP_wavelet_varchg', x, )
        elif self.level is None:
            return eng.run_function(1, 'CP_wavelet_varchg', x, self.wname)
        elif self.maxnchpts is None:
            return eng.run_function(1, 'CP_wavelet_varchg', x, self.wname, self.level)
        elif self.minDelay is None:
            return eng.run_function(1, 'CP_wavelet_varchg', x, self.wname, self.level, self.maxnchpts)
        return eng.run_function(1, 'CP_wavelet_varchg', x, self.wname, self.level, self.maxnchpts, self.minDelay)


class DN_Burstiness(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Returns the 'burstiness' statistic from:
    %
    % Goh and Barabasi, 'Burstiness and memory in complex systems' Europhys. Lett.
    % 81, 48002 (2008)
    %
    %---INPUT:
    % y, the input time series
    %
    %---OUTPUT:
    % The burstiness statistic, B.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13,)

    TAGS = ('distribution', 'locdep', 'raw')

    def __init__(self):
        super(DN_Burstiness, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'DN_Burstiness', x, )


class DN_CompareKSFit(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Returns simple statistics on the discrepancy between the
    % kernel-smoothed distribution of the time-series values, and the distribution
    % fitted to it by some model: Gaussian (using normfifit from Matlab's
    % Statistics Toolbox), Extreme Value (evfifit), Uniform (unififit), Beta
    % (betafifit), Rayleigh (raylfifit), Exponential (expfifit), Gamma (gamfit),
    % LogNormal (lognfifit), and Weibull (wblfifit).
    %
    %---INPUTS:
    % x, the input data vector
    % whatDistn, the type of distribution to fit to the data:
    %           'norm' (normal), 'ev' (extreme value), 'uni' (uniform),
    %           'beta' (Beta), 'rayleigh' (Rayleigh), 'exp' (exponential),
    %           'gamma' (Gamma), 'logn' (Log-Normal), 'wbl' (Weibull).
    %
    %---OUTPUTS: include the absolute area between the two distributions, the peak
    % separation, overlap integral, and relative entropy.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (20, 18, 19, 21, 20, 20, 24, 19, 19)

    TAGS = ('beta', 'distribution', 'ev', 'exp', 'gamma', 'ksdensity', 'locdep', 'lognormal'
            'norm', 'posOnly', 'raw', 'rayleigh', 'spreaddep', 'uni', 'weibull')

    def __init__(self, whatDistn='norm'):
        super(DN_CompareKSFit, self).__init__()
        self.whatDistn = whatDistn

    def _eval_hook(self, eng, x):
        if self.whatDistn is None:
            return eng.run_function(1, 'DN_CompareKSFit', x, )
        return eng.run_function(1, 'DN_CompareKSFit', x, self.whatDistn)


class DN_Cumulants(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Very simple function that uses the skewness and kurtosis functions in
    % Matlab's Statistics Toolbox to calculate these higher order moments of input
    % time series, y
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % cumWhatMay, the type of higher order moment:
    %           (i) 'skew1', skewness
    %           (ii) 'skew2', skewness correcting for bias
    %           (iii) 'kurt1', kurtosis
    %           (iv) 'kurt2', kurtosis correcting for bias
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, cumWhatMay=None):
        super(DN_Cumulants, self).__init__()
        self.cumWhatMay = cumWhatMay

    def _eval_hook(self, eng, x):
        if self.cumWhatMay is None:
            return eng.run_function(1, 'DN_Cumulants', x, )
        return eng.run_function(1, 'DN_Cumulants', x, self.cumWhatMay)


class DN_CustomSkewness(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Compute the Pearson or Bowley skewness
    %
    %---INPUTS:
    % y, the input time series
    %
    % whatSkew, the skewness measure to calculate, either 'pearson' or 'bowley'
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (24, 25)

    TAGS = ('distribution', 'locdep', 'moment', 'raw', 'shape')

    def __init__(self, whatSkew='pearson'):
        super(DN_CustomSkewness, self).__init__()
        self.whatSkew = whatSkew

    def _eval_hook(self, eng, x):
        if self.whatSkew is None:
            return eng.run_function(1, 'DN_CustomSkewness', x, )
        return eng.run_function(1, 'DN_CustomSkewness', x, self.whatSkew)


class DN_FitKernelSmooth(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % x, the input data vector
    % <can also produce additional outputs with the following optional settings>
    % [opt] 'numcross': number of times the distribution crosses the given threshold
    %           e.g., usage: DN_FitKernelSmooth(x,'numcross',[0.5,0.7]) for
    %                        thresholds of 0.5 and 0.7
    % [opt] 'area': area under where the distribution crosses the given thresholds.
    %               Usage as for 'numcross' above
    % [opt] 'arclength': arclength between where the distribution passes given
    %       thresholds. Usage as above.
    %
    %---EXAMPLE USAGE:
    % DN_FitKernelSmooth(x,'numcross',[0.05,0.1],'area',[0.1,0.2,0.4],'arclength',[0.5,1,2])
    % returns all the basic outputs, plus those for numcross, area, and arclength
    % for the thresholds given
    %
    %---OUTPUTS: a set of statistics summarizing the obtained distribution, including
    % the number of peaks, the distributional entropy, the number of times the curve
    % crosses fifixed probability thresholds, the area under the curve for fifixed
    % probability thresholds, the arc length, and the symmetry of probability
    % density above and below the mean.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21, 24)

    TAGS = ('arclength', 'areaconst', 'crossconst', 'distribution', 'entropy', 'ksdensity'
            'raw', 'spreaddep', 'symmetry')

    def __init__(self, varargin='numcross'):
        super(DN_FitKernelSmooth, self).__init__()
        self.varargin = varargin

    def _eval_hook(self, eng, x):
        if self.varargin is None:
            return eng.run_function(1, 'DN_FitKernelSmooth', x, )
        return eng.run_function(1, 'DN_FitKernelSmooth', x, self.varargin + ('_celltrick_',) if isinstance(self.varargin,
                                tuple) else (self.varargin, '_celltrick_'))


class DN_Fit_mle(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Fits either a Gaussian, Uniform, or Geometric distribution to the data using
    % maximum likelihood estimation via the Matlab function mle
    % from the Statistics Toolbox.
    %
    %---INPUTS:
    % y, the input data vector.
    % fitWhat, the type of fit to do: 'gaussian', 'uniform', or 'geometric'.
    %
    %---OUTPUTS: parameters from the fit.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (20,)

    TAGS = ('distribution', 'fit')

    def __init__(self, fitWhat='geometric'):
        super(DN_Fit_mle, self).__init__()
        self.fitWhat = fitWhat

    def _eval_hook(self, eng, x):
        if self.fitWhat is None:
            return eng.run_function(1, 'DN_Fit_mle', x, )
        return eng.run_function(1, 'DN_Fit_mle', x, self.fitWhat)


class DN_HighLowMu(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The highlowmu statistic is the ratio of the mean of the data that is above the
    % (global) mean compared to the mean of the data that is below the global mean.
    %
    %---INPUTS:
    % y, the input data vector
    
    %---NOTES:
    % Somehow measures the same information as SB_MotifTwo(y,'mean') -> u, i.e.,
    % contains the same information as the proportion of the data that is above the
    % mean. This indicates that you cannot independently control the proportion of
    % data that is above the mean and the ratio of the means of the data above and
    % below the mean. This is not immediately obvious...
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (12,)

    TAGS = ('distribution', 'locdep', 'raw', 'spreaddep')

    def __init__(self):
        super(DN_HighLowMu, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'DN_HighLowMu', x, )


class DN_HistogramMode(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Measures the mode of the data vector using histograms with a given number
    % of bins.
    %
    %---INPUTS:
    %
    % y, the input data vector
    %
    % numBins, the number of bins to use in the histogram.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (19, 19, 18)

    TAGS = ('distribution', 'location')

    def __init__(self, numBins=5.0):
        super(DN_HistogramMode, self).__init__()
        self.numBins = numBins

    def _eval_hook(self, eng, x):
        if self.numBins is None:
            return eng.run_function(1, 'DN_HistogramMode', x, )
        return eng.run_function(1, 'DN_HistogramMode', x, self.numBins)


class DN_Mean(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    %
    % y, the input data vector
    %
    % meanType, (i) 'norm' or 'arithmetic', arithmetic mean
    %           (ii) 'median', median
    %           (iii) 'geom', geometric mean
    %           (iv) 'harm', harmonic mean
    %           (v) 'rms', root-mean-square
    %           (vi) 'iqm', interquartile mean
    %           (vii) 'midhinge', midhinge
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (8, 7, 9, 11, 6)

    TAGS = ('distribution', 'location', 'locdep', 'raw', 'spreaddep')

    def __init__(self, meanType='norm'):
        super(DN_Mean, self).__init__()
        self.meanType = meanType

    def _eval_hook(self, eng, x):
        if self.meanType is None:
            return eng.run_function(1, 'DN_Mean', x, )
        return eng.run_function(1, 'DN_Mean', x, self.meanType)


class DN_MinMax(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    %
    % y, the input data vector
    %
    % minOrMax, either 'min' or 'max' to return either the minimum or maximum of y
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (6, 6)

    TAGS = ('distribution',)

    def __init__(self, minOrMax='max'):
        super(DN_MinMax, self).__init__()
        self.minOrMax = minOrMax

    def _eval_hook(self, eng, x):
        if self.minOrMax is None:
            return eng.run_function(1, 'DN_MinMax', x, )
        return eng.run_function(1, 'DN_MinMax', x, self.minOrMax)


class DN_Moments(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Normalizes by the standard deviation
    % Uses the moment function from Matlab's Statistics Toolbox
    %
    %---INPUTS:
    % y, the input data vector
    % theMom, the moment to calculate (a scalar)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13, 13, 12, 12, 12, 12, 12, 12, 12, 17, 17, 16, 16, 16, 16, 16, 16, 16)

    TAGS = ('distribution', 'moment', 'raw', 'shape', 'spreaddep')

    def __init__(self, theMom=3.0):
        super(DN_Moments, self).__init__()
        self.theMom = theMom

    def _eval_hook(self, eng, x):
        if self.theMom is None:
            return eng.run_function(1, 'DN_Moments', x, )
        return eng.run_function(1, 'DN_Moments', x, self.theMom)


class DN_OutlierInclude(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Measures a range of different statistics about the time series as more and
    % more outliers are included in the calculation according to a specified rule:
    %
    % (i) 'abs': outliers are furthest from the mean,
    % (ii) 'p': outliers are the greatest positive deviations from the mean, or
    % (iii) 'n': outliers are the greatest negative deviations from the mean.
    %
    % The threshold for including time-series data points in the analysis increases
    % from zero to the maximum deviation, in increments of 0.01*sigma (by default),
    % where sigma is the standard deviation of the time series.
    %
    % At each threshold, the mean, standard error, proportion of time series points
    % included, median, and standard deviation are calculated, and outputs from the
    % algorithm measure how these statistical quantities change as more extreme
    % points are included in the calculation.
    %
    %---INPUTS:
    % y, the input time series (ideally z-scored)
    %
    % thresholdHow, the method of how to determine outliers: 'abs', 'p', or 'n' (see above
    %           for descriptions)
    %
    % inc, the increment to move through (fraction of std if input time series is
    %       z-scored)
    %
    % Most of the outputs measure either exponential, i.e., f(x) = Aexp(Bx)+C, or
    % linear, i.e., f(x) = Ax + B, fits to the sequence of statistics obtained in
    % this way.
    %
    % [future: could compare differences in outputs obtained with 'p', 'n', and
    %               'abs' -- could give an idea as to asymmetries/nonstationarities??]
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21, 19, 19)

    TAGS = ('distribution', 'outliers')

    def __init__(self, thresholdHow='abs', inc=None):
        super(DN_OutlierInclude, self).__init__()
        self.thresholdHow = thresholdHow
        self.inc = inc

    def _eval_hook(self, eng, x):
        if self.thresholdHow is None:
            return eng.run_function(1, 'DN_OutlierInclude', x, )
        elif self.inc is None:
            return eng.run_function(1, 'DN_OutlierInclude', x, self.thresholdHow)
        return eng.run_function(1, 'DN_OutlierInclude', x, self.thresholdHow, self.inc)


class DN_OutlierTest(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Removes the p% of highest and lowest values in the time series (i.e., 2*p%
    % removed from the time series in total) and returns the ratio of either the
    % mean or the standard deviation of the time series, before and after this
    % transformation.
    %
    %---INPUTS:
    % y, the input data vector
    % p, the percentage of values to remove beyond upper and lower percentiles
    % justMe [opt], just returns a number:
    %               (i) 'mean' -- returns the mean of the middle portion of the data
    %               (ii) 'std' -- returns the std of the middle portion of the data
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (16, 15, 15)

    TAGS = ('distribution', 'outliers', 'spread')

    def __init__(self, p=2.0, justMe=None):
        super(DN_OutlierTest, self).__init__()
        self.p = p
        self.justMe = justMe

    def _eval_hook(self, eng, x):
        if self.p is None:
            return eng.run_function(1, 'DN_OutlierTest', x, )
        elif self.justMe is None:
            return eng.run_function(1, 'DN_OutlierTest', x, self.p)
        return eng.run_function(1, 'DN_OutlierTest', x, self.p, self.justMe)


class DN_ProportionValues(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Returns statistics on the values of the data vector: the proportion of zeros,
    % the proportion of positive values, and the proportion of values greater than or
    % equal to zero.
    %
    %---INPUTS:
    % x, the input time series
    %
    % propWhat, the proportion of a given type of value in the time series:
    %           (i) 'zeros': values that equal zero
    %           (ii) 'positive': values that are strictly positive
    %           (iii) 'geq0': values that are greater than or equal to zero
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (24, 28, 25)

    TAGS = ('distribution', 'locdep', 'raw')

    def __init__(self, propWhat='zeros'):
        super(DN_ProportionValues, self).__init__()
        self.propWhat = propWhat

    def _eval_hook(self, eng, x):
        if self.propWhat is None:
            return eng.run_function(1, 'DN_ProportionValues', x, )
        return eng.run_function(1, 'DN_ProportionValues', x, self.propWhat)


class DN_Quantile(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Calculates the quantile value at a specified proportion, p, using the
    % Statistics Toolbox function, quantile.
    %
    %---INPUTS:
    % y, the input data vector
    % p, the quantile proportion
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13, 14, 13, 14, 13, 14, 13, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14)

    TAGS = ('distribution',)

    def __init__(self, p=0.01):
        super(DN_Quantile, self).__init__()
        self.p = p

    def _eval_hook(self, eng, x):
        if self.p is None:
            return eng.run_function(1, 'DN_Quantile', x, )
        return eng.run_function(1, 'DN_Quantile', x, self.p)


class DN_RemovePoints(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % A proportion, p, of points are removed from the time series according to some
    % rule, and a set of statistics are computed before and after the change.
    %
    %---INPUTS:
    % y, the input time series
    % removeHow, how to remove points from the time series:
    %               (i) 'absclose': those that are the closest to the mean,
    %               (ii) 'absfar': those that are the furthest from the mean,
    %               (iii) 'min': the lowest values,
    %               (iv) 'max': the highest values,
    %               (v) 'random': at random.
    %
    % p, the proportion of points to remove
    %
    %---OUTPUTS: Statistics include the change in autocorrelation, time scales, mean,
    % spread, and skewness.
    %
    % NOTE: This is a similar idea to that implemented in DN_OutlierInclude.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (27, 27, 27, 25, 25, 22, 22, 22, 22, 22)

    TAGS = ('correlation', 'distribution', 'outliers')

    def __init__(self, removeHow='absclose', p=0.1):
        super(DN_RemovePoints, self).__init__()
        self.removeHow = removeHow
        self.p = p

    def _eval_hook(self, eng, x):
        if self.removeHow is None:
            return eng.run_function(1, 'DN_RemovePoints', x, )
        elif self.p is None:
            return eng.run_function(1, 'DN_RemovePoints', x, self.removeHow)
        return eng.run_function(1, 'DN_RemovePoints', x, self.removeHow, self.p)


class DN_SimpleFit(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the 'fit' function from Matlab's Curve Fitting Toolbox.
    %
    % The distribution of time-series values is estimated using either a
    % kernel-smoothed density via the Matlab function ksdensity with the default
    % width parameter, or by a histogram with a specified number of bins, numBins.
    %
    %---INPUTS:
    % x, the input time series
    %
    % dmodel, the model to fit:
    %       (I) distribution models:
    %           (i) 'gauss1'
    %           (ii) 'gauss2'
    %           (iii) 'exp1'
    %           (iv) 'power1'
    %       (II) simple time-series models:
    %           (i) 'sin1'
    %           (ii) 'sin2'
    %           (iii) 'sin3'
    %           (iv) 'fourier1'
    %           (v) 'fourier2'
    %           (vi) 'fourier3'
    %
    % numBins, the number of bins for a histogram-estimate of the distribution of
    %       time-series values. If numBins = 0, uses ksdensity instead of histogram.
    %
    %
    %---OUTPUTS: the goodness of fifit, R^2, rootmean square error, the
    % autocorrelation of the residuals, and a runs test on the residuals.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (23, 20, 25, 22, 25, 22, 25, 17, 17, 17)

    TAGS = ('distribution', 'exp1', 'gauss1', 'gauss2', 'gof', 'model', 'posOnly', 'power1'
            'raw', 'sin1', 'sin2', 'sin3')

    def __init__(self, dmodel='gauss1', numBins=0.0):
        super(DN_SimpleFit, self).__init__()
        self.dmodel = dmodel
        self.numBins = numBins

    def _eval_hook(self, eng, x):
        if self.dmodel is None:
            return eng.run_function(1, 'DN_SimpleFit', x, )
        elif self.numBins is None:
            return eng.run_function(1, 'DN_SimpleFit', x, self.dmodel)
        return eng.run_function(1, 'DN_SimpleFit', x, self.dmodel, self.numBins)


class DN_Spread(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Returns the spread of the raw data vector, as the standard deviation,
    % inter-quartile range, mean absolute deviation, or median absolute deviation.
    %
    %---INPUTS:
    % y, the input data vector
    %
    % spreadMeasure, the spead measure:
    %               (i) 'std': standard deviation
    %               (ii) 'iqr': interquartile range
    %               (iii) 'mad': mean absolute deviation
    %               (iv) 'mead': median absolute deviation
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13, 13, 14, 13)

    TAGS = ('distribution', 'raw', 'spread', 'spreaddep')

    def __init__(self, spreadMeasure='std'):
        super(DN_Spread, self).__init__()
        self.spreadMeasure = spreadMeasure

    def _eval_hook(self, eng, x):
        if self.spreadMeasure is None:
            return eng.run_function(1, 'DN_Spread', x, )
        return eng.run_function(1, 'DN_Spread', x, self.spreadMeasure)


class DN_TrimmedMean(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, the input time series
    % n, the percent of highest and lowest values in y to exclude from the mean
    %     calculation
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (16, 17, 17, 16, 17)

    TAGS = ('distribution', 'location', 'locdep', 'raw')

    def __init__(self, n=1.0):
        super(DN_TrimmedMean, self).__init__()
        self.n = n

    def _eval_hook(self, eng, x):
        if self.n is None:
            return eng.run_function(1, 'DN_TrimmedMean', x, )
        return eng.run_function(1, 'DN_TrimmedMean', x, self.n)


class DN_Withinp(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % x, the input data vector
    % p, the number (proportion) of standard deviations.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13, 13, 13, 13, 13, 13)

    TAGS = ('distribution', 'spread')

    def __init__(self, p=0.5):
        super(DN_Withinp, self).__init__()
        self.p = p

    def _eval_hook(self, eng, x):
        if self.p is None:
            return eng.run_function(1, 'DN_Withinp', x, )
        return eng.run_function(1, 'DN_Withinp', x, self.p)


class DN_cv(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Coefficient of variation of order k is sigma^k / mu^k (for sigma, standard
    % deviation and mu, mean) of a data vector, x
    %
    %---INPUTS:
    %
    % x, the input data vector
    %
    % k, the order of coefficient of variation (k = 1 is default)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (7, 7)

    TAGS = ('cv', 'distribution', 'locdep', 'raw', 'spread', 'spreaddep')

    def __init__(self, k=1.0):
        super(DN_cv, self).__init__()
        self.k = k

    def _eval_hook(self, eng, x):
        if self.k is None:
            return eng.run_function(1, 'DN_cv', x, )
        return eng.run_function(1, 'DN_cv', x, self.k)


class DN_nlogL_norm(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Fits a Gaussian distribution to the data using the normfit function in
    % Matlab's Statistics Toolbox and returns the negative log likelihood of the
    % data coming from a Gaussian distribution using the normlike function.
    %
    %---INPUT:
    % y, a vector of data
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self):
        super(DN_nlogL_norm, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'DN_nlogL_norm', x, )


class DN_pleft(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Measures the maximum distance from the mean at which a given fixed proportion,
    % p, of the time-series data points are further.
    % Normalizes by the standard deviation of the time series
    % (could generalize to separate positive and negative deviations in future)
    % Uses the quantile function from Matlab's Statistics Toolbox
    %
    %---INPUTS:
    % y, the input data vector
    % th, the proportion of data further than p from the mean
    %           (output p, normalized by standard deviation)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (12, 11, 11, 11, 11, 11)

    TAGS = ('distribution', 'spread')

    def __init__(self, th=0.05):
        super(DN_pleft, self).__init__()
        self.th = th

    def _eval_hook(self, eng, x):
        if self.th is None:
            return eng.run_function(1, 'DN_pleft', x, )
        return eng.run_function(1, 'DN_pleft', x, self.th)


class DT_IsSeasonal(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Fits a 'sin1' model to the time series using fit function from the Curve Fitting
    % Toolbox. The output is binary: 1 if the goodness of fit, R^2, exceeds 0.3 and
    % the amplitude of the fitted periodic component exceeds 0.5, and 0 otherwise.
    %
    %---INPUTS:
    % y, the input time series
    %
    %---OUTPUT: Binary: 1 (= seasonal), 0 (= non-seasonal)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13,)

    TAGS = ('periodicity',)

    def __init__(self):
        super(DT_IsSeasonal, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'DT_IsSeasonal', x, )


class EN_ApEn(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % ApEn(m,r).
    %
    % cf. S. M. Pincus, "Approximate entropy as a measure of system complexity",
    % P. Natl. Acad. Sci. USA, 88(6) 2297 (1991)
    %
    % For more information, cf. http://physionet.org/physiotools/ApEn/
    %
    %---INPUTS:
    % y, the input time series
    % mnom, the embedding dimension
    % rth, the threshold for judging closeness/similarity
    %
    %---NOTES:
    % No record of where this was code was derived from :-/
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (8, 8, 8, 8)

    TAGS = ('entropy',)

    def __init__(self, mnom=1.0, rth=0.1):
        super(EN_ApEn, self).__init__()
        self.mnom = mnom
        self.rth = rth

    def _eval_hook(self, eng, x):
        if self.mnom is None:
            return eng.run_function(1, 'EN_ApEn', x, )
        elif self.rth is None:
            return eng.run_function(1, 'EN_ApEn', x, self.mnom)
        return eng.run_function(1, 'EN_ApEn', x, self.mnom, self.rth)


class EN_CID(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Estimates of 'complexity' of a time series as the stretched-out length of the
    % lines resulting from a line-graph of the time series.
    %
    % cf. Batista, G. E. A. P. A., Keogh, E. J., Tataw, O. M. & de Souza, V. M. A.
    % CID: an efficient complexity-invariant distance for time series. Data Min.
    % Knowl. Disc. 28, 634–669 (2014).
    %
    %---INPUTS:
    %
    % y, the input time series
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (6,)

    TAGS = ('complexity', 'distribution', 'entropy')

    def __init__(self):
        super(EN_CID, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'EN_CID', x, )


class EN_DistributionEntropy(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Estimates of entropy from the distribution of a data vector. The
    % distribution is estimated either using a histogram with numBins bins, or as a
    % kernel-smoothed distribution, using the ksdensity function from Matlab's
    % Statistics Toolbox with width parameter, w (specified as the iunput numBins).
    %
    % An optional additional parameter can be used to remove a proportion of the
    % most extreme positive and negative deviations from the mean as an initial
    % pre-processing.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % histOrKS: 'hist' for histogram, or 'ks' for ksdensity
    %
    % numBins: (*) (for 'hist'): an integer, uses a histogram with that many bins
    %          (*) (for 'ks'): a positive real number, for the width parameter for
    %                       ksdensity (can also be empty for default width
    %                                       parameter, optimum for Gaussian)
    %
    % olremp [opt]: the proportion of outliers at both extremes to remove
    %               (e.g., if olremp = 0.01; keeps only the middle 98% of data; 0
    %               keeps all data. This parameter ought to be less than 0.5, which
    %               keeps none of the data).
    %               If olremp is specified, returns the difference in entropy from
    %               removing the outliers.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (32, 34, 34, 34, 33, 33, 33, 32, 32, 31, 34, 36, 36, 36, 35, 35, 35, 32, 34, 37, 31, 30, 30, 29, 30, 30, 30, 29, 29, 29, 34, 34, 34, 33, 33, 33)

    TAGS = ('entropy', 'raw', 'spreaddep')

    def __init__(self, histOrKS='ks', numBins=0.01, olremp=0.0):
        super(EN_DistributionEntropy, self).__init__()
        self.histOrKS = histOrKS
        self.numBins = numBins
        self.olremp = olremp

    def _eval_hook(self, eng, x):
        if self.histOrKS is None:
            return eng.run_function(1, 'EN_DistributionEntropy', x, )
        elif self.numBins is None:
            return eng.run_function(1, 'EN_DistributionEntropy', x, self.histOrKS)
        elif self.olremp is None:
            return eng.run_function(1, 'EN_DistributionEntropy', x, self.histOrKS, self.numBins)
        return eng.run_function(1, 'EN_DistributionEntropy', x, self.histOrKS, self.numBins, self.olremp)


class EN_MS_LZcomplexity(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, the input time series
    % n, the (integer) number of bits to encode the data into
    % preProc [opt], first apply a given preProcessing to the time series. For now,
    %               just 'diff' is implemented, which zscores incremental
    %               differences and then applies the complexity method.
    %
    %---OUTPUT: the normalized Lempel-Ziv complexity: i.e., the number of distinct
    %           symbol sequences in the time series divided by the expected number
    %           of distinct symbols for a noise sequence.
    
    % Uses Michael Small's code: 'complexity' (renamed MS_complexity here).
    %
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Code is available at http://small.eie.polyu.edu.hk/matlab/
    %
    % The code is a wrapper for Michael Small's original code and uses the
    % associated mex file compiled from complexitybs.c (renamed MS_complexitybs.c
    % here).
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21, 26, 20, 25, 20, 25, 20, 25, 20, 25, 20, 25, 20, 25, 20, 25, 20, 25)

    TAGS = ('LempelZiv', 'MichaelSmall', 'complexity', 'mex')

    def __init__(self, n=2.0, preProc=array((), dtype=float64)):
        super(EN_MS_LZcomplexity, self).__init__()
        self.n = n
        self.preProc = preProc

    def _eval_hook(self, eng, x):
        if self.n is None:
            return eng.run_function(1, 'EN_MS_LZcomplexity', x, )
        elif self.preProc is None:
            return eng.run_function(1, 'EN_MS_LZcomplexity', x, self.n)
        return eng.run_function(1, 'EN_MS_LZcomplexity', x, self.n, self.preProc)


class EN_MS_shannon(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses an nbin-bin encoding and depth-symbol sequences.
    % Uniform population binning is used, and the implementation uses Michael Small's code
    % MS_shannon.m (renamed from the original, simply shannon.m)
    %
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Michael Small's code is available at available at http://small.eie.polyu.edu.hk/matlab/
    %
    % In this wrapper function, you can evaluate the code at a given n and d, and
    % also across a range of depth and nbin to return statistics on how the obtained
    % entropies change.
    %
    %---INPUTS:
    % y, the input time series
    % nbin, the number of bins to discretize the time series into (i.e., alphabet size)
    % depth, the length of strings to analyze
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (17, 14, 14, 17, 17, 17, 17, 14, 17)

    TAGS = ('MichaelSmall', 'entropy', 'mex', 'shannon')

    def __init__(self, nbin=2.0, depth=2.0):
        super(EN_MS_shannon, self).__init__()
        self.nbin = nbin
        self.depth = depth

    def _eval_hook(self, eng, x):
        if self.nbin is None:
            return eng.run_function(1, 'EN_MS_shannon', x, )
        elif self.depth is None:
            return eng.run_function(1, 'EN_MS_shannon', x, self.nbin)
        return eng.run_function(1, 'EN_MS_shannon', x, self.nbin, self.depth)


class EN_PermEn(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    % EN_PermEn     Permutation Entropy of a time series.
    %
    % "Permutation Entropy: A Natural Complexity Measure for Time Series"
    % C. Bandt and B. Pompe, Phys. Rev. Lett. 88(17) 174102 (2002)
    %
    %---INPUTS:
    % y, the input time series
    % m, the embedding dimension (or order of the permutation entropy)
    % tau, the time-delay for the embedding
    %
    %---OUTPUT:
    % Outputs the permutation entropy and normalized version computed according to
    % different implementations
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13, 13, 14, 13, 13, 14, 13, 13, 14, 13, 13, 14)

    TAGS = ('entropy',)

    def __init__(self, m=2.0, tau=1.0):
        super(EN_PermEn, self).__init__()
        self.m = m
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.m is None:
            return eng.run_function(1, 'EN_PermEn', x, )
        elif self.tau is None:
            return eng.run_function(1, 'EN_PermEn', x, self.m)
        return eng.run_function(1, 'EN_PermEn', x, self.m, self.tau)


class EN_RM_entropy(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Original code, now RM_entropy, was obtained from:
    % http://www.cs.rug.nl/~rudy/matlab/
    %
    % The above website has code and documentation for the function.
    %
    %---INPUTS:
    % y, the input time series
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (10,)

    TAGS = ('entropy',)

    def __init__(self):
        super(EN_RM_entropy, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'EN_RM_entropy', x, )


class EN_Randomize(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Progressively randomizes the input time series according to a specified
    % randomization procedure
    %
    % The procedure is repeated 2N times, where N is the length of the time series.
    %
    %---INPUTS:
    % y, the input (z-scored) time series
    %
    % randomizeHow, specifies the randomization scheme for each iteration:
    %      (i) 'statdist' -- substitutes a random element of the time series with
    %                           one from the original time-series distribution
    %      (ii) 'dyndist' -- overwrites a random element of the time
    %                       series with another random element
    %      (iii) 'permute' -- permutes pairs of elements of the time
    %                       series randomly
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %
    %---OUTPUTS: summarize how the properties change as one of these
    % randomization procedures is iterated, including the cross correlation with the
    % original time series, the autocorrelation of the randomized time series, its
    % entropy, and stationarity.
    %
    % These statistics are calculated every N/10 iterations, and thus 20 times
    % throughout the process in total.
    %
    % Most statistics measure how these properties decay with randomization, by
    % fitting a function f(x) = Aexp(Bx).
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (20, 20, 21)

    TAGS = ('entropy', 'lengthdep', 'slow')

    def __init__(self, randomizeHow='statdist', randomSeed='default'):
        super(EN_Randomize, self).__init__()
        self.randomizeHow = randomizeHow
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.randomizeHow is None:
            return eng.run_function(1, 'EN_Randomize', x, )
        elif self.randomSeed is None:
            return eng.run_function(1, 'EN_Randomize', x, self.randomizeHow)
        return eng.run_function(1, 'EN_Randomize', x, self.randomizeHow, self.randomSeed)


class EN_SampEn(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % SampEn(m,r), using code from PhysioNet.
    % Uses a compiled C version of the code if available, otherwise uses a (slower)
    % Matlab implementation (which can actually be faster for shorter time series
    % due to overheads of reading/writing to disk)
    %
    % The publicly-available PhysioNet Matlab code, sampenc (renamed here to
    % RN_sampenc) is available from:
    % http://www.physionet.org/physiotools/sampen/matlab/1.1/sampenc.m
    %
    % cf. "Physiological time-series analysis using approximate entropy and sample
    % entropy", J. S. Richman and J. R. Moorman, Am. J. Physiol. Heart Circ.
    % Physiol., 278(6) H2039 (2000)
    %
    % This function can also calculate the SampEn of successive increments of time
    % series, i.e., we using an incremental differencing pre-processing, as
    % used in the so-called Control Entropy quantity:
    %
    % "Control Entropy: A complexity measure for nonstationary signals"
    % E. M. Bollt and J. Skufca, Math. Biosci. Eng., 6(1) 1 (2009)
    %
    %---INPUTS:
    % y, the input time series
    % M, the embedding dimension
    % r, the threshold
    % preProcessHow [opt], (i) 'diff1', incremental differencing (as per 'Control Entropy').
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (15, 14, 15, 20, 14, 20, 14)

    TAGS = ('controlen', 'entropy', 'sampen')

    def __init__(self, M=5.0, r=0.05, preProcessHow=None):
        super(EN_SampEn, self).__init__()
        self.M = M
        self.r = r
        self.preProcessHow = preProcessHow

    def _eval_hook(self, eng, x):
        if self.M is None:
            return eng.run_function(1, 'EN_SampEn', x, )
        elif self.r is None:
            return eng.run_function(1, 'EN_SampEn', x, self.M)
        elif self.preProcessHow is None:
            return eng.run_function(1, 'EN_SampEn', x, self.M, self.r)
        return eng.run_function(1, 'EN_SampEn', x, self.M, self.r, self.preProcessHow)


class EN_mse(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % As per "Multiscale entropy analysis of biological signals",
    % Costa, Goldberger and Peng, PRE, 71, 021906 (2005)
    % http://physionet.comp.nus.edu.sg/physiotools/mse/papers/pre-2005.pdf
    %
    %---INPUTS:
    % scaleRange: a vector of scales (default: 1:10)
    % m: embedding dimension/length of sequence to match (default: 2)
    % r: similarity threshold for matching (default: 0.15)
    % preProcessHow: how to preprocess the data (default: do not)
    %
    %
    % Original C implementation and docs here:
    % http://physionet.org/physiotools/mse/tutorial/node3.html
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (17, 23, 29)

    TAGS = ('controlen', 'entropy', 'mse', 'sampen')

    def __init__(self, scaleRange=MatlabSequence('1:10'), m=2.0, r=0.15, preProcessHow=None):
        super(EN_mse, self).__init__()
        self.scaleRange = scaleRange
        self.m = m
        self.r = r
        self.preProcessHow = preProcessHow

    def _eval_hook(self, eng, x):
        if self.scaleRange is None:
            return eng.run_function(1, 'EN_mse', x, )
        elif self.m is None:
            return eng.run_function(1, 'EN_mse', x, self.scaleRange)
        elif self.r is None:
            return eng.run_function(1, 'EN_mse', x, self.scaleRange, self.m)
        elif self.preProcessHow is None:
            return eng.run_function(1, 'EN_mse', x, self.scaleRange, self.m, self.r)
        return eng.run_function(1, 'EN_mse', x, self.scaleRange, self.m, self.r, self.preProcessHow)


class EN_rpde(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    % EN_rpde   Recurrence period density entropy (RPDE).
    %
    % Fast RPDE analysis on an input signal to obtain an estimate of the H_norm value
    % and other related statistics.
    %
    % Based on Max Little's code rpde (see below)
    %
    %---USAGE:
    % [H_norm, rpd] = rpde(x, m, tau)
    % [H_norm, rpd] = rpde(x, m, tau, epsilon)
    % [H_norm, rpd] = rpde(x, m, tau, epsilon, T_max)
    %
    %---INPUTS:
    %    x       - input signal: must be a row vector
    %    m       - embedding dimension
    %    tau     - embedding time delay
    %
    %---OPTIONAL INPUTS:
    %    epsilon - recurrence neighbourhood radius
    %              (If not specified, then a suitable value is chosen automatically)
    %    T_max   - maximum recurrence time
    %              (If not specified, then all recurrence times are returned)
    %---OUTPUTS:
    %    H_norm  - Estimated RPDE value
    %    rpd     - Estimated recurrence period density
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (11, 12)

    TAGS = ('entropy',)

    def __init__(self, m=3.0, tau=1.0, epsilon=None, T_max=None):
        super(EN_rpde, self).__init__()
        self.m = m
        self.tau = tau
        self.epsilon = epsilon
        self.T_max = T_max

    def _eval_hook(self, eng, x):
        if self.m is None:
            return eng.run_function(1, 'EN_rpde', x, )
        elif self.tau is None:
            return eng.run_function(1, 'EN_rpde', x, self.m)
        elif self.epsilon is None:
            return eng.run_function(1, 'EN_rpde', x, self.m, self.tau)
        elif self.T_max is None:
            return eng.run_function(1, 'EN_rpde', x, self.m, self.tau, self.epsilon)
        return eng.run_function(1, 'EN_rpde', x, self.m, self.tau, self.epsilon, self.T_max)


class EN_wentropy(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the wentropy function from Matlab's Wavelet toolbox.
    %
    %--INPUTS:
    % y, the input time series
    % whaten, the entropy type:
    %               'shannon',
    %               'logenergy',
    %               'threshold' (with a given threshold),
    %               'sure' (with a given parameter).
    %               (see the wentropy documentaiton for information)
    % p, the additional parameter needed for threshold and sure entropies
    %
    %---NOTE:
    % It seems likely that this implementation of wentropy is nonsense.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (20, 19)

    TAGS = ('entropy', 'shannon')

    def __init__(self, whaten='shannon', p=None):
        super(EN_wentropy, self).__init__()
        self.whaten = whaten
        self.p = p

    def _eval_hook(self, eng, x):
        if self.whaten is None:
            return eng.run_function(1, 'EN_wentropy', x, )
        elif self.p is None:
            return eng.run_function(1, 'EN_wentropy', x, self.whaten)
        return eng.run_function(1, 'EN_wentropy', x, self.whaten, self.p)


class EX_MovingThreshold(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Inspired by an idea contained in:
    % "Reactions to extreme events: Moving threshold model"
    % Altmann et al., Physica A 364, 435--444 (2006)
    %
    % This algorithm is based on this idea: it uses the occurrence of extreme events
    % to modify a hypothetical 'barrier' that classes new points as 'extreme' or not.
    % The barrier begins at sigma, and if the absolute value of the next data point
    % is greater than the barrier, the barrier is increased by a proportion 'a',
    % otherwise the position of the barrier is decreased by a proportion 'b'.
    %
    %---INPUTS:
    % y, the input (z-scored) time series
    % a, the barrier jump parameter (in extreme event)
    % b, the barrier decay proportion (in absence of extreme event)
    %
    %---OUTPUTS: the mean, spread, maximum, and minimum of the time series for the
    % barrier, the mean of the difference between the barrier and the time series
    % values, and statistics on the occurrence of 'kicks' (times at which the
    % threshold is modified), and by how much the threshold changes on average.
    %
    % In future could make a variant operation that optimizes a and b to minimize the
    % quantity meanqover/pkick (hugged the shape as close as possible with the
    % minimum number of kicks), and returns a and b...?
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (25, 24, 24, 23)

    TAGS = ('outliers',)

    def __init__(self, a=0.1, b=0.1):
        super(EX_MovingThreshold, self).__init__()
        self.a = a
        self.b = b

    def _eval_hook(self, eng, x):
        if self.a is None:
            return eng.run_function(1, 'EX_MovingThreshold', x, )
        elif self.b is None:
            return eng.run_function(1, 'EX_MovingThreshold', x, self.a)
        return eng.run_function(1, 'EX_MovingThreshold', x, self.a, self.b)


class FC_LocalSimple(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Simple predictors using the past trainLength values of the time series to
    % predict its next value.
    %
    %---INPUTS:
    % y, the input time series
    %
    % forecastMeth, the forecasting method:
    %          (i) 'mean': local mean prediction using the past trainLength time-series
    %                       values,
    %          (ii) 'median': local median prediction using the past trainLength
    %                         time-series values
    %          (iii) 'lfit': local linear prediction using the past trainLength
    %                         time-series values.
    %
    % trainLength, the number of time-series values to use to forecast the next value
    %
    %---OUTPUTS: the mean error, stationarity of residuals, Gaussianity of
    % residuals, and their autocorrelation structure.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (20, 20, 20, 20, 22, 20, 20, 20, 20, 22, 22, 22, 22)

    TAGS = ('forecasting',)

    def __init__(self, forecastMeth='mean', trainLength=1.0):
        super(FC_LocalSimple, self).__init__()
        self.forecastMeth = forecastMeth
        self.trainLength = trainLength

    def _eval_hook(self, eng, x):
        if self.forecastMeth is None:
            return eng.run_function(1, 'FC_LocalSimple', x, )
        elif self.trainLength is None:
            return eng.run_function(1, 'FC_LocalSimple', x, self.forecastMeth)
        return eng.run_function(1, 'FC_LocalSimple', x, self.forecastMeth, self.trainLength)


class FC_LoopLocalSimple(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Analyzes the outputs of FC_LocalSimple for a range of local window lengths, l.
    % Loops over the length of the data to use for FC_LocalSimple prediction
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % forecastMeth, the prediction method:
    %            (i) 'mean', local mean prediction
    %            (ii) 'median', local median prediction
    %
    %---OUTPUTS:
    % Statistics including whether the mean square error increases or decreases,
    % testing for peaks, variability, autocorrelation, stationarity, and a fit of
    % exponential decay, f(x) = A*exp(Bx) + C, to the variation.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (23,)

    TAGS = ('forecasting',)

    def __init__(self, forecastMeth='mean'):
        super(FC_LoopLocalSimple, self).__init__()
        self.forecastMeth = forecastMeth

    def _eval_hook(self, eng, x):
        if self.forecastMeth is None:
            return eng.run_function(1, 'FC_LoopLocalSimple', x, )
        return eng.run_function(1, 'FC_LoopLocalSimple', x, self.forecastMeth)


class FC_Surprise(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Coarse-grains the time series, turning it into a sequence of symbols of a
    % given alphabet size, numGroups, and quantifies measures of surprise of a
    % process with local memory of the past memory values of the symbolic string.
    %
    % We then consider a memory length, memory, of the time series, and
    % use the data in the proceeding memory samples to inform our expectations of
    % the following sample.
    %
    % The 'information gained', log(1/p), at each sample using expectations
    % calculated from the previous memory samples, is estimated.
    %
    %---INPUTS:
    % y, the input time series
    %
    % whatPrior, the type of information to store in memory:
    %           (i) 'dist': the values of the time series in the previous memory
    %                       samples,
    %           (ii) 'T1': the one-point transition probabilities in the previous
    %                       memory samples, and
    %           (iii) 'T2': the two-point transition probabilities in the previous
    %                       memory samples.
    %
    % memory, the memory length (either number of samples, or a proportion of the
    %           time-series length, if between 0 and 1)
    %
    % numGroups, the number of groups to coarse-grain the time series into
    %
    % cgmeth, the coarse-graining, or symbolization method:
    %          (i) 'quantile': an equiprobable alphabet by the value of each
    %                          time-series datapoint,
    %          (ii) 'updown': an equiprobable alphabet by the value of incremental
    %                         changes in the time-series values, and
    %          (iii) 'embed2quadrants': by the quadrant each data point resides in
    %                          in a two-dimensional embedding space.
    %
    % numIters, the number of iterations to repeat the procedure for.
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %
    %---OUTPUTS: summaries of this series of information gains, including the
    %            minimum, maximum, mean, median, lower and upper quartiles, and
    %            standard deviation.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (26, 28, 26, 28, 30, 32, 25, 27, 32, 25, 27, 32, 26, 28, 26, 28, 33, 25, 27, 25, 27, 32, 28, 30, 28, 30, 34, 27, 29, 34, 27, 29, 31, 26, 28, 33)

    TAGS = ('information', 'symbolic')

    def __init__(self, whatPrior='dist', memory=5.0, numGroups=2.0, cgmeth='quantile',
                 numIters=500.0, randomSeed='default'):
        super(FC_Surprise, self).__init__()
        self.whatPrior = whatPrior
        self.memory = memory
        self.numGroups = numGroups
        self.cgmeth = cgmeth
        self.numIters = numIters
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.whatPrior is None:
            return eng.run_function(1, 'FC_Surprise', x, )
        elif self.memory is None:
            return eng.run_function(1, 'FC_Surprise', x, self.whatPrior)
        elif self.numGroups is None:
            return eng.run_function(1, 'FC_Surprise', x, self.whatPrior, self.memory)
        elif self.cgmeth is None:
            return eng.run_function(1, 'FC_Surprise', x, self.whatPrior, self.memory, self.numGroups)
        elif self.numIters is None:
            return eng.run_function(1, 'FC_Surprise', x, self.whatPrior, self.memory, self.numGroups, self.cgmeth)
        elif self.randomSeed is None:
            return eng.run_function(1, 'FC_Surprise', x, self.whatPrior, self.memory, self.numGroups,
                                    self.cgmeth, self.numIters)
        return eng.run_function(1, 'FC_Surprise', x, self.whatPrior, self.memory, self.numGroups,
                                self.cgmeth, self.numIters, self.randomSeed)


class HT_DistributionTest(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Fits a distribution to the data and then performs an appropriate hypothesis
    % test to quantify the difference between the two distributions.
    %
    % We fit Gaussian, Extreme Value, Uniform, Beta, Rayleigh, Exponential, Gamma,
    % Log-Normal, and Weibull distributions, using code described for DN_M_kscomp.
    %
    %---INPUTS:
    % x, the input data vector
    % theTest, the hypothesis test to perform:
    %           (i) 'chi2gof': chi^2 goodness of fit test
    %           (ii) 'ks': Kolmogorov-Smirnov test
    %           (iii) 'lillie': Lilliefors test
    %
    % theDistn, the distribution to fit:
    %           (i) 'norm' (Normal)
    %           (ii) 'ev' (Extreme value)
    %           (iii) 'uni' (Uniform)
    %           (iv) 'beta' (Beta)
    %           (v) 'rayleigh' (Rayleigh)
    %           (vi) 'exp' (Exponential)
    %           (vii) 'gamma' (Gamma)
    %           (viii) 'logn' (Log-normal)
    %           (ix) 'wbl' (Weibull)
    %
    % numBins, the number of bins to use for the chi2 goodness of fit test
    %
    % All of these functions for hypothesis testing are implemented in Matlab's
    % Statistics Toolbox.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (30, 31, 30, 29, 30, 28, 29, 28, 27, 28, 29, 30, 29, 28, 29, 29, 30, 29, 28, 29, 30, 31, 30, 29, 30, 30, 31, 30, 29, 30, 30, 31, 30, 29, 30, 29, 30, 29, 28, 29, 29, 30, 29, 28, 29, 27, 25, 26, 26, 27, 27, 27, 26, 26, 29, 30, 31)

    TAGS = ('chi2gof', 'distribution', 'hypothesistest', 'ks', 'lillie', 'locdep', 'posOnly', 'raw')

    def __init__(self, theTest='chi2gof', theDistn='norm', numBins=5.0):
        super(HT_DistributionTest, self).__init__()
        self.theTest = theTest
        self.theDistn = theDistn
        self.numBins = numBins

    def _eval_hook(self, eng, x):
        if self.theTest is None:
            return eng.run_function(1, 'HT_DistributionTest', x, )
        elif self.theDistn is None:
            return eng.run_function(1, 'HT_DistributionTest', x, self.theTest)
        elif self.numBins is None:
            return eng.run_function(1, 'HT_DistributionTest', x, self.theTest, self.theDistn)
        return eng.run_function(1, 'HT_DistributionTest', x, self.theTest, self.theDistn, self.numBins)


class HT_HypothesisTest(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Tests are implemented as functions in Matlab's Statistics Toolbox.
    % (except Ljung-Box Q-test, which uses the Econometrics Toolbox)
    %
    %---INPUTS:
    % x, the input time series
    %
    % theTest, the hypothesis test to perform:
    %           (i) sign test ('signtest'),
    %           (ii) runs test ('runstest'),
    %           (iii) variance test ('vartest'),
    %           (iv) Z-test ('ztest'),
    %           (v) Wilcoxon signed rank test for a zero median ('signrank'),
    %           (vi) Jarque-Bera test of composite normality ('jbtest').
    %           (vii) Ljung-Box Q-test for residual autocorrelation ('lbq')
    %
    %---OUTPUT:
    % p-value from the specified statistical test
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (24, 25, 28, 26, 26, 26, 23)

    TAGS = ('econometricstoolbox', 'hypothesistest', 'randomness', 'raw', 'signtest')

    def __init__(self, theTest='signtest'):
        super(HT_HypothesisTest, self).__init__()
        self.theTest = theTest

    def _eval_hook(self, eng, x):
        if self.theTest is None:
            return eng.run_function(1, 'HT_HypothesisTest', x, )
        return eng.run_function(1, 'HT_HypothesisTest', x, self.theTest)


class IN_AutoMutualInfo(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    %
    % y: input time series
    %
    % timeDelay: time lag for automutual information calculation
    %
    % estMethod: the estimation method used to compute the mutual information:
    %           (*) 'gaussian'
    %           (*) 'kernel'
    %           (*) 'kraskov1'
    %           (*) 'kraskov2'
    %
    % cf. Kraskov, A., Stoegbauer, H., Grassberger, P., Estimating mutual
    % information: http://dx.doi.org/10.1103/PhysRevE.69.066138
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, timeDelay=None, estMethod=None, extraParam=None):
        super(IN_AutoMutualInfo, self).__init__()
        self.timeDelay = timeDelay
        self.estMethod = estMethod
        self.extraParam = extraParam

    def _eval_hook(self, eng, x):
        if self.timeDelay is None:
            return eng.run_function(1, 'IN_AutoMutualInfo', x, )
        elif self.estMethod is None:
            return eng.run_function(1, 'IN_AutoMutualInfo', x, self.timeDelay)
        elif self.extraParam is None:
            return eng.run_function(1, 'IN_AutoMutualInfo', x, self.timeDelay, self.estMethod)
        return eng.run_function(1, 'IN_AutoMutualInfo', x, self.timeDelay, self.estMethod, self.extraParam)


class IN_AutoMutualInfoStats(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, column vector of time series data
    %
    % maxTau, maximal time delay
    %
    % estMethod, extraParam -- cf. inputs to IN_AutoMutualInfo.m
    %
    %---OUTPUTS:
    % out, a structure containing statistics on the AMIs and their pattern across
    %       the range of specified time delays.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (34, 36, 39, 41)

    TAGS = ('AMI', 'correlation', 'information')

    def __init__(self, maxTau=40.0, estMethod='gaussian', extraParam=None):
        super(IN_AutoMutualInfoStats, self).__init__()
        self.maxTau = maxTau
        self.estMethod = estMethod
        self.extraParam = extraParam

    def _eval_hook(self, eng, x):
        if self.maxTau is None:
            return eng.run_function(1, 'IN_AutoMutualInfoStats', x, )
        elif self.estMethod is None:
            return eng.run_function(1, 'IN_AutoMutualInfoStats', x, self.maxTau)
        elif self.extraParam is None:
            return eng.run_function(1, 'IN_AutoMutualInfoStats', x, self.maxTau, self.estMethod)
        return eng.run_function(1, 'IN_AutoMutualInfoStats', x, self.maxTau, self.estMethod, self.extraParam)


class IN_Initialize_MI(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    %
    % estMethod: the estimation method used to compute the mutual information:
    %           (*) 'gaussian'
    %           (*) 'kernel'
    %           (*) 'kraskov1'
    %           (*) 'kraskov2'
    %
    % cf. Kraskov, A., Stoegbauer, H., Grassberger, P., Estimating mutual
    % information: http://dx.doi.org/10.1103/PhysRevE.69.066138
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, extraParam=None):
        super(IN_Initialize_MI, self).__init__()
        self.extraParam = extraParam

    def _eval_hook(self, eng, x):
        if self.extraParam is None:
            return eng.run_function(1, 'IN_Initialize_MI', x, )
        return eng.run_function(1, 'IN_Initialize_MI', x, self.extraParam)


class IN_MutualInfo(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the information dynamics toolkit implementation.
    %
    %---INPUTS:
    %
    % y1: input time series 1
    % y2: input time series 2
    %
    % estMethod: the estimation method used to compute the mutual information:
    %           (*) 'gaussian'
    %           (*) 'kernel'
    %           (*) 'kraskov1'
    %           (*) 'kraskov2'
    %
    % cf. Kraskov, A., Stoegbauer, H., Grassberger, P., Estimating mutual
    % information: http://dx.doi.org/10.1103/PhysRevE.69.066138
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, y2=None, estMethod=None, extraParam=None):
        super(IN_MutualInfo, self).__init__()
        self.y2 = y2
        self.estMethod = estMethod
        self.extraParam = extraParam

    def _eval_hook(self, eng, x):
        if self.y2 is None:
            return eng.run_function(1, 'IN_MutualInfo', x, )
        elif self.estMethod is None:
            return eng.run_function(1, 'IN_MutualInfo', x, self.y2)
        elif self.extraParam is None:
            return eng.run_function(1, 'IN_MutualInfo', x, self.y2, self.estMethod)
        return eng.run_function(1, 'IN_MutualInfo', x, self.y2, self.estMethod, self.extraParam)


class MD_hrv_classic(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Typically assumes an NN/RR time series in units of seconds.
    %
    %---INPUTS:
    % y, the input time series.
    %
    % Includes:
    %  (i) pNNx
    %  cf. "The pNNx files: re-examining a widely used heart rate variability
    %           measure", J.E. Mietus et al., Heart 88(4) 378 (2002)
    %
    %  (ii) Power spectral density ratios in different frequency ranges
    %   cf. "Heart rate variability: Standards of measurement, physiological
    %       interpretation, and clinical use",
    %       M. Malik et al., Eur. Heart J. 17(3) 354 (1996)
    %
    %  (iii) Triangular histogram index, and
    %
    %  (iv) Poincare plot measures
    %  cf. "Do existing measures of Poincare plot geometry reflect nonlinear
    %       features of heart rate variability?"
    %       M. Brennan, et al., IEEE T. Bio.-Med. Eng. 48(11) 1342 (2001)
    %
    % Code is heavily derived from that provided by Max A. Little:
    % http://www.maxlittle.net/
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (14,)

    TAGS = ('medical',)

    def __init__(self):
        super(MD_hrv_classic, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'MD_hrv_classic', x, )


class MD_pNN(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Applies pNNx measures to time series assumed to represent sequences of
    % consecutive RR intervals measured in milliseconds.
    %
    % cf. "The pNNx files: re-examining a widely used heart rate variability
    %           measure", J.E. Mietus et al., Heart 88(4) 378 (2002)
    %
    %---INPUTS:
    % x, the input time series
    %
    % This code is derived from MD_hrv_classic.m becuase it doesn't make medical
    % sense to do PNN on a z-scored time series.
    %
    % But now PSD doesn't make too much sense, so we just evaluate the pNN measures.
    %
    % Code is heavily derived from that provided by Max A. Little:
    % http://www.maxlittle.net/
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (10,)

    TAGS = ('medical', 'raw', 'spreaddep')

    def __init__(self):
        super(MD_pNN, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'MD_pNN', x, )


class MD_polvar(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Measures the probability of obtaining a sequence of consecutive ones or zeros.
    % 
    % The first mention may be in Wessel et al., PRE (2000), called Plvar
    % cf. "Short-term forecasting of life-threatening cardiac arrhythmias based on
    % symbolic dynamics and finite-time growth rates",
    %       N. Wessel et al., Phys. Rev. E 61(1) 733 (2000)
    %
    % Although the original measure used raw thresholds, d, on RR interval sequences
    % (measured in milliseconds), this code can be applied to general z-scored time
    % series. So now d is not the time difference in milliseconds, but in units of
    % std.
    %
    % The measure was originally applied to sequences of RR intervals and this code
    % is heavily derived from that provided by Max A. Little, January 2009.
    % cf. http://www.maxlittle.net/
    %
    %---INPUTS:
    % x, the input time series
    % d, the symbolic coding (amplitude) difference,
    % D, the word length (classically words of length 6).
    %
    %---OUPUT:
    % p - probability of obtaining a sequence of consecutive ones/zeros
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13)

    TAGS = ('medical', 'symbolic')

    def __init__(self, d=1.0, D=3.0):
        super(MD_polvar, self).__init__()
        self.d = d
        self.D = D

    def _eval_hook(self, eng, x):
        if self.d is None:
            return eng.run_function(1, 'MD_polvar', x, )
        elif self.D is None:
            return eng.run_function(1, 'MD_polvar', x, self.d)
        return eng.run_function(1, 'MD_polvar', x, self.d, self.D)


class MD_rawHRVmeas(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Computes the triangular histogram index and Poincare plot measures to a time
    % series assumed to measure sequences of consecutive RR intervals measured in
    % milliseconds. Doesn't make much sense for other time series
    %
    % cf. "Do existing measures of Poincare plot geometry reflect nonlinear
    %      features of heart rate variability?"
    %      M. Brennan, et al., IEEE T. Bio.-Med. Eng. 48(11) 1342 (2001)
    %
    % Note that pNNx is not done here, but in MD_pNN.m
    %
    % This code is heavily derived from Max Little's hrv_classic.m code
    % Max Little: http://www.maxlittle.net/
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13,)

    TAGS = ('medical', 'raw', 'spreaddep')

    def __init__(self):
        super(MD_rawHRVmeas, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'MD_rawHRVmeas', x, )


class MF_ARMA_orders(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Given a set of AR orders, p, and a set of MA orders, q, this operation fits
    % ARMA(p,q) models to the time series and evaluates the goodness of fit from all
    % combinations (p,q).
    %
    % Uses functions iddata, armax, and aic from Matlab's System Identification toolbox
    %
    %---INPUTS:
    % y, the input time series
    % pr, a vector specifying the range of AR model orders to analyze
    % qr, a vector specifying the range of MA model orders to analyze
    %
    %---OUTPUTS: statistics on the appropriateness of different types of models,
    % including the goodness of fit from the best model, and the optimal orders of
    % fitted ARMA(p,q) models.
    %
    % ** Future improvements **
    % (1) May want to quantify where a particular order starts to stand out, i.e.,
    % may be quite sensitive to wanting p > 2, but may be quite indiscriminate
    % when it comes to the MA order.
    % (2) May want to do some prediction and get more statistics on quality of
    % model rather than just in-sample FPE/AIC...
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (22,)

    TAGS = ('arma', 'model', 'systemidentificationtoolbox')

    def __init__(self, pr=MatlabSequence('1:6'), qr=MatlabSequence('1:4')):
        super(MF_ARMA_orders, self).__init__()
        self.pr = pr
        self.qr = qr

    def _eval_hook(self, eng, x):
        if self.pr is None:
            return eng.run_function(1, 'MF_ARMA_orders', x, )
        elif self.qr is None:
            return eng.run_function(1, 'MF_ARMA_orders', x, self.pr)
        return eng.run_function(1, 'MF_ARMA_orders', x, self.pr, self.qr)


class MF_AR_arcov(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses arcov code from Matlab's Signal Processing Toolbox.
    %
    %---INPUTS:
    % y, the input time series
    % p, the AR model order
    %
    %---OUTPUTS: include the parameters of the fitted model, the variance estimate
    % of a white noise input to the AR model, the root-mean-square (RMS) error of a
    % reconstructed time series, and the autocorrelation of residuals.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13, 13, 13, 13, 13)

    TAGS = ('ar', 'fit', 'gof', 'model')

    def __init__(self, p=1.0):
        super(MF_AR_arcov, self).__init__()
        self.p = p

    def _eval_hook(self, eng, x):
        if self.p is None:
            return eng.run_function(1, 'MF_AR_arcov', x, )
        return eng.run_function(1, 'MF_AR_arcov', x, self.p)


class MF_CompareAR(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses functions from Matlab's System Identification Toolbox: iddata, arxstruc,
    % and selstruc
    %
    %---INPUTS:
    % y, vector of time-series data
    % orders, a vector of possible model orders
    % testHow, specify a fraction, or provide a string 'all' to train and test on
    %            all the data
    %
    %---OUTPUTS: statistics on the loss at each model order, which are obtained by
    % applying the model trained on the training data to the testing data.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (20, 21)

    TAGS = ('ar', 'model', 'systemidentificationtoolbox')

    def __init__(self, orders=MatlabSequence('1:10'), testHow='all'):
        super(MF_CompareAR, self).__init__()
        self.orders = orders
        self.testHow = testHow

    def _eval_hook(self, eng, x):
        if self.orders is None:
            return eng.run_function(1, 'MF_CompareAR', x, )
        elif self.testHow is None:
            return eng.run_function(1, 'MF_CompareAR', x, self.orders)
        return eng.run_function(1, 'MF_CompareAR', x, self.orders, self.testHow)


class MF_CompareTestSets(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Robustness is quantified over different samples in the time series from
    % fitting a specified time-series model.
    %
    % Similar to MF_FitSubsegments, except fits the model on the full time
    % series and compares how well it predicts time series in different local
    % time-series segments.
    %
    % Says something of stationarity in spread of values, and something of the
    % suitability of model in level of values.
    %
    % Uses function iddata and predict from Matlab's System Identification Toolbox,
    % as well as either ar, n4sid, or armax from Matlab's System Identification
    % Toolbox to fit the models, depending on the specified model to fit to the data.
    %
    %---INPUTS:
    % y, the input time series
    %
    % theModel, the type of time-series model to fit:
    %           (i) 'ar', fits an AR model
    %           (ii) 'ss', first a state-space model
    %           (iii) 'arma', first an ARMA model
    %
    % ord, the order of the specified model to fit
    %
    % subsetHow, how to select random subsets of the time series to fit:
    %           (i) 'rand', select at random
    %           (ii) 'uniform', uniformly distributed segments throughout the time
    %                   series
    %
    % samplep, a two-vector specifying the sampling parameters
    %           e.g., [20, 0.1] repeats 20 times for segments 10% the length of the
    %                           time series
    %
    % steps, the number of steps ahead to do the predictions.
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %               (when 'rand' specified for subsetHow)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (38, 44, 41, 44)

    TAGS = ('ar', 'arfit', 'model', 'prediction', 'statespace', 'systemidentificationtoolbox')

    def __init__(self, theModel='ar', ordd=4.0, subsetHow='rand', samplep=array((25. ,
                 0.1)), steps=1.0, randomSeed='default'):
        super(MF_CompareTestSets, self).__init__()
        self.theModel = theModel
        self.ordd = ordd
        self.subsetHow = subsetHow
        self.samplep = samplep
        self.steps = steps
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.theModel is None:
            return eng.run_function(1, 'MF_CompareTestSets', x, )
        elif self.ordd is None:
            return eng.run_function(1, 'MF_CompareTestSets', x, self.theModel)
        elif self.subsetHow is None:
            return eng.run_function(1, 'MF_CompareTestSets', x, self.theModel, self.ordd)
        elif self.samplep is None:
            return eng.run_function(1, 'MF_CompareTestSets', x, self.theModel, self.ordd, self.subsetHow)
        elif self.steps is None:
            return eng.run_function(1, 'MF_CompareTestSets', x, self.theModel, self.ordd, self.subsetHow, self.samplep)
        elif self.randomSeed is None:
            return eng.run_function(1, 'MF_CompareTestSets', x, self.theModel, self.ordd, self.subsetHow,
                                    self.samplep, self.steps)
        return eng.run_function(1, 'MF_CompareTestSets', x, self.theModel, self.ordd, self.subsetHow,
                                self.samplep, self.steps, self.randomSeed)


class MF_ExpSmoothing(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Fits an exponential smoothing model to the time series using a training set to
    % fit the optimal smoothing parameter, alpha, and then applies the result to the
    % try to predict the rest of the time series.
    %
    % cf. "The Analysis of Time Series", C. Chatfield, CRC Press LLC (2004)
    %
    % Code is adapted from original code provided by Siddharth Arora:
    % Siddharth.Arora@sbs.ox.ac.uk
    %
    %---INPUTS:
    % x, the input time series
    %
    % ntrain, the number of samples to use for training (can be a proportion of the
    %           time-series length)
    %
    % alpha, the exponential smoothing parameter
    %
    %---OUTPUTS: include the fitted alpha, and statistics on the residuals from the
    % prediction phase.
    %
    % Future alteration could take a number of training sets and average to some
    % optimal alpha, for example, rather than just fitting it in an initial portion
    % of the time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (23,)

    TAGS = ('expsmoothing', 'model')

    def __init__(self, ntrain=0.5, alpha='best'):
        super(MF_ExpSmoothing, self).__init__()
        self.ntrain = ntrain
        self.alpha = alpha

    def _eval_hook(self, eng, x):
        if self.ntrain is None:
            return eng.run_function(1, 'MF_ExpSmoothing', x, )
        elif self.alpha is None:
            return eng.run_function(1, 'MF_ExpSmoothing', x, self.ntrain)
        return eng.run_function(1, 'MF_ExpSmoothing', x, self.ntrain, self.alpha)


class MF_FitSubsegments(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The spread of parameters obtained (including in-sample goodness of fit
    % statistics) provide some indication of stationarity.
    %
    % Values of goodness of fit provide some indication of model suitability.
    %
    % N.B., this code inherits strongly from this MF_CompareTestSets
    %
    %---INPUTS:
    % y, the input time series.
    %
    % model, the model to fit in each segments of the time series:
    %           'arsbc': fits an AR model using the ARfit package. Outputs
    %                       statistics are on how the optimal order, p_{opt}, and
    %                       the goodness of fit varies in different parts of the
    %                       time series.
    %           'ar': fits an AR model of a specified order using the code
    %                   ar from Matlab's System Identification Toolbox. Outputs are
    %                   on how Akaike's Final Prediction Error (FPE), and the fitted
    %                   AR parameters vary across the different segments of time
    %                   series.
    %           'ss': fits a state space model of a given order using the code
    %                   n4sid from Matlab's System Identification Toolbox. Outputs
    %                   are on how the FPE varies.
    %           'arma': fits an ARMA model using armax code from Matlab's System
    %                   Identification Toolbox. Outputs are statistics on the FPE,
    %                   and fitted AR and MA parameters.
    %
    % subsetHow, how to choose segments from the time series, either 'uniform'
    %               (uniformly) or 'rand' (at random).
    %
    % samplep, a two-vector specifying how many segments to take and of what length.
    %           Of the form [nsamples, length], where length can be a proportion of
    %           the time-series length. e.g., [20,0.1] takes 20 segments of 10% the
    %           time-series length.
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %               (for when subsetHow is 'rand')
    %
    %---OUTPUTS: depend on the model, as described above.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (36, 40, 34, 37, 36)

    TAGS = ('ar', 'arfit', 'arma', 'model', 'prediction', 'statespace', 'systemidentificationtoolbox')

    def __init__(self, model='arsbc', order=array((), dtype=float64),
                 subsetHow='uniform', samplep=array((25. ,  0.1)), randomSeed=None):
        super(MF_FitSubsegments, self).__init__()
        self.model = model
        self.order = order
        self.subsetHow = subsetHow
        self.samplep = samplep
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.model is None:
            return eng.run_function(1, 'MF_FitSubsegments', x, )
        elif self.order is None:
            return eng.run_function(1, 'MF_FitSubsegments', x, self.model)
        elif self.subsetHow is None:
            return eng.run_function(1, 'MF_FitSubsegments', x, self.model, self.order)
        elif self.samplep is None:
            return eng.run_function(1, 'MF_FitSubsegments', x, self.model, self.order, self.subsetHow)
        elif self.randomSeed is None:
            return eng.run_function(1, 'MF_FitSubsegments', x, self.model, self.order, self.subsetHow, self.samplep)
        return eng.run_function(1, 'MF_FitSubsegments', x, self.model, self.order, self.subsetHow,
                                self.samplep, self.randomSeed)


class MF_GARCHcompare(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % This code fits a set of GARCH(p,q) models to the time series and
    % returns statistics on the goodness of fits across a range of p and
    % q parameters.
    %
    % Uses the following functions from Matlab's Econometrics Toolbox: archtest,
    % lbqtest, autocorr, parcorr, garchset, garchfit, garchcount, aicbic
    %
    % Compares all combinations of p and q and output statistics are on the models
    % with the best fit.
    %
    % This operation focuses on the GARCH/variance component, and therefore
    % attempts to pre-whiten and assumes a constant mean process (applies a linear
    % detrending).
    %
    %---INPUTS:
    % y, the input time series
    % preProc, a preprocessing to apply:
    %           (i) 'none': no preprocessing is performed
    %           (ii) 'ar': performs a preprocessing that maximizes AR(2) whiteness,
    %
    % pr, a vector of model orders, p, to compare
    %
    % qr, a vector of model orders, q, to compare
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %
    %
    %---OUTPUTS: include log-likelihoods, Bayesian Information  Criteria (BIC),
    % Akaike's Information Criteria (AIC), outputs from Engle's ARCH test and the
    % Ljung-Box Q-test, and estimates of optimal model orders.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (27,)

    TAGS = ('aic', 'bic', 'econometricstoolbox', 'garch', 'model')

    def __init__(self, preProc='ar', pr=MatlabSequence('1:3'), qr=MatlabSequence('1:3'),
                 randomSeed=None, beVocal=None):
        super(MF_GARCHcompare, self).__init__()
        self.preProc = preProc
        self.pr = pr
        self.qr = qr
        self.randomSeed = randomSeed
        self.beVocal = beVocal

    def _eval_hook(self, eng, x):
        if self.preProc is None:
            return eng.run_function(1, 'MF_GARCHcompare', x, )
        elif self.pr is None:
            return eng.run_function(1, 'MF_GARCHcompare', x, self.preProc)
        elif self.qr is None:
            return eng.run_function(1, 'MF_GARCHcompare', x, self.preProc, self.pr)
        elif self.randomSeed is None:
            return eng.run_function(1, 'MF_GARCHcompare', x, self.preProc, self.pr, self.qr)
        elif self.beVocal is None:
            return eng.run_function(1, 'MF_GARCHcompare', x, self.preProc, self.pr, self.qr, self.randomSeed)
        return eng.run_function(1, 'MF_GARCHcompare', x, self.preProc, self.pr, self.qr, self.randomSeed, self.beVocal)


class MF_GARCHfit(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Simulates a procedure for fitting Generalized Autoregressive Conditional
    % Heteroskedasticity (GARCH) models to a time series, namely:
    %
    % (1) Preprocessing the data to remove strong trends,
    % (2) Pre-estimation to calculate initial correlation properties of the time
    %       series and motivate a GARCH model,
    % (3) Fitting a GARCH model, returning goodness of fit statistics and parameters
    %           of the fitted model, and
    % (4) Post-estimation, involves calculating statistics on residuals and
    %           standardized residuals.
    %
    % The idea is that all of these stages can be pre-specified or skipped using
    % arguments to the function.
    %
    % Uses functions from Matlab's Econometrics Toolbox: archtest, lbqtest,
    % autocorr, parcorr, garchset, garchfit, garchcount, aicbic
    %
    % All methods implemented are from Matlab's Econometrics Toolbox, including
    % Engle's ARCH test (archtest), the Ljung-Box Q-test (lbqtest), estimating the
    % partial autocorrelation function (parcorr), as well as specifying (garchset)
    % and fitting (garchfit) the GARCH model to the time series.
    %
    % As part of this code, a very basic automatic pre-processing routine,
    % PP_ModelFit, is implemented, that applies a range of pre-processings and
    % returns the preprocessing of the time series that shows the worst fit to an
    % AR(2) model.
    %
    % In the case that no simple transformations of the time series are
    % significantly more stationary/less trivially correlated than the original time
    % series (by more than 5%), the original time series is simply used without
    % transformation.
    %
    % Where r and m are the autoregressive and moving average orders of the model,
    % respectively, and p and q control the conditional variance parameters.
    %
    %---INPUTS:
    % y, the input time series
    %
    % preproc, the preprocessing to apply, can be 'ar' or 'none'
    %
    % P, the GARCH model order
    %
    % Q, the ARCH model order
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %               (for pre-processing: PP_PreProcess)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (20, 20)

    TAGS = ('econometricstoolbox', 'garch', 'model')

    def __init__(self, preproc='ar', P=1.0, Q=1.0, randomSeed=None):
        super(MF_GARCHfit, self).__init__()
        self.preproc = preproc
        self.P = P
        self.Q = Q
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.preproc is None:
            return eng.run_function(1, 'MF_GARCHfit', x, )
        elif self.P is None:
            return eng.run_function(1, 'MF_GARCHfit', x, self.preproc)
        elif self.Q is None:
            return eng.run_function(1, 'MF_GARCHfit', x, self.preproc, self.P)
        elif self.randomSeed is None:
            return eng.run_function(1, 'MF_GARCHfit', x, self.preproc, self.P, self.Q)
        return eng.run_function(1, 'MF_GARCHfit', x, self.preproc, self.P, self.Q, self.randomSeed)


class MF_GP_FitAcross(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Trains a Gaussian Process model on equally-spaced points throughout the time
    % series and uses the model to predict its intermediate values.
    %
    % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    %
    %---INPUTS:
    % y, the input time series
    % covFunc, the covariance function (structured in the standard way for the gpml toolbox)
    % npoints, the number of points through the time series to fit the GP model to
    %
    %---OUTPUTS: summarize the error and fitted hyperparameters.
    %
    % In future could do a better job of the sampling of points -- perhaps to take
    % into account the autocorrelation of the time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (36,)

    TAGS = ('gaussianprocess',)

    def __init__(self, covFunc=array(('covSum', array(('covSEiso', 'covNoise'),
                          dtype=object)),
      dtype=object), npoints=20.0):
        super(MF_GP_FitAcross, self).__init__()
        self.covFunc = covFunc
        self.npoints = npoints

    def _eval_hook(self, eng, x):
        if self.covFunc is None:
            return eng.run_function(1, 'MF_GP_FitAcross', x, )
        elif self.npoints is None:
            return eng.run_function(1, 'MF_GP_FitAcross', x, self.covFunc)
        return eng.run_function(1, 'MF_GP_FitAcross', x, self.covFunc, self.npoints)


class MF_GP_LearnHyperp(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Used by main Gaussian Process model fitting operations.
    %
    % References code 'minimize' from the GAUSSIAN PROCESS REGRESSION AND
    % CLASSIFICATION Toolbox version 3.2, which is avilable at:
    % http://gaussianprocess.org/gpml/code
    %
    %---INPUTS:
    %
    % t,             time
    % y,             data
    % covFunc,       the covariance function, formatted as gpml likes it
    % meanFunc, the mean function, formatted as gpml likes it
    % likFunc, the likelihood function, formatted as gpml likes it
    % infAlg, the inference algorithm (in gpml form)
    % nfevals,       the number of function evaluations
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, y=None, covFunc=None, meanFunc=None, likFunc=None,
                 infAlg=None, nfevals=None, hyp=None):
        super(MF_GP_LearnHyperp, self).__init__()
        self.y = y
        self.covFunc = covFunc
        self.meanFunc = meanFunc
        self.likFunc = likFunc
        self.infAlg = infAlg
        self.nfevals = nfevals
        self.hyp = hyp

    def _eval_hook(self, eng, x):
        if self.y is None:
            return eng.run_function(1, 'MF_GP_LearnHyperp', x, )
        elif self.covFunc is None:
            return eng.run_function(1, 'MF_GP_LearnHyperp', x, self.y)
        elif self.meanFunc is None:
            return eng.run_function(1, 'MF_GP_LearnHyperp', x, self.y, self.covFunc)
        elif self.likFunc is None:
            return eng.run_function(1, 'MF_GP_LearnHyperp', x, self.y, self.covFunc, self.meanFunc)
        elif self.infAlg is None:
            return eng.run_function(1, 'MF_GP_LearnHyperp', x, self.y, self.covFunc, self.meanFunc, self.likFunc)
        elif self.nfevals is None:
            return eng.run_function(1, 'MF_GP_LearnHyperp', x, self.y, self.covFunc, self.meanFunc,
                                    self.likFunc, self.infAlg)
        elif self.hyp is None:
            return eng.run_function(1, 'MF_GP_LearnHyperp', x, self.y, self.covFunc, self.meanFunc,
                                    self.likFunc, self.infAlg, self.nfevals)
        return eng.run_function(1, 'MF_GP_LearnHyperp', x, self.y, self.covFunc, self.meanFunc,
                                self.likFunc, self.infAlg, self.nfevals, self.hyp)


class MF_GP_LocalPrediction(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Fits a given Gaussian Process model to a section of the time series and uses
    % it to predict to the subsequent datapoint.
    %
    %---INPUTS:
    % y, the input time series
    %
    % covFunc, covariance function in the standard form for the gpml package.
    %           E.g., covFunc = {'covSum', {'covSEiso','covNoise'}} combines squared
    %           exponential and noise terms
    %
    % numTrain, the number of training samples (for each iteration)
    %
    % numTest, the number of testing samples (for each interation)
    %
    % numPreds, the number of predictions to make
    %
    % pmode, the prediction mode:
    %       (i) 'beforeafter': predicts the preceding time series values by training
    %                           on the following values,
    %       (ii) 'frombefore': predicts the following values of the time series by
    %                    training on preceding values, and
    %       (iii) 'randomgap': predicts random values within a segment of time
    %                    series by training on the other values in that segment.
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %               (for 'randomgap' prediction)
    %
    %---OUTPUTS: summaries of the quality of predictions made, the mean and
    % spread of obtained hyperparameter values, and marginal likelihoods.
    
    % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (58, 57, 58)

    TAGS = ('gaussianprocess',)

    def __init__(self, covFunc=array(('covSum', array(('covSEiso',
                                'covNoise'), dtype=object)),
      dtype=object), numTrain=5.0, numTest=3.0, numPreds=10.0, pmode='beforeafter', randomSeed=None):
        super(MF_GP_LocalPrediction, self).__init__()
        self.covFunc = covFunc
        self.numTrain = numTrain
        self.numTest = numTest
        self.numPreds = numPreds
        self.pmode = pmode
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.covFunc is None:
            return eng.run_function(1, 'MF_GP_LocalPrediction', x, )
        elif self.numTrain is None:
            return eng.run_function(1, 'MF_GP_LocalPrediction', x, self.covFunc)
        elif self.numTest is None:
            return eng.run_function(1, 'MF_GP_LocalPrediction', x, self.covFunc, self.numTrain)
        elif self.numPreds is None:
            return eng.run_function(1, 'MF_GP_LocalPrediction', x, self.covFunc, self.numTrain, self.numTest)
        elif self.pmode is None:
            return eng.run_function(1, 'MF_GP_LocalPrediction', x, self.covFunc, self.numTrain,
                                    self.numTest, self.numPreds)
        elif self.randomSeed is None:
            return eng.run_function(1, 'MF_GP_LocalPrediction', x, self.covFunc, self.numTrain,
                                    self.numTest, self.numPreds, self.pmode)
        return eng.run_function(1, 'MF_GP_LocalPrediction', x, self.covFunc, self.numTrain,
                                self.numTest, self.numPreds, self.pmode, self.randomSeed)


class MF_GP_hyperparameters(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    %
    % The code can accomodate a range of covariance functions, e.g.:
    % (i) a sum of squared exponential and noise terms, and
    % (ii) a sum of squared exponential, periodic, and noise terms.
    %
    % The model is fitted to <> samples from the time series, which are
    % chosen by:
    % (i) resampling the time series down to this many data points,
    % (ii) taking the first 200 samples from the time series, or
    % (iii) taking random samples from the time series.
    %
    %---INPUTS:
    % y, the input time series
    %
    % covFunc, the covariance function, in the standard form of the gmpl package
    %
    % squishorsquash, whether to squash onto the unit interval, or spread across 1:N
    %
    % maxN, the maximum length of time series to consider -- inputs greater than
    %           this length are resampled down to maxN
    %
    % resampleHow, specifies the method of how to resample time series longer than maxN
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed,
    %             for settings of resampleHow that involve random number generation
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (51, 54, 53, 63, 66)

    TAGS = ('gaussianprocess',)

    def __init__(self, covFunc=array(('covSum', array(('covSEiso',
                                'covNoise'), dtype=object)),
      dtype=object), squishorsquash=1.0, maxN=200.0, resampleHow='resample', randomSeed=None):
        super(MF_GP_hyperparameters, self).__init__()
        self.covFunc = covFunc
        self.squishorsquash = squishorsquash
        self.maxN = maxN
        self.resampleHow = resampleHow
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.covFunc is None:
            return eng.run_function(1, 'MF_GP_hyperparameters', x, )
        elif self.squishorsquash is None:
            return eng.run_function(1, 'MF_GP_hyperparameters', x, self.covFunc)
        elif self.maxN is None:
            return eng.run_function(1, 'MF_GP_hyperparameters', x, self.covFunc, self.squishorsquash)
        elif self.resampleHow is None:
            return eng.run_function(1, 'MF_GP_hyperparameters', x, self.covFunc, self.squishorsquash, self.maxN)
        elif self.randomSeed is None:
            return eng.run_function(1, 'MF_GP_hyperparameters', x, self.covFunc, self.squishorsquash,
                                    self.maxN, self.resampleHow)
        return eng.run_function(1, 'MF_GP_hyperparameters', x, self.covFunc, self.squishorsquash,
                                self.maxN, self.resampleHow, self.randomSeed)


class MF_ResidualAnalysis(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Given an input residual time series residuals, e, this function returns a
    % structure with fields corresponding to statistical tests on the residuals.
    % These are motivated by a general expectation of model residuals to be
    % uncorrelated.
    %
    %---INPUT:
    % e, should be raw residuals as prediction minus data (e = yp - y) as a column
    %       vector.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self):
        super(MF_ResidualAnalysis, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'MF_ResidualAnalysis', x, )


class MF_StateSpaceCompOrder(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Fits state space models using n4sid (from Matlab's System Identification
    % Toolbox) of orders 1, 2, ..., maxOrder and returns statistics on how the
    % goodness of fit changes across this range.
    %
    % c.f., MF_CompareAR -- does a similar thing for AR models
    % Uses the functions iddata, n4sid, and aic from Matlab's System Identification
    % Toolbox
    %
    %---INPUTS:
    % y, the input time series
    % maxOrder, the maximum model order to consider.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (24,)

    TAGS = ('model', 'statespace', 'systemidentificationtoolbox')

    def __init__(self, maxOrder=8.0):
        super(MF_StateSpaceCompOrder, self).__init__()
        self.maxOrder = maxOrder

    def _eval_hook(self, eng, x):
        if self.maxOrder is None:
            return eng.run_function(1, 'MF_StateSpaceCompOrder', x, )
        return eng.run_function(1, 'MF_StateSpaceCompOrder', x, self.maxOrder)


class MF_StateSpace_n4sid(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % First fits the model to the whole time series, then trains it on the first
    % portion and tries to predict the rest.
    %
    % In the second portion of this code, the state space model is fitted to the
    % first p*N samples of the time series, where p is a given proportion and N is
    % the length of the time series.
    %
    % This model is then used to predict the latter portion of the time
    % series (i.e., the subsequent (1-p)*N samples).
    %
    %---INPUTS:
    % y, the input time series
    % ord, the order of state-space model to implement (can also be the string 'best')
    % ptrain, the proportion of the time series to use for training
    % steps, the number of steps ahead to predict
    %
    %---OUTPUTS: parameters from the model fitted to the entire time series, and
    % goodness of fit and residual analysis from n4sid prediction.
    %
    % Uses the functions iddata, n4sid, aic, and predict from Matlab's System
    % Identification Toolbox
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (26, 26, 26)

    TAGS = ('model', 'statespace', 'systemidentificationtoolbox')

    def __init__(self, ordd=1.0, ptrain=0.5, steps=1.0):
        super(MF_StateSpace_n4sid, self).__init__()
        self.ordd = ordd
        self.ptrain = ptrain
        self.steps = steps

    def _eval_hook(self, eng, x):
        if self.ordd is None:
            return eng.run_function(1, 'MF_StateSpace_n4sid', x, )
        elif self.ptrain is None:
            return eng.run_function(1, 'MF_StateSpace_n4sid', x, self.ordd)
        elif self.steps is None:
            return eng.run_function(1, 'MF_StateSpace_n4sid', x, self.ordd, self.ptrain)
        return eng.run_function(1, 'MF_StateSpace_n4sid', x, self.ordd, self.ptrain, self.steps)


class MF_arfit(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses various functions implemented in the ARfit package, which is
    % freely-available at http://www.gps.caltech.edu/~tapio/arfit/
    %
    % cf. "Estimation of parameters and eigenmodes of multivariate autoregressive
    %       models", A. Neumaier and T. Schneider, ACM Trans. Math. Softw. 27, 27 (2001)
    %
    % cf. "Algorithm 808: ARFIT---a Matlab package for the estimation of parameters
    %      and eigenmodes of multivariate autoregressive models",
    %      T. Schneider and A. Neumaier, ACM Trans. Math. Softw. 27, 58 (2001)
    %
    % Autoregressive (AR) models are fitted with orders p = pmin, pmin + 1, ..., pmax.
    %
    % The optimal model order is selected using Schwartz's Bayesian Criterion (SBC).
    %
    %---INPUTS:
    % y, the input time series
    % pmin, the minimum AR model order to fit
    % pmax, the maximum AR model order to fit
    % selector, crierion to select optimal time-series model order (e.g., 'sbc', cf.
    %           ARFIT package documentation)
    %
    %---OUTPUTS: include the model coefficients obtained, the SBCs at each model order,
    % various tests on residuals, and statistics from an eigendecomposition of the
    % time series using the estimated AR model.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (16,)

    TAGS = ('arfit', 'modelfit')

    def __init__(self, pmin=1.0, pmax=8.0, selector='sbc'):
        super(MF_arfit, self).__init__()
        self.pmin = pmin
        self.pmax = pmax
        self.selector = selector

    def _eval_hook(self, eng, x):
        if self.pmin is None:
            return eng.run_function(1, 'MF_arfit', x, )
        elif self.pmax is None:
            return eng.run_function(1, 'MF_arfit', x, self.pmin)
        elif self.selector is None:
            return eng.run_function(1, 'MF_arfit', x, self.pmin, self.pmax)
        return eng.run_function(1, 'MF_arfit', x, self.pmin, self.pmax, self.selector)


class MF_armax(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the functions iddata, armax, aic, and predict from Matlab's System
    % Identification Toolbox
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % orders, a two-vector for p and q, the AR and MA components of the model,
    %           respectively,
    %
    % ptrain, the proportion of data to train the model on (the remainder is used
    %           for testing),
    %
    % numSteps, number of steps to predict into the future for testing the model.
    %
    %
    %---OUTPUTS: include the fitted AR and MA coefficients, the goodness of fit in
    % the training data, and statistics on the residuals from using the fitted model
    % to predict the testing data.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (17, 17, 17)

    TAGS = ('arma', 'model', 'systemidentificationtoolbox')

    def __init__(self, orders=array((3, 1)), ptrain=0.5, numSteps=1.0):
        super(MF_armax, self).__init__()
        self.orders = orders
        self.ptrain = ptrain
        self.numSteps = numSteps

    def _eval_hook(self, eng, x):
        if self.orders is None:
            return eng.run_function(1, 'MF_armax', x, )
        elif self.ptrain is None:
            return eng.run_function(1, 'MF_armax', x, self.orders)
        elif self.numSteps is None:
            return eng.run_function(1, 'MF_armax', x, self.orders, self.ptrain)
        return eng.run_function(1, 'MF_armax', x, self.orders, self.ptrain, self.numSteps)


class MF_hmm_CompareNStates(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Fits HMMs with different numbers of states, and compares the resulting
    % test-set likelihoods.
    %
    % The code relies on Zoubin Gharamani's implementation of HMMs for real-valued
    % Gassian-distributed observations, including the hmm and hmm_cl routines (
    % renamed ZG_hmm and ZG_hmm_cl here).
    % Implementation of HMMs for real-valued Gaussian observations:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software.html
    % or, specifically:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software/hmm.tar.gz
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % trainp, the initial proportion of the time series to train the model on
    %
    % nstater, the vector of state numbers to compare. E.g., (2:4) compares a number
    %               of states 2, 3, and 4.
    %
    %---OUTPUTS: statistics on how the log likelihood of the test data changes with
    % the number of states n_{states}$. We implement the code for p_{train} = 0.6$
    % as n_{states}$ varies across the range n_{states} = 2, 3, 4$.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (27,)

    TAGS = ('gharamani', 'hmm', 'model')

    def __init__(self, trainp=0.6, nstater=MatlabSequence('2:4')):
        super(MF_hmm_CompareNStates, self).__init__()
        self.trainp = trainp
        self.nstater = nstater

    def _eval_hook(self, eng, x):
        if self.trainp is None:
            return eng.run_function(1, 'MF_hmm_CompareNStates', x, )
        elif self.nstater is None:
            return eng.run_function(1, 'MF_hmm_CompareNStates', x, self.trainp)
        return eng.run_function(1, 'MF_hmm_CompareNStates', x, self.trainp, self.nstater)


class MF_hmm_fit(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, the input time series
    % trainp, the proportion of data to train on, 0 < trainp < 1
    % numStates, the number of states in the HMM
    
    % Uses Zoubin Gharamani's implementation of HMMs for real-valued Gaussian
    % observations:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software.html
    % or, specifically:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software/hmm.tar.gz
    %
    % Uses ZG_hmm (renamed from hmm) and ZG_hmm_cl (renamed from hmm_cl)
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (11, 11)

    TAGS = ('gharamani', 'hmm', 'model')

    def __init__(self, trainp=0.7, numStates=3.0):
        super(MF_hmm_fit, self).__init__()
        self.trainp = trainp
        self.numStates = numStates

    def _eval_hook(self, eng, x):
        if self.trainp is None:
            return eng.run_function(1, 'MF_hmm_fit', x, )
        elif self.numStates is None:
            return eng.run_function(1, 'MF_hmm_fit', x, self.trainp)
        return eng.run_function(1, 'MF_hmm_fit', x, self.trainp, self.numStates)


class MF_steps_ahead(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Given a model, characterizes the variation in goodness of model predictions
    % across a range of prediction lengths, l, which is made to vary from
    % 1-step ahead to maxSteps steps-ahead predictions.
    %
    % Models are fit using code from Matlab's System Identification Toolbox:
    % (i) AR models using the ar function,
    % (ii) ARMA models using armax code, and
    % (iii) state-space models using n4sid code.
    %
    % The model is fitted on the full time series and then used to predict the same
    % data.
    %
    %---INPUTS:
    % y, the input time series
    % model, the time-series model to fit: 'ar', 'arma', or 'ss'
    % order, the order of the model to fit
    % maxSteps, the maximum number of steps ahead to predict
    %
    %---OUTPUTS: include the errors, for prediction lengths l = 1, 2, ..., maxSteps,
    % returned for each model relative to the best performance from basic null
    % predictors, including sliding 1- and 2-sample mean predictors and simply
    % predicting each point as the mean of the full time series. Additional outputs
    % quantify how the errors change as the prediction length increases from l = 1,
    % ..., maxSteps (relative to a simple predictor).
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21, 24, 25, 24)

    TAGS = ('ar', 'arfit', 'arma', 'model', 'prediction', 'statespace', 'systemidentificationtoolbox')

    def __init__(self, model='ar', order=2.0, maxSteps=6.0):
        super(MF_steps_ahead, self).__init__()
        self.model = model
        self.order = order
        self.maxSteps = maxSteps

    def _eval_hook(self, eng, x):
        if self.model is None:
            return eng.run_function(1, 'MF_steps_ahead', x, )
        elif self.order is None:
            return eng.run_function(1, 'MF_steps_ahead', x, self.model)
        elif self.maxSteps is None:
            return eng.run_function(1, 'MF_steps_ahead', x, self.model, self.order)
        return eng.run_function(1, 'MF_steps_ahead', x, self.model, self.order, self.maxSteps)


class NL_BoxCorrDim(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % References TSTOOL code, corrdim, to estimate the correlation dimension of a
    % time-delay embedded time series using a box-counting approach.
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    % y, column vector of time series data
    % numBins, maximum number of partitions per axis
    % embedParams [opt], embedding parameters as {tau,m} in 2-entry cell for a
    %                   time-delay, tau, and embedding dimension, m. As inputs to BF_embed.
    %
    %---OUTPUTS: Simple summaries of the outputs from corrdim.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21,)

    TAGS = ('corrdim', 'correlation', 'nonlinear', 'tstool')

    def __init__(self, numBins=50.0, embedParams=array(('ac', 5), dtype=object)):
        super(NL_BoxCorrDim, self).__init__()
        self.numBins = numBins
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.numBins is None:
            return eng.run_function(1, 'NL_BoxCorrDim', x, )
        elif self.embedParams is None:
            return eng.run_function(1, 'NL_BoxCorrDim', x, self.numBins)
        return eng.run_function(1, 'NL_BoxCorrDim', x, self.numBins, self.embedParams)


class NL_CaosMethod(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % References TSTOOL code cao, to determine the minimum embedding dimension for a
    % time series using Cao's method:
    %
    % "Practical method for determining the minimum embedding dimension of a scalar
    % time series", L. Cao, Physica D 110(1-2) 43 (1997)
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    % It computes the quantities E and E* for a range of embedding dimensions
    % m = 1, ..., m_{max}.
    %
    %---INPUTS:
    % y, time series as a column vector
    % maxdim, maximum embedding dimension to consider
    % tau, time delay (can also be 'ac' or 'mi' for first zero-crossing of the
    %          autocorrelation function or the first minimum of the automutual information
    %          function)
    % NNR, number of nearest neighbours to use
    % Nref, number of reference points (can also be a fraction; of data length)
    % justanum [opt]: if not empty can just return a number, the embedding
    %                   dimension, based on the specified criterion:
    %                 (i) 'thresh', caoo1 passes above a threshold, th (given as
    %                               justanum = {'thresh',th})
    %                 (ii) 'mthresh', when gradient passes below a threshold (levels
    %                                 off), given as {'mthresh',mthresh}, where
    %                                 mthresh in the threshold
    %                 (iii) 'mmthresh', analyzes incremental differences to find
    %                                   level-off point
    %
    %---OUTPUTS: statistics on the result, including when the output quantity first
    % passes a given threshold, and the m at which it levels off.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, maxdim=None, tau=None, NNR=None, Nref=None, justanum=None):
        super(NL_CaosMethod, self).__init__()
        self.maxdim = maxdim
        self.tau = tau
        self.NNR = NNR
        self.Nref = Nref
        self.justanum = justanum

    def _eval_hook(self, eng, x):
        if self.maxdim is None:
            return eng.run_function(1, 'NL_CaosMethod', x, )
        elif self.tau is None:
            return eng.run_function(1, 'NL_CaosMethod', x, self.maxdim)
        elif self.NNR is None:
            return eng.run_function(1, 'NL_CaosMethod', x, self.maxdim, self.tau)
        elif self.Nref is None:
            return eng.run_function(1, 'NL_CaosMethod', x, self.maxdim, self.tau, self.NNR)
        elif self.justanum is None:
            return eng.run_function(1, 'NL_CaosMethod', x, self.maxdim, self.tau, self.NNR, self.Nref)
        return eng.run_function(1, 'NL_CaosMethod', x, self.maxdim, self.tau, self.NNR, self.Nref, self.justanum)


class NL_DVV(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    % NL_DVV 	Delay Vector Variance method for real and complex signals.
    %
    % Uses predictability of the signal in phase space to characterize the
    % time series.
    %
    % This function uses the original code from the DVV toolbox to do the computation
    % and produces statistics on the outputs -- comparing the DVV curves for both
    % the real and the surrogate data.
    %
    %---USAGE:
    % outputStats = NL_DVV(x, m, numDVs, nd, Ntv, numSurr, randomSeed)
    %
    %---INPUTS:
    % x:            Original real-valued or complex time series
    % m:            Delay embedding dimension
    % Ntv:          Number of points on horizontal axes
    % numDVs:	    Number of reference DVs to consider
    % nd:           Span over which to perform DVV
    % Ntv:          Number of points on the horizontal axis
    % numSurr:      Number of surrogates to compare to
    % randomSeed:   How to control the random seed for reproducibility
    %
    % A Delay Vector Variance (DVV) toolbox for MATLAB
    % (c) Copyright Danilo P. Mandic 2008
    % http://www.commsp.ee.ic.ac.uk/~mandic/dvv.htm
    % http://www.commsp.ee.ic.ac.uk/~mandic/dvv/papers/dvv_proj.pdf
    %
    % Modified by Ben Fulcher, 2015-05-13, for use in hctsa.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   This program is free software; you can redistribute it and/or modify it
    %   under the terms of the GNU General Public License as published by the Free
    %   Software Foundation; either version 2 of the License, or (at your option)
    %   any later version.
    %
    %   This program is distributed in the hope that it will be useful, but WITHOUT
    %   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    %   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    %   more details.
    %
    %   You can obtain a copy of the GNU General Public License from
    %   http://www.gnu.org/copyleft/gpl.html or by writing to Free Software
    %   Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Plot output plot:
    doPlot = 0;
    
    % Talk to me:
    beVocal = 0;
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (28,)

    TAGS = ('delayVectorVariance',)

    def __init__(self, m=3.0, numDVs=100.0, nd=2.0, Ntv=50.0, numSurr=10.0, randomSeed='default'):
        super(NL_DVV, self).__init__()
        self.m = m
        self.numDVs = numDVs
        self.nd = nd
        self.Ntv = Ntv
        self.numSurr = numSurr
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.m is None:
            return eng.run_function(1, 'NL_DVV', x, )
        elif self.numDVs is None:
            return eng.run_function(1, 'NL_DVV', x, self.m)
        elif self.nd is None:
            return eng.run_function(1, 'NL_DVV', x, self.m, self.numDVs)
        elif self.Ntv is None:
            return eng.run_function(1, 'NL_DVV', x, self.m, self.numDVs, self.nd)
        elif self.numSurr is None:
            return eng.run_function(1, 'NL_DVV', x, self.m, self.numDVs, self.nd, self.Ntv)
        elif self.randomSeed is None:
            return eng.run_function(1, 'NL_DVV', x, self.m, self.numDVs, self.nd, self.Ntv, self.numSurr)
        return eng.run_function(1, 'NL_DVV', x, self.m, self.numDVs, self.nd, self.Ntv, self.numSurr, self.randomSeed)


class NL_MS_fnn(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Determines the number of false nearest neighbors for the embedded time series
    % using Michael Small's false nearest neighbor code, fnn (renamed MS_fnn here).
    %
    % False nearest neighbors are judged using a ratio of the distances between the
    % next k points and the neighboring points of a given datapoint.
    %
    %---INPUTS:
    % y, the input time series
    %
    % de, the embedding dimensions to compare across (a vector)
    %
    % tau, the time-delay (can be 'ac' or 'mi' to be the first zero-crossing of ACF,
    %                       or first minimum of AMI, respectively)
    %
    % th, the distance threshold for neighbours
    %
    % kth, the the distance to next points
    %
    % [opt] justBest, can be set to 1 to just return the best embedding dimension, m_{best}
    %
    % [opt] bestp, if justBest = 1, can set bestp as the proportion of false nearest
    %              neighbours at which the optimal embedding dimension is selected.
    %
    % This function returns statistics on the proportion of false nearest neighbors
    % as a function of the embedding dimension m = m_{min}, m_{min}+1, ..., m_{max}
    % for a given time lag tau, and distance threshold for neighbors, d_{th}.
    %
    %---OUTPUTS: include the proportion of false nearest neighbors at each m, the mean
    % and spread, and the smallest m at which the proportion of false nearest
    % neighbors drops below each of a set of fixed thresholds.
    
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Code available at http://small.eie.polyu.edu.hk/matlab/
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21,)

    TAGS = ('MichaelSmall', 'fnn', 'nonlinear', 'slow')

    def __init__(self, de=MatlabSequence('1:10'), tau='mi', th=5.0, kth=1.0, justBest=None, bestp=None):
        super(NL_MS_fnn, self).__init__()
        self.de = de
        self.tau = tau
        self.th = th
        self.kth = kth
        self.justBest = justBest
        self.bestp = bestp

    def _eval_hook(self, eng, x):
        if self.de is None:
            return eng.run_function(1, 'NL_MS_fnn', x, )
        elif self.tau is None:
            return eng.run_function(1, 'NL_MS_fnn', x, self.de)
        elif self.th is None:
            return eng.run_function(1, 'NL_MS_fnn', x, self.de, self.tau)
        elif self.kth is None:
            return eng.run_function(1, 'NL_MS_fnn', x, self.de, self.tau, self.th)
        elif self.justBest is None:
            return eng.run_function(1, 'NL_MS_fnn', x, self.de, self.tau, self.th, self.kth)
        elif self.bestp is None:
            return eng.run_function(1, 'NL_MS_fnn', x, self.de, self.tau, self.th, self.kth, self.justBest)
        return eng.run_function(1, 'NL_MS_fnn', x, self.de, self.tau, self.th, self.kth, self.justBest, self.bestp)


class NL_MS_nlpe(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Computes the nlpe for a time-delay embedded time series using Michael Small's
    % code, nlpe (renamed MS_nlpe here):
    %
    %---INPUTS:
    % y, the input time series
    % de, the embedding dimension (can be an integer, or 'fnn' to select as the
    %       point where the proportion of false nearest neighbors falls below 5%
    %       using NL_MS_fnn)
    % tau, the time-delay (can be an integer or 'ac' to be the first zero-crossing
    %       of the ACF or 'mi' to be the first minimum of the automutual information
    %       function)
    %
    %---OUTPUTS: include measures of the meanerror of the nonlinear predictor, and a
    % set of measures on the correlation, Gaussianity, etc. of the residuals.
    %
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    %
    % Michael Small's Matlab code is available at http://small.eie.polyu.edu.hk/matlab/
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (15, 15, 17)

    TAGS = ('MichaelSmall', 'model', 'nlpe', 'nonlinear', 'slow')

    def __init__(self, de=2.0, tau='mi', maxN=None):
        super(NL_MS_nlpe, self).__init__()
        self.de = de
        self.tau = tau
        self.maxN = maxN

    def _eval_hook(self, eng, x):
        if self.de is None:
            return eng.run_function(1, 'NL_MS_nlpe', x, )
        elif self.tau is None:
            return eng.run_function(1, 'NL_MS_nlpe', x, self.de)
        elif self.maxN is None:
            return eng.run_function(1, 'NL_MS_nlpe', x, self.de, self.tau)
        return eng.run_function(1, 'NL_MS_nlpe', x, self.de, self.tau, self.maxN)


class NL_TISEAN_c1(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Implements the c1 and c2d routines from the TISEAN nonlinear time-series
    % analysis package that compute curves for the fixed mass computation of the
    % information dimension.
    %
    %---INPUTS:
    %
    % y, the time series to analyze
    %
    % tau, the time-delay (can be 'ac' or 'mi' for the first zero-crossing of the
    %           autocorrelation function or first minimum of the automutual
    %           information function)
    %
    % mmm, a two-vector specifying the minimum and maximum embedding dimensions,
    %       e.g., [2,10] for m = 2 up to m = 10
    %
    % tsep, time separation (can be between 0 and 1 for a proportion of the
    %       time-series length)
    %
    % Nref, the number of reference points (can also be between 0 and 1 to specify a
    %       proportion of the time-series length)
    %
    %---OUTPUTS: optimal scaling ranges and dimension estimates for a time delay,
    % tau, embedding dimensions, m, ranging from m_{min} to m_{max}, a time
    % separation, tsep, and a number of reference points, Nref.
    %
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package" Hegger, R. and Kantz, H. and Schreiber, T., Chaos 9(2) 413 (1999)
    %
    % Available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    %
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (25, 24)

    TAGS = ('dimension', 'entropy', 'nonlinear', 'tisean')

    def __init__(self, tau=1.0, mmm=array((1, 7)), tsep=0.02, Nref=0.5):
        super(NL_TISEAN_c1, self).__init__()
        self.tau = tau
        self.mmm = mmm
        self.tsep = tsep
        self.Nref = Nref

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'NL_TISEAN_c1', x, )
        elif self.mmm is None:
            return eng.run_function(1, 'NL_TISEAN_c1', x, self.tau)
        elif self.tsep is None:
            return eng.run_function(1, 'NL_TISEAN_c1', x, self.tau, self.mmm)
        elif self.Nref is None:
            return eng.run_function(1, 'NL_TISEAN_c1', x, self.tau, self.mmm, self.tsep)
        return eng.run_function(1, 'NL_TISEAN_c1', x, self.tau, self.mmm, self.tsep, self.Nref)


class NL_TISEAN_d2(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The function estimates the correlation sum, the correlation dimension and
    % the correlation entropy of a given time series, y. Our code uses the outputs
    % from this algorithm to return a set of informative features about the results.
    %
    %---INPUTS:
    %
    % y, input time series
    %
    % tau, time-delay (can be 'ac' or 'mi' for first zero-crossing of
    %       autocorrelation function, or first minimum of the automutual
    %       information)
    %
    % maxm, the maximum embedding dimension
    %
    % theilerWin, the Theiler window
    
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package", R. Hegger, H. Kantz, and T. Schreiber, Chaos 9(2) 413 (1999)
    %
    % The TISEAN package is available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    %
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    %
    % cf. "Spurious dimension from correlation algorithms applied to limited
    % time-series data", J. Theiler, Phys. Rev. A, 34(3) 2427 (1986)
    %
    % cf. "Nonlinear Time Series Analysis", Cambridge University Press, H. Kantz
    % and T. Schreiber (2004)
    %
    % Taken's estimator is computed for the correlation dimension, as well as related
    % statistics, including other dimension estimates by finding appropriate scaling
    % ranges, and searching for a flat region in the output of TISEAN's h2
    % algorithm, which indicates determinism/deterministic chaos.
    %
    % To find a suitable scaling range, a penalized regression procedure is used to
    % determine an optimal scaling range that simultaneously spans the greatest
    % range of scales and shows the best fit to the data, and return the range, a
    % goodness of fit statistic, and a dimension estimate.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (19, 22)

    TAGS = ('dimension', 'nonlinear', 'tisean')

    def __init__(self, tau=1.0, maxm=10.0, theilerWin=0.0):
        super(NL_TISEAN_d2, self).__init__()
        self.tau = tau
        self.maxm = maxm
        self.theilerWin = theilerWin

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'NL_TISEAN_d2', x, )
        elif self.maxm is None:
            return eng.run_function(1, 'NL_TISEAN_d2', x, self.tau)
        elif self.theilerWin is None:
            return eng.run_function(1, 'NL_TISEAN_d2', x, self.tau, self.maxm)
        return eng.run_function(1, 'NL_TISEAN_d2', x, self.tau, self.maxm, self.theilerWin)


class NL_TISEAN_fnn(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, the input time series
    % tau, the time delay
    % maxm, the maximum embedding dimension
    % theilerWin, the Theiler window
    % justBest, if 1 just outputs a scalar estimate of embedding dimension
    % bestp, only used if justBest==1 -- the fnn threshold for picking an embedding
    %                dimension
    %
    %---OUTPUTS: individual false nearest neighbors proportions, as well as
    % summaries of neighborhood size, and embedding dimensions at which the
    % proportion of nearest neighbours falls below a range of thresholds
    
    % Uses the false_nearest routine from the TISEAN package for nonlinear time-series
    % analysis.
    %
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package", R. Hegger, H. Kantz, and T. Schreiber, Chaos 9(2) 413 (1999)
    %
    % Available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    %
    % Documentation here:
    % http://www.mpipks-dresden.mpg.de/~tisean/TISEAN_2.1/docs/docs_c/false_nearest.html
    %
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (22, 23, 23)

    TAGS = ('dimension', 'nonlinear', 'tisean')

    def __init__(self, tau='ac', maxm=10.0, theilerWin=0.05, justBest=0.0, bestp=None):
        super(NL_TISEAN_fnn, self).__init__()
        self.tau = tau
        self.maxm = maxm
        self.theilerWin = theilerWin
        self.justBest = justBest
        self.bestp = bestp

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'NL_TISEAN_fnn', x, )
        elif self.maxm is None:
            return eng.run_function(1, 'NL_TISEAN_fnn', x, self.tau)
        elif self.theilerWin is None:
            return eng.run_function(1, 'NL_TISEAN_fnn', x, self.tau, self.maxm)
        elif self.justBest is None:
            return eng.run_function(1, 'NL_TISEAN_fnn', x, self.tau, self.maxm, self.theilerWin)
        elif self.bestp is None:
            return eng.run_function(1, 'NL_TISEAN_fnn', x, self.tau, self.maxm, self.theilerWin, self.justBest)
        return eng.run_function(1, 'NL_TISEAN_fnn', x, self.tau, self.maxm, self.theilerWin, self.justBest, self.bestp)


class NL_TSTL_FractalDimensions(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, column vector of time series data
    % kmin, minimum number of neighbours for each reference point
    % kmax, maximum number of neighbours for each reference point
    % Nref, number of randomly-chosen reference points (-1: use all points)
    % gstart, starting value for moments
    % gend, end value for moments
    % past [opt], number of samples to exclude before an after each reference
    %             index (default=0)
    % steps [opt], number of moments to calculate (default=32);
    % embedParams, how to embed the time series using a time-delay reconstruction
    %
    %---OUTPUTS: include basic statistics of D(q) and q, statistics from a linear fit,
    % and an exponential fit of the form D(q) = Aexp(Bq) + C.
    
    % Computes the fractal dimension spectrum, D(q), using moments of neighbor
    % distances for time-delay embedded time series by referencing the code,
    % fracdims, from the TSTOOL package.
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (48, 49, 47, 49, 53)

    TAGS = ('correlation', 'dimension', 'nonlinear', 'stochastic', 'tstool')

    def __init__(self, kmin=5.0, kmax=20.0, Nref=0.2, gstart=1.0,
                 gend=10.0, past=0.0, steps=32.0, embedParams=array(('ac', 'fnnmar'), dtype=object)):
        super(NL_TSTL_FractalDimensions, self).__init__()
        self.kmin = kmin
        self.kmax = kmax
        self.Nref = Nref
        self.gstart = gstart
        self.gend = gend
        self.past = past
        self.steps = steps
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.kmin is None:
            return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, )
        elif self.kmax is None:
            return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, self.kmin)
        elif self.Nref is None:
            return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, self.kmin, self.kmax)
        elif self.gstart is None:
            return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, self.kmin, self.kmax, self.Nref)
        elif self.gend is None:
            return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, self.kmin, self.kmax, self.Nref, self.gstart)
        elif self.past is None:
            return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, self.kmin, self.kmax,
                                    self.Nref, self.gstart, self.gend)
        elif self.steps is None:
            return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, self.kmin, self.kmax,
                                    self.Nref, self.gstart, self.gend, self.past)
        elif self.embedParams is None:
            return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, self.kmin, self.kmax,
                                    self.Nref, self.gstart, self.gend, self.past, self.steps)
        return eng.run_function(1, 'NL_TSTL_FractalDimensions', x, self.kmin, self.kmax, self.Nref,
                                self.gstart, self.gend, self.past, self.steps, self.embedParams)


class HCTSA_NL_TSTL_GPCorrSum(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, column vector of time-series data
    % Nref, number of (randomly-chosen) reference points (-1: use all points,
    %       if a decimal, then use this fraction of the time series length)
    % r, maximum search radius relative to attractor size, 0 < r < 1
    % thwin, number of samples to exclude before and after each reference index
    %        (~ Theiler window)
    % nbins, number of partitioned bins
    % embedParams, embedding parameters to feed BF_embed.m for embedding the
    %               signal in the form {tau,m}
    % doTwo, if this is set to 1, will use corrsum, if set to 2, will use corrsum2.
    %           For corrsum2, n specifies the number of pairs per bin. Default is 1,
    %           to use corrsum.
    %
    %---OUTPUTS: basic statistics on the outputs of corrsum, including iteratively
    % re-weighted least squares linear fits to log-log plots using the robustfit
    % function in Matlab's Statistics Toolbox.
    %
    % Uses TSTOOL code corrsum (or corrsum2) to compute scaling of the correlation
    % sum for a time-delay reconstructed time series by the Grassberger-Proccacia
    % algorithm using fast nearest-neighbor search.
    %
    % cf. "Characterization of Strange Attractors", P. Grassberger and I. Procaccia,
    % Phys. Rev. Lett. 50(5) 346 (1983)
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (40, 40, 41, 40)

    TAGS = ('correlation', 'corrsum', 'nonlinear', 'stochastic', 'tstool')

    def __init__(self, Nref=-1.0, r=0.5, thwin=40.0, nbins=20.0, embedParams=array(('ac', 'fnnmar'), dtype=object), doTwo=1.0):
        super(HCTSA_NL_TSTL_GPCorrSum, self).__init__()
        self.Nref = Nref
        self.r = r
        self.thwin = thwin
        self.nbins = nbins
        self.embedParams = embedParams
        self.doTwo = doTwo

    def _eval_hook(self, eng, x):
        if self.Nref is None:
            return eng.run_function(1, 'NL_TSTL_GPCorrSum', x, )
        elif self.r is None:
            return eng.run_function(1, 'NL_TSTL_GPCorrSum', x, self.Nref)
        elif self.thwin is None:
            return eng.run_function(1, 'NL_TSTL_GPCorrSum', x, self.Nref, self.r)
        elif self.nbins is None:
            return eng.run_function(1, 'NL_TSTL_GPCorrSum', x, self.Nref, self.r, self.thwin)
        elif self.embedParams is None:
            return eng.run_function(1, 'NL_TSTL_GPCorrSum', x, self.Nref, self.r, self.thwin, self.nbins)
        elif self.doTwo is None:
            return eng.run_function(1, 'NL_TSTL_GPCorrSum', x, self.Nref, self.r, self.thwin,
                                    self.nbins, self.embedParams)
        return eng.run_function(1, 'NL_TSTL_GPCorrSum', x, self.Nref, self.r, self.thwin, self.nbins,
                                self.embedParams, self.doTwo)


class NL_TSTL_LargestLyap(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, the time series to analyze
    % Nref, number of randomly-chosen reference points (-1 == all)
    % maxtstep, maximum prediction length (samples)
    % past, exclude -- Theiler window idea
    % NNR, number of nearest neighbours
    % embedParams, input to BF_embed, how to time-delay-embed the time series, in
    %               the form {tau,m}, where string specifiers can indicate standard
    %               methods of determining tau or m.
    %
    %---OUTPUTS: a range of statistics on the outputs from this function, including
    % a penalized linear regression to the scaling range in an attempt to fit to as
    % much of the range of scales as possible while simultaneously achieving the
    % best possible linear fit.
    
    % Uses the TSTOOL code 'largelyap'.
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    % The algorithm used (using formula (1.5) in Parlitz Nonlinear Time Series
    % Analysis book) is very similar to the Wolf algorithm:
    % "Determining Lyapunov exponents from a time series", A. Wolf et al., Physica D
    % 16(3) 285 (1985)
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (35, 41)

    TAGS = ('largelyap', 'nonlinear', 'tstool')

    def __init__(self, Nref=-1.0, maxtstep=0.1, past=0.01, NNR=3.0,
                 embedParams=array(('mi', 'fnnmar'), dtype=object)):
        super(NL_TSTL_LargestLyap, self).__init__()
        self.Nref = Nref
        self.maxtstep = maxtstep
        self.past = past
        self.NNR = NNR
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.Nref is None:
            return eng.run_function(1, 'NL_TSTL_LargestLyap', x, )
        elif self.maxtstep is None:
            return eng.run_function(1, 'NL_TSTL_LargestLyap', x, self.Nref)
        elif self.past is None:
            return eng.run_function(1, 'NL_TSTL_LargestLyap', x, self.Nref, self.maxtstep)
        elif self.NNR is None:
            return eng.run_function(1, 'NL_TSTL_LargestLyap', x, self.Nref, self.maxtstep, self.past)
        elif self.embedParams is None:
            return eng.run_function(1, 'NL_TSTL_LargestLyap', x, self.Nref, self.maxtstep, self.past, self.NNR)
        return eng.run_function(1, 'NL_TSTL_LargestLyap', x, self.Nref, self.maxtstep, self.past,
                                self.NNR, self.embedParams)


class NL_TSTL_PoincareSection(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Obtains a Poincare section of the time-delay embedded time series, producing a
    % set of vector points projected orthogonal to the tangential vector at the
    % specified index using TSTOOL code 'poincare'.
    %
    %---INPUTS:
    % y, the input time series
    %
    % ref: the reference point. Can be an absolute number (2 takes the second point
    %      in the (embedded) time series) or a string like 'max' or 'min' that takes
    %      the first maximum, ... in the (scalar) time series, ...
    %
    % embedParams: the usual thing to give BF_embed for the time-delay embedding, as
    %               {tau,m}. A common choice for m is 3 -- i.e., embed in a 3
    %               dimensional space so that the Poincare section is 2-dimensional.
    %
    %---OUTPUTS: include statistics on the x- and y- components of these vectors on the
    % Poincare surface, on distances between adjacent points, distances from the
    % mean position, and the entropy of the vector cloud.
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    
    % Another thing that could be cool to do is to analyze variation in the plots as
    % ref changes... (not done here)
    %
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (29, 30, 30)

    TAGS = ('nonlinear', 'poincare', 'tstool')

    def __init__(self, ref='max', embedParams=array(('mi', 3), dtype=object)):
        super(NL_TSTL_PoincareSection, self).__init__()
        self.ref = ref
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.ref is None:
            return eng.run_function(1, 'NL_TSTL_PoincareSection', x, )
        elif self.embedParams is None:
            return eng.run_function(1, 'NL_TSTL_PoincareSection', x, self.ref)
        return eng.run_function(1, 'NL_TSTL_PoincareSection', x, self.ref, self.embedParams)


class HCTSA_NL_TSTL_ReturnTime(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Return times are the time taken for the time series to return to a similar
    % location in phase space for a given reference point
    %
    % Strong peaks in the histogram are indicative of periodicities in the data.
    %
    %---INPUTS:
    %
    % y, scalar time series as a column vector
    % NNR, number of nearest neighbours
    % maxT, maximum return time to consider
    % past, Theiler window
    % Nref, number of reference indicies
    % embedParams, to feed into BF_embed
    %
    %---OUTPUTS: include basic measures from the histogram, including the occurrence of
    % peaks, spread, proportion of zeros, and the distributional entropy.
    
    % Uses the code, return_time, from TSTOOL.
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (35, 33, 32)

    TAGS = ('nonlinear', 'returntime', 'tstool')

    def __init__(self, NNR=10.0, maxT=1.0, past=1.0, Nref=-1.0, embedParams=array(('ac', 8), dtype=object)):
        super(HCTSA_NL_TSTL_ReturnTime, self).__init__()
        self.NNR = NNR
        self.maxT = maxT
        self.past = past
        self.Nref = Nref
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.NNR is None:
            return eng.run_function(1, 'NL_TSTL_ReturnTime', x, )
        elif self.maxT is None:
            return eng.run_function(1, 'NL_TSTL_ReturnTime', x, self.NNR)
        elif self.past is None:
            return eng.run_function(1, 'NL_TSTL_ReturnTime', x, self.NNR, self.maxT)
        elif self.Nref is None:
            return eng.run_function(1, 'NL_TSTL_ReturnTime', x, self.NNR, self.maxT, self.past)
        elif self.embedParams is None:
            return eng.run_function(1, 'NL_TSTL_ReturnTime', x, self.NNR, self.maxT, self.past, self.Nref)
        return eng.run_function(1, 'NL_TSTL_ReturnTime', x, self.NNR, self.maxT, self.past,
                                self.Nref, self.embedParams)


class NL_TSTL_TakensEstimator(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % cf. "Detecting strange attractors in turbulence", F. Takens.
    % Lect. Notes Math. 898 p366 (1981)
    %
    %---INPUTS:
    % y, the input time series
    % Nref, the number of reference points (can be -1 to use all points)
    % rad, the maximum search radius (as a proportion of the attractor size)
    % past, the Theiler window
    % embedParams, the embedding parameters for BF_embed, in the form {tau,m}
    %
    %---OUTPUT: the Taken's estimator of the correlation dimension, d2.
    %
    % Uses the TSTOOL code, takens_estimator.
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (38, 38, 39, 39, 44, 38)

    TAGS = ('crptool', 'dimension', 'nonlinear', 'scaling', 'takens', 'tstool')

    def __init__(self, Nref=-1.0, rad=0.05, past=0.05, embedParams=array((1,
                 8), dtype=object), randomSeed=None):
        super(NL_TSTL_TakensEstimator, self).__init__()
        self.Nref = Nref
        self.rad = rad
        self.past = past
        self.embedParams = embedParams
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.Nref is None:
            return eng.run_function(1, 'NL_TSTL_TakensEstimator', x, )
        elif self.rad is None:
            return eng.run_function(1, 'NL_TSTL_TakensEstimator', x, self.Nref)
        elif self.past is None:
            return eng.run_function(1, 'NL_TSTL_TakensEstimator', x, self.Nref, self.rad)
        elif self.embedParams is None:
            return eng.run_function(1, 'NL_TSTL_TakensEstimator', x, self.Nref, self.rad, self.past)
        elif self.randomSeed is None:
            return eng.run_function(1, 'NL_TSTL_TakensEstimator', x, self.Nref, self.rad, self.past, self.embedParams)
        return eng.run_function(1, 'NL_TSTL_TakensEstimator', x, self.Nref, self.rad, self.past,
                                self.embedParams, self.randomSeed)


class NL_TSTL_acp(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The documentation isn't crystal clear, but this function seems to be related
    % to cross-prediction.
    %
    %---INPUTS:
    % y, time series
    % tau, delay time
    % past, number of samples to exclude before and after each index (to avoid
    %               correlation effects ~ Theiler window)
    % maxDelay, maximal delay (<< length(y))
    % maxDim, maximal dimension to use
    % Nref, number of reference points
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %
    %---OUTPUTS: statistics summarizing the output of the routine.
    
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    % May in future want to also make outputs normalized by first value; so get
    % metrics on both absolute values at each dimension but also some
    % indication of the shape
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (27, 20)

    TAGS = ('acp', 'correlation', 'nonlinear')

    def __init__(self, tau='mi', past=1.0, maxDelay=array((), dtype=float64),
                 maxDim=10.0, Nref=array((), dtype=float64), randomSeed=None):
        super(NL_TSTL_acp, self).__init__()
        self.tau = tau
        self.past = past
        self.maxDelay = maxDelay
        self.maxDim = maxDim
        self.Nref = Nref
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'NL_TSTL_acp', x, )
        elif self.past is None:
            return eng.run_function(1, 'NL_TSTL_acp', x, self.tau)
        elif self.maxDelay is None:
            return eng.run_function(1, 'NL_TSTL_acp', x, self.tau, self.past)
        elif self.maxDim is None:
            return eng.run_function(1, 'NL_TSTL_acp', x, self.tau, self.past, self.maxDelay)
        elif self.Nref is None:
            return eng.run_function(1, 'NL_TSTL_acp', x, self.tau, self.past, self.maxDelay, self.maxDim)
        elif self.randomSeed is None:
            return eng.run_function(1, 'NL_TSTL_acp', x, self.tau, self.past, self.maxDelay, self.maxDim, self.Nref)
        return eng.run_function(1, 'NL_TSTL_acp', x, self.tau, self.past, self.maxDelay, self.maxDim,
                                self.Nref, self.randomSeed)


class NL_TSTL_dimensions(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Computes the box counting, information, and correlation dimension of a
    % time-delay embedded time series using the TSTOOL code 'dimensions'.
    % This function contains extensive code for estimating the best scaling range to
    % estimate the dimension using a penalized regression procedure.
    %
    %---INPUTS:
    % y, column vector of time series data
    % nbins, maximum number of partitions per axis
    % embedParams, embedding parameters to feed BF_embed.m for embedding the
    %              signal in the form {tau,m}
    %
    %---OUTPUTS:
    % A range of statistics are returned about how each dimension estimate changes
    % with m, the scaling range in r, and the embedding dimension at which the best
    % fit is obtained.
    
    % cf. TSTOOL, http://www.physik3.gwdg.de/tstool/
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (31,)

    TAGS = ('dimension', 'nonlinear', 'scaling', 'tstool')

    def __init__(self, nbins=50.0, embedParams=array(('ac', 'fnnmar'), dtype=object)):
        super(NL_TSTL_dimensions, self).__init__()
        self.nbins = nbins
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.nbins is None:
            return eng.run_function(1, 'NL_TSTL_dimensions', x, )
        elif self.embedParams is None:
            return eng.run_function(1, 'NL_TSTL_dimensions', x, self.nbins)
        return eng.run_function(1, 'NL_TSTL_dimensions', x, self.nbins, self.embedParams)


class NL_crptool_fnn(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, the input time series
    % maxm, the maximum embedding dimension to consider
    % r, the threshold; neighbourhood criterion
    % taum, the method of determining the time delay, 'corr' for first zero-crossing
    %       of autocorrelation function, or 'mi' for the first minimum of the mutual
    %       information
    %
    % th [opt], returns the first time the number of false nearest neighbours drops
    %           under this threshold
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    
    % Computation uses N. Marwan's code from the CRP Toolbox:
    % http://tocsy.pik-potsdam.de/CRPtoolbox/
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21, 22, 22)

    TAGS = ('crptool', 'dimension', 'nonlinear')

    def __init__(self, maxm=10.0, r=2.0, taum='mi', th=array((), dtype=float64), randomSeed='default'):
        super(NL_crptool_fnn, self).__init__()
        self.maxm = maxm
        self.r = r
        self.taum = taum
        self.th = th
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.maxm is None:
            return eng.run_function(1, 'NL_crptool_fnn', x, )
        elif self.r is None:
            return eng.run_function(1, 'NL_crptool_fnn', x, self.maxm)
        elif self.taum is None:
            return eng.run_function(1, 'NL_crptool_fnn', x, self.maxm, self.r)
        elif self.th is None:
            return eng.run_function(1, 'NL_crptool_fnn', x, self.maxm, self.r, self.taum)
        elif self.randomSeed is None:
            return eng.run_function(1, 'NL_crptool_fnn', x, self.maxm, self.r, self.taum, self.th)
        return eng.run_function(1, 'NL_crptool_fnn', x, self.maxm, self.r, self.taum, self.th, self.randomSeed)


class NL_embed_PCA(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Reconstructs the time series as a time-delay embedding, and performs Principal
    % Components Analysis on the result using princomp code from
    % Matlab's Bioinformatics Toolbox.
    %
    % This technique is known as singular spectrum analysis.
    %
    % "Extracting qualitative dynamics from experimental data"
    % D. S. Broomhead and G. P. King, Physica D 20(2-3) 217 (1986)
    %
    %---INPUTS:
    % y, the input time series
    %
    % tau, the time-delay, can be an integer or 'ac', or 'mi' for first
    %               zero-crossing of the autocorrelation function or first minimum
    %               of the automutual information, respectively
    %
    % m, the embedding dimension
    %
    % OUTPUTS: Various statistics summarizing the obtained eigenvalue distribution.
    %
    % The suggestion to implement this idea was provided by Siddarth Arora.
    % (Siddharth Arora, <arora@maths.ox.ac.uk>)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (17, 17, 18)

    TAGS = ('pca', 'tdembedding')

    def __init__(self, tau='mi', m=10.0):
        super(NL_embed_PCA, self).__init__()
        self.tau = tau
        self.m = m

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'NL_embed_PCA', x, )
        elif self.m is None:
            return eng.run_function(1, 'NL_embed_PCA', x, self.tau)
        return eng.run_function(1, 'NL_embed_PCA', x, self.tau, self.m)


class NW_VisibilityGraph(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Constructs a visibility graph of the time series and returns various
    % statistics on the properties of the resulting network.
    %
    % cf.: "From time series to complex networks: The visibility graph"
    % Lacasa, Lucas and Luque, Bartolo and Ballesteros, Fernando and Luque, Jordi
    % and Nuno, Juan Carlos P. Natl. Acad. Sci. USA. 105(13) 4972 (2008)
    %
    % "Horizontal visibility graphs: Exact results for random time series"
    % Luque, B. and Lacasa, L. and Ballesteros, F. and Luque, J.
    % Phys. Rev. E. 80(4) 046103 (2009)
    %
    %---INPUTS:
    % y, the time series (a column vector)
    %
    % meth, the method for constructing:
    % 			(i) 'norm': the normal visibility definition
    % 			(ii) 'horiz': uses only horizonatal lines to link nodes/datums
    %
    % maxL, the maximum number of samples to consider. Due to memory constraints,
    %               only the first maxL (6000 by default) points of time series are
    %               analyzed. Longer time series are reduced to their first maxL
    %               samples.
    %
    %---OUTPUTS:
    % Statistics on the degree distribution, including the mode, mean,
    % spread, histogram entropy, and fits to gaussian, exponential, and powerlaw
    % distributions.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (24, 23)

    TAGS = ('lengthdep', 'network', 'visibilitygraph')

    def __init__(self, meth='horiz', maxL=None):
        super(NW_VisibilityGraph, self).__init__()
        self.meth = meth
        self.maxL = maxL

    def _eval_hook(self, eng, x):
        if self.meth is None:
            return eng.run_function(1, 'NW_VisibilityGraph', x, )
        elif self.maxL is None:
            return eng.run_function(1, 'NW_VisibilityGraph', x, self.meth)
        return eng.run_function(1, 'NW_VisibilityGraph', x, self.meth, self.maxL)


class PD_PeriodicityWang(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Implements an idea based on the periodicity extraction measure proposed in:
    %
    % "Structure-based Statistical Features and Multivariate Time Series Clustering"
    % Wang, X. and Wirth, A. and Wang, L.
    % Seventh IEEE International Conference on Data Mining, 351--360 (2007)
    % DOI: 10.1109/ICDM.2007.103
    %
    % Detrends the time series using a single-knot cubic regression spline
    % and then computes autocorrelations up to one third of the length of
    % the time series. The frequency is the first peak in the autocorrelation
    % function satisfying a set of conditions.
    %
    %---INPUT:
    % y, the input time series.
    %
    % The single threshold of 0.01 was considered in the original paper, this code
    % uses a range of thresholds: 0, 0.01, 0.1, 0.2, 1\sqrt{N}, 5\sqrt{N}, and
    % 10\sqrt{N}, where N is the length of the time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (18,)

    TAGS = ('periodicity', 'spline')

    def __init__(self):
        super(PD_PeriodicityWang, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'PD_PeriodicityWang', x, )


class PH_ForcePotential(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The input time series forces a particle in the given potential well.
    %
    % The time series contributes to a forcing term on a simulated particle in a:
    %
    % (i) Quartic double-well potential with potential energy V(x) = x^4/4 - alpha^2
    %           x^2/2, or a force F(x) = -x^3 + alpha^2 x
    %
    % (ii) Sinusoidal potential with V(x) = -cos(x/alpha), or F(x) = sin(x/alpha)/alpha
    %
    %---INPUTS:
    % y, the input time series
    %
    % whatPotential, the potential function to simulate:
    %               (i) 'dblwell' (a double well potential function)
    %               (ii) 'sine' (a sinusoidal potential function)
    %
    % params, the parameters for simulation, should be in the form:
    %                   params = [alpha, kappa, deltat]
    %
    %           (i) The double-well potential has three parameters:
    %               * alpha controls the relative positions of the wells,
    %               * kappa is the coefficient of friction,
    %               * deltat sets the time step for the simulation.
    %
    %           (ii) The sinusoidal potential also has three parameters:
    %               * alpha controls the period of oscillations in the potential
    %               * kappa is the coefficient of friction,
    %               * deltat sets the time step for the simulation.
    %
    %---OUTPUTS:
    % Statistics summarizing the trajectory of the simulated particle,
    % including its mean, the range, proportion positive, proportion of times it
    % crosses zero, its autocorrelation, final position, and standard deviation.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (33, 33, 34, 34, 32, 28, 29)

    TAGS = ('dblwell', 'dynsys', 'sine')

    def __init__(self, whatPotential='dblwell', params=array((1. , 0.2, 0.1))):
        super(PH_ForcePotential, self).__init__()
        self.whatPotential = whatPotential
        self.params = params

    def _eval_hook(self, eng, x):
        if self.whatPotential is None:
            return eng.run_function(1, 'PH_ForcePotential', x, )
        elif self.params is None:
            return eng.run_function(1, 'PH_ForcePotential', x, self.whatPotential)
        return eng.run_function(1, 'PH_ForcePotential', x, self.whatPotential, self.params)


class PH_Walker(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The hypothetical particle (or 'walker') moves in response to values of the
    % time series at each point.
    %
    % Outputs from this operation are summaries of the walker's motion, and
    % comparisons of it to the original time series.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % walkerRule, the kinematic rule by which the walker moves in response to the
    %             time series over time:
    %
    %            (i) 'prop': the walker narrows the gap between its value and that
    %                        of the time series by a given proportion p.
    %                        walkerParams = p;
    %
    %            (ii) 'biasprop': the walker is biased to move more in one
    %                         direction; when it is being pushed up by the time
    %                         series, it narrows the gap by a proportion p_{up},
    %                         and when it is being pushed down by the time series,
    %                         it narrows the gap by a (potentially different)
    %                         proportion p_{down}. walkerParams = [pup,pdown].
    %
    %            (iii) 'momentum': the walker moves as if it has mass m and inertia
    %                         from the previous time step and the time series acts
    %                         as a force altering its motion in a classical
    %                         Newtonian dynamics framework. [walkerParams = m], the mass.
    %
    %             (iv) 'runningvar': the walker moves with inertia as above, but
    %                         its values are also adjusted so as to match the local
    %                         variance of time series by a multiplicative factor.
    %                         walkerParams = [m,wl], where m is the inertial mass and wl
    %                         is the window length.
    %
    % walkerParams, the parameters for the specified walkerRule, explained above.
    %
    %---OUTPUTS: include the mean, spread, maximum, minimum, and autocorrelation of
    % the walker's trajectory, the number of crossings between the walker and the
    % original time series, the ratio or difference of some basic summary statistics
    % between the original time series and the walker, an Ansari-Bradley test
    % comparing the distributions of the walker and original time series, and
    % various statistics summarizing properties of the residuals between the
    % walker's trajectory and the original time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (24, 24, 20, 20, 17, 17, 17, 17, 26)

    TAGS = ('trend',)

    def __init__(self, walkerRule='prop', walkerParams=0.1):
        super(PH_Walker, self).__init__()
        self.walkerRule = walkerRule
        self.walkerParams = walkerParams

    def _eval_hook(self, eng, x):
        if self.walkerRule is None:
            return eng.run_function(1, 'PH_Walker', x, )
        elif self.walkerParams is None:
            return eng.run_function(1, 'PH_Walker', x, self.walkerRule)
        return eng.run_function(1, 'PH_Walker', x, self.walkerRule, self.walkerParams)


class PP_Compare(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Applies a given pre-processing transformation to the time series, and returns
    % statistics on how various time-series properties change as a result.
    %
    % Inputs are structured in a clunky way, unfortunately:
    %
    %---INPUTS:
    % y, the input time series
    % detrndmeth, the method to use for detrending:
    %      (i) 'poly': polynomial detrendings, both linear and quadratic. Can
    %                  be of the following forms:
    %            (a) polynomial of given order: 'poly1', 'poly2', 'poly3',
    %                'poly4', 'poly5', 'poly6', 'poly7', 'poly8', 'poly9'
    %            (b) fit best polynomial: 'polybest' determines 'best' by
    %                            various tests (e.g., whiteness of residuals,
    %                            etc.)
    %            (c) 'fitstrong' only fits if a 'strong' trend.
    %      (ii) 'sin': sinusoidal detrending with either one or two frequency
    %                components,
    %            (a) fit a sine series of a given order
    %               Fits a form like: a1*sin(b1*x+c1) + a2*sin(b2*x+c2) + ...
    %               Additional number determines how many terms to include in the
    %               series: 'sin1', 'sin2', 'sin3', 'sin4', 'sin5', 'sin6', 'sin7',
    %               'sin8'
    %            (b) 'sin_st1': fit only if a strong trend (i.e., if the amplitudes
    %                                           are above a given threshold)
    %      (iii) 'spline': removes a least squares spline using Matlab's
    %                      Spline Toolbox function spap2
    %                      Input of the form 'spline<nknots><interpolant_order>'
    %                      e.g., 'spline45' uses four knots and 5th order
    %                      interpolants (Implements a least squares spline via the
    %                      spline toolbox function spap2)
    %      (iv) 'diff': takes incremental differences of the time series. Of form
    %                 'diff<ndiff>' where ndiff is the number of differencings to
    %                 perform. e.g., 'diff3' performs three recursive differences
    %      (v) 'medianf': a running median filter using a given window lengths
    %                   Of form 'medianf<n>' where n is the window length.
    %                   e.g., 'medianf3' takes a running median using the median of
    %                     every 3 consecutive values as a point in the filtered
    %                     time series. Uses the Signal Processing Toolbox
    %                     function medfilt1
    %      (vi) 'rav': running mean filter of a given window length.
    %                  Uses Matlab's filter function to perform a running
    %                  average of order n. Of form 'rav<n>' where n is the order of
    %                  the running average.
    %      (vii) 'resample': resamples the data by a given ratio using the resample
    %                        function in Matlab.
    %                        Of form 'resample_<p>_<q>', where the ratio p/q is the
    %                        new sampling rate e.g., 'resample_1_2' will downsample
    %                        the signal by one half e.g., resample_10_1' will
    %                        resample the signal to 10 times its original length
    %      (viii) 'logr': takes log returns of the data. Only valid for positive
    %                       data, else returns a NaN.
    %      (ix) 'boxcox': makes a Box-Cox transformation of the data. Only valid for
    %                     positive only data; otherwise returns a NaN.
    %
    % If multiple detrending methods are specified, they should be in a cell of
    % strings; the methods will be executed in order, e.g., {'poly1','sin1'} does a
    % linear polynomial then a simple one-frequency seasonal detrend (in that order)
    %
    %---OUTPUTS: include comparisons of stationarity and distributional measures
    % between the original and transformed time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (16, 16, 20, 19, 19, 16, 16, 16, 15, 15, 15, 23, 23, 15, 15, 19, 19, 19)

    TAGS = ('locdep', 'preprocessing', 'raw', 'spline')

    def __init__(self, detrndmeth='poly1'):
        super(PP_Compare, self).__init__()
        self.detrndmeth = detrndmeth

    def _eval_hook(self, eng, x):
        if self.detrndmeth is None:
            return eng.run_function(1, 'PP_Compare', x, )
        return eng.run_function(1, 'PP_Compare', x, self.detrndmeth)


class PP_Iterate(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The pre-processing transformation is iteratively applied to the time series.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % dtMeth, the detrending method to apply:
    %           (i) 'spline' removes a spine fit,
    %           (ii) 'diff' takes incremental differences,
    %           (iii) 'medianf' applies a median filter,
    %           (iv) 'rav' applies a running mean filter,
    %           (v) 'resampleup' progressively upsamples the time series,
    %           (vi) 'resampledown' progressively downsamples the time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (15,)

    TAGS = ('preprocessing', 'raw')

    def __init__(self, dtMeth='diff'):
        super(PP_Iterate, self).__init__()
        self.dtMeth = dtMeth

    def _eval_hook(self, eng, x):
        if self.dtMeth is None:
            return eng.run_function(1, 'PP_Iterate', x, )
        return eng.run_function(1, 'PP_Iterate', x, self.dtMeth)


class PP_ModelFit(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % After performing the range of transformations to the time series, returns the
    % in-sample root-mean-square (RMS) prediction errors for an AR model on each
    % transformed time series as a ratio of the RMS prediction error of the original
    % time series.
    %
    % PP_PreProcess.m is used to perform the preprocessings
    %
    % The AR model is fitted using the function ar and pe from Matlab's System
    % Identification Toolbox
    %
    % Transformations performed include:
    % (i) incremental differencing,
    % (ii) filtering of the power spectral density function,
    % (iii) removal of piece-wise polynomial trends, and
    % (iv) rank mapping the values of the time series to a Gaussian distribution.
    %
    %---INPUTS:
    %
    % y, the input time series
    % model, the time-series model to fit to the transformed time series (currently
    %           'ar' is the only option)
    %
    % order, the order of the AR model to fit to the data
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (16,)

    TAGS = ('preprocessing', 'trend')

    def __init__(self, model='ar', order=2.0, randomSeed='default'):
        super(PP_ModelFit, self).__init__()
        self.model = model
        self.order = order
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.model is None:
            return eng.run_function(1, 'PP_ModelFit', x, )
        elif self.order is None:
            return eng.run_function(1, 'PP_ModelFit', x, self.model)
        elif self.randomSeed is None:
            return eng.run_function(1, 'PP_ModelFit', x, self.model, self.order)
        return eng.run_function(1, 'PP_ModelFit', x, self.model, self.order, self.randomSeed)


class SB_BinaryStats(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Binary symbolization of the time series is a symbolic string of 0s and 1s.
    %
    % Provides information about the coarse-grained behavior of the time series
    %
    %---INPUTS:
    % y, the input time series
    %
    % binaryMethod, the symbolization rule:
    %         (i) 'diff': by whether incremental differences of the time series are
    %                      positive (1), or negative (0),
    %          (ii) 'mean': by whether each point is above (1) or below the mean (0)
    %          (iii) 'iqr': by whether the time series is within the interquartile range
    %                      (1), or not (0).
    %
    %---OUTPUTS:
    % Include the Shannon entropy of the string, the longest stretches of 0s
    % or 1s, the mean length of consecutive 0s or 1s, and the spread of consecutive
    % strings of 0s or 1s.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (19, 18, 19)

    TAGS = ('distribution', 'stationarity')

    def __init__(self, binaryMethod='diff'):
        super(SB_BinaryStats, self).__init__()
        self.binaryMethod = binaryMethod

    def _eval_hook(self, eng, x):
        if self.binaryMethod is None:
            return eng.run_function(1, 'SB_BinaryStats', x, )
        return eng.run_function(1, 'SB_BinaryStats', x, self.binaryMethod)


class SB_BinaryStretch(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % (DOESN'T ACTUALLY, see note) measure the longest stretch of consecutive zeros or ones in
    % a symbolized time series as a proportion of the time-series length.
    %
    % The time series is symbolized to a binary string by whether it's above (1) or
    % below (0) its mean.
    %
    %---INPUTS:
    %
    % x, the input time series
    %
    % stretchWhat, (i) 'lseq1', measures something related to consecutive 1s
    %              (ii) 'lseq0', measures something related to consecutive 0s
    %
    %---NOTES:
    % It doesn't actually measure what it's supposed to measure correctly, due to an
    % implementation error, but it's still kind of an interesting operation...!
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (22, 22)

    TAGS = ('binary',)

    def __init__(self, stretchWhat='lseq1'):
        super(SB_BinaryStretch, self).__init__()
        self.stretchWhat = stretchWhat

    def _eval_hook(self, eng, x):
        if self.stretchWhat is None:
            return eng.run_function(1, 'SB_BinaryStretch', x, )
        return eng.run_function(1, 'SB_BinaryStretch', x, self.stretchWhat)


class SB_CoarseGrain(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % howtocg, the method of coarse-graining
    %
    % numGroups, either specifies the size of the alphabet for 'quantile' and 'updown'
    %       or sets the timedelay for the embedding subroutines
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, howtocg=None, numGroups=None):
        super(SB_CoarseGrain, self).__init__()
        self.howtocg = howtocg
        self.numGroups = numGroups

    def _eval_hook(self, eng, x):
        if self.howtocg is None:
            return eng.run_function(1, 'SB_CoarseGrain', x, )
        elif self.numGroups is None:
            return eng.run_function(1, 'SB_CoarseGrain', x, self.howtocg)
        return eng.run_function(1, 'SB_CoarseGrain', x, self.howtocg, self.numGroups)


class SB_MotifThree(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % (As SB_MotifTwo but with a 3-letter alphabet)
    %
    %---INPUTS:
    % y, time series to analyze
    % cgHow, the coarse-graining method to use:
    %       (i) 'quantile': equiprobable alphabet by time-series value
    %       (ii) 'diffquant': equiprobably alphabet by time-series increments
    %
    %---OUTPUTS:
    % Statistics on words of length 1, 2, 3, and 4.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (23, 22)

    TAGS = ('motifs',)

    def __init__(self, cgHow='quantile'):
        super(SB_MotifThree, self).__init__()
        self.cgHow = cgHow

    def _eval_hook(self, eng, x):
        if self.cgHow is None:
            return eng.run_function(1, 'SB_MotifThree', x, )
        return eng.run_function(1, 'SB_MotifThree', x, self.cgHow)


class SB_MotifTwo(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Coarse-graining is performed by a given binarization method.
    %
    %---INPUTS:
    % y, the input time series
    % binarizeHow, the binary transformation method:
    %       (i) 'diff': incremental time-series increases are encoded as 1, and
    %                   decreases as 0,
    %       (ii) 'mean': time-series values above its mean are given 1, and those
    %                    below the mean are 0,
    %       (iii) 'median': time-series values above the median are given 1, and
    %       those below the median 0.
    %
    %---OUTPUTS:
    % Probabilities of words in the binary alphabet of lengths 1, 2, 3, and 4, and
    % their entropies.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (16, 16, 18)

    TAGS = ('motifs',)

    def __init__(self, binarizeHow='diff'):
        super(SB_MotifTwo, self).__init__()
        self.binarizeHow = binarizeHow

    def _eval_hook(self, eng, x):
        if self.binarizeHow is None:
            return eng.run_function(1, 'SB_MotifTwo', x, )
        return eng.run_function(1, 'SB_MotifTwo', x, self.binarizeHow)


class SB_TransitionMatrix(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The time series is coarse-grained according to a given method.
    %
    % The input time series is transformed into a symbolic string using an
    % equiprobable alphabet of numGroups letters. The transition probabilities are
    % calculated at a lag tau.
    %
    %---INPUTS:
    % y, the input time series
    %
    % howtocg, the method of discretization (currently 'quantile' is the only
    %           option; could incorporate SB_CoarseGrain for more options in future)
    %
    % numGroups: number of groups in the course-graining
    %
    % tau: analyze transition matricies corresponding to this lag. We
    %      could either downsample the time series at this lag and then do the
    %      discretization as normal, or do the discretization and then just
    %      look at this dicrete lag. Here we do the former. Can also set tau to 'ac'
    %      to set tau to the first zero-crossing of the autocorrelation function.
    %
    %---OUTPUTS: include the transition probabilities themselves, as well as the trace
    % of the transition matrix, measures of asymmetry, and eigenvalues of the
    % transition matrix.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (22, 23, 22, 23, 22, 23, 22, 23)

    TAGS = ('transitionmat',)

    def __init__(self, howtocg='quantile', numGroups=2.0, tau=1.0):
        super(SB_TransitionMatrix, self).__init__()
        self.howtocg = howtocg
        self.numGroups = numGroups
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.howtocg is None:
            return eng.run_function(1, 'SB_TransitionMatrix', x, )
        elif self.numGroups is None:
            return eng.run_function(1, 'SB_TransitionMatrix', x, self.howtocg)
        elif self.tau is None:
            return eng.run_function(1, 'SB_TransitionMatrix', x, self.howtocg, self.numGroups)
        return eng.run_function(1, 'SB_TransitionMatrix', x, self.howtocg, self.numGroups, self.tau)


class SB_TransitionpAlphabet(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Discretization is done by quantile separation.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % numGroups, the number of groups in the coarse-graining (scalar for constant, or a
    %       vector of numGroups to compare across this range)
    %
    % tau: the time-delay; transition matricies corresponding to this time-delay. We
    %      can either downsample the time series at this lag and then do the
    %      discretization as normal, or do the discretization and then just
    %      look at this dicrete lag. Here we do the former. (scalar for
    %      constant tau, vector for range to vary across)
    %
    %---OUTPUTS: include the decay rate of the sum, mean, and maximum of diagonal
    % elements of the transition matrices, changes in symmetry, and the eigenvalues
    % of the transition matrix.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (28, 27)

    TAGS = ('transitionmat',)

    def __init__(self, numGroups=MatlabSequence('2:40'), tau=1.0):
        super(SB_TransitionpAlphabet, self).__init__()
        self.numGroups = numGroups
        self.tau = tau

    def _eval_hook(self, eng, x):
        if self.numGroups is None:
            return eng.run_function(1, 'SB_TransitionpAlphabet', x, )
        elif self.tau is None:
            return eng.run_function(1, 'SB_TransitionpAlphabet', x, self.numGroups)
        return eng.run_function(1, 'SB_TransitionpAlphabet', x, self.numGroups, self.tau)


class SC_FluctAnal(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Much of our implementation is based on the well-explained discussion of
    % scaling methods in:
    % "Power spectrum and detrended fluctuation analysis: Application to daily
    % temperatures" P. Talkner and R. O. Weber, Phys. Rev. E 62(1) 150 (2000)
    %
    % The main difference between algorithms for estimating scaling exponents amount
    % to differences in how fluctuations, F, are quantified in time-series segments.
    % Many alternatives are implemented in this function.
    %
    %---INPUTS:
    % y, the input time series
    %
    % q, the parameter in the fluctuation function q = 2 (usual) gives RMS fluctuations.
    %
    % wtf, (what to fluctuate)
    %       (i) 'endptdiff', calculates the differences in end points in each segment
    %       (ii) 'range' calculates the range in each segment
    %       (iii) 'std' takes the standard deviation in each segment
    %
    %           cf. "Evaluating scaled windowed variance methods for estimating the
    %               Hurst coefficient of time series", M. J. Cannon et al. Physica A
    %               241(3-4) 606 (1997)
    %
    %       (iv) 'iqr' takes the interquartile range in each segment
    %       (v) 'dfa' removes a polynomial trend of order k in each segment,
    %       (vi) 'rsrange' returns the range after removing a straight line fit
    %
    %           cf. "Analyzing exact fractal time series: evaluating dispersional
    %           analysis and rescaled range methods",  D. C. Caccia et al., Physica
    %           A 246(3-4) 609 (1997)
    %
    %       (vii) 'rsrangefit' fits a polynomial of order k and then returns the
    %           range. The parameter q controls the order of fluctuations, for which
    %           we mostly use the standard choice, q = 2, corresponding to root mean
    %           square fluctuations.
    %           An optional input parameter to this operation is a timelag for
    %           computing the cumulative sum (or integrated profile), as suggested
    %           by: "Using detrended fluctuation analysis for lagged correlation
    %           analysis of nonstationary signals" J. Alvarez-Ramirez et al. Phys.
    %           Rev. E 79(5) 057202 (2009)
    %
    % tauStep, increments in tau for linear range (i.e., if logInc = 0), or number of tau
    %           steps in logarithmic range if login = 1
    %           The spacing of time scales, tau, is commonly logarithmic through a range from
    %           5 samples to a quarter of the length of the time series, as suggested in
    %           "Statistical properties of DNA sequences", C.-K. Peng et al. Physica A
    %           221(1-3) 180 (1995)
    %
    %           Max A. Little's fractal paper used L = 4 to L = N/2:
    %           "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection"
    %           M. A. Little et al. Biomed. Eng. Online 6(1) 23 (2007)
    %
    % k, polynomial order of detrending for 'dfa', 'rsrangefit'
    %
    % lag, optional time-lag, as in Alvarez-Ramirez (see (vii) above)
    %
    % logInc, whether to use logarithmic increments in tau (it should be logarithmic).
    %
    %---OUTPUTS: include statistics of fitting a linear function to a plot of log(F) as
    % a function of log(tau), and for fitting two straight lines to the same data,
    % choosing the split point at tau = tau_{split} as that which minimizes the
    % combined fitting errors.
    %
    % This function can also be applied to the absolute deviations of the time
    % series from its mean, and also for just the sign of deviations from the mean
    % (i.e., converting the time series into a series of +1, when the time series is
    % above its mean, and -1 when the time series is below its mean).
    %
    % All results are obtained with both linearly, and logarithmically-spaced time
    % scales tau.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (28, 30, 30, 28, 28, 28, 32, 26, 30, 28, 30, 35, 26, 32, 33)

    TAGS = ('dfa', 'fa', 'rsrange', 'scaling')

    def __init__(self, q=2.0, wtf='nothing', tauStep=50.0, k=array((), dtype=float64),
                 lag=array((), dtype=float64), logInc=1.0):
        super(SC_FluctAnal, self).__init__()
        self.q = q
        self.wtf = wtf
        self.tauStep = tauStep
        self.k = k
        self.lag = lag
        self.logInc = logInc

    def _eval_hook(self, eng, x):
        if self.q is None:
            return eng.run_function(1, 'SC_FluctAnal', x, )
        elif self.wtf is None:
            return eng.run_function(1, 'SC_FluctAnal', x, self.q)
        elif self.tauStep is None:
            return eng.run_function(1, 'SC_FluctAnal', x, self.q, self.wtf)
        elif self.k is None:
            return eng.run_function(1, 'SC_FluctAnal', x, self.q, self.wtf, self.tauStep)
        elif self.lag is None:
            return eng.run_function(1, 'SC_FluctAnal', x, self.q, self.wtf, self.tauStep, self.k)
        elif self.logInc is None:
            return eng.run_function(1, 'SC_FluctAnal', x, self.q, self.wtf, self.tauStep, self.k, self.lag)
        return eng.run_function(1, 'SC_FluctAnal', x, self.q, self.wtf, self.tauStep, self.k, self.lag, self.logInc)


class SC_MMA(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Scale-dependent estimates of multifractal scaling in a time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (14,)

    TAGS = ('fractal', 'scaling')

    def __init__(self, doOverlap=0.0, scaleRange=array((), dtype=float64), qRange=array((-5,  5))):
        super(SC_MMA, self).__init__()
        self.doOverlap = doOverlap
        self.scaleRange = scaleRange
        self.qRange = qRange

    def _eval_hook(self, eng, x):
        if self.doOverlap is None:
            return eng.run_function(1, 'SC_MMA', x, )
        elif self.scaleRange is None:
            return eng.run_function(1, 'SC_MMA', x, self.doOverlap)
        elif self.qRange is None:
            return eng.run_function(1, 'SC_MMA', x, self.doOverlap, self.scaleRange)
        return eng.run_function(1, 'SC_MMA', x, self.doOverlap, self.scaleRange, self.qRange)


class SC_fastdfa(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Measures the scaling exponent of the time series using a fast implementation
    % of detrended fluctuation analysis (DFA).
    %
    %---INPUT:
    % y, the input time series, is fed straight into the fastdfa script.
    
    % The original fastdfa code is by Max A. Little and publicly-available at
    % http://www.maxlittle.net/software/index.php
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (10,)

    TAGS = ('dfa', 'mex', 'scaling')

    def __init__(self):
        super(SC_fastdfa, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'SC_fastdfa', x, )


class SD_MakeSurrogates(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Method described relatively clearly in Guarin Lopez et al. (arXiv, 2010)
    % Used bits of aaft code that references (and presumably was obtained from)
    % "ìSurrogate data test for nonlinearity including monotonic
    % transformations", D. Kugiumtzis, Phys. Rev. E, vol. 62, no. 1, 2000.
    %
    % Note that many other surrogate data methods exist that could later be
    % implemented, cf. references in "Improvements to surrogate data methods for
    % nonstationary time series", J. H. Lucio et al., Phys. Rev. E 85, 056202 (2012)
    %
    %---INPUTS:
    % x, the input time series
    %
    % surrMethod, the method for generating surrogates:
    %             (i) 'RP' -- random phase surrogates
    %             (ii) 'AAFT' -- amplitude adjusted Fourier transform
    %             (iii) 'TFT' -- truncated Fourier transform
    %
    % numSurrs, the number of surrogates to generate
    %
    % extraParams, extra parameters required by the selected surrogate generation method
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, surrMethod=None, numSurrs=None, extraParams=None, randomSeed=None):
        super(SD_MakeSurrogates, self).__init__()
        self.surrMethod = surrMethod
        self.numSurrs = numSurrs
        self.extraParams = extraParams
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.surrMethod is None:
            return eng.run_function(1, 'SD_MakeSurrogates', x, )
        elif self.numSurrs is None:
            return eng.run_function(1, 'SD_MakeSurrogates', x, self.surrMethod)
        elif self.extraParams is None:
            return eng.run_function(1, 'SD_MakeSurrogates', x, self.surrMethod, self.numSurrs)
        elif self.randomSeed is None:
            return eng.run_function(1, 'SD_MakeSurrogates', x, self.surrMethod, self.numSurrs, self.extraParams)
        return eng.run_function(1, 'SD_MakeSurrogates', x, self.surrMethod, self.numSurrs, self.extraParams,
                                self.randomSeed)


class SD_SurrogateTest(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % This function is based on information found in:
    % "Surrogate data test for nonlinearity including nonmonotonic transforms"
    % D. Kugiumtzis Phys. Rev. E 62(1) R25 (2000)
    %
    % The generation of surrogates is done by the periphery function,
    % SD_MakeSurrogates
    %
    %---INPUTS:
    % x, the input time series
    %
    % surrMeth, the method for generating surrogate time series:
    %       (i) 'RP': random phase surrogates that maintain linear correlations in
    %                 the data but destroy any nonlinear structure through phase
    %                 randomization
    %       (ii) 'AAFT': the amplitude-adjusted Fourier transform method maintains
    %                    linear correlations but destroys nonlinear structure
    %                    through phase randomization, yet preserves the approximate
    %                    amplitude distribution,
    %       (iii) 'TFT': preserves low-frequency phases but randomizes high-frequency phases (as a way of dealing
    %                    with non-stationarity, cf.:
    %               "A new surrogate data method for nonstationary time series",
    %                   D. L. Guarin Lopez et al., arXiv 1008.1804 (2010)
    %
    % numSurrs, the number of surrogates to compute (default is 99 for a 0.01
    %         significance level 1-sided test)
    %
    % extrap, extra parameter, the cut-off frequency for 'TFT'
    %
    % theTestStat, the test statistic to evalute on all surrogates and the original
    %           time series. Can specify multiple options in a cell and will return
    %           output for each specified test statistic:
    %           (i) 'ami': the automutual information at lag 1, cf.
    %                 "Testing for nonlinearity in irregular fluctuations with
    %                 long-term trends" T. Nakamura and M. Small and Y. Hirata,
    %                 Phys. Rev. E 74(2) 026205 (2006)
    %           (ii) 'fmmi': the first minimum of the automutual information
    %                       function
    %           (iii) 'o3': a third-order statistic used in: "Surrogate time
    %                 series", T. Schreiber and A. Schmitz, Physica D 142(3-4) 346
    %                 (2000)
    %           (iv) 'tc3': a time-reversal asymmetry measure. Outputs of the
    %                 function include a z-test between the two distributions, and
    %                 some comparative rank-based statistics.
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (22,)

    TAGS = ('nonlinearity', 'surrogatedata')

    def __init__(self, surrMeth='RP', numSurrs=99.0, extrap=array((),
                 dtype=float64), theTestStat=array(('ami1', 'fmmi', 'o3', 'tc3'), dtype=object), randomSeed='default'):
        super(SD_SurrogateTest, self).__init__()
        self.surrMeth = surrMeth
        self.numSurrs = numSurrs
        self.extrap = extrap
        self.theTestStat = theTestStat
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.surrMeth is None:
            return eng.run_function(1, 'SD_SurrogateTest', x, )
        elif self.numSurrs is None:
            return eng.run_function(1, 'SD_SurrogateTest', x, self.surrMeth)
        elif self.extrap is None:
            return eng.run_function(1, 'SD_SurrogateTest', x, self.surrMeth, self.numSurrs)
        elif self.theTestStat is None:
            return eng.run_function(1, 'SD_SurrogateTest', x, self.surrMeth, self.numSurrs, self.extrap)
        elif self.randomSeed is None:
            return eng.run_function(1, 'SD_SurrogateTest', x, self.surrMeth, self.numSurrs,
                                    self.extrap, self.theTestStat)
        return eng.run_function(1, 'SD_SurrogateTest', x, self.surrMeth, self.numSurrs, self.extrap,
                                self.theTestStat, self.randomSeed)


class SD_TSTL_surrogates(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Generates surrogate time series and tests them against the original time
    % series according to some test statistics: T_{C3}, using the TSTOOL code tc3 or
    % T_{rev}, using TSTOOL code trev.
    %
    %---INPUTS:
    % y, the input time series
    %
    % tau, the autocorrelation lag length <x_n x_{n-tau} x_{n-2tau)>/abs(<x_n
    %                                                   x_{n-tau}|^3/2
    % nsurr, the number of surrogates to generate
    %
    % surrmeth, the method of generating surrogates:
    %               (i) 1: randomizes phases of fourier spectrum
    %               (ii) 2:  (see Theiler algorithm II)
    %               (iii) 3: permutes samples randomly
    %
    % surrfn, the surrogate statistic to evaluate on all surrogates, either 'tc3' or
    %           'trev'
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    %
    %---OUTPUTS: include the Gaussianity of the test statistics, a z-test, and
    % various tests based on fitted kernel densities.
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (30, 31, 30, 31, 30, 31, 31, 32, 31, 32, 31, 32)

    TAGS = ('correlation', 'nonlinear', 'surrogate', 'tstool')

    def __init__(self, tau=1.0, nsurr=100.0, surrMethod=1.0, surrfn='tc3', randomSeed='default'):
        super(SD_TSTL_surrogates, self).__init__()
        self.tau = tau
        self.nsurr = nsurr
        self.surrMethod = surrMethod
        self.surrfn = surrfn
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.tau is None:
            return eng.run_function(1, 'SD_TSTL_surrogates', x, )
        elif self.nsurr is None:
            return eng.run_function(1, 'SD_TSTL_surrogates', x, self.tau)
        elif self.surrMethod is None:
            return eng.run_function(1, 'SD_TSTL_surrogates', x, self.tau, self.nsurr)
        elif self.surrfn is None:
            return eng.run_function(1, 'SD_TSTL_surrogates', x, self.tau, self.nsurr, self.surrMethod)
        elif self.randomSeed is None:
            return eng.run_function(1, 'SD_TSTL_surrogates', x, self.tau, self.nsurr, self.surrMethod, self.surrfn)
        return eng.run_function(1, 'SD_TSTL_surrogates', x, self.tau, self.nsurr, self.surrMethod,
                                self.surrfn, self.randomSeed)


class SP_Summaries(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The estimation can be done using a periodogram, using the periodogram code in
    % Matlab's Signal Processing Toolbox, or a fast fourier transform, implemented
    % using Matlab's fft code.
    %
    %---INPUTS:
    % y, the input time series
    %
    % psdmeth, the method of obtaining the spectrum from the signal:
    %               (i) 'periodogram': periodogram
    %               (ii) 'fft': fast fourier transform
    %
    % wmeth, the window to use:
    %               (i) 'boxcar'
    %               (iii) 'bartlett'
    %               (iv) 'hann'
    %               (v) 'hamming'
    %               (vi) 'none'
    %
    % nf, the number of frequency components to include, if
    %           empty (default), it's approx length(y)
    %
    % dologabs, if 1, takes log amplitude of the signal before
    %           transforming to the frequency domain.
    %
    % doPower, analyzes the power spectrum rather than amplitudes of a Fourier
    %          transform
    %
    %---OUTPUTS:
    % Statistics summarizing various properties of the spectrum,
    % including its maximum, minimum, spread, correlation, centroid, area in certain
    % (normalized) frequency bands, moments of the spectrum, Shannon spectral
    % entropy, a spectral flatness measure, power-law fits, and the number of
    % crossings of the spectrum at various amplitude thresholds.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (23, 23, 23)

    TAGS = ('spectral',)

    def __init__(self, psdmeth='periodogram', wmeth='hamming', nf=array((), dtype=float64), dologabs=0.0):
        super(SP_Summaries, self).__init__()
        self.psdmeth = psdmeth
        self.wmeth = wmeth
        self.nf = nf
        self.dologabs = dologabs

    def _eval_hook(self, eng, x):
        if self.psdmeth is None:
            return eng.run_function(1, 'SP_Summaries', x, )
        elif self.wmeth is None:
            return eng.run_function(1, 'SP_Summaries', x, self.psdmeth)
        elif self.nf is None:
            return eng.run_function(1, 'SP_Summaries', x, self.psdmeth, self.wmeth)
        elif self.dologabs is None:
            return eng.run_function(1, 'SP_Summaries', x, self.psdmeth, self.wmeth, self.nf)
        return eng.run_function(1, 'SP_Summaries', x, self.psdmeth, self.wmeth, self.nf, self.dologabs)


class ST_FitPolynomial(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Usually kind of a stupid thing to do with a time series, but it's sometimes
    % somehow informative for time series with large trends.
    %
    %---INPUTS:
    % y, the input time series.
    % k, the order of the polynomial to fit to y.
    %
    %---OUTPUT:
    % RMS error of the fit.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (18, 18, 18, 18)

    TAGS = ('raw', 'spreaddep', 'trend')

    def __init__(self, k=1.0):
        super(ST_FitPolynomial, self).__init__()
        self.k = k

    def _eval_hook(self, eng, x):
        if self.k is None:
            return eng.run_function(1, 'ST_FitPolynomial', x, )
        return eng.run_function(1, 'ST_FitPolynomial', x, self.k)


class ST_Length(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUT:
    % y, data vector
    %
    %---OUTPUT: the length of the time series
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (9,)

    TAGS = ('lengthdep', 'misc', 'raw')

    def __init__(self):
        super(ST_Length, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'ST_Length', x, )


class ST_LocalExtrema(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Finds maximums and minimums within given segments of the time series and
    % analyses the results.
    %
    %---INPUTS:
    % y, the input time series
    %
    % lorf, whether to use:
    %     (i) 'l', windows of a given length (in which case the third input, n
    %             specifies the length)
    %     (ii) 'n', a specified number of windows to break the time series up into
    %               (in which case the third input, n specifies this number)
    %     (iii) 'tau', sets a window length equal to the correlation length of the
    %                 time series, the first zero-crossing of the autocorrelation
    %                 function.
    %
    % n, somehow specifies the window length given the setting of lorf above.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (20, 19, 20, 19, 19)

    TAGS = ('distribution', 'stationarity')

    def __init__(self, lorf='l', n=50.0):
        super(ST_LocalExtrema, self).__init__()
        self.lorf = lorf
        self.n = n

    def _eval_hook(self, eng, x):
        if self.lorf is None:
            return eng.run_function(1, 'ST_LocalExtrema', x, )
        elif self.n is None:
            return eng.run_function(1, 'ST_LocalExtrema', x, self.lorf)
        return eng.run_function(1, 'ST_LocalExtrema', x, self.lorf, self.n)


class ST_MomentCorr(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Idea to implement by Nick S. Jones.
    %
    %---INPUTS:
    % x, the input time series
    %
    % windowLength, the sliding window length (can be a fraction to specify a proportion of
    %       the time-series length)
    %
    % wOverlap, the overlap between consecutive windows as a fraction of the window
    %       length,
    %
    % mom1, mom2: the statistics to investigate correlations between (in each window):
    %               (i) 'iqr': interquartile range
    %               (ii) 'median': median
    %               (iii) 'std': standard deviation (about the local mean)
    %               (iv) 'mean': mean
    %
    % whatTransform: the pre-processing whatTransformormation to apply to the time series before
    %         analyzing it:
    %               (i) 'abs': takes absolute values of all data points
    %               (ii) 'sqrt': takes the square root of absolute values of all
    %                            data points
    %               (iii) 'sq': takes the square of every data point
    %               (iv) 'none': does no whatTransformormation
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (33, 34, 34, 35, 36, 36)

    TAGS = ('statistics',)

    def __init__(self, windowLength=0.02, wOverlap=0.2, mom1='mean', mom2='std', whatTransform='none'):
        super(ST_MomentCorr, self).__init__()
        self.windowLength = windowLength
        self.wOverlap = wOverlap
        self.mom1 = mom1
        self.mom2 = mom2
        self.whatTransform = whatTransform

    def _eval_hook(self, eng, x):
        if self.windowLength is None:
            return eng.run_function(1, 'ST_MomentCorr', x, )
        elif self.wOverlap is None:
            return eng.run_function(1, 'ST_MomentCorr', x, self.windowLength)
        elif self.mom1 is None:
            return eng.run_function(1, 'ST_MomentCorr', x, self.windowLength, self.wOverlap)
        elif self.mom2 is None:
            return eng.run_function(1, 'ST_MomentCorr', x, self.windowLength, self.wOverlap, self.mom1)
        elif self.whatTransform is None:
            return eng.run_function(1, 'ST_MomentCorr', x, self.windowLength, self.wOverlap, self.mom1, self.mom2)
        return eng.run_function(1, 'ST_MomentCorr', x, self.windowLength, self.wOverlap, self.mom1,
                                self.mom2, self.whatTransform)


class ST_SimpleStats(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % x, the input time series
    %
    % whatStat, the statistic to return:
    %          (i) 'zcross': the proportionof zero-crossings of the time series
    %                        (z-scored input thus returns mean-crossings)
    %          (ii) 'maxima': the proportion of the time series that is a local maximum
    %          (iii) 'minima': the proportion of the time series that is a local minimum
    %          (iv) 'pmcross': the ratio of the number of times that the (ideally
    %                          z-scored) time-series crosses +1 (i.e., 1 standard
    %                          deviation above the mean) to the number of times
    %                          that it crosses -1 (i.e., 1 standard deviation below
    %                          the mean)
    %          (v) 'zsczcross': the ratio of zero crossings of raw to detrended
    %                           time series where the raw has zero mean
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (22, 21)

    TAGS = ('distribution', 'noisiness')

    def __init__(self, whatStat='zcross'):
        super(ST_SimpleStats, self).__init__()
        self.whatStat = whatStat

    def _eval_hook(self, eng, x):
        if self.whatStat is None:
            return eng.run_function(1, 'ST_SimpleStats', x, )
        return eng.run_function(1, 'ST_SimpleStats', x, self.whatStat)


class SY_DriftingMean(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Splits the time series into segments, computes the mean and variance in each
    % segment and compares the maximum and minimum mean to the mean variance.
    %
    % This function implements an idea found in the Matlab Central forum:
    % http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    %
    % >> It seems to me that you are looking for a measure for a drifting mean.
    % >> If so, this is what I would try:
    % >>
    % >> - Decide on a frame length N
    % >> - Split your signal in a number of frames of length N
    % >> - Compute the means of each frame
    % >> - Compute the variance for each frame
    % >> - Compare the ratio of maximum and minimum mean
    % >>   with the mean variance of the frames.
    % >>
    % >> Rune
    %
    %---INPUTS:
    % y, the input time series
    %
    % howl, (i) 'fix': fixed-length segments (of length l)
    %       (ii) 'num': a given number, l, of segments
    %
    % l, either the length ('fix') or number of segments ('num')
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (18, 17, 17, 18, 17)

    TAGS = ('stationarity',)

    def __init__(self, howl='fix', l=20.0):
        super(SY_DriftingMean, self).__init__()
        self.howl = howl
        self.l = l

    def _eval_hook(self, eng, x):
        if self.howl is None:
            return eng.run_function(1, 'SY_DriftingMean', x, )
        elif self.l is None:
            return eng.run_function(1, 'SY_DriftingMean', x, self.howl)
        return eng.run_function(1, 'SY_DriftingMean', x, self.howl, self.l)


class SY_DynWin(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Specifically, variation in a range of local measures are implemented: mean,
    % standard deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1),
    % AC(2), and the first zero-crossing of the autocorrelation function.
    %
    % The standard deviation of local estimates of these quantities across the time
    % series are calculated as an estimate of the stationarity in this quantity as a
    % function of the number of splits, n_{seg}, of the time series.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % maxnseg, the maximum number of segments to consider. Will sweep from 2
    %           segments to maxnseg.
    %
    %
    %---OUTPUTS:
    %
    % The standard deviation of this set of 'stationarity' estimates
    % across these window sizes.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (11,)

    TAGS = ('stationarity',)

    def __init__(self, maxnseg=10.0):
        super(SY_DynWin, self).__init__()
        self.maxnseg = maxnseg

    def _eval_hook(self, eng, x):
        if self.maxnseg is None:
            return eng.run_function(1, 'SY_DynWin', x, )
        return eng.run_function(1, 'SY_DynWin', x, self.maxnseg)


class SY_KPSStest(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The KPSS stationarity test, of Kwiatkowski, Phillips, Schmidt, and Shin:
    % "Testing the null hypothesis of stationarity against the alternative of a
    % unit root: How sure are we that economic time series have a unit root?"
    % Kwiatkowski, Denis and Phillips, Peter C. B. and Schmidt, Peter and Shin, Yongcheol
    % J. Econometrics, 54(1-3) 159 (2002)
    %
    % Uses the function kpsstest from Matlab's Econometrics Toolbox. The null
    % hypothesis is that a univariate time series is trend stationary, the
    % alternative hypothesis is that it is a non-stationary unit-root process.
    %
    % The code can implemented for a specific time lag, tau. Alternatively, measures
    % of change in p-values and test statistics will be outputted if the input is a
    % vector of time lags.
    %
    %---INPUTS:
    % y, the input time series
    % lags, can be either a scalar (returns basic test statistic and p-value), or
    %                   vector (returns statistics on changes across these time lags)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13, 16, 13, 13)

    TAGS = ('econometricstoolbox', 'hypothesistest', 'kpsstest', 'pvalue', 'stationarity')

    def __init__(self, lags=0.0):
        super(SY_KPSStest, self).__init__()
        self.lags = lags

    def _eval_hook(self, eng, x):
        if self.lags is None:
            return eng.run_function(1, 'SY_KPSStest', x, )
        return eng.run_function(1, 'SY_KPSStest', x, self.lags)


class SY_LocalDistributions(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Returns the sum of differences between each kernel-smoothed distributions
    % (using the Matlab function ksdensity).
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % nseg, the number of segments to break the time series into
    %
    % eachOrPar, (i) 'par': compares each local distribution to the parent (full time
    %                       series) distribution
    %            (ii) 'each': compare each local distribution to all other local
    %                         distributions
    %
    % numPoints, number of points to compute the distribution across (in each local
    %          segments)
    %
    % The operation behaves in one of two modes: each compares the distribution in
    % each segment to that in every other segment, and par compares each
    % distribution to the so-called 'parent' distribution, that of the full signal.
    %
    %---OUTPUTS: measures of the sum of absolute deviations between distributions
    % across the different pairwise comparisons.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (28, 27, 28, 27, 28, 27, 28, 27)

    TAGS = ('stationarity',)

    def __init__(self, nseg=2.0, eachOrPar='each', numPoints=None):
        super(SY_LocalDistributions, self).__init__()
        self.nseg = nseg
        self.eachOrPar = eachOrPar
        self.numPoints = numPoints

    def _eval_hook(self, eng, x):
        if self.nseg is None:
            return eng.run_function(1, 'SY_LocalDistributions', x, )
        elif self.eachOrPar is None:
            return eng.run_function(1, 'SY_LocalDistributions', x, self.nseg)
        elif self.numPoints is None:
            return eng.run_function(1, 'SY_LocalDistributions', x, self.nseg, self.eachOrPar)
        return eng.run_function(1, 'SY_LocalDistributions', x, self.nseg, self.eachOrPar, self.numPoints)


class SY_LocalGlobal(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUTS:
    % y, the time series to analyze
    %
    % subsetHow, the local subset of time series to study:
    %             (i) 'l': the first n points in a time series,
    %             (ii) 'p': an initial proportion of the full time series, n
    %             (iii) 'unicg': n evenly-spaced points throughout the time series
    %             (iv) 'randcg': n randomly-chosen points from the time series
    %                               (chosen with replacement)
    %
    % n, the parameter for the method specified above
    %
    % randomSeed, an option for whether (and how) to reset the random seed, for the
    % 'randcg' input
    %
    %---OUTPUTS: the mean, standard deviation, median, interquartile range,
    % skewness, kurtosis, AC(1), and SampEn(1,0.1).
    %
    % This is not the most reliable or systematic operation because only a single
    % sample is taken from the time series and compared to the full time series.
    % A better approach would be to repeat over many local subsets and compare the
    % statistics of these local regions to the full time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (18, 19, 18, 18, 19, 18, 18, 17, 17, 22, 23, 22, 22, 23)

    TAGS = ('stationarity',)

    def __init__(self, subsetHow='l', n=10.0, randomSeed=None):
        super(SY_LocalGlobal, self).__init__()
        self.subsetHow = subsetHow
        self.n = n
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.subsetHow is None:
            return eng.run_function(1, 'SY_LocalGlobal', x, )
        elif self.n is None:
            return eng.run_function(1, 'SY_LocalGlobal', x, self.subsetHow)
        elif self.randomSeed is None:
            return eng.run_function(1, 'SY_LocalGlobal', x, self.subsetHow, self.n)
        return eng.run_function(1, 'SY_LocalGlobal', x, self.subsetHow, self.n, self.randomSeed)


class SY_PPtest(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the pptest code from Matlab's Econometrics Toolbox.
    %
    %---INPUTS:
    % y, the input time series
    %
    % lags, a vector of lags
    %
    % model, a specified model:
    %               'ar': autoregressive
    %               'ard': autoregressive with drift, or
    %               'ts': trend stationary,
    %               (see Matlab documentation for information)
    %
    % testStatistic, the test statistic:
    %               't1': the standard t-statistic, or
    %               't2' a lag-adjusted, 'unStudentized' t statistic.
    %               (see Matlab documentation for information)
    %
    %---OUTPUTS: statistics on the p-values and lags obtained from the set of tests, as
    % well as measures of the regression statistics.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (19,)

    TAGS = ('bic', 'econometricstoolbox', 'pptest', 'pvalue', 'rmse', 'unitroot')

    def __init__(self, lags=MatlabSequence('0:5'), model='ar', testStatistic='t1'):
        super(SY_PPtest, self).__init__()
        self.lags = lags
        self.model = model
        self.testStatistic = testStatistic

    def _eval_hook(self, eng, x):
        if self.lags is None:
            return eng.run_function(1, 'SY_PPtest', x, )
        elif self.model is None:
            return eng.run_function(1, 'SY_PPtest', x, self.lags)
        elif self.testStatistic is None:
            return eng.run_function(1, 'SY_PPtest', x, self.lags, self.model)
        return eng.run_function(1, 'SY_PPtest', x, self.lags, self.model, self.testStatistic)


class SY_RangeEvolve(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Measures of the range of the time series as a function of time,
    % i.e., range(x_{1:i}) for i = 1, 2, ..., N, where N is the length of the time
    % series.
    %
    %---INPUT:
    % y, the time series
    %
    %---OUTPUTS: based on the dynamics of how new extreme events occur with time.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (14,)

    TAGS = ('stationarity',)

    def __init__(self):
        super(SY_RangeEvolve, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'SY_RangeEvolve', x, )


class SY_SlidingWindow(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % This function is based on sliding a window along the time series, measuring
    % some quantity in each window, and outputting some summary of this set of local
    % estimates of that quantity.
    %
    % Another way of saying it: calculate 'windowStat' in each window, and computes
    % 'acrossWinStat' for the set of statistics calculated in each window.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % windowStat, the measure to calculate in each window:
    %               (i) 'mean', mean
    %               (ii) 'std', standard deviation
    %               (iii) 'ent', distribution entropy
    %               (iv) 'mom3', skewness
    %               (v) 'mom4', kurtosis
    %               (vi) 'mom5', the fifth moment of the distribution
    %               (vii) 'lillie', the p-value for a Lilliefors Gaussianity test
    %               (viii) 'AC1', the lag-1 autocorrelation
    %               (ix) 'apen', Approximate Entropy, ApEn(1,0.2)
    %               (ix) 'sampen', Sample Entropy, SampEn(2,0.1)
    %
    % acrossWinStat, controls how the obtained sequence of local estimates is
    %                   compared (as a ratio to the full time series):
    %                       (i) 'std': standard deviation
    %                       (ii) 'ent' histogram entropy
    %                       (iii) 'apen': Approximate Entropy, ApEn(1,0.2)
    %                       (iii) 'sampen': Sample Entropy, SampEn(2,0.1)
    %
    % numSeg, the number of segments to divide the time series up into, thus
    %       controlling the window length
    %
    % incMove, the increment to move the window at each iteration, as 1/fraction of the
    %       window length (e.g., incMove = 2, means the window moves half the length of the
    %       window at each increment)
    %
    % NOTE: SY_SlidingWindow(y,'mean','std',X,1) is the same as StatAvX, computed as
    %                       SY_StatAv(y,'seg',X);
    % cf. "Heart rate control in normal and aborted-SIDS infants", S. M. Pincus et al.
    %           Am J. Physiol. Regul. Integr. Comp. Physiol. 264(3) R638 (1993)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (28, 29, 28, 27, 28, 27, 27, 28, 27, 26, 27, 26, 25, 26, 25, 25, 26, 25, 31, 32, 31, 31, 31, 28, 29, 28, 27, 28, 27, 27, 28, 27, 26, 27, 26, 25, 26, 25, 25, 26, 25, 32, 31, 31, 31, 28, 29, 28, 27, 28, 27, 27, 28, 27, 26, 27, 26, 25, 26, 25, 25, 26, 25, 31, 32, 31, 31, 31, 26, 27, 26, 25, 26, 25, 25, 26, 25, 25, 24, 24, 23, 24, 23, 25, 24, 24, 24, 24, 24, 24, 24, 24, 29, 30, 29, 29, 29, 29, 30, 29, 28, 29, 28, 28, 29, 28, 27, 28, 27, 26, 27, 26, 26, 27, 26, 32, 33, 32, 32, 32, 29, 30, 29, 28, 29, 28, 28, 29, 28, 27, 28, 27, 26, 27, 26, 26, 27, 26, 32, 33, 32, 32, 32, 26, 27, 26, 25, 26, 25, 25, 26, 25, 24, 25, 24, 23, 24, 23, 23, 24, 23, 29, 30, 29, 29, 29, 31, 32, 31, 30, 31, 30, 30, 31, 30, 29, 30, 29, 28, 29, 28, 28, 29, 28, 34, 35, 34, 34, 34)

    TAGS = ('StatAv', 'slidingwin', 'stationarity')

    def __init__(self, windowStat='mean', acrossWinStat='std', numSeg=2.0, incMove=1.0):
        super(SY_SlidingWindow, self).__init__()
        self.windowStat = windowStat
        self.acrossWinStat = acrossWinStat
        self.numSeg = numSeg
        self.incMove = incMove

    def _eval_hook(self, eng, x):
        if self.windowStat is None:
            return eng.run_function(1, 'SY_SlidingWindow', x, )
        elif self.acrossWinStat is None:
            return eng.run_function(1, 'SY_SlidingWindow', x, self.windowStat)
        elif self.numSeg is None:
            return eng.run_function(1, 'SY_SlidingWindow', x, self.windowStat, self.acrossWinStat)
        elif self.incMove is None:
            return eng.run_function(1, 'SY_SlidingWindow', x, self.windowStat, self.acrossWinStat, self.numSeg)
        return eng.run_function(1, 'SY_SlidingWindow', x, self.windowStat, self.acrossWinStat,
                                self.numSeg, self.incMove)


class SY_SpreadRandomLocal(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % numSegs time-series segments of length l are selected at random from the time
    % series and in each segment some statistic is calculated: mean, standard
    % deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1), AC(2), and the
    % first zero-crossing of the autocorrelation function.
    % Outputs summarize how these quantities vary in different local segments of the
    % time series.
    %
    %---INPUTS:
    % y, the input time series
    %
    % l, the length of local time-series segments to analyze as a positive integer.
    %    Can also be a specified character string:
    %       (i) 'ac2': twice the first zero-crossing of the autocorrelation function
    %       (ii) 'ac5': five times the first zero-crossing of the autocorrelation function
    %
    % numSegs, the number of randomly-selected local segments to analyze
    %
    % randomSeed, the input to BF_ResetSeed to control reproducibility
    %
    %---OUTPUTS: the mean and also the standard deviation of this set of 100 local
    % estimates.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (28, 28, 27, 28, 28)

    TAGS = ('stationarity',)

    def __init__(self, l='ac2', numSegs=100.0, randomSeed='default'):
        super(SY_SpreadRandomLocal, self).__init__()
        self.l = l
        self.numSegs = numSegs
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.l is None:
            return eng.run_function(1, 'SY_SpreadRandomLocal', x, )
        elif self.numSegs is None:
            return eng.run_function(1, 'SY_SpreadRandomLocal', x, self.l)
        elif self.randomSeed is None:
            return eng.run_function(1, 'SY_SpreadRandomLocal', x, self.l, self.numSegs)
        return eng.run_function(1, 'SY_SpreadRandomLocal', x, self.l, self.numSegs, self.randomSeed)


class SY_StatAv(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % The StatAv measure divides the time series into non-overlapping subsegments,
    % calculates the mean in each of these segments and returns the standard deviation
    % of this set of means.
    %
    % cf. "Heart rate control in normal and aborted-SIDS infants", S. M. Pincus et al.
    % Am J. Physiol. Regul. Integr. Comp. Physiol. 264(3) R638 (1993)
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % whatType, the type of StatAv to perform:
    %           (i) 'seg': divide the time series into n segments
    %           (ii) 'len': divide the time series into segments of length n
    %
    % n, either the number of subsegments ('seg') or their length ('len')
    
    % Might be nicer to use the 'buffer' function for this...?
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (10, 11, 10, 10, 9, 10, 9, 10)

    TAGS = ('StatAv', 'stationarity')

    def __init__(self, whatType='len', n=25.0):
        super(SY_StatAv, self).__init__()
        self.whatType = whatType
        self.n = n

    def _eval_hook(self, eng, x):
        if self.whatType is None:
            return eng.run_function(1, 'SY_StatAv', x, )
        elif self.n is None:
            return eng.run_function(1, 'SY_StatAv', x, self.whatType)
        return eng.run_function(1, 'SY_StatAv', x, self.whatType, self.n)


class SY_StdNthDer(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Based on an idea by Vladimir Vassilevsky, a DSP and Mixed Signal Design
    % Consultant in a Matlab forum, who stated that You can measure the standard
    % deviation of the nth derivative, if you like".
    %
    % cf. http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    %
    % The derivative is estimated very simply by simply taking successive increments
    % of the time series; the process is repeated to obtain higher order
    % derivatives.
    %
    % Note that this idea is popular in the heart-rate variability literature, cf.
    % cf. "Do Existing Measures ... ", Brennan et. al. (2001), IEEE Trans Biomed Eng 48(11)
    % (and function MD_hrv_classic)
    %
    %---INPUTS:
    %
    % y, time series to analyze
    %
    % n, the order of derivative to analyze
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (14, 15, 14, 14, 14, 14)

    TAGS = ('entropy',)

    def __init__(self, n=1.0):
        super(SY_StdNthDer, self).__init__()
        self.n = n

    def _eval_hook(self, eng, x):
        if self.n is None:
            return eng.run_function(1, 'SY_StdNthDer', x, )
        return eng.run_function(1, 'SY_StdNthDer', x, self.n)


class SY_StdNthDerChange(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Order parameter controls the derivative of the signal.
    %
    % Operation inspired by a comment on the Matlab Central forum: "You can
    % measure the standard deviation of the n-th derivative, if you like." --
    % Vladimir Vassilevsky, DSP and Mixed Signal Design Consultant from
    % http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    %
    %---INPUTS:
    % y, the input time series
    %
    % maxd, the maximum derivative to take.
    %
    %---OUTPUTS:
    % An exponential function, f(x) = Aexp(bx), is fitted to the variation across
    % successive derivatives; outputs are the parameters and quality of this fit.
    %
    % Typically an excellent fit to exponential: regular signals decrease, irregular
    % signals increase...?
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (18,)

    TAGS = ('entropy',)

    def __init__(self, maxd=None):
        super(SY_StdNthDerChange, self).__init__()
        self.maxd = maxd

    def _eval_hook(self, eng, x):
        if self.maxd is None:
            return eng.run_function(1, 'SY_StdNthDerChange', x, )
        return eng.run_function(1, 'SY_StdNthDerChange', x, self.maxd)


class SY_TISEAN_nstat_z(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the nstat_z routine from the TISEAN package for nonlinear time-series
    % analysis to calculate cross-forecast errors of zeroth-order models for the
    % time-delay embedded time series.
    %
    % The program looks for nonstationarity in a time series by dividing it
    % into a number of segments and calculating the cross-forecast errors
    % between the different segments. The model used for the forecast is
    % zeroth order model as proposed by Schreiber.
    %
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package", R. Hegger, H. Kantz, and T. Schreiber, Chaos 9(2) 413 (1999)
    %
    % Available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    %
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % numSeg, the number of equally-spaced segments to divide the time series into,
    %       and used to predict the other time series segments
    %
    % embedParams, in the form {tau,m}, as usual for BF_embed. That is, for an
    %               embedding dimension, tau, and embedding dimension, m. E.g.,
    %               {1,3} has a time-delay of 1 and embedding dimension of 3.
    %
    %
    %---OUTPUTS: include the trace of the cross-prediction error matrix, the mean,
    % minimum, and maximum cross-prediction error, the minimum off-diagonal
    % cross-prediction error, and eigenvalues of the cross-prediction error matrix.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (23, 23)

    TAGS = ('model', 'nonlinear', 'stationarity', 'tisean')

    def __init__(self, numSeg=4.0, embedParams=array((1, 3), dtype=object)):
        super(SY_TISEAN_nstat_z, self).__init__()
        self.numSeg = numSeg
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.numSeg is None:
            return eng.run_function(1, 'SY_TISEAN_nstat_z', x, )
        elif self.embedParams is None:
            return eng.run_function(1, 'SY_TISEAN_nstat_z', x, self.numSeg)
        return eng.run_function(1, 'SY_TISEAN_nstat_z', x, self.numSeg, self.embedParams)


class SY_Trend(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    %---INPUT:
    % y, the input time series.
    %
    %---OUTPUTS:
    % Linearly detrends the time series using detrend, and returns the ratio of
    % standard deviations before and after the linear detrending. If a strong linear
    % trend is present in the time series, this operation should output a low value.
    %
    % Also fits a line and gives parameters from that fit, as well as statistics on
    % a cumulative sum of the time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (8,)

    TAGS = ('stationarity',)

    def __init__(self):
        super(SY_Trend, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'SY_Trend', x, )


class SY_VarRatioTest(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Implemented using the vratiotest function from Matlab's Econometrics Toolbox.
    %
    % The test assesses the null hypothesis of a random walk in the time series,
    % which is rejected for some critical p-value.
    %
    %---INPUTS:
    % y, the input time series
    %
    % periods, a vector (or scalar) of period(s)
    %
    % IIDs, a vector (or scalar) representing boolean values indicating whether to
    %       assume independent and identically distributed (IID) innovations for
    %       each period.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (33, 19, 19, 19, 19)

    TAGS = ('econometricstoolbox', 'pvalue', 'vratiotest')

    def __init__(self, periods=2.0, IIDs=0.0):
        super(SY_VarRatioTest, self).__init__()
        self.periods = periods
        self.IIDs = IIDs

    def _eval_hook(self, eng, x):
        if self.periods is None:
            return eng.run_function(1, 'SY_VarRatioTest', x, )
        elif self.IIDs is None:
            return eng.run_function(1, 'SY_VarRatioTest', x, self.periods)
        return eng.run_function(1, 'SY_VarRatioTest', x, self.periods, self.IIDs)


class TSTL_delaytime(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the TSTOOL code delaytime (this method is specified in the TSTOOL
    % documentation but without reference).
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    % y, column vector of time series data
    %
    % maxDelay, maximum value of the delay to consider (can also specify a
    %           proportion of time series length)
    %
    % past, the TSTOOL documentation describes this parameter as "?", which is
    %       relatively uninformative.
    %
    % randomSeed, whether (and how) to reset the random seed, using BF_ResetSeed
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (19,)

    TAGS = ('correlation', 'nonlinear', 'tau', 'tstool')

    def __init__(self, maxDelay=0.1, past=1.0, randomSeed='default'):
        super(TSTL_delaytime, self).__init__()
        self.maxDelay = maxDelay
        self.past = past
        self.randomSeed = randomSeed

    def _eval_hook(self, eng, x):
        if self.maxDelay is None:
            return eng.run_function(1, 'TSTL_delaytime', x, )
        elif self.past is None:
            return eng.run_function(1, 'TSTL_delaytime', x, self.maxDelay)
        elif self.randomSeed is None:
            return eng.run_function(1, 'TSTL_delaytime', x, self.maxDelay, self.past)
        return eng.run_function(1, 'TSTL_delaytime', x, self.maxDelay, self.past, self.randomSeed)


class TSTL_localdensity(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % TSTOOL code localdensity is very poorly documented in the TSTOOL
    % package, can assume it returns local density estimates in the
    % time-delay embedding space.
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    %
    % y, the time series as a column vector
    %
    % NNR, number of nearest neighbours to compute
    %
    % past, number of time-correlated points to discard (samples)
    %
    % embedParams, the embedding parameters, inputs to BF_embed as {tau,m}, where
    %               tau and m can be characters specifying a given automatic method
    %               of determining tau and/or m (see BF_embed).
    %
    %---OUTPUTS: various statistics on the local density estimates at each point in
    % the time-delay embedding, including the minimum and maximum values, the range,
    % the standard deviation, mean, median, and autocorrelation.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (26, 27, 32)

    TAGS = ('localdensity', 'nonlinear', 'tstool')

    def __init__(self, NNR=5.0, past=40.0, embedParams=array(('ac', 'fnnmar'), dtype=object)):
        super(TSTL_localdensity, self).__init__()
        self.NNR = NNR
        self.past = past
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.NNR is None:
            return eng.run_function(1, 'TSTL_localdensity', x, )
        elif self.past is None:
            return eng.run_function(1, 'TSTL_localdensity', x, self.NNR)
        elif self.embedParams is None:
            return eng.run_function(1, 'TSTL_localdensity', x, self.NNR, self.past)
        return eng.run_function(1, 'TSTL_localdensity', x, self.NNR, self.past, self.embedParams)


class TSTL_predict(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % References TSTOOL code 'predict', which does local constant iterative
    % prediction for scalar data using fast nearest neighbour searching.
    % There are four modes available for the prediction output.
    %
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    %
    % y, scalar column vector time series
    %
    % plen, prediction length in samples or as proportion of time series length NNR,
    %
    % NNR, number of nearest neighbours
    %
    % stepSize, number of samples to step for each prediction
    %
    % pmode, prediction mode, four options:
    %           (i) 0: output vectors are means of images of nearest neighbours
    %           (ii) 1: output vectors are distance-weighted means of images
    %                     nearest neighbours
    %           (iii) 2: output vectors are calculated using local flow and the
    %                    mean of the images of the neighbours
    %           (iv) 3: output vectors are calculated using local flow and the
    %                    weighted mean of the images of the neighbours
    % embedParams, as usual to feed into BF_embed, except that now you can set
    %              to zero to not embed.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = ()

    TAGS = ()

    def __init__(self, plen=None, NNR=None, stepSize=None, pmode=None, embedParams=None):
        super(TSTL_predict, self).__init__()
        self.plen = plen
        self.NNR = NNR
        self.stepSize = stepSize
        self.pmode = pmode
        self.embedParams = embedParams

    def _eval_hook(self, eng, x):
        if self.plen is None:
            return eng.run_function(1, 'TSTL_predict', x, )
        elif self.NNR is None:
            return eng.run_function(1, 'TSTL_predict', x, self.plen)
        elif self.stepSize is None:
            return eng.run_function(1, 'TSTL_predict', x, self.plen, self.NNR)
        elif self.pmode is None:
            return eng.run_function(1, 'TSTL_predict', x, self.plen, self.NNR, self.stepSize)
        elif self.embedParams is None:
            return eng.run_function(1, 'TSTL_predict', x, self.plen, self.NNR, self.stepSize, self.pmode)
        return eng.run_function(1, 'TSTL_predict', x, self.plen, self.NNR, self.stepSize, self.pmode, self.embedParams)


class WL_DetailCoeffs(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Compares the detail coefficients obtained at each level of the wavelet
    % decomposition from 1 to the maximum possible level for the wavelet given the
    % length of the input time series (computed using wmaxlev from
    % Matlab's Wavelet Toolbox).
    %
    %---INPUTS:
    % y, the input time series
    %
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    %
    % maxlevel, the maximum wavelet decomposition level (can also set to 'max' to be
    %               that determined by wmaxlev)
    %
    %---OUTPUTS:
    % Statistics on the detail coefficients.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (23,)

    TAGS = ('wavelet', 'waveletTB')

    def __init__(self, wname='db3', maxlevel='max'):
        super(WL_DetailCoeffs, self).__init__()
        self.wname = wname
        self.maxlevel = maxlevel

    def _eval_hook(self, eng, x):
        if self.wname is None:
            return eng.run_function(1, 'WL_DetailCoeffs', x, )
        elif self.maxlevel is None:
            return eng.run_function(1, 'WL_DetailCoeffs', x, self.wname)
        return eng.run_function(1, 'WL_DetailCoeffs', x, self.wname, self.maxlevel)


class WL_coeffs(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Performs a wavelet decomposition of the time series using a given wavelet at a
    % given level and returns a set of statistics on the coefficients obtained.
    %
    % Uses Matlab's Wavelet Toolbox.
    %
    %---INPUTS:
    % y, the input time series
    %
    % wname, the wavelet name, e.g., 'db3' (see Wavelet Toolbox Documentation for
    %                                       all options)
    %
    % level, the level of wavelet decomposition
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (15, 15, 15, 15, 15, 17)

    TAGS = ('lengthdep', 'wavelet', 'waveletTB')

    def __init__(self, wname='db3', level=1.0):
        super(WL_coeffs, self).__init__()
        self.wname = wname
        self.level = level

    def _eval_hook(self, eng, x):
        if self.wname is None:
            return eng.run_function(1, 'WL_coeffs', x, )
        elif self.level is None:
            return eng.run_function(1, 'WL_coeffs', x, self.wname)
        return eng.run_function(1, 'WL_coeffs', x, self.wname, self.level)


class WL_cwt(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the function cwt from Matlab's Wavelet Toolbox.
    %
    %---INPUTS:
    % y, the input time series
    %
    % wname, the wavelet name, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet),
    %                           etc. (see Wavelet Toolbox Documentation for all
    %                           options)
    %
    % maxScale, the maximum scale of wavelet analysis.
    %
    %---OUTPUTS: statistics on the coefficients, entropy, and results of
    % coefficients summed across scales.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (13, 14)

    TAGS = ('cwt', 'statTB', 'wavelet', 'waveletTB')

    def __init__(self, wname='db3', maxScale=32.0):
        super(WL_cwt, self).__init__()
        self.wname = wname
        self.maxScale = maxScale

    def _eval_hook(self, eng, x):
        if self.wname is None:
            return eng.run_function(1, 'WL_cwt', x, )
        elif self.maxScale is None:
            return eng.run_function(1, 'WL_cwt', x, self.wname)
        return eng.run_function(1, 'WL_cwt', x, self.wname, self.maxScale)


class WL_dwtcoeff(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Decomposes the time series using a given wavelet and outputs statistics on the
    % coefficients obtained up to a maximum level, level.
    %
    %---INPUTS:
    %
    % y, the input time series
    %
    % wname, the mother wavelet, e.g., 'db3', 'sym2' (see Wavelet Toolbox
    %           Documentation)
    %
    % level, the level of wavelet decomposition (can be set to 'max' for the maximum
    %               level determined by wmaxlev)
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (17, 18)

    TAGS = ('dwt', 'wavelet', 'waveletTB')

    def __init__(self, wname='db3', level=5.0):
        super(WL_dwtcoeff, self).__init__()
        self.wname = wname
        self.level = level

    def _eval_hook(self, eng, x):
        if self.wname is None:
            return eng.run_function(1, 'WL_dwtcoeff', x, )
        elif self.level is None:
            return eng.run_function(1, 'WL_dwtcoeff', x, self.wname)
        return eng.run_function(1, 'WL_dwtcoeff', x, self.wname, self.level)


class WL_fBM(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Uses the wfbmesti function from Matlab's Wavelet Toolbox
    %
    %---INPUT:
    % y, the time series to analyze.
    %
    %---OUTPUTS: All three outputs of wfbmesti are returned from this function.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (6,)

    TAGS = ('wavelet', 'waveletTB')

    def __init__(self):
        super(WL_fBM, self).__init__()

    def _eval_hook(self, eng, x):
        return eng.run_function(1, 'WL_fBM', x, )


class WL_scal2frq(HCTSASuper):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Estimates frequency components using functions from Matlab's Wavelet Toolbox,
    % including the scal2frq function.
    %
    %---INPUTS:
    % y, the input time series
    %
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    %
    % amax, the maximum scale / level (can be 'max' to set according to wmaxlev)
    %
    % delta, the sampling period
    %
    %---OUTPUTS: the level with the highest energy coefficients, the dominant
    % period, and the dominant pseudo-frequency.
    %
    % Adapted from example in Matlab Wavelet Toolbox documentation. It's kind of a
    % weird idea to apply the method to generic time series.
    
    ----------------------------------------
    """

    KNOWN_OUTPUTS_SIZES = (21,)

    TAGS = ('wavelet', 'waveletTB')

    def __init__(self, wname='db3', amax='max', delta=1.0):
        super(WL_scal2frq, self).__init__()
        self.wname = wname
        self.amax = amax
        self.delta = delta

    def _eval_hook(self, eng, x):
        if self.wname is None:
            return eng.run_function(1, 'WL_scal2frq', x, )
        elif self.amax is None:
            return eng.run_function(1, 'WL_scal2frq', x, self.wname)
        elif self.delta is None:
            return eng.run_function(1, 'WL_scal2frq', x, self.wname, self.amax)
        return eng.run_function(1, 'WL_scal2frq', x, self.wname, self.amax, self.delta)


HCTSA_ALL_CLASSES = (
    CO_AddNoise,
    CO_AutoCorr,
    CO_AutoCorrShape,
    CO_CompareMinAMI,
    CO_Embed2,
    CO_Embed2_AngleTau,
    CO_Embed2_Basic,
    CO_Embed2_Dist,
    CO_Embed2_Shapes,
    CO_FirstMin,
    CO_FirstZero,
    CO_HistogramAMI,
    CO_NonlinearAutocorr,
    CO_RM_AMInformation,
    CO_StickAngles,
    CO_TSTL_amutual,
    CO_TranslateShape,
    CO_f1ecac,
    CO_fzcglscf,
    CO_glscf,
    CO_tc3,
    CO_trev,
    CP_ML_StepDetect,
    CP_l1pwc_sweep_lambda,
    CP_wavelet_varchg,
    DN_Burstiness,
    DN_CompareKSFit,
    DN_Cumulants,
    DN_CustomSkewness,
    DN_FitKernelSmooth,
    DN_Fit_mle,
    DN_HighLowMu,
    DN_HistogramMode,
    DN_Mean,
    DN_MinMax,
    DN_Moments,
    DN_OutlierInclude,
    DN_OutlierTest,
    DN_ProportionValues,
    DN_Quantile,
    DN_RemovePoints,
    DN_SimpleFit,
    DN_Spread,
    DN_TrimmedMean,
    DN_Withinp,
    DN_cv,
    DN_nlogL_norm,
    DN_pleft,
    DT_IsSeasonal,
    EN_ApEn,
    EN_CID,
    EN_DistributionEntropy,
    EN_MS_LZcomplexity,
    EN_MS_shannon,
    EN_PermEn,
    EN_RM_entropy,
    EN_Randomize,
    EN_SampEn,
    EN_mse,
    EN_rpde,
    EN_wentropy,
    EX_MovingThreshold,
    FC_LocalSimple,
    FC_LoopLocalSimple,
    FC_Surprise,
    HT_DistributionTest,
    HT_HypothesisTest,
    IN_AutoMutualInfo,
    IN_AutoMutualInfoStats,
    IN_Initialize_MI,
    IN_MutualInfo,
    MD_hrv_classic,
    MD_pNN,
    MD_polvar,
    MD_rawHRVmeas,
    MF_ARMA_orders,
    MF_AR_arcov,
    MF_CompareAR,
    MF_CompareTestSets,
    MF_ExpSmoothing,
    MF_FitSubsegments,
    MF_GARCHcompare,
    MF_GARCHfit,
    MF_GP_FitAcross,
    MF_GP_LearnHyperp,
    MF_GP_LocalPrediction,
    MF_GP_hyperparameters,
    MF_ResidualAnalysis,
    MF_StateSpaceCompOrder,
    MF_StateSpace_n4sid,
    MF_arfit,
    MF_armax,
    MF_hmm_CompareNStates,
    MF_hmm_fit,
    MF_steps_ahead,
    NL_BoxCorrDim,
    NL_CaosMethod,
    NL_DVV,
    NL_MS_fnn,
    NL_MS_nlpe,
    NL_TISEAN_c1,
    NL_TISEAN_d2,
    NL_TISEAN_fnn,
    NL_TSTL_FractalDimensions,
    HCTSA_NL_TSTL_GPCorrSum,
    NL_TSTL_LargestLyap,
    NL_TSTL_PoincareSection,
    HCTSA_NL_TSTL_ReturnTime,
    NL_TSTL_TakensEstimator,
    NL_TSTL_acp,
    NL_TSTL_dimensions,
    NL_crptool_fnn,
    NL_embed_PCA,
    NW_VisibilityGraph,
    PD_PeriodicityWang,
    PH_ForcePotential,
    PH_Walker,
    PP_Compare,
    PP_Iterate,
    PP_ModelFit,
    SB_BinaryStats,
    SB_BinaryStretch,
    SB_CoarseGrain,
    SB_MotifThree,
    SB_MotifTwo,
    SB_TransitionMatrix,
    SB_TransitionpAlphabet,
    SC_FluctAnal,
    SC_MMA,
    SC_fastdfa,
    SD_MakeSurrogates,
    SD_SurrogateTest,
    SD_TSTL_surrogates,
    SP_Summaries,
    ST_FitPolynomial,
    ST_Length,
    ST_LocalExtrema,
    ST_MomentCorr,
    ST_SimpleStats,
    SY_DriftingMean,
    SY_DynWin,
    SY_KPSStest,
    SY_LocalDistributions,
    SY_LocalGlobal,
    SY_PPtest,
    SY_RangeEvolve,
    SY_SlidingWindow,
    SY_SpreadRandomLocal,
    SY_StatAv,
    SY_StdNthDer,
    SY_StdNthDerChange,
    SY_TISEAN_nstat_z,
    SY_Trend,
    SY_VarRatioTest,
    TSTL_delaytime,
    TSTL_localdensity,
    TSTL_predict,
    WL_DetailCoeffs,
    WL_coeffs,
    WL_cwt,
    WL_dwtcoeff,
    WL_fBM,
    WL_scal2frq,
)


class HCTSAOperations(object):
    """Namespace for HCTSA selected operations."""

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,1,_,q
    # outs: u,a,n,t,i
    # outs: l,e,s,_,1
    # outs: 0
    # tags: AMI,correlation,entropy
    CO_AddNoise_1_quantiles_10 = HCTSAOperation(
        'CO_AddNoise_1_quantiles_10',
        "CO_AddNoise(y,1,'quantiles',10,'default')",
        CO_AddNoise(tau=1, amiMethod='quantiles', extraParam=10, randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,a,c,_
    # outs: q,u,a,n,t
    # outs: i,l,e,s,_
    # outs: 1,0
    # tags: AMI,correlation,entropy
    CO_AddNoise_ac_quantiles_10 = HCTSAOperation(
        'CO_AddNoise_ac_quantiles_10',
        "CO_AddNoise(y,'ac','quantiles',10,'default')",
        CO_AddNoise(tau='ac', amiMethod='quantiles', extraParam=10, randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,1,_,e
    # outs: v,e,n,_,1
    # outs: 0
    # tags: AMI,correlation,entropy
    CO_AddNoise_1_even_10 = HCTSAOperation(
        'CO_AddNoise_1_even_10',
        "CO_AddNoise(y,1,'even',10,'default')",
        CO_AddNoise(tau=1, amiMethod='even', extraParam=10, randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,a,c,_
    # outs: e,v,e,n,_
    # outs: 1,0
    # tags: AMI,correlation,entropy
    CO_AddNoise_ac_even_10 = HCTSAOperation(
        'CO_AddNoise_ac_even_10',
        "CO_AddNoise(y,'ac','even',10,'default')",
        CO_AddNoise(tau='ac', amiMethod='even', extraParam=10, randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,1,_,s
    # outs: t,d,1,_,1
    # outs: 0
    # tags: AMI,correlation,entropy
    CO_AddNoise_1_std1_10 = HCTSAOperation(
        'CO_AddNoise_1_std1_10',
        "CO_AddNoise(y,1,'std1',10,'default')",
        CO_AddNoise(tau=1, amiMethod='std1', extraParam=10, randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,a,c,_
    # outs: s,t,d,1,_
    # outs: 1,0
    # tags: AMI,correlation,entropy
    CO_AddNoise_ac_std1_10 = HCTSAOperation(
        'CO_AddNoise_ac_std1_10',
        "CO_AddNoise(y,'ac','std1',10,'default')",
        CO_AddNoise(tau='ac', amiMethod='std1', extraParam=10, randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,1,_,g
    # outs: a,u,s,s,i
    # outs: a,n
    # tags: AMI,correlation,entropy
    CO_AddNoise_1_gaussian = HCTSAOperation(
        'CO_AddNoise_1_gaussian',
        "CO_AddNoise(y,1,'gaussian',[],'default')",
        CO_AddNoise(tau=1, amiMethod='gaussian', extraParam=array([], dtype=float64), randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,a,c,_
    # outs: g,a,u,s,s
    # outs: i,a,n
    # tags: AMI,correlation,entropy
    CO_AddNoise_ac_gaussian = HCTSAOperation(
        'CO_AddNoise_ac_gaussian',
        "CO_AddNoise(y,'ac','gaussian',[],'default')",
        CO_AddNoise(tau='ac', amiMethod='gaussian', extraParam=array([], dtype=float64), randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,1,_,k
    # outs: r,a,s,k,o
    # outs: v,1,_,4
    # tags: AMI,correlation,entropy
    CO_AddNoise_1_kraskov1_4 = HCTSAOperation(
        'CO_AddNoise_1_kraskov1_4',
        "CO_AddNoise(y,1,'kraskov1','4','default')",
        CO_AddNoise(tau=1, amiMethod='kraskov1', extraParam='4', randomSeed='default'))

    # outs: C,O,_,A,d
    # outs: d,N,o,i,s
    # outs: e,_,a,c,_
    # outs: k,r,a,s,k
    # outs: o,v,1,_,4
    # tags: AMI,correlation,entropy
    CO_AddNoise_ac_kraskov1_4 = HCTSAOperation(
        'CO_AddNoise_ac_kraskov1_4',
        "CO_AddNoise(y,'ac','kraskov1','4','default')",
        CO_AddNoise(tau='ac', amiMethod='kraskov1', extraParam='4', randomSeed='default'))

    # outs: A,C,_,1
    # tags: autocorrelation,correlation
    AC_1 = HCTSAOperation(
        'AC_1',
        "CO_AutoCorr(y,1,'Fourier')",
        CO_AutoCorr(tau=1, whatMethod='Fourier'))

    # outs: A,C,_,2
    # tags: autocorrelation,correlation
    AC_2 = HCTSAOperation(
        'AC_2',
        "CO_AutoCorr(y,2,'Fourier')",
        CO_AutoCorr(tau=2, whatMethod='Fourier'))

    # outs: A,C,_,3
    # tags: autocorrelation,correlation
    AC_3 = HCTSAOperation(
        'AC_3',
        "CO_AutoCorr(y,3,'Fourier')",
        CO_AutoCorr(tau=3, whatMethod='Fourier'))

    # outs: A,C,_,4
    # tags: autocorrelation,correlation
    AC_4 = HCTSAOperation(
        'AC_4',
        "CO_AutoCorr(y,4,'Fourier')",
        CO_AutoCorr(tau=4, whatMethod='Fourier'))

    # outs: A,C,_,5
    # tags: autocorrelation,correlation
    AC_5 = HCTSAOperation(
        'AC_5',
        "CO_AutoCorr(y,5,'Fourier')",
        CO_AutoCorr(tau=5, whatMethod='Fourier'))

    # outs: A,C,_,6
    # tags: autocorrelation,correlation
    AC_6 = HCTSAOperation(
        'AC_6',
        "CO_AutoCorr(y,6,'Fourier')",
        CO_AutoCorr(tau=6, whatMethod='Fourier'))

    # outs: A,C,_,7
    # tags: autocorrelation,correlation
    AC_7 = HCTSAOperation(
        'AC_7',
        "CO_AutoCorr(y,7,'Fourier')",
        CO_AutoCorr(tau=7, whatMethod='Fourier'))

    # outs: A,C,_,8
    # tags: autocorrelation,correlation
    AC_8 = HCTSAOperation(
        'AC_8',
        "CO_AutoCorr(y,8,'Fourier')",
        CO_AutoCorr(tau=8, whatMethod='Fourier'))

    # outs: A,C,_,9
    # tags: autocorrelation,correlation
    AC_9 = HCTSAOperation(
        'AC_9',
        "CO_AutoCorr(y,9,'Fourier')",
        CO_AutoCorr(tau=9, whatMethod='Fourier'))

    # outs: A,C,_,1,0
    # tags: autocorrelation,correlation
    AC_10 = HCTSAOperation(
        'AC_10',
        "CO_AutoCorr(y,10,'Fourier')",
        CO_AutoCorr(tau=10, whatMethod='Fourier'))

    # outs: A,C,_,1,1
    # tags: autocorrelation,correlation
    AC_11 = HCTSAOperation(
        'AC_11',
        "CO_AutoCorr(y,11,'Fourier')",
        CO_AutoCorr(tau=11, whatMethod='Fourier'))

    # outs: A,C,_,1,2
    # tags: autocorrelation,correlation
    AC_12 = HCTSAOperation(
        'AC_12',
        "CO_AutoCorr(y,12,'Fourier')",
        CO_AutoCorr(tau=12, whatMethod='Fourier'))

    # outs: A,C,_,1,3
    # tags: autocorrelation,correlation
    AC_13 = HCTSAOperation(
        'AC_13',
        "CO_AutoCorr(y,13,'Fourier')",
        CO_AutoCorr(tau=13, whatMethod='Fourier'))

    # outs: A,C,_,1,4
    # tags: autocorrelation,correlation
    AC_14 = HCTSAOperation(
        'AC_14',
        "CO_AutoCorr(y,14,'Fourier')",
        CO_AutoCorr(tau=14, whatMethod='Fourier'))

    # outs: A,C,_,1,5
    # tags: autocorrelation,correlation
    AC_15 = HCTSAOperation(
        'AC_15',
        "CO_AutoCorr(y,15,'Fourier')",
        CO_AutoCorr(tau=15, whatMethod='Fourier'))

    # outs: A,C,_,1,6
    # tags: autocorrelation,correlation
    AC_16 = HCTSAOperation(
        'AC_16',
        "CO_AutoCorr(y,16,'Fourier')",
        CO_AutoCorr(tau=16, whatMethod='Fourier'))

    # outs: A,C,_,1,7
    # tags: autocorrelation,correlation
    AC_17 = HCTSAOperation(
        'AC_17',
        "CO_AutoCorr(y,17,'Fourier')",
        CO_AutoCorr(tau=17, whatMethod='Fourier'))

    # outs: A,C,_,1,8
    # tags: autocorrelation,correlation
    AC_18 = HCTSAOperation(
        'AC_18',
        "CO_AutoCorr(y,18,'Fourier')",
        CO_AutoCorr(tau=18, whatMethod='Fourier'))

    # outs: A,C,_,1,9
    # tags: autocorrelation,correlation
    AC_19 = HCTSAOperation(
        'AC_19',
        "CO_AutoCorr(y,19,'Fourier')",
        CO_AutoCorr(tau=19, whatMethod='Fourier'))

    # outs: A,C,_,2,0
    # tags: autocorrelation,correlation
    AC_20 = HCTSAOperation(
        'AC_20',
        "CO_AutoCorr(y,20,'Fourier')",
        CO_AutoCorr(tau=20, whatMethod='Fourier'))

    # outs: A,C,_,2,1
    # tags: autocorrelation,correlation
    AC_21 = HCTSAOperation(
        'AC_21',
        "CO_AutoCorr(y,21,'Fourier')",
        CO_AutoCorr(tau=21, whatMethod='Fourier'))

    # outs: A,C,_,2,2
    # tags: autocorrelation,correlation
    AC_22 = HCTSAOperation(
        'AC_22',
        "CO_AutoCorr(y,22,'Fourier')",
        CO_AutoCorr(tau=22, whatMethod='Fourier'))

    # outs: A,C,_,2,3
    # tags: autocorrelation,correlation
    AC_23 = HCTSAOperation(
        'AC_23',
        "CO_AutoCorr(y,23,'Fourier')",
        CO_AutoCorr(tau=23, whatMethod='Fourier'))

    # outs: A,C,_,2,4
    # tags: autocorrelation,correlation
    AC_24 = HCTSAOperation(
        'AC_24',
        "CO_AutoCorr(y,24,'Fourier')",
        CO_AutoCorr(tau=24, whatMethod='Fourier'))

    # outs: A,C,_,2,5
    # tags: autocorrelation,correlation
    AC_25 = HCTSAOperation(
        'AC_25',
        "CO_AutoCorr(y,25,'Fourier')",
        CO_AutoCorr(tau=25, whatMethod='Fourier'))

    # outs: A,C,_,2,6
    # tags: autocorrelation,correlation
    AC_26 = HCTSAOperation(
        'AC_26',
        "CO_AutoCorr(y,26,'Fourier')",
        CO_AutoCorr(tau=26, whatMethod='Fourier'))

    # outs: A,C,_,2,7
    # tags: autocorrelation,correlation
    AC_27 = HCTSAOperation(
        'AC_27',
        "CO_AutoCorr(y,27,'Fourier')",
        CO_AutoCorr(tau=27, whatMethod='Fourier'))

    # outs: A,C,_,2,8
    # tags: autocorrelation,correlation
    AC_28 = HCTSAOperation(
        'AC_28',
        "CO_AutoCorr(y,28,'Fourier')",
        CO_AutoCorr(tau=28, whatMethod='Fourier'))

    # outs: A,C,_,2,9
    # tags: autocorrelation,correlation
    AC_29 = HCTSAOperation(
        'AC_29',
        "CO_AutoCorr(y,29,'Fourier')",
        CO_AutoCorr(tau=29, whatMethod='Fourier'))

    # outs: A,C,_,3,0
    # tags: autocorrelation,correlation
    AC_30 = HCTSAOperation(
        'AC_30',
        "CO_AutoCorr(y,30,'Fourier')",
        CO_AutoCorr(tau=30, whatMethod='Fourier'))

    # outs: A,C,_,3,1
    # tags: autocorrelation,correlation
    AC_31 = HCTSAOperation(
        'AC_31',
        "CO_AutoCorr(y,31,'Fourier')",
        CO_AutoCorr(tau=31, whatMethod='Fourier'))

    # outs: A,C,_,3,2
    # tags: autocorrelation,correlation
    AC_32 = HCTSAOperation(
        'AC_32',
        "CO_AutoCorr(y,32,'Fourier')",
        CO_AutoCorr(tau=32, whatMethod='Fourier'))

    # outs: A,C,_,3,3
    # tags: autocorrelation,correlation
    AC_33 = HCTSAOperation(
        'AC_33',
        "CO_AutoCorr(y,33,'Fourier')",
        CO_AutoCorr(tau=33, whatMethod='Fourier'))

    # outs: A,C,_,3,4
    # tags: autocorrelation,correlation
    AC_34 = HCTSAOperation(
        'AC_34',
        "CO_AutoCorr(y,34,'Fourier')",
        CO_AutoCorr(tau=34, whatMethod='Fourier'))

    # outs: A,C,_,3,5
    # tags: autocorrelation,correlation
    AC_35 = HCTSAOperation(
        'AC_35',
        "CO_AutoCorr(y,35,'Fourier')",
        CO_AutoCorr(tau=35, whatMethod='Fourier'))

    # outs: A,C,_,3,6
    # tags: autocorrelation,correlation
    AC_36 = HCTSAOperation(
        'AC_36',
        "CO_AutoCorr(y,36,'Fourier')",
        CO_AutoCorr(tau=36, whatMethod='Fourier'))

    # outs: A,C,_,3,7
    # tags: autocorrelation,correlation
    AC_37 = HCTSAOperation(
        'AC_37',
        "CO_AutoCorr(y,37,'Fourier')",
        CO_AutoCorr(tau=37, whatMethod='Fourier'))

    # outs: A,C,_,3,8
    # tags: autocorrelation,correlation
    AC_38 = HCTSAOperation(
        'AC_38',
        "CO_AutoCorr(y,38,'Fourier')",
        CO_AutoCorr(tau=38, whatMethod='Fourier'))

    # outs: A,C,_,3,9
    # tags: autocorrelation,correlation
    AC_39 = HCTSAOperation(
        'AC_39',
        "CO_AutoCorr(y,39,'Fourier')",
        CO_AutoCorr(tau=39, whatMethod='Fourier'))

    # outs: A,C,_,4,0
    # tags: autocorrelation,correlation
    AC_40 = HCTSAOperation(
        'AC_40',
        "CO_AutoCorr(y,40,'Fourier')",
        CO_AutoCorr(tau=40, whatMethod='Fourier'))

    # outs: C,O,_,A,u
    # outs: t,o,C,o,r
    # outs: r,S,h,a,p
    # outs: e,_,d,r,o
    # outs: w,n
    # tags: correlation
    CO_AutoCorrShape_drown = HCTSAOperation(
        'CO_AutoCorrShape_drown',
        "CO_AutoCorrShape(y,'drown')",
        CO_AutoCorrShape(stopWhen='drown'))

    # outs: C,O,_,C,o
    # outs: m,p,a,r,e
    # outs: M,i,n,A,M
    # outs: I,_,s,t,d
    # outs: 1,_,2,_,8
    # outs: 0
    # tags: AMI,correlation
    CO_CompareMinAMI_std1_2_80 = HCTSAOperation(
        'CO_CompareMinAMI_std1_2_80',
        "CO_CompareMinAMI(y,'std1',[2:80])",
        CO_CompareMinAMI(binMethod='std1', numBins=array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                         12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
       70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80])))

    # outs: C,O,_,C,o
    # outs: m,p,a,r,e
    # outs: M,i,n,A,M
    # outs: I,_,q,u,a
    # outs: n,t,i,l,e
    # outs: s,_,2,_,8
    # outs: 0
    # tags: AMI,correlation
    CO_CompareMinAMI_quantiles_2_80 = HCTSAOperation(
        'CO_CompareMinAMI_quantiles_2_80',
        "CO_CompareMinAMI(y,'quantiles',[2:80])",
        CO_CompareMinAMI(binMethod='quantiles', numBins=array([ 2,  3,  4,  5,  6,  7,  8,  9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
       70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80])))

    # outs: C,O,_,C,o
    # outs: m,p,a,r,e
    # outs: M,i,n,A,M
    # outs: I,_,e,v,e
    # outs: n,_,2,_,8
    # outs: 0
    # tags: AMI,correlation
    CO_CompareMinAMI_even_2_80 = HCTSAOperation(
        'CO_CompareMinAMI_even_2_80',
        "CO_CompareMinAMI(y,'even',[2:80])",
        CO_CompareMinAMI(binMethod='even', numBins=array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                         12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
       70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80])))

    # outs: C,O,_,C,o
    # outs: m,p,a,r,e
    # outs: M,i,n,A,M
    # outs: I,_,s,t,d
    # outs: 2,_,2,_,8
    # outs: 0
    # tags: AMI,correlation
    CO_CompareMinAMI_std2_2_80 = HCTSAOperation(
        'CO_CompareMinAMI_std2_2_80',
        "CO_CompareMinAMI(y,'std2',[2:80])",
        CO_CompareMinAMI(binMethod='std2', numBins=array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                         12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
       70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80])))

    # outs: C,O,_,E,m
    # outs: b,e,d,2,_
    # outs: t,a,u
    # tags: correlation,embedding
    CO_Embed2_tau = HCTSAOperation(
        'CO_Embed2_tau',
        "CO_Embed2(y,'tau')",
        CO_Embed2(tau='tau'))

    # outs: C,O,_,E,m
    # outs: b,e,d,2,_
    # outs: A,n,g,l,e
    # outs: T,a,u,_,5
    # outs: 0
    # tags: correlation,embedding
    CO_Embed2_AngleTau_50 = HCTSAOperation(
        'CO_Embed2_AngleTau_50',
        'CO_Embed2_AngleTau(y,50)',
        CO_Embed2_AngleTau(maxTau=50))

    # outs: C,O,_,E,m
    # outs: b,e,d,2,_
    # outs: B,a,s,i,c
    # outs: _,1
    # tags: correlation
    CO_Embed2_Basic_1 = HCTSAOperation(
        'CO_Embed2_Basic_1',
        'CO_Embed2_Basic(y,1)',
        CO_Embed2_Basic(tau=1))

    # outs: C,O,_,E,m
    # outs: b,e,d,2,_
    # outs: B,a,s,i,c
    # outs: _,t,a,u
    # tags: correlation
    CO_Embed2_Basic_tau = HCTSAOperation(
        'CO_Embed2_Basic_tau',
        "CO_Embed2_Basic(y,'tau')",
        CO_Embed2_Basic(tau='tau'))

    # outs: C,O,_,E,m
    # outs: b,e,d,2,_
    # outs: D,i,s,t,_
    # outs: t,a,u
    # tags: correlation,embedding
    CO_Embed2_Dist_tau = HCTSAOperation(
        'CO_Embed2_Dist_tau',
        "CO_Embed2_Dist(y,'tau')",
        CO_Embed2_Dist(tau='tau'))

    # outs: C,O,_,E,m
    # outs: b,e,d,2,_
    # outs: S,h,a,p,e
    # outs: s,_,t,a,u
    # outs: _,c,i,r,c
    # outs: l,e,_,0,1
    # tags: correlation,embedding
    CO_Embed2_Shapes_tau_circle_01 = HCTSAOperation(
        'CO_Embed2_Shapes_tau_circle_01',
        "CO_Embed2_Shapes(y,'tau','circle',0.1)",
        CO_Embed2_Shapes(tau='tau', shape='circle', r=0.1))

    # outs: C,O,_,E,m
    # outs: b,e,d,2,_
    # outs: S,h,a,p,e
    # outs: s,_,t,a,u
    # outs: _,c,i,r,c
    # outs: l,e,_,1
    # tags: correlation,embedding
    CO_Embed2_Shapes_tau_circle_1 = HCTSAOperation(
        'CO_Embed2_Shapes_tau_circle_1',
        "CO_Embed2_Shapes(y,'tau','circle',1)",
        CO_Embed2_Shapes(tau='tau', shape='circle', r=1))

    # outs: C,O,_,F,i
    # outs: r,s,t,M,i
    # outs: n,_,a,c
    # tags: autocorrelation,correlation,tau
    CO_FirstMin_ac = HCTSAOperation(
        'CO_FirstMin_ac',
        "CO_FirstMin(y,'ac')",
        CO_FirstMin(minWhat='ac'))

    # outs: C,O,_,F,i
    # outs: r,s,t,M,i
    # outs: n,_,m,i,_
    # outs: g,a,u,s,s
    # outs: i,a,n
    # tags: AMI,correlation
    CO_FirstMin_mi_gaussian = HCTSAOperation(
        'CO_FirstMin_mi_gaussian',
        "CO_FirstMin(y,'mi-gaussian')",
        CO_FirstMin(minWhat='mi-gaussian'))

    # outs: C,O,_,F,i
    # outs: r,s,t,M,i
    # outs: n,_,m,i,_
    # outs: k,e,r,n,e
    # outs: l
    # tags: AMI,correlation
    CO_FirstMin_mi_kernel = HCTSAOperation(
        'CO_FirstMin_mi_kernel',
        "CO_FirstMin(y,'mi-kernel')",
        CO_FirstMin(minWhat='mi-kernel'))

    # outs: C,O,_,F,i
    # outs: r,s,t,M,i
    # outs: n,_,m,i,_
    # outs: k,r,a,s,k
    # outs: o,v,1,_,4
    # tags: AMI,correlation
    CO_FirstMin_mi_kraskov1_4 = HCTSAOperation(
        'CO_FirstMin_mi_kraskov1_4',
        "CO_FirstMin(y,'mi-kraskov1','4')",
        CO_FirstMin(minWhat='mi-kraskov1', extraParam='4'))

    # outs: C,O,_,F,i
    # outs: r,s,t,M,i
    # outs: n,_,m,i,_
    # outs: k,r,a,s,k
    # outs: o,v,2,_,4
    # tags: AMI,correlation
    CO_FirstMin_mi_kraskov2_4 = HCTSAOperation(
        'CO_FirstMin_mi_kraskov2_4',
        "CO_FirstMin(y,'mi-kraskov2','4')",
        CO_FirstMin(minWhat='mi-kraskov2', extraParam='4'))

    # outs: C,O,_,F,i
    # outs: r,s,t,M,i
    # outs: n,_,m,i,_
    # outs: h,i,s,t,_
    # outs: 5
    # tags: AMI,correlation
    CO_FirstMin_mi_hist_5 = HCTSAOperation(
        'CO_FirstMin_mi_hist_5',
        "CO_FirstMin(y,'mi-hist',5)",
        CO_FirstMin(minWhat='mi-hist', extraParam=5))

    # outs: C,O,_,F,i
    # outs: r,s,t,M,i
    # outs: n,_,m,i,_
    # outs: h,i,s,t,_
    # outs: 1,0
    # tags: AMI,correlation
    CO_FirstMin_mi_hist_10 = HCTSAOperation(
        'CO_FirstMin_mi_hist_10',
        "CO_FirstMin(y,'mi-hist',10)",
        CO_FirstMin(minWhat='mi-hist', extraParam=10))

    # outs: C,O,_,F,i
    # outs: r,s,t,Z,e
    # outs: r,o,_,a,c
    # tags: autocorrelation,correlation,tau
    CO_FirstZero_ac = HCTSAOperation(
        'CO_FirstZero_ac',
        "CO_FirstZero(y,'ac')",
        CO_FirstZero(corrFun='ac'))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,s,t,d,1
    # outs: _,2
    # tags: AMI,correlation,information
    CO_HistogramAMI_std1_2 = HCTSAOperation(
        'CO_HistogramAMI_std1_2',
        "CO_HistogramAMI(y,1:5,'std1',2)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='std1', numBins=2))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,s,t,d,1
    # outs: _,5
    # tags: AMI,correlation,information
    CO_HistogramAMI_std1_5 = HCTSAOperation(
        'CO_HistogramAMI_std1_5',
        "CO_HistogramAMI(y,1:5,'std1',5)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='std1', numBins=5))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,s,t,d,1
    # outs: _,1,0
    # tags: AMI,correlation,information
    CO_HistogramAMI_std1_10 = HCTSAOperation(
        'CO_HistogramAMI_std1_10',
        "CO_HistogramAMI(y,1:5,'std1',10)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='std1', numBins=10))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,s,t,d,2
    # outs: _,2
    # tags: AMI,correlation,information
    CO_HistogramAMI_std2_2 = HCTSAOperation(
        'CO_HistogramAMI_std2_2',
        "CO_HistogramAMI(y,1:5,'std2',2)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='std2', numBins=2))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,s,t,d,2
    # outs: _,5
    # tags: AMI,correlation,information
    CO_HistogramAMI_std2_5 = HCTSAOperation(
        'CO_HistogramAMI_std2_5',
        "CO_HistogramAMI(y,1:5,'std2',5)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='std2', numBins=5))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,s,t,d,2
    # outs: _,1,0
    # tags: AMI,correlation,information
    CO_HistogramAMI_std2_10 = HCTSAOperation(
        'CO_HistogramAMI_std2_10',
        "CO_HistogramAMI(y,1:5,'std2',10)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='std2', numBins=10))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,e,v,e,n
    # outs: _,2
    # tags: AMI,correlation,information
    CO_HistogramAMI_even_2 = HCTSAOperation(
        'CO_HistogramAMI_even_2',
        "CO_HistogramAMI(y,1:5,'even',2)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='even', numBins=2))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,e,v,e,n
    # outs: _,5
    # tags: AMI,correlation,information
    CO_HistogramAMI_even_5 = HCTSAOperation(
        'CO_HistogramAMI_even_5',
        "CO_HistogramAMI(y,1:5,'even',5)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='even', numBins=5))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,e,v,e,n
    # outs: _,1,0
    # tags: AMI,correlation,information
    CO_HistogramAMI_even_10 = HCTSAOperation(
        'CO_HistogramAMI_even_10',
        "CO_HistogramAMI(y,1:5,'even',10)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='even', numBins=10))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,q,u,a,n
    # outs: t,i,l,e,s
    # outs: _,2
    # tags: AMI,correlation,information
    CO_HistogramAMI_quantiles_2 = HCTSAOperation(
        'CO_HistogramAMI_quantiles_2',
        "CO_HistogramAMI(y,1:5,'quantiles',2)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='quantiles', numBins=2))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,q,u,a,n
    # outs: t,i,l,e,s
    # outs: _,5
    # tags: AMI,correlation,information
    CO_HistogramAMI_quantiles_5 = HCTSAOperation(
        'CO_HistogramAMI_quantiles_5',
        "CO_HistogramAMI(y,1:5,'quantiles',5)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='quantiles', numBins=5))

    # outs: C,O,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,A,M,I
    # outs: _,q,u,a,n
    # outs: t,i,l,e,s
    # outs: _,1,0
    # tags: AMI,correlation,information
    CO_HistogramAMI_quantiles_10 = HCTSAOperation(
        'CO_HistogramAMI_quantiles_10',
        "CO_HistogramAMI(y,1:5,'quantiles',10)",
        CO_HistogramAMI(tau=MatlabSequence('1:5'), meth='quantiles', numBins=10))

    # outs: A,C,_,n,l
    # outs: _,1,2
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_12 = HCTSAOperation(
        'AC_nl_12',
        'CO_NonlinearAutocorr(y,[1,2])',
        CO_NonlinearAutocorr(taus=array([1, 2])))

    # outs: A,C,_,n,l
    # outs: _,1,2,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_123 = HCTSAOperation(
        'AC_nl_123',
        'CO_NonlinearAutocorr(y,[1,2,3])',
        CO_NonlinearAutocorr(taus=array([1, 2, 3])))

    # outs: A,C,_,n,l
    # outs: _,1,2,3,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_1234 = HCTSAOperation(
        'AC_nl_1234',
        'CO_NonlinearAutocorr(y,[1,2,3,4])',
        CO_NonlinearAutocorr(taus=array([1, 2, 3, 4])))

    # outs: A,C,_,n,l
    # outs: _,1,2,3,4
    # outs: 5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_12345 = HCTSAOperation(
        'AC_nl_12345',
        'CO_NonlinearAutocorr(y,[1,2,3,4,5])',
        CO_NonlinearAutocorr(taus=array([1, 2, 3, 4, 5])))

    # outs: A,C,_,n,l
    # outs: _,1,2,3,4
    # outs: 5,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_123456 = HCTSAOperation(
        'AC_nl_123456',
        'CO_NonlinearAutocorr(y,[1,2,3,4,5,6])',
        CO_NonlinearAutocorr(taus=array([1, 2, 3, 4, 5, 6])))

    # outs: A,C,_,n,l
    # outs: _,1,2,3,4
    # outs: 5,6,7
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_1234567 = HCTSAOperation(
        'AC_nl_1234567',
        'CO_NonlinearAutocorr(y,[1,2,3,4,5,6,7])',
        CO_NonlinearAutocorr(taus=array([1, 2, 3, 4, 5, 6, 7])))

    # outs: A,C,_,n,l
    # outs: _,1,2,3,4
    # outs: 5,6,7,8
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_12345678 = HCTSAOperation(
        'AC_nl_12345678',
        'CO_NonlinearAutocorr(y,[1,2,3,4,5,6,7,8])',
        CO_NonlinearAutocorr(taus=array([1, 2, 3, 4, 5, 6, 7, 8])))

    # outs: A,C,_,n,l
    # outs: _,1,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_13 = HCTSAOperation(
        'AC_nl_13',
        'CO_NonlinearAutocorr(y,[1,3])',
        CO_NonlinearAutocorr(taus=array([1, 3])))

    # outs: A,C,_,n,l
    # outs: _,2,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_24 = HCTSAOperation(
        'AC_nl_24',
        'CO_NonlinearAutocorr(y,[2,4])',
        CO_NonlinearAutocorr(taus=array([2, 4])))

    # outs: A,C,_,n,l
    # outs: _,1,3,5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_135 = HCTSAOperation(
        'AC_nl_135',
        'CO_NonlinearAutocorr(y,[1,3,5])',
        CO_NonlinearAutocorr(taus=array([1, 3, 5])))

    # outs: A,C,_,n,l
    # outs: _,2,4,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_246 = HCTSAOperation(
        'AC_nl_246',
        'CO_NonlinearAutocorr(y,[2,4,6])',
        CO_NonlinearAutocorr(taus=array([2, 4, 6])))

    # outs: A,C,_,n,l
    # outs: _,1,3,5,7
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_1357 = HCTSAOperation(
        'AC_nl_1357',
        'CO_NonlinearAutocorr(y,[1,3,5,7])',
        CO_NonlinearAutocorr(taus=array([1, 3, 5, 7])))

    # outs: A,C,_,n,l
    # outs: _,2,4,6,8
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_2468 = HCTSAOperation(
        'AC_nl_2468',
        'CO_NonlinearAutocorr(y,[2,4,6,8])',
        CO_NonlinearAutocorr(taus=array([2, 4, 6, 8])))

    # outs: A,C,_,n,l
    # outs: _,1,2,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_124 = HCTSAOperation(
        'AC_nl_124',
        'CO_NonlinearAutocorr(y,[1,2,4])',
        CO_NonlinearAutocorr(taus=array([1, 2, 4])))

    # outs: A,C,_,n,l
    # outs: _,1,3,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_134 = HCTSAOperation(
        'AC_nl_134',
        'CO_NonlinearAutocorr(y,[1,3,4])',
        CO_NonlinearAutocorr(taus=array([1, 3, 4])))

    # outs: A,C,_,n,l
    # outs: _,1,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_14 = HCTSAOperation(
        'AC_nl_14',
        'CO_NonlinearAutocorr(y,[1,4])',
        CO_NonlinearAutocorr(taus=array([1, 4])))

    # outs: A,C,_,n,l
    # outs: _,1,1
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_11 = HCTSAOperation(
        'AC_nl_11',
        'CO_NonlinearAutocorr(y,[1,1])',
        CO_NonlinearAutocorr(taus=array([1, 1])))

    # outs: A,C,_,n,l
    # outs: _,2,2
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_22 = HCTSAOperation(
        'AC_nl_22',
        'CO_NonlinearAutocorr(y,[2,2])',
        CO_NonlinearAutocorr(taus=array([2, 2])))

    # outs: A,C,_,n,l
    # outs: _,3,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_33 = HCTSAOperation(
        'AC_nl_33',
        'CO_NonlinearAutocorr(y,[3,3])',
        CO_NonlinearAutocorr(taus=array([3, 3])))

    # outs: A,C,_,n,l
    # outs: _,4,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_44 = HCTSAOperation(
        'AC_nl_44',
        'CO_NonlinearAutocorr(y,[4,4])',
        CO_NonlinearAutocorr(taus=array([4, 4])))

    # outs: A,C,_,n,l
    # outs: _,5,5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_55 = HCTSAOperation(
        'AC_nl_55',
        'CO_NonlinearAutocorr(y,[5,5])',
        CO_NonlinearAutocorr(taus=array([5, 5])))

    # outs: A,C,_,n,l
    # outs: _,6,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_66 = HCTSAOperation(
        'AC_nl_66',
        'CO_NonlinearAutocorr(y,[6,6])',
        CO_NonlinearAutocorr(taus=array([6, 6])))

    # outs: A,C,_,n,l
    # outs: _,0,1,1
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_011 = HCTSAOperation(
        'AC_nl_011',
        'CO_NonlinearAutocorr(y,[0,1,1])',
        CO_NonlinearAutocorr(taus=array([0, 1, 1])))

    # outs: A,C,_,n,l
    # outs: _,0,2,2
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_022 = HCTSAOperation(
        'AC_nl_022',
        'CO_NonlinearAutocorr(y,[0,2,2])',
        CO_NonlinearAutocorr(taus=array([0, 2, 2])))

    # outs: A,C,_,n,l
    # outs: _,0,3,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_033 = HCTSAOperation(
        'AC_nl_033',
        'CO_NonlinearAutocorr(y,[0,3,3])',
        CO_NonlinearAutocorr(taus=array([0, 3, 3])))

    # outs: A,C,_,n,l
    # outs: _,0,4,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_044 = HCTSAOperation(
        'AC_nl_044',
        'CO_NonlinearAutocorr(y,[0,4,4])',
        CO_NonlinearAutocorr(taus=array([0, 4, 4])))

    # outs: A,C,_,n,l
    # outs: _,0,5,5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_055 = HCTSAOperation(
        'AC_nl_055',
        'CO_NonlinearAutocorr(y,[0,5,5])',
        CO_NonlinearAutocorr(taus=array([0, 5, 5])))

    # outs: A,C,_,n,l
    # outs: _,0,6,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_066 = HCTSAOperation(
        'AC_nl_066',
        'CO_NonlinearAutocorr(y,[0,6,6])',
        CO_NonlinearAutocorr(taus=array([0, 6, 6])))

    # outs: A,C,_,n,l
    # outs: _,0,0,1
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_001 = HCTSAOperation(
        'AC_nl_001',
        'CO_NonlinearAutocorr(y,[0,0,1])',
        CO_NonlinearAutocorr(taus=array([0, 0, 1])))

    # outs: A,C,_,n,l
    # outs: _,0,0,2
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_002 = HCTSAOperation(
        'AC_nl_002',
        'CO_NonlinearAutocorr(y,[0,0,2])',
        CO_NonlinearAutocorr(taus=array([0, 0, 2])))

    # outs: A,C,_,n,l
    # outs: _,0,0,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_003 = HCTSAOperation(
        'AC_nl_003',
        'CO_NonlinearAutocorr(y,[0,0,3])',
        CO_NonlinearAutocorr(taus=array([0, 0, 3])))

    # outs: A,C,_,n,l
    # outs: _,0,0,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_004 = HCTSAOperation(
        'AC_nl_004',
        'CO_NonlinearAutocorr(y,[0,0,4])',
        CO_NonlinearAutocorr(taus=array([0, 0, 4])))

    # outs: A,C,_,n,l
    # outs: _,0,0,5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_005 = HCTSAOperation(
        'AC_nl_005',
        'CO_NonlinearAutocorr(y,[0,0,5])',
        CO_NonlinearAutocorr(taus=array([0, 0, 5])))

    # outs: A,C,_,n,l
    # outs: _,0,0,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_006 = HCTSAOperation(
        'AC_nl_006',
        'CO_NonlinearAutocorr(y,[0,0,6])',
        CO_NonlinearAutocorr(taus=array([0, 0, 6])))

    # outs: A,C,_,n,l
    # outs: _,0,1,2
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_012 = HCTSAOperation(
        'AC_nl_012',
        'CO_NonlinearAutocorr(y,[0,1,2])',
        CO_NonlinearAutocorr(taus=array([0, 1, 2])))

    # outs: A,C,_,n,l
    # outs: _,0,1,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_013 = HCTSAOperation(
        'AC_nl_013',
        'CO_NonlinearAutocorr(y,[0,1,3])',
        CO_NonlinearAutocorr(taus=array([0, 1, 3])))

    # outs: A,C,_,n,l
    # outs: _,0,2,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_023 = HCTSAOperation(
        'AC_nl_023',
        'CO_NonlinearAutocorr(y,[0,2,3])',
        CO_NonlinearAutocorr(taus=array([0, 2, 3])))

    # outs: A,C,_,n,l
    # outs: _,0,1,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_014 = HCTSAOperation(
        'AC_nl_014',
        'CO_NonlinearAutocorr(y,[0,1,4])',
        CO_NonlinearAutocorr(taus=array([0, 1, 4])))

    # outs: A,C,_,n,l
    # outs: _,0,2,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_024 = HCTSAOperation(
        'AC_nl_024',
        'CO_NonlinearAutocorr(y,[0,2,4])',
        CO_NonlinearAutocorr(taus=array([0, 2, 4])))

    # outs: A,C,_,n,l
    # outs: _,0,3,4
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_034 = HCTSAOperation(
        'AC_nl_034',
        'CO_NonlinearAutocorr(y,[0,3,4])',
        CO_NonlinearAutocorr(taus=array([0, 3, 4])))

    # outs: A,C,_,n,l
    # outs: _,0,1,5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_015 = HCTSAOperation(
        'AC_nl_015',
        'CO_NonlinearAutocorr(y,[0,1,5])',
        CO_NonlinearAutocorr(taus=array([0, 1, 5])))

    # outs: A,C,_,n,l
    # outs: _,0,2,5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_025 = HCTSAOperation(
        'AC_nl_025',
        'CO_NonlinearAutocorr(y,[0,2,5])',
        CO_NonlinearAutocorr(taus=array([0, 2, 5])))

    # outs: A,C,_,n,l
    # outs: _,0,3,5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_035 = HCTSAOperation(
        'AC_nl_035',
        'CO_NonlinearAutocorr(y,[0,3,5])',
        CO_NonlinearAutocorr(taus=array([0, 3, 5])))

    # outs: A,C,_,n,l
    # outs: _,0,4,5
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_045 = HCTSAOperation(
        'AC_nl_045',
        'CO_NonlinearAutocorr(y,[0,4,5])',
        CO_NonlinearAutocorr(taus=array([0, 4, 5])))

    # outs: A,C,_,n,l
    # outs: _,0,1,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_016 = HCTSAOperation(
        'AC_nl_016',
        'CO_NonlinearAutocorr(y,[0,1,6])',
        CO_NonlinearAutocorr(taus=array([0, 1, 6])))

    # outs: A,C,_,n,l
    # outs: _,0,2,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_026 = HCTSAOperation(
        'AC_nl_026',
        'CO_NonlinearAutocorr(y,[0,2,6])',
        CO_NonlinearAutocorr(taus=array([0, 2, 6])))

    # outs: A,C,_,n,l
    # outs: _,0,3,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_036 = HCTSAOperation(
        'AC_nl_036',
        'CO_NonlinearAutocorr(y,[0,3,6])',
        CO_NonlinearAutocorr(taus=array([0, 3, 6])))

    # outs: A,C,_,n,l
    # outs: _,0,4,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_046 = HCTSAOperation(
        'AC_nl_046',
        'CO_NonlinearAutocorr(y,[0,4,6])',
        CO_NonlinearAutocorr(taus=array([0, 4, 6])))

    # outs: A,C,_,n,l
    # outs: _,0,5,6
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_056 = HCTSAOperation(
        'AC_nl_056',
        'CO_NonlinearAutocorr(y,[0,5,6])',
        CO_NonlinearAutocorr(taus=array([0, 5, 6])))

    # outs: A,C,_,n,l
    # outs: _,1,1,2
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_112 = HCTSAOperation(
        'AC_nl_112',
        'CO_NonlinearAutocorr(y,[1,1,2])',
        CO_NonlinearAutocorr(taus=array([1, 1, 2])))

    # outs: A,C,_,n,l
    # outs: _,1,1,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_113 = HCTSAOperation(
        'AC_nl_113',
        'CO_NonlinearAutocorr(y,[1,1,3])',
        CO_NonlinearAutocorr(taus=array([1, 1, 3])))

    # outs: A,C,_,n,l
    # outs: _,1,2,2
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_122 = HCTSAOperation(
        'AC_nl_122',
        'CO_NonlinearAutocorr(y,[1,2,2])',
        CO_NonlinearAutocorr(taus=array([1, 2, 2])))

    # outs: A,C,_,n,l
    # outs: _,1,3,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_133 = HCTSAOperation(
        'AC_nl_133',
        'CO_NonlinearAutocorr(y,[1,3,3])',
        CO_NonlinearAutocorr(taus=array([1, 3, 3])))

    # outs: A,C,_,n,l
    # outs: _,2,2,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_223 = HCTSAOperation(
        'AC_nl_223',
        'CO_NonlinearAutocorr(y,[2,2,3])',
        CO_NonlinearAutocorr(taus=array([2, 2, 3])))

    # outs: A,C,_,n,l
    # outs: _,2,3,3
    # tags: autocorrelation,correlation,nonlinearautocorr
    AC_nl_233 = HCTSAOperation(
        'AC_nl_233',
        'CO_NonlinearAutocorr(y,[2,3,3])',
        CO_NonlinearAutocorr(taus=array([2, 3, 3])))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 0
    # tags: AMI,correlation,information
    CO_RM_AMInformation_0 = HCTSAOperation(
        'CO_RM_AMInformation_0',
        'CO_RM_AMInformation(y,0)',
        CO_RM_AMInformation(tau=0))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 1
    # tags: AMI,correlation,information
    CO_RM_AMInformation_1 = HCTSAOperation(
        'CO_RM_AMInformation_1',
        'CO_RM_AMInformation(y,1)',
        CO_RM_AMInformation(tau=1))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 2
    # tags: AMI,correlation,information
    CO_RM_AMInformation_2 = HCTSAOperation(
        'CO_RM_AMInformation_2',
        'CO_RM_AMInformation(y,2)',
        CO_RM_AMInformation(tau=2))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 3
    # tags: AMI,correlation,information
    CO_RM_AMInformation_3 = HCTSAOperation(
        'CO_RM_AMInformation_3',
        'CO_RM_AMInformation(y,3)',
        CO_RM_AMInformation(tau=3))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 4
    # tags: AMI,correlation,information
    CO_RM_AMInformation_4 = HCTSAOperation(
        'CO_RM_AMInformation_4',
        'CO_RM_AMInformation(y,4)',
        CO_RM_AMInformation(tau=4))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 5
    # tags: AMI,correlation,information
    CO_RM_AMInformation_5 = HCTSAOperation(
        'CO_RM_AMInformation_5',
        'CO_RM_AMInformation(y,5)',
        CO_RM_AMInformation(tau=5))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 6
    # tags: AMI,correlation,information
    CO_RM_AMInformation_6 = HCTSAOperation(
        'CO_RM_AMInformation_6',
        'CO_RM_AMInformation(y,6)',
        CO_RM_AMInformation(tau=6))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 7
    # tags: AMI,correlation,information
    CO_RM_AMInformation_7 = HCTSAOperation(
        'CO_RM_AMInformation_7',
        'CO_RM_AMInformation(y,7)',
        CO_RM_AMInformation(tau=7))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 8
    # tags: AMI,correlation,information
    CO_RM_AMInformation_8 = HCTSAOperation(
        'CO_RM_AMInformation_8',
        'CO_RM_AMInformation(y,8)',
        CO_RM_AMInformation(tau=8))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 9
    # tags: AMI,correlation,information
    CO_RM_AMInformation_9 = HCTSAOperation(
        'CO_RM_AMInformation_9',
        'CO_RM_AMInformation(y,9)',
        CO_RM_AMInformation(tau=9))

    # outs: C,O,_,R,M
    # outs: _,A,M,I,n
    # outs: f,o,r,m,a
    # outs: t,i,o,n,_
    # outs: 1,0
    # tags: AMI,correlation,information
    CO_RM_AMInformation_10 = HCTSAOperation(
        'CO_RM_AMInformation_10',
        'CO_RM_AMInformation(y,10)',
        CO_RM_AMInformation(tau=10))

    # outs: C,O,_,S,t
    # outs: i,c,k,A,n
    # outs: g,l,e,s,_
    # outs: y
    # tags: correlation
    CO_StickAngles_y = HCTSAOperation(
        'CO_StickAngles_y',
        'CO_StickAngles(y)',
        CO_StickAngles())

    # outs: C,O,_,T,r
    # outs: a,n,s,l,a
    # outs: t,e,S,h,a
    # outs: p,e,_,c,i
    # outs: r,c,l,e,_
    # outs: 1,5,_,p,t
    # outs: s
    # tags: correlation
    CO_TranslateShape_circle_15_pts = HCTSAOperation(
        'CO_TranslateShape_circle_15_pts',
        "CO_TranslateShape(y,'circle',1.5,'pts')",
        CO_TranslateShape(shape='circle', d=1.5, howToMove='pts'))

    # outs: C,O,_,T,r
    # outs: a,n,s,l,a
    # outs: t,e,S,h,a
    # outs: p,e,_,c,i
    # outs: r,c,l,e,_
    # outs: 2,5,_,p,t
    # outs: s
    # tags: correlation
    CO_TranslateShape_circle_25_pts = HCTSAOperation(
        'CO_TranslateShape_circle_25_pts',
        "CO_TranslateShape(y,'circle',2.5,'pts')",
        CO_TranslateShape(shape='circle', d=2.5, howToMove='pts'))

    # outs: C,O,_,T,r
    # outs: a,n,s,l,a
    # outs: t,e,S,h,a
    # outs: p,e,_,c,i
    # outs: r,c,l,e,_
    # outs: 3,5,_,p,t
    # outs: s
    # tags: correlation
    CO_TranslateShape_circle_35_pts = HCTSAOperation(
        'CO_TranslateShape_circle_35_pts',
        "CO_TranslateShape(y,'circle',3.5,'pts')",
        CO_TranslateShape(shape='circle', d=3.5, howToMove='pts'))

    # outs: C,O,_,T,r
    # outs: a,n,s,l,a
    # outs: t,e,S,h,a
    # outs: p,e,_,r,e
    # outs: c,t,a,n,g
    # outs: l,e,_,2,_
    # outs: p,t,s
    # tags: correlation
    CO_TranslateShape_rectangle_2_pts = HCTSAOperation(
        'CO_TranslateShape_rectangle_2_pts',
        "CO_TranslateShape(y,'rectangle',2,'pts')",
        CO_TranslateShape(shape='rectangle', d=2, howToMove='pts'))

    # outs: C,O,_,f,1
    # outs: e,c,a,c
    # tags: autocorrelation,correlation,tau
    CO_f1ecac = HCTSAOperation(
        'CO_f1ecac',
        'CO_f1ecac(y)',
        CO_f1ecac())

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,1
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_1 = HCTSAOperation(
        'CO_fzcglscf_1_1',
        'CO_fzcglscf(y,1,1)',
        CO_fzcglscf(alpha=1, beta=1))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,2
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_2 = HCTSAOperation(
        'CO_fzcglscf_1_2',
        'CO_fzcglscf(y,1,2)',
        CO_fzcglscf(alpha=1, beta=2))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,3
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_3 = HCTSAOperation(
        'CO_fzcglscf_1_3',
        'CO_fzcglscf(y,1,3)',
        CO_fzcglscf(alpha=1, beta=3))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,4
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_4 = HCTSAOperation(
        'CO_fzcglscf_1_4',
        'CO_fzcglscf(y,1,4)',
        CO_fzcglscf(alpha=1, beta=4))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,5
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_5 = HCTSAOperation(
        'CO_fzcglscf_1_5',
        'CO_fzcglscf(y,1,5)',
        CO_fzcglscf(alpha=1, beta=5))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,6
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_6 = HCTSAOperation(
        'CO_fzcglscf_1_6',
        'CO_fzcglscf(y,1,6)',
        CO_fzcglscf(alpha=1, beta=6))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,7
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_7 = HCTSAOperation(
        'CO_fzcglscf_1_7',
        'CO_fzcglscf(y,1,7)',
        CO_fzcglscf(alpha=1, beta=7))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,8
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_8 = HCTSAOperation(
        'CO_fzcglscf_1_8',
        'CO_fzcglscf(y,1,8)',
        CO_fzcglscf(alpha=1, beta=8))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,9
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_9 = HCTSAOperation(
        'CO_fzcglscf_1_9',
        'CO_fzcglscf(y,1,9)',
        CO_fzcglscf(alpha=1, beta=9))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,_,1
    # outs: 0
    # tags: correlation,glscf,tau
    CO_fzcglscf_1_10 = HCTSAOperation(
        'CO_fzcglscf_1_10',
        'CO_fzcglscf(y,1,10)',
        CO_fzcglscf(alpha=1, beta=10))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,2,_,2
    # tags: correlation,glscf,tau
    CO_fzcglscf_2_2 = HCTSAOperation(
        'CO_fzcglscf_2_2',
        'CO_fzcglscf(y,2,2)',
        CO_fzcglscf(alpha=2, beta=2))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,2,_,5
    # tags: correlation,glscf,tau
    CO_fzcglscf_2_5 = HCTSAOperation(
        'CO_fzcglscf_2_5',
        'CO_fzcglscf(y,2,5)',
        CO_fzcglscf(alpha=2, beta=5))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,2,_,1
    # outs: 0
    # tags: correlation,glscf,tau
    CO_fzcglscf_2_10 = HCTSAOperation(
        'CO_fzcglscf_2_10',
        'CO_fzcglscf(y,2,10)',
        CO_fzcglscf(alpha=2, beta=10))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,5,_,5
    # tags: correlation,glscf,tau
    CO_fzcglscf_5_5 = HCTSAOperation(
        'CO_fzcglscf_5_5',
        'CO_fzcglscf(y,5,5)',
        CO_fzcglscf(alpha=5, beta=5))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,5,_,1
    # outs: 0
    # tags: correlation,glscf,tau
    CO_fzcglscf_5_10 = HCTSAOperation(
        'CO_fzcglscf_5_10',
        'CO_fzcglscf(y,5,10)',
        CO_fzcglscf(alpha=5, beta=10))

    # outs: C,O,_,f,z
    # outs: c,g,l,s,c
    # outs: f,_,1,0,_
    # outs: 1,0
    # tags: correlation,glscf,tau
    CO_fzcglscf_10_10 = HCTSAOperation(
        'CO_fzcglscf_10_10',
        'CO_fzcglscf(y,10,10)',
        CO_fzcglscf(alpha=10, beta=10))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,_,1
    # tags: correlation,glscf
    CO_glscf_1_1_1 = HCTSAOperation(
        'CO_glscf_1_1_1',
        'CO_glscf(y,1,1,1)',
        CO_glscf(alpha=1, beta=1, tau=1))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,_,2
    # tags: correlation,glscf
    CO_glscf_1_1_2 = HCTSAOperation(
        'CO_glscf_1_1_2',
        'CO_glscf(y,1,1,2)',
        CO_glscf(alpha=1, beta=1, tau=2))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,_,3
    # tags: correlation,glscf
    CO_glscf_1_1_3 = HCTSAOperation(
        'CO_glscf_1_1_3',
        'CO_glscf(y,1,1,3)',
        CO_glscf(alpha=1, beta=1, tau=3))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,_,4
    # tags: correlation,glscf
    CO_glscf_1_1_4 = HCTSAOperation(
        'CO_glscf_1_1_4',
        'CO_glscf(y,1,1,4)',
        CO_glscf(alpha=1, beta=1, tau=4))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,_,5
    # tags: correlation,glscf
    CO_glscf_1_1_5 = HCTSAOperation(
        'CO_glscf_1_1_5',
        'CO_glscf(y,1,1,5)',
        CO_glscf(alpha=1, beta=1, tau=5))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,_,t,a
    # outs: u
    # tags: correlation,glscf
    CO_glscf_1_1_tau = HCTSAOperation(
        'CO_glscf_1_1_tau',
        "CO_glscf(y,1,1,'tau')",
        CO_glscf(alpha=1, beta=1, tau='tau'))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,2,_,1
    # tags: correlation,glscf
    CO_glscf_1_2_1 = HCTSAOperation(
        'CO_glscf_1_2_1',
        'CO_glscf(y,1,2,1)',
        CO_glscf(alpha=1, beta=2, tau=1))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,2,_,2
    # tags: correlation,glscf
    CO_glscf_1_2_2 = HCTSAOperation(
        'CO_glscf_1_2_2',
        'CO_glscf(y,1,2,2)',
        CO_glscf(alpha=1, beta=2, tau=2))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,2,_,3
    # tags: correlation,glscf
    CO_glscf_1_2_3 = HCTSAOperation(
        'CO_glscf_1_2_3',
        'CO_glscf(y,1,2,3)',
        CO_glscf(alpha=1, beta=2, tau=3))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,2,_,4
    # tags: correlation,glscf
    CO_glscf_1_2_4 = HCTSAOperation(
        'CO_glscf_1_2_4',
        'CO_glscf(y,1,2,4)',
        CO_glscf(alpha=1, beta=2, tau=4))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,2,_,5
    # tags: correlation,glscf
    CO_glscf_1_2_5 = HCTSAOperation(
        'CO_glscf_1_2_5',
        'CO_glscf(y,1,2,5)',
        CO_glscf(alpha=1, beta=2, tau=5))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,2,_,t,a
    # outs: u
    # tags: correlation,glscf
    CO_glscf_1_2_tau = HCTSAOperation(
        'CO_glscf_1_2_tau',
        "CO_glscf(y,1,2,'tau')",
        CO_glscf(alpha=1, beta=2, tau='tau'))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,5,_,1
    # tags: correlation,glscf
    CO_glscf_1_5_1 = HCTSAOperation(
        'CO_glscf_1_5_1',
        'CO_glscf(y,1,5,1)',
        CO_glscf(alpha=1, beta=5, tau=1))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,5,_,2
    # tags: correlation,glscf
    CO_glscf_1_5_2 = HCTSAOperation(
        'CO_glscf_1_5_2',
        'CO_glscf(y,1,5,2)',
        CO_glscf(alpha=1, beta=5, tau=2))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,5,_,3
    # tags: correlation,glscf
    CO_glscf_1_5_3 = HCTSAOperation(
        'CO_glscf_1_5_3',
        'CO_glscf(y,1,5,3)',
        CO_glscf(alpha=1, beta=5, tau=3))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,5,_,4
    # tags: correlation,glscf
    CO_glscf_1_5_4 = HCTSAOperation(
        'CO_glscf_1_5_4',
        'CO_glscf(y,1,5,4)',
        CO_glscf(alpha=1, beta=5, tau=4))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,5,_,5
    # tags: correlation,glscf
    CO_glscf_1_5_5 = HCTSAOperation(
        'CO_glscf_1_5_5',
        'CO_glscf(y,1,5,5)',
        CO_glscf(alpha=1, beta=5, tau=5))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,5,_,t,a
    # outs: u
    # tags: correlation,glscf
    CO_glscf_1_5_tau = HCTSAOperation(
        'CO_glscf_1_5_tau',
        "CO_glscf(y,1,5,'tau')",
        CO_glscf(alpha=1, beta=5, tau='tau'))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,0,_,1
    # tags: correlation,glscf
    CO_glscf_1_10_1 = HCTSAOperation(
        'CO_glscf_1_10_1',
        'CO_glscf(y,1,10,1)',
        CO_glscf(alpha=1, beta=10, tau=1))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,0,_,2
    # tags: correlation,glscf
    CO_glscf_1_10_2 = HCTSAOperation(
        'CO_glscf_1_10_2',
        'CO_glscf(y,1,10,2)',
        CO_glscf(alpha=1, beta=10, tau=2))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,0,_,3
    # tags: correlation,glscf
    CO_glscf_1_10_3 = HCTSAOperation(
        'CO_glscf_1_10_3',
        'CO_glscf(y,1,10,3)',
        CO_glscf(alpha=1, beta=10, tau=3))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,0,_,4
    # tags: correlation,glscf
    CO_glscf_1_10_4 = HCTSAOperation(
        'CO_glscf_1_10_4',
        'CO_glscf(y,1,10,4)',
        CO_glscf(alpha=1, beta=10, tau=4))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,0,_,5
    # tags: correlation,glscf
    CO_glscf_1_10_5 = HCTSAOperation(
        'CO_glscf_1_10_5',
        'CO_glscf(y,1,10,5)',
        CO_glscf(alpha=1, beta=10, tau=5))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,1
    # outs: _,1,0,_,t
    # outs: a,u
    # tags: correlation,glscf
    CO_glscf_1_10_tau = HCTSAOperation(
        'CO_glscf_1_10_tau',
        "CO_glscf(y,1,10,'tau')",
        CO_glscf(alpha=1, beta=10, tau='tau'))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,2,_,1
    # tags: correlation,glscf
    CO_glscf_2_2_1 = HCTSAOperation(
        'CO_glscf_2_2_1',
        'CO_glscf(y,2,2,1)',
        CO_glscf(alpha=2, beta=2, tau=1))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,2,_,2
    # tags: correlation,glscf
    CO_glscf_2_2_2 = HCTSAOperation(
        'CO_glscf_2_2_2',
        'CO_glscf(y,2,2,2)',
        CO_glscf(alpha=2, beta=2, tau=2))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,2,_,3
    # tags: correlation,glscf
    CO_glscf_2_2_3 = HCTSAOperation(
        'CO_glscf_2_2_3',
        'CO_glscf(y,2,2,3)',
        CO_glscf(alpha=2, beta=2, tau=3))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,2,_,4
    # tags: correlation,glscf
    CO_glscf_2_2_4 = HCTSAOperation(
        'CO_glscf_2_2_4',
        'CO_glscf(y,2,2,4)',
        CO_glscf(alpha=2, beta=2, tau=4))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,2,_,5
    # tags: correlation,glscf
    CO_glscf_2_2_5 = HCTSAOperation(
        'CO_glscf_2_2_5',
        'CO_glscf(y,2,2,5)',
        CO_glscf(alpha=2, beta=2, tau=5))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,2,_,t,a
    # outs: u
    # tags: correlation,glscf
    CO_glscf_2_2_tau = HCTSAOperation(
        'CO_glscf_2_2_tau',
        "CO_glscf(y,2,2,'tau')",
        CO_glscf(alpha=2, beta=2, tau='tau'))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,5,_,1
    # tags: correlation,glscf
    CO_glscf_2_5_1 = HCTSAOperation(
        'CO_glscf_2_5_1',
        'CO_glscf(y,2,5,1)',
        CO_glscf(alpha=2, beta=5, tau=1))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,5,_,2
    # tags: correlation,glscf
    CO_glscf_2_5_2 = HCTSAOperation(
        'CO_glscf_2_5_2',
        'CO_glscf(y,2,5,2)',
        CO_glscf(alpha=2, beta=5, tau=2))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,5,_,3
    # tags: correlation,glscf
    CO_glscf_2_5_3 = HCTSAOperation(
        'CO_glscf_2_5_3',
        'CO_glscf(y,2,5,3)',
        CO_glscf(alpha=2, beta=5, tau=3))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,5,_,4
    # tags: correlation,glscf
    CO_glscf_2_5_4 = HCTSAOperation(
        'CO_glscf_2_5_4',
        'CO_glscf(y,2,5,4)',
        CO_glscf(alpha=2, beta=5, tau=4))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,5,_,5
    # tags: correlation,glscf
    CO_glscf_2_5_5 = HCTSAOperation(
        'CO_glscf_2_5_5',
        'CO_glscf(y,2,5,5)',
        CO_glscf(alpha=2, beta=5, tau=5))

    # outs: C,O,_,g,l
    # outs: s,c,f,_,2
    # outs: _,5,_,t,a
    # outs: u
    # tags: correlation,glscf
    CO_glscf_2_5_tau = HCTSAOperation(
        'CO_glscf_2_5_tau',
        "CO_glscf(y,2,5,'tau')",
        CO_glscf(alpha=2, beta=5, tau='tau'))

    # outs: C,O,_,t,c
    # outs: 3,_,1
    # tags: autocorrelation,correlation,nonlinear
    CO_tc3_1 = HCTSAOperation(
        'CO_tc3_1',
        'CO_tc3(y,1)',
        CO_tc3(tau=1))

    # outs: C,O,_,t,c
    # outs: 3,_,2
    # tags: autocorrelation,correlation,nonlinear
    CO_tc3_2 = HCTSAOperation(
        'CO_tc3_2',
        'CO_tc3(y,2)',
        CO_tc3(tau=2))

    # outs: C,O,_,t,c
    # outs: 3,_,3
    # tags: autocorrelation,correlation,nonlinear
    CO_tc3_3 = HCTSAOperation(
        'CO_tc3_3',
        'CO_tc3(y,3)',
        CO_tc3(tau=3))

    # outs: C,O,_,t,c
    # outs: 3,_,a,c
    # tags: autocorrelation,correlation,nonlinear
    CO_tc3_ac = HCTSAOperation(
        'CO_tc3_ac',
        "CO_tc3(y,'ac')",
        CO_tc3(tau='ac'))

    # outs: C,O,_,t,c
    # outs: 3,_,m,i
    # tags: autocorrelation,correlation,nonlinear
    CO_tc3_mi = HCTSAOperation(
        'CO_tc3_mi',
        "CO_tc3(y,'mi')",
        CO_tc3(tau='mi'))

    # outs: C,O,_,t,r
    # outs: e,v,_,1
    # tags: autocorrelation,correlation,nonlinear
    CO_trev_1 = HCTSAOperation(
        'CO_trev_1',
        'CO_trev(y,1)',
        CO_trev(tau=1))

    # outs: C,O,_,t,r
    # outs: e,v,_,2
    # tags: autocorrelation,correlation,nonlinear
    CO_trev_2 = HCTSAOperation(
        'CO_trev_2',
        'CO_trev(y,2)',
        CO_trev(tau=2))

    # outs: C,O,_,t,r
    # outs: e,v,_,3
    # tags: autocorrelation,correlation,nonlinear
    CO_trev_3 = HCTSAOperation(
        'CO_trev_3',
        'CO_trev(y,3)',
        CO_trev(tau=3))

    # outs: C,O,_,t,r
    # outs: e,v,_,a,c
    # tags: autocorrelation,correlation,nonlinear
    CO_trev_ac = HCTSAOperation(
        'CO_trev_ac',
        "CO_trev(y,'ac')",
        CO_trev(tau='ac'))

    # outs: C,O,_,t,r
    # outs: e,v,_,m,i
    # tags: autocorrelation,correlation,nonlinear
    CO_trev_mi = HCTSAOperation(
        'CO_trev_mi',
        "CO_trev(y,'mi')",
        CO_trev(tau='mi'))

    # outs: C,P,_,M,L
    # outs: _,S,t,e,p
    # outs: D,e,t,e,c
    # outs: t,_,l,1,p
    # outs: w,c,_,1,0
    # tags: l1pwc,stepdetection
    CP_ML_StepDetect_l1pwc_10 = HCTSAOperation(
        'CP_ML_StepDetect_l1pwc_10',
        "CP_ML_StepDetect(y,'l1pwc',10)",
        CP_ML_StepDetect(method='l1pwc', params=10))

    # outs: C,P,_,M,L
    # outs: _,S,t,e,p
    # outs: D,e,t,e,c
    # outs: t,_,l,1,p
    # outs: w,c,_,0,0
    # outs: 5
    # tags: l1pwc,lengthdep,stepdetection
    CP_ML_StepDetect_l1pwc_005 = HCTSAOperation(
        'CP_ML_StepDetect_l1pwc_005',
        "CP_ML_StepDetect(y,'l1pwc',0.05)",
        CP_ML_StepDetect(method='l1pwc', params=0.05))

    # outs: C,P,_,M,L
    # outs: _,S,t,e,p
    # outs: D,e,t,e,c
    # outs: t,_,l,1,p
    # outs: w,c,_,0,2
    # tags: l1pwc,lengthdep,stepdetection
    CP_ML_StepDetect_l1pwc_02 = HCTSAOperation(
        'CP_ML_StepDetect_l1pwc_02',
        "CP_ML_StepDetect(y,'l1pwc',0.2)",
        CP_ML_StepDetect(method='l1pwc', params=0.2))

    # outs: C,P,_,l,1
    # outs: p,w,c,_,s
    # outs: w,e,e,p,_
    # outs: l,a,m,b,d
    # outs: a,_,0,_,0
    # outs: 0,5,_,0,9
    # outs: 5
    # tags: l1pwc,stepdetection
    CP_l1pwc_sweep_lambda_0_005_095 = HCTSAOperation(
        'CP_l1pwc_sweep_lambda_0_005_095',
        'CP_l1pwc_sweep_lambda(y,0:0.05:0.95)',
        CP_l1pwc_sweep_lambda(lambdar=MatlabSequence('0:0.05:0.95')))

    # outs: C,P,_,w,a
    # outs: v,e,l,e,t
    # outs: _,v,a,r,c
    # outs: h,g,_,d,b
    # outs: 3,_,2,_,1
    # outs: 0,_,0,0,1
    # tags: varchg,wavelet,waveletTB
    CP_wavelet_varchg_db3_2_10_001 = HCTSAOperation(
        'CP_wavelet_varchg_db3_2_10_001',
        "CP_wavelet_varchg(y,'db3',2,10,0.01)",
        CP_wavelet_varchg(wname='db3', level=2, maxnchpts=10, minDelay=0.01))

    # outs: C,P,_,w,a
    # outs: v,e,l,e,t
    # outs: _,v,a,r,c
    # outs: h,g,_,d,b
    # outs: 3,_,3,_,1
    # outs: 0,_,0,0,1
    # tags: varchg,wavelet,waveletTB
    CP_wavelet_varchg_db3_3_10_001 = HCTSAOperation(
        'CP_wavelet_varchg_db3_3_10_001',
        "CP_wavelet_varchg(y,'db3',3,10,0.01)",
        CP_wavelet_varchg(wname='db3', level=3, maxnchpts=10, minDelay=0.01))

    # outs: C,P,_,w,a
    # outs: v,e,l,e,t
    # outs: _,v,a,r,c
    # outs: h,g,_,d,b
    # outs: 3,_,4,_,1
    # outs: 0,_,0,0,1
    # tags: varchg,wavelet,waveletTB
    CP_wavelet_varchg_db3_4_10_001 = HCTSAOperation(
        'CP_wavelet_varchg_db3_4_10_001',
        "CP_wavelet_varchg(y,'db3',4,10,0.01)",
        CP_wavelet_varchg(wname='db3', level=4, maxnchpts=10, minDelay=0.01))

    # outs: C,P,_,w,a
    # outs: v,e,l,e,t
    # outs: _,v,a,r,c
    # outs: h,g,_,d,b
    # outs: 3,_,5,_,1
    # outs: 0,_,0,0,1
    # tags: varchg,wavelet,waveletTB
    CP_wavelet_varchg_db3_5_10_001 = HCTSAOperation(
        'CP_wavelet_varchg_db3_5_10_001',
        "CP_wavelet_varchg(y,'db3',5,10,0.01)",
        CP_wavelet_varchg(wname='db3', level=5, maxnchpts=10, minDelay=0.01))

    # outs: C,P,_,w,a
    # outs: v,e,l,e,t
    # outs: _,v,a,r,c
    # outs: h,g,_,s,y
    # outs: m,2,_,2,_
    # outs: 1,0,_,0,0
    # outs: 1
    # tags: varchg,wavelet,waveletTB
    CP_wavelet_varchg_sym2_2_10_001 = HCTSAOperation(
        'CP_wavelet_varchg_sym2_2_10_001',
        "CP_wavelet_varchg(y,'sym2',2,10,0.01)",
        CP_wavelet_varchg(wname='sym2', level=2, maxnchpts=10, minDelay=0.01))

    # outs: C,P,_,w,a
    # outs: v,e,l,e,t
    # outs: _,v,a,r,c
    # outs: h,g,_,s,y
    # outs: m,2,_,3,_
    # outs: 1,0,_,0,0
    # outs: 1
    # tags: varchg,wavelet,waveletTB
    CP_wavelet_varchg_sym2_3_10_001 = HCTSAOperation(
        'CP_wavelet_varchg_sym2_3_10_001',
        "CP_wavelet_varchg(y,'sym2',3,10,0.01)",
        CP_wavelet_varchg(wname='sym2', level=3, maxnchpts=10, minDelay=0.01))

    # outs: C,P,_,w,a
    # outs: v,e,l,e,t
    # outs: _,v,a,r,c
    # outs: h,g,_,s,y
    # outs: m,2,_,4,_
    # outs: 1,0,_,0,0
    # outs: 1
    # tags: varchg,wavelet,waveletTB
    CP_wavelet_varchg_sym2_4_10_001 = HCTSAOperation(
        'CP_wavelet_varchg_sym2_4_10_001',
        "CP_wavelet_varchg(y,'sym2',4,10,0.01)",
        CP_wavelet_varchg(wname='sym2', level=4, maxnchpts=10, minDelay=0.01))

    # outs: C,P,_,w,a
    # outs: v,e,l,e,t
    # outs: _,v,a,r,c
    # outs: h,g,_,s,y
    # outs: m,2,_,5,_
    # outs: 1,0,_,0,0
    # outs: 1
    # tags: varchg,wavelet,waveletTB
    CP_wavelet_varchg_sym2_5_10_001 = HCTSAOperation(
        'CP_wavelet_varchg_sym2_5_10_001',
        "CP_wavelet_varchg(y,'sym2',5,10,0.01)",
        CP_wavelet_varchg(wname='sym2', level=5, maxnchpts=10, minDelay=0.01))

    # outs: D,N,_,b,u
    # outs: r,s,t,i,n
    # outs: e,s,s
    # tags: distribution,locdep,raw
    DN_burstiness = HCTSAOperation(
        'DN_burstiness',
        'DN_Burstiness(x)',
        DN_Burstiness())

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,n,o,r,m
    # tags: distribution,ksdensity,norm,raw
    DN_CompareKSFit_norm = HCTSAOperation(
        'DN_CompareKSFit_norm',
        "DN_CompareKSFit(x,'norm')",
        DN_CompareKSFit(whatDistn='norm'))

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,e,v
    # tags: distribution,ev,ksdensity,locdep,raw
    DN_CompareKSFit_ev = HCTSAOperation(
        'DN_CompareKSFit_ev',
        "DN_CompareKSFit(x,'ev')",
        DN_CompareKSFit(whatDistn='ev'))

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,u,n,i
    # tags: distribution,ksdensity,locdep,raw,uni
    DN_CompareKSFit_uni = HCTSAOperation(
        'DN_CompareKSFit_uni',
        "DN_CompareKSFit(x,'uni')",
        DN_CompareKSFit(whatDistn='uni'))

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,b,e,t,a
    # tags: beta,distribution,ksdensity,raw
    DN_CompareKSFit_beta = HCTSAOperation(
        'DN_CompareKSFit_beta',
        "DN_CompareKSFit(x,'beta')",
        DN_CompareKSFit(whatDistn='beta'))

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,r,a,y,l
    # outs: e,i,g,h
    # tags: distribution,ksdensity,locdep,posOnly,raw,rayleigh
    DN_CompareKSFit_rayleigh = HCTSAOperation(
        'DN_CompareKSFit_rayleigh',
        "DN_CompareKSFit(x,'rayleigh')",
        DN_CompareKSFit(whatDistn='rayleigh'))

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,e,x,p
    # tags: distribution,exp,ksdensity,locdep,posOnly,raw,spreaddep
    DN_CompareKSFit_exp = HCTSAOperation(
        'DN_CompareKSFit_exp',
        "DN_CompareKSFit(x,'exp')",
        DN_CompareKSFit(whatDistn='exp'))

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,g,a,m,m
    # outs: a
    # tags: distribution,gamma,ksdensity,locdep,posOnly,raw
    DN_CompareKSFit_gamma = HCTSAOperation(
        'DN_CompareKSFit_gamma',
        "DN_CompareKSFit(x,'gamma')",
        DN_CompareKSFit(whatDistn='gamma'))

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,l,o,g,n
    # tags: distribution,ksdensity,locdep,lognormal,posOnly,raw
    DN_CompareKSFit_logn = HCTSAOperation(
        'DN_CompareKSFit_logn',
        "DN_CompareKSFit(x,'logn')",
        DN_CompareKSFit(whatDistn='logn'))

    # outs: D,N,_,C,o
    # outs: m,p,a,r,e
    # outs: K,S,F,i,t
    # outs: _,w,b,l
    # tags: distribution,ksdensity,locdep,posOnly,raw,weibull
    DN_CompareKSFit_wbl = HCTSAOperation(
        'DN_CompareKSFit_wbl',
        "DN_CompareKSFit(x,'wbl')",
        DN_CompareKSFit(whatDistn='wbl'))

    # outs: D,N,_,C,u
    # outs: s,t,o,m,S
    # outs: k,e,w,n,e
    # outs: s,s,_,p,e
    # outs: a,r,s,o,n
    # tags: distribution,locdep,moment,raw,shape
    DN_CustomSkewness_pearson = HCTSAOperation(
        'DN_CustomSkewness_pearson',
        "DN_CustomSkewness(x,'pearson')",
        DN_CustomSkewness(whatSkew='pearson'))

    # outs: D,N,_,C,u
    # outs: s,t,o,m,S
    # outs: k,e,w,n,e
    # outs: s,s,_,b,o
    # outs: w,l,e,y
    # tags: distribution,moment,shape
    DN_CustomSkewness_bowley = HCTSAOperation(
        'DN_CustomSkewness_bowley',
        "DN_CustomSkewness(y,'bowley')",
        DN_CustomSkewness(whatSkew='bowley'))

    # outs: D,N,_,F,i
    # outs: t,K,e,r,n
    # outs: e,l,S,m,o
    # outs: o,t,h,r,a
    # outs: w
    # tags: arclength,areaconst,crossconst,distribution,entropy,ksdensity,raw,spreaddep,symmetry
    DN_FitKernelSmoothraw = HCTSAOperation(
        'DN_FitKernelSmoothraw',
        "DN_FitKernelSmooth(x,'numcross',[0.05,0.1,0.2,0.3,0.4,0.5],'area',[0.05,0.1,0.2,0.3,0.4,0.5],'arclength',[0.1,0.5,1,2])",
        DN_FitKernelSmooth(varargin='numcross'))

    # outs: D,N,_,F,i
    # outs: t,K,e,r,n
    # outs: e,l,S,m,o
    # outs: o,t,h,z,s
    # outs: c,o,r,e
    # tags: distribution,entropy,ksdensity
    DN_FitKernelSmoothzscore = HCTSAOperation(
        'DN_FitKernelSmoothzscore',
        'DN_FitKernelSmooth(y)',
        DN_FitKernelSmooth())

    # outs: D,N,_,F,i
    # outs: t,_,m,l,e
    # outs: _,g,e,o,m
    # outs: e,t,r,i,c
    # tags: distribution,fit
    DN_Fit_mle_geometric = HCTSAOperation(
        'DN_Fit_mle_geometric',
        "DN_Fit_mle(y,'geometric')",
        DN_Fit_mle(fitWhat='geometric'))

    # outs: D,N,_,H,i
    # outs: g,h,L,o,w
    # outs: M,u
    # tags: distribution,locdep,raw,spreaddep
    DN_HighLowMu = HCTSAOperation(
        'DN_HighLowMu',
        'DN_HighLowMu(x)',
        DN_HighLowMu())

    # outs: D,N,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,M,o,d
    # outs: e,_,5
    # tags: distribution,location
    DN_HistogramMode_5 = HCTSAOperation(
        'DN_HistogramMode_5',
        'DN_HistogramMode(y,5)',
        DN_HistogramMode(numBins=5))

    # outs: D,N,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,M,o,d
    # outs: e,_,1,0
    # tags: distribution,location
    DN_HistogramMode_10 = HCTSAOperation(
        'DN_HistogramMode_10',
        'DN_HistogramMode(y,10)',
        DN_HistogramMode(numBins=10))

    # outs: D,N,_,H,i
    # outs: s,t,o,g,r
    # outs: a,m,M,o,d
    # outs: e,_,2,0
    # tags: distribution,location
    DN_HistogramMode_20 = HCTSAOperation(
        'DN_HistogramMode_20',
        'DN_HistogramMode(y,20)',
        DN_HistogramMode(numBins=20))

    # outs: D,N,_,m,e
    # outs: a,n
    # tags: distribution,location,locdep,raw
    DN_mean = HCTSAOperation(
        'DN_mean',
        "DN_Mean(x,'norm')",
        DN_Mean(meanType='norm'))

    # outs: D,N,_,h,m
    # outs: e,a,n
    # tags: distribution,location,locdep,raw
    DN_hmean = HCTSAOperation(
        'DN_hmean',
        "DN_Mean(x,'harm')",
        DN_Mean(meanType='harm'))

    # outs: D,N,_,r,m
    # outs: s
    # tags: distribution,location,locdep,raw,spreaddep
    DN_rms = HCTSAOperation(
        'DN_rms',
        "DN_Mean(x,'rms')",
        DN_Mean(meanType='rms'))

    # outs: D,N,_,m,e
    # outs: d,i,a,n
    # tags: distribution,location,locdep,raw
    DN_median = HCTSAOperation(
        'DN_median',
        "DN_Mean(x,'median')",
        DN_Mean(meanType='median'))

    # outs: D,N,_,m,i
    # outs: d,h,i,n,g
    # outs: e
    # tags: distribution,location,locdep,raw
    DN_midhinge = HCTSAOperation(
        'DN_midhinge',
        "DN_Mean(x,'midhinge')",
        DN_Mean(meanType='midhinge'))

    # outs: D,N,_,m,a
    # outs: x
    # tags: distribution
    DN_max = HCTSAOperation(
        'DN_max',
        "DN_MinMax(y,'max')",
        DN_MinMax(minOrMax='max'))

    # outs: D,N,_,m,i
    # outs: n
    # tags: distribution
    DN_min = HCTSAOperation(
        'DN_min',
        "DN_MinMax(y,'min')",
        DN_MinMax(minOrMax='min'))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,3
    # tags: distribution,moment,shape
    DN_Moments_3 = HCTSAOperation(
        'DN_Moments_3',
        'DN_Moments(y,3)',
        DN_Moments(theMom=3))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,4
    # tags: distribution,moment,shape
    DN_Moments_4 = HCTSAOperation(
        'DN_Moments_4',
        'DN_Moments(y,4)',
        DN_Moments(theMom=4))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,5
    # tags: distribution,moment,shape
    DN_Moments_5 = HCTSAOperation(
        'DN_Moments_5',
        'DN_Moments(y,5)',
        DN_Moments(theMom=5))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,6
    # tags: distribution,moment,shape
    DN_Moments_6 = HCTSAOperation(
        'DN_Moments_6',
        'DN_Moments(y,6)',
        DN_Moments(theMom=6))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,7
    # tags: distribution,moment,shape
    DN_Moments_7 = HCTSAOperation(
        'DN_Moments_7',
        'DN_Moments(y,7)',
        DN_Moments(theMom=7))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,8
    # tags: distribution,moment,shape
    DN_Moments_8 = HCTSAOperation(
        'DN_Moments_8',
        'DN_Moments(y,8)',
        DN_Moments(theMom=8))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,9
    # tags: distribution,moment,shape
    DN_Moments_9 = HCTSAOperation(
        'DN_Moments_9',
        'DN_Moments(y,9)',
        DN_Moments(theMom=9))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,1,0
    # tags: distribution,moment,shape
    DN_Moments_10 = HCTSAOperation(
        'DN_Moments_10',
        'DN_Moments(y,10)',
        DN_Moments(theMom=10))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,1,1
    # tags: distribution,moment,shape
    DN_Moments_11 = HCTSAOperation(
        'DN_Moments_11',
        'DN_Moments(y,11)',
        DN_Moments(theMom=11))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 3
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_3 = HCTSAOperation(
        'DN_Moments_raw_3',
        'DN_Moments(x,3)',
        DN_Moments(theMom=3))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 4
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_4 = HCTSAOperation(
        'DN_Moments_raw_4',
        'DN_Moments(x,4)',
        DN_Moments(theMom=4))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 5
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_5 = HCTSAOperation(
        'DN_Moments_raw_5',
        'DN_Moments(x,5)',
        DN_Moments(theMom=5))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 6
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_6 = HCTSAOperation(
        'DN_Moments_raw_6',
        'DN_Moments(x,6)',
        DN_Moments(theMom=6))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 7
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_7 = HCTSAOperation(
        'DN_Moments_raw_7',
        'DN_Moments(x,7)',
        DN_Moments(theMom=7))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 8
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_8 = HCTSAOperation(
        'DN_Moments_raw_8',
        'DN_Moments(x,8)',
        DN_Moments(theMom=8))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 9
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_9 = HCTSAOperation(
        'DN_Moments_raw_9',
        'DN_Moments(x,9)',
        DN_Moments(theMom=9))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 1,0
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_10 = HCTSAOperation(
        'DN_Moments_raw_10',
        'DN_Moments(x,10)',
        DN_Moments(theMom=10))

    # outs: D,N,_,M,o
    # outs: m,e,n,t,s
    # outs: _,r,a,w,_
    # outs: 1,1
    # tags: distribution,moment,raw,shape,spreaddep
    DN_Moments_raw_11 = HCTSAOperation(
        'DN_Moments_raw_11',
        'DN_Moments(x,11)',
        DN_Moments(theMom=11))

    # outs: D,N,_,O,u
    # outs: t,l,i,e,r
    # outs: I,n,c,l,u
    # outs: d,e,_,a,b
    # outs: s
    # tags: distribution,outliers
    DN_OutlierInclude_abs = HCTSAOperation(
        'DN_OutlierInclude_abs',
        "DN_OutlierInclude(y,'abs')",
        DN_OutlierInclude(thresholdHow='abs'))

    # outs: D,N,_,O,u
    # outs: t,l,i,e,r
    # outs: I,n,c,l,u
    # outs: d,e,_,p
    # tags: distribution,outliers
    DN_OutlierInclude_p = HCTSAOperation(
        'DN_OutlierInclude_p',
        "DN_OutlierInclude(y,'p')",
        DN_OutlierInclude(thresholdHow='p'))

    # outs: D,N,_,O,u
    # outs: t,l,i,e,r
    # outs: I,n,c,l,u
    # outs: d,e,_,n
    # tags: distribution,outliers
    DN_OutlierInclude_n = HCTSAOperation(
        'DN_OutlierInclude_n',
        "DN_OutlierInclude(y,'n')",
        DN_OutlierInclude(thresholdHow='n'))

    # outs: D,N,_,O,u
    # outs: t,l,i,e,r
    # outs: T,e,s,t,2
    # tags: distribution,outliers,spread
    DN_OutlierTest2 = HCTSAOperation(
        'DN_OutlierTest2',
        'DN_OutlierTest(y,2)',
        DN_OutlierTest(p=2))

    # outs: D,N,_,O,u
    # outs: t,l,i,e,r
    # outs: T,e,s,t,5
    # tags: distribution,outliers,spread
    DN_OutlierTest5 = HCTSAOperation(
        'DN_OutlierTest5',
        'DN_OutlierTest(y,5)',
        DN_OutlierTest(p=5))

    # outs: D,N,_,O,u
    # outs: t,l,i,e,r
    # outs: T,e,s,t,1
    # outs: 0
    # tags: distribution,outliers,spread
    DN_OutlierTest10 = HCTSAOperation(
        'DN_OutlierTest10',
        'DN_OutlierTest(y,10)',
        DN_OutlierTest(p=10))

    # outs: D,N,_,P,r
    # outs: o,p,o,r,t
    # outs: i,o,n,V,a
    # outs: l,u,e,s,_
    # outs: z,e,r,o,s
    # tags: distribution,raw
    DN_ProportionValues_zeros = HCTSAOperation(
        'DN_ProportionValues_zeros',
        "DN_ProportionValues(x,'zeros')",
        DN_ProportionValues(propWhat='zeros'))

    # outs: D,N,_,P,r
    # outs: o,p,o,r,t
    # outs: i,o,n,V,a
    # outs: l,u,e,s,_
    # outs: g,e,q,0
    # tags: distribution,locdep,raw
    DN_ProportionValues_geq0 = HCTSAOperation(
        'DN_ProportionValues_geq0',
        "DN_ProportionValues(x,'geq0')",
        DN_ProportionValues(propWhat='geq0'))

    # outs: D,N,_,P,r
    # outs: o,p,o,r,t
    # outs: i,o,n,V,a
    # outs: l,u,e,s,_
    # outs: p,o,s,i,t
    # outs: i,v,e
    # tags: distribution,locdep,raw
    DN_ProportionValues_positive = HCTSAOperation(
        'DN_ProportionValues_positive',
        "DN_ProportionValues(x,'positive')",
        DN_ProportionValues(propWhat='positive'))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,1
    # tags: distribution
    DN_Quantile_1 = HCTSAOperation(
        'DN_Quantile_1',
        'DN_Quantile(y,0.01)',
        DN_Quantile(p=0.01))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,2
    # tags: distribution
    DN_Quantile_2 = HCTSAOperation(
        'DN_Quantile_2',
        'DN_Quantile(y,0.02)',
        DN_Quantile(p=0.02))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,3
    # tags: distribution
    DN_Quantile_3 = HCTSAOperation(
        'DN_Quantile_3',
        'DN_Quantile(y,0.03)',
        DN_Quantile(p=0.03))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,4
    # tags: distribution
    DN_Quantile_4 = HCTSAOperation(
        'DN_Quantile_4',
        'DN_Quantile(y,0.04)',
        DN_Quantile(p=0.04))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,5
    # tags: distribution
    DN_Quantile_5 = HCTSAOperation(
        'DN_Quantile_5',
        'DN_Quantile(y,0.05)',
        DN_Quantile(p=0.05))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,1,0
    # tags: distribution
    DN_Quantile_10 = HCTSAOperation(
        'DN_Quantile_10',
        'DN_Quantile(y,0.10)',
        DN_Quantile(p=0.1))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,2,0
    # tags: distribution
    DN_Quantile_20 = HCTSAOperation(
        'DN_Quantile_20',
        'DN_Quantile(y,0.20)',
        DN_Quantile(p=0.2))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,3,0
    # tags: distribution
    DN_Quantile_30 = HCTSAOperation(
        'DN_Quantile_30',
        'DN_Quantile(y,0.30)',
        DN_Quantile(p=0.3))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,4,0
    # tags: distribution
    DN_Quantile_40 = HCTSAOperation(
        'DN_Quantile_40',
        'DN_Quantile(y,0.40)',
        DN_Quantile(p=0.4))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,5,0
    # tags: distribution
    DN_Quantile_50 = HCTSAOperation(
        'DN_Quantile_50',
        'DN_Quantile(y,0.50)',
        DN_Quantile(p=0.5))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,6,0
    # tags: distribution
    DN_Quantile_60 = HCTSAOperation(
        'DN_Quantile_60',
        'DN_Quantile(y,0.60)',
        DN_Quantile(p=0.6))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,7,0
    # tags: distribution
    DN_Quantile_70 = HCTSAOperation(
        'DN_Quantile_70',
        'DN_Quantile(y,0.70)',
        DN_Quantile(p=0.7))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,8,0
    # tags: distribution
    DN_Quantile_80 = HCTSAOperation(
        'DN_Quantile_80',
        'DN_Quantile(y,0.80)',
        DN_Quantile(p=0.8))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,0
    # tags: distribution
    DN_Quantile_90 = HCTSAOperation(
        'DN_Quantile_90',
        'DN_Quantile(y,0.90)',
        DN_Quantile(p=0.9))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,1
    # tags: distribution
    DN_Quantile_91 = HCTSAOperation(
        'DN_Quantile_91',
        'DN_Quantile(y,0.91)',
        DN_Quantile(p=0.91))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,2
    # tags: distribution
    DN_Quantile_92 = HCTSAOperation(
        'DN_Quantile_92',
        'DN_Quantile(y,0.92)',
        DN_Quantile(p=0.92))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,3
    # tags: distribution
    DN_Quantile_93 = HCTSAOperation(
        'DN_Quantile_93',
        'DN_Quantile(y,0.93)',
        DN_Quantile(p=0.93))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,4
    # tags: distribution
    DN_Quantile_94 = HCTSAOperation(
        'DN_Quantile_94',
        'DN_Quantile(y,0.94)',
        DN_Quantile(p=0.94))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,5
    # tags: distribution
    DN_Quantile_95 = HCTSAOperation(
        'DN_Quantile_95',
        'DN_Quantile(y,0.95)',
        DN_Quantile(p=0.95))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,6
    # tags: distribution
    DN_Quantile_96 = HCTSAOperation(
        'DN_Quantile_96',
        'DN_Quantile(y,0.96)',
        DN_Quantile(p=0.96))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,7
    # tags: distribution
    DN_Quantile_97 = HCTSAOperation(
        'DN_Quantile_97',
        'DN_Quantile(y,0.97)',
        DN_Quantile(p=0.97))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,8
    # tags: distribution
    DN_Quantile_98 = HCTSAOperation(
        'DN_Quantile_98',
        'DN_Quantile(y,0.98)',
        DN_Quantile(p=0.98))

    # outs: D,N,_,Q,u
    # outs: a,n,t,i,l
    # outs: e,_,9,9
    # tags: distribution
    DN_Quantile_99 = HCTSAOperation(
        'DN_Quantile_99',
        'DN_Quantile(y,0.99)',
        DN_Quantile(p=0.99))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,a,b,s,c
    # outs: l,o,s,e,_
    # outs: 0,1
    # tags: correlation,distribution,outliers
    DN_RemovePoints_absclose_01 = HCTSAOperation(
        'DN_RemovePoints_absclose_01',
        "DN_RemovePoints(y,'absclose',0.1)",
        DN_RemovePoints(removeHow='absclose', p=0.1))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,a,b,s,c
    # outs: l,o,s,e,_
    # outs: 0,5
    # tags: correlation,distribution,outliers
    DN_RemovePoints_absclose_05 = HCTSAOperation(
        'DN_RemovePoints_absclose_05',
        "DN_RemovePoints(y,'absclose',0.5)",
        DN_RemovePoints(removeHow='absclose', p=0.5))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,a,b,s,c
    # outs: l,o,s,e,_
    # outs: 0,8
    # tags: correlation,distribution,outliers
    DN_RemovePoints_absclose_08 = HCTSAOperation(
        'DN_RemovePoints_absclose_08',
        "DN_RemovePoints(y,'absclose',0.8)",
        DN_RemovePoints(removeHow='absclose', p=0.8))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,a,b,s,f
    # outs: a,r,_,0,1
    # tags: correlation,distribution,outliers
    DN_RemovePoints_absfar_01 = HCTSAOperation(
        'DN_RemovePoints_absfar_01',
        "DN_RemovePoints(y,'absfar',0.1)",
        DN_RemovePoints(removeHow='absfar', p=0.1))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,a,b,s,f
    # outs: a,r,_,0,8
    # tags: correlation,distribution,outliers
    DN_RemovePoints_absfar_08 = HCTSAOperation(
        'DN_RemovePoints_absfar_08',
        "DN_RemovePoints(y,'absfar',0.8)",
        DN_RemovePoints(removeHow='absfar', p=0.8))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,m,i,n,_
    # outs: 0,1
    # tags: correlation,distribution,outliers
    DN_RemovePoints_min_01 = HCTSAOperation(
        'DN_RemovePoints_min_01',
        "DN_RemovePoints(y,'min',0.1)",
        DN_RemovePoints(removeHow='min', p=0.1))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,m,i,n,_
    # outs: 0,5
    # tags: correlation,distribution,outliers
    DN_RemovePoints_min_05 = HCTSAOperation(
        'DN_RemovePoints_min_05',
        "DN_RemovePoints(y,'min',0.5)",
        DN_RemovePoints(removeHow='min', p=0.5))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,m,i,n,_
    # outs: 0,8
    # tags: correlation,distribution,outliers
    DN_RemovePoints_min_08 = HCTSAOperation(
        'DN_RemovePoints_min_08',
        "DN_RemovePoints(y,'min',0.8)",
        DN_RemovePoints(removeHow='min', p=0.8))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,m,a,x,_
    # outs: 0,1
    # tags: correlation,distribution,outliers
    DN_RemovePoints_max_01 = HCTSAOperation(
        'DN_RemovePoints_max_01',
        "DN_RemovePoints(y,'max',0.1)",
        DN_RemovePoints(removeHow='max', p=0.1))

    # outs: D,N,_,R,e
    # outs: m,o,v,e,P
    # outs: o,i,n,t,s
    # outs: _,m,a,x,_
    # outs: 0,8
    # tags: correlation,distribution,outliers
    DN_RemovePoints_max_08 = HCTSAOperation(
        'DN_RemovePoints_max_08',
        "DN_RemovePoints(y,'max',0.8)",
        DN_RemovePoints(removeHow='max', p=0.8))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,g,a
    # outs: u,s,s,1,_
    # outs: k,s
    # tags: distribution,gauss1,gof
    DN_SimpleFit_gauss1_ks = HCTSAOperation(
        'DN_SimpleFit_gauss1_ks',
        "DN_SimpleFit(y,'gauss1',0)",
        DN_SimpleFit(dmodel='gauss1', numBins=0))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,g,a
    # outs: u,s,s,1,_
    # outs: h,s,q,r,t
    # tags: distribution,gauss1,gof
    DN_SimpleFit_gauss1_hsqrt = HCTSAOperation(
        'DN_SimpleFit_gauss1_hsqrt',
        "DN_SimpleFit(y,'gauss1','sqrt')",
        DN_SimpleFit(dmodel='gauss1', numBins='sqrt'))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,g,a
    # outs: u,s,s,2,_
    # outs: k,s
    # tags: distribution,gauss2,gof
    DN_SimpleFit_gauss2_ks = HCTSAOperation(
        'DN_SimpleFit_gauss2_ks',
        "DN_SimpleFit(y,'gauss2',0)",
        DN_SimpleFit(dmodel='gauss2', numBins=0))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,g,a
    # outs: u,s,s,2,_
    # outs: h,s,q,r,t
    # tags: distribution,gauss2,gof
    DN_SimpleFit_gauss2_hsqrt = HCTSAOperation(
        'DN_SimpleFit_gauss2_hsqrt',
        "DN_SimpleFit(y,'gauss2','sqrt')",
        DN_SimpleFit(dmodel='gauss2', numBins='sqrt'))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,e,x
    # outs: p,1,_,k,s
    # tags: distribution,exp1,gof
    DN_SimpleFit_exp1_ks = HCTSAOperation(
        'DN_SimpleFit_exp1_ks',
        "DN_SimpleFit(y,'exp1',0)",
        DN_SimpleFit(dmodel='exp1', numBins=0))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,e,x
    # outs: p,1,_,h,s
    # outs: q,r,t
    # tags: distribution,exp1,gof
    DN_SimpleFit_exp1_hsqrt = HCTSAOperation(
        'DN_SimpleFit_exp1_hsqrt',
        "DN_SimpleFit(y,'exp1','sqrt')",
        DN_SimpleFit(dmodel='exp1', numBins='sqrt'))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,p,o
    # outs: w,e,r,1,_
    # outs: h,s,q,r,t
    # tags: distribution,gof,posOnly,power1,raw
    DN_SimpleFit_power1_hsqrt = HCTSAOperation(
        'DN_SimpleFit_power1_hsqrt',
        "DN_SimpleFit(x,'power1','sqrt')",
        DN_SimpleFit(dmodel='power1', numBins='sqrt'))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,s,i
    # outs: n,1
    # tags: gof,model,sin1
    DN_SimpleFit_sin1 = HCTSAOperation(
        'DN_SimpleFit_sin1',
        "DN_SimpleFit(y,'sin1')",
        DN_SimpleFit(dmodel='sin1'))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,s,i
    # outs: n,2
    # tags: gof,model,sin2
    DN_SimpleFit_sin2 = HCTSAOperation(
        'DN_SimpleFit_sin2',
        "DN_SimpleFit(y,'sin2')",
        DN_SimpleFit(dmodel='sin2'))

    # outs: D,N,_,S,i
    # outs: m,p,l,e,F
    # outs: i,t,_,s,i
    # outs: n,3
    # tags: gof,model,sin3
    DN_SimpleFit_sin3 = HCTSAOperation(
        'DN_SimpleFit_sin3',
        "DN_SimpleFit(y,'sin3')",
        DN_SimpleFit(dmodel='sin3'))

    # outs: D,N,_,S,p
    # outs: r,e,a,d,_
    # outs: s,t,d
    # tags: distribution,raw,spread,spreaddep
    DN_Spread_std = HCTSAOperation(
        'DN_Spread_std',
        "DN_Spread(x,'std')",
        DN_Spread(spreadMeasure='std'))

    # outs: D,N,_,S,p
    # outs: r,e,a,d,_
    # outs: m,a,d
    # tags: distribution,raw,spread,spreaddep
    DN_Spread_mad = HCTSAOperation(
        'DN_Spread_mad',
        "DN_Spread(x,'mad')",
        DN_Spread(spreadMeasure='mad'))

    # outs: D,N,_,S,p
    # outs: r,e,a,d,_
    # outs: i,q,r
    # tags: distribution,raw,spread,spreaddep
    DN_Spread_iqr = HCTSAOperation(
        'DN_Spread_iqr',
        "DN_Spread(x,'iqr')",
        DN_Spread(spreadMeasure='iqr'))

    # outs: D,N,_,S,p
    # outs: r,e,a,d,_
    # outs: m,e,a,d
    # tags: distribution,raw,spread,spreaddep
    DN_Spread_mead = HCTSAOperation(
        'DN_Spread_mead',
        "DN_Spread(x,'mead')",
        DN_Spread(spreadMeasure='mead'))

    # outs: D,N,_,T,r
    # outs: i,m,m,e,d
    # outs: M,e,a,n,_
    # outs: 1
    # tags: distribution,location,locdep,raw
    DN_TrimmedMean_1 = HCTSAOperation(
        'DN_TrimmedMean_1',
        'DN_TrimmedMean(x,1)',
        DN_TrimmedMean(n=1))

    # outs: D,N,_,T,r
    # outs: i,m,m,e,d
    # outs: M,e,a,n,_
    # outs: 5
    # tags: distribution,location,locdep,raw
    DN_TrimmedMean_5 = HCTSAOperation(
        'DN_TrimmedMean_5',
        'DN_TrimmedMean(x,5)',
        DN_TrimmedMean(n=5))

    # outs: D,N,_,T,r
    # outs: i,m,m,e,d
    # outs: M,e,a,n,_
    # outs: 1,0
    # tags: distribution,location,locdep,raw
    DN_TrimmedMean_10 = HCTSAOperation(
        'DN_TrimmedMean_10',
        'DN_TrimmedMean(x,10)',
        DN_TrimmedMean(n=10))

    # outs: D,N,_,T,r
    # outs: i,m,m,e,d
    # outs: M,e,a,n,_
    # outs: 2,5
    # tags: distribution,location,locdep,raw
    DN_TrimmedMean_25 = HCTSAOperation(
        'DN_TrimmedMean_25',
        'DN_TrimmedMean(x,25)',
        DN_TrimmedMean(n=25))

    # outs: D,N,_,T,r
    # outs: i,m,m,e,d
    # outs: M,e,a,n,_
    # outs: 5,0
    # tags: distribution,location,locdep,raw
    DN_TrimmedMean_50 = HCTSAOperation(
        'DN_TrimmedMean_50',
        'DN_TrimmedMean(x,50)',
        DN_TrimmedMean(n=50))

    # outs: D,N,_,W,i
    # outs: t,h,i,n,p
    # outs: _,0,5
    # tags: distribution,spread
    DN_Withinp_05 = HCTSAOperation(
        'DN_Withinp_05',
        'DN_Withinp(y,0.5)',
        DN_Withinp(p=0.5))

    # outs: D,N,_,W,i
    # outs: t,h,i,n,p
    # outs: _,1,0
    # tags: distribution,spread
    DN_Withinp_10 = HCTSAOperation(
        'DN_Withinp_10',
        'DN_Withinp(y,1)',
        DN_Withinp(p=1))

    # outs: D,N,_,W,i
    # outs: t,h,i,n,p
    # outs: _,1,5
    # tags: distribution,spread
    DN_Withinp_15 = HCTSAOperation(
        'DN_Withinp_15',
        'DN_Withinp(y,1.5)',
        DN_Withinp(p=1.5))

    # outs: D,N,_,W,i
    # outs: t,h,i,n,p
    # outs: _,2,0
    # tags: distribution,spread
    DN_Withinp_20 = HCTSAOperation(
        'DN_Withinp_20',
        'DN_Withinp(y,2)',
        DN_Withinp(p=2))

    # outs: D,N,_,W,i
    # outs: t,h,i,n,p
    # outs: _,2,5
    # tags: distribution,spread
    DN_Withinp_25 = HCTSAOperation(
        'DN_Withinp_25',
        'DN_Withinp(y,2.5)',
        DN_Withinp(p=2.5))

    # outs: D,N,_,W,i
    # outs: t,h,i,n,p
    # outs: _,3,0
    # tags: distribution,spread
    DN_Withinp_30 = HCTSAOperation(
        'DN_Withinp_30',
        'DN_Withinp(y,3)',
        DN_Withinp(p=3))

    # outs: D,N,_,c,v
    # outs: _,1
    # tags: cv,distribution,locdep,raw,spread,spreaddep
    DN_cv_1 = HCTSAOperation(
        'DN_cv_1',
        'DN_cv(x,1)',
        DN_cv(k=1))

    # outs: D,N,_,c,v
    # outs: _,2
    # tags: cv,distribution,locdep,raw,spread,spreaddep
    DN_cv_2 = HCTSAOperation(
        'DN_cv_2',
        'DN_cv(x,2)',
        DN_cv(k=2))

    # outs: D,N,_,p,l
    # outs: e,f,t,_,0
    # outs: 0,5
    # tags: distribution,spread
    DN_pleft_005 = HCTSAOperation(
        'DN_pleft_005',
        'DN_pleft(y,0.05)',
        DN_pleft(th=0.05))

    # outs: D,N,_,p,l
    # outs: e,f,t,_,0
    # outs: 1
    # tags: distribution,spread
    DN_pleft_01 = HCTSAOperation(
        'DN_pleft_01',
        'DN_pleft(y,0.1)',
        DN_pleft(th=0.1))

    # outs: D,N,_,p,l
    # outs: e,f,t,_,0
    # outs: 2
    # tags: distribution,spread
    DN_pleft_02 = HCTSAOperation(
        'DN_pleft_02',
        'DN_pleft(y,0.2)',
        DN_pleft(th=0.2))

    # outs: D,N,_,p,l
    # outs: e,f,t,_,0
    # outs: 3
    # tags: distribution,spread
    DN_pleft_03 = HCTSAOperation(
        'DN_pleft_03',
        'DN_pleft(y,0.3)',
        DN_pleft(th=0.3))

    # outs: D,N,_,p,l
    # outs: e,f,t,_,0
    # outs: 4
    # tags: distribution,spread
    DN_pleft_04 = HCTSAOperation(
        'DN_pleft_04',
        'DN_pleft(y,0.4)',
        DN_pleft(th=0.4))

    # outs: D,N,_,p,l
    # outs: e,f,t,_,0
    # outs: 5
    # tags: distribution,spread
    DN_pleft_05 = HCTSAOperation(
        'DN_pleft_05',
        'DN_pleft(y,0.5)',
        DN_pleft(th=0.5))

    # outs: D,T,_,I,s
    # outs: S,e,a,s,o
    # outs: n,a,l
    # tags: periodicity
    DT_IsSeasonal = HCTSAOperation(
        'DT_IsSeasonal',
        'DT_IsSeasonal(y)',
        DT_IsSeasonal())

    # outs: A,p,E,n,1
    # outs: _,0,1
    # tags: entropy
    ApEn1_01 = HCTSAOperation(
        'ApEn1_01',
        'EN_ApEn(y,1,0.1)',
        EN_ApEn(mnom=1, rth=0.1))

    # outs: A,p,E,n,1
    # outs: _,0,2
    # tags: entropy
    ApEn1_02 = HCTSAOperation(
        'ApEn1_02',
        'EN_ApEn(y,1,0.2)',
        EN_ApEn(mnom=1, rth=0.2))

    # outs: A,p,E,n,2
    # outs: _,0,1
    # tags: entropy
    ApEn2_01 = HCTSAOperation(
        'ApEn2_01',
        'EN_ApEn(y,2,0.1)',
        EN_ApEn(mnom=2, rth=0.1))

    # outs: A,p,E,n,2
    # outs: _,0,2
    # tags: entropy
    ApEn2_02 = HCTSAOperation(
        'ApEn2_02',
        'EN_ApEn(y,2,0.2)',
        EN_ApEn(mnom=2, rth=0.2))

    # outs: E,N,_,C,I
    # outs: D
    # tags: complexity,distribution,entropy
    EN_CID = HCTSAOperation(
        'EN_CID',
        'EN_CID(y)',
        EN_CID())

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,0,0,1,_
    # outs: 0
    # tags: entropy
    EN_DistributionEntropy_ks_001_0 = HCTSAOperation(
        'EN_DistributionEntropy_ks_001_0',
        "EN_DistributionEntropy(y,'ks',0.01,0)",
        EN_DistributionEntropy(histOrKS='ks', numBins=0.01, olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,0,1,_,0
    # tags: entropy
    EN_DistributionEntropy_ks_01_0 = HCTSAOperation(
        'EN_DistributionEntropy_ks_01_0',
        "EN_DistributionEntropy(y,'ks',0.1,0)",
        EN_DistributionEntropy(histOrKS='ks', numBins=0.1, olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,0,5,_,0
    # tags: entropy
    EN_DistributionEntropy_ks_05_0 = HCTSAOperation(
        'EN_DistributionEntropy_ks_05_0',
        "EN_DistributionEntropy(y,'ks',0.5,0)",
        EN_DistributionEntropy(histOrKS='ks', numBins=0.5, olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,1,_,0
    # tags: entropy
    EN_DistributionEntropy_ks_1_0 = HCTSAOperation(
        'EN_DistributionEntropy_ks_1_0',
        "EN_DistributionEntropy(y,'ks',1,0)",
        EN_DistributionEntropy(histOrKS='ks', numBins=1, olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,_,0,0,1
    # tags: entropy
    EN_DistributionEntropy_ks__001 = HCTSAOperation(
        'EN_DistributionEntropy_ks__001',
        "EN_DistributionEntropy(y,'ks',[],0.01)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.01))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,_,0,0,2
    # tags: entropy
    EN_DistributionEntropy_ks__002 = HCTSAOperation(
        'EN_DistributionEntropy_ks__002',
        "EN_DistributionEntropy(y,'ks',[],0.02)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.02))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,_,0,0,5
    # tags: entropy
    EN_DistributionEntropy_ks__005 = HCTSAOperation(
        'EN_DistributionEntropy_ks__005',
        "EN_DistributionEntropy(y,'ks',[],0.05)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.05))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,_,0,1
    # tags: entropy
    EN_DistributionEntropy_ks__01 = HCTSAOperation(
        'EN_DistributionEntropy_ks__01',
        "EN_DistributionEntropy(y,'ks',[],0.1)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.1))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,_,0,2
    # tags: entropy
    EN_DistributionEntropy_ks__02 = HCTSAOperation(
        'EN_DistributionEntropy_ks__02',
        "EN_DistributionEntropy(y,'ks',[],0.2)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.2))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,k,s
    # outs: _,_,0,3
    # tags: entropy
    EN_DistributionEntropy_ks__03 = HCTSAOperation(
        'EN_DistributionEntropy_ks__03',
        "EN_DistributionEntropy(y,'ks',[],0.3)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.3))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,r,a
    # outs: w,_,k,s,_
    # outs: _,0,0,1
    # tags: entropy,raw,spreaddep
    EN_DistributionEntropy_raw_ks__001 = HCTSAOperation(
        'EN_DistributionEntropy_raw_ks__001',
        "EN_DistributionEntropy(x,'ks',[],0.01)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.01))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,r,a
    # outs: w,_,k,s,_
    # outs: _,0,0,2
    # tags: entropy,raw,spreaddep
    EN_DistributionEntropy_raw_ks__002 = HCTSAOperation(
        'EN_DistributionEntropy_raw_ks__002',
        "EN_DistributionEntropy(x,'ks',[],0.02)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.02))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,r,a
    # outs: w,_,k,s,_
    # outs: _,0,0,5
    # tags: entropy,raw,spreaddep
    EN_DistributionEntropy_raw_ks__005 = HCTSAOperation(
        'EN_DistributionEntropy_raw_ks__005',
        "EN_DistributionEntropy(x,'ks',[],0.05)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.05))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,r,a
    # outs: w,_,k,s,_
    # outs: _,0,1
    # tags: entropy,raw,spreaddep
    EN_DistributionEntropy_raw_ks__01 = HCTSAOperation(
        'EN_DistributionEntropy_raw_ks__01',
        "EN_DistributionEntropy(x,'ks',[],0.1)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.1))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,r,a
    # outs: w,_,k,s,_
    # outs: _,0,2
    # tags: entropy,raw,spreaddep
    EN_DistributionEntropy_raw_ks__02 = HCTSAOperation(
        'EN_DistributionEntropy_raw_ks__02',
        "EN_DistributionEntropy(x,'ks',[],0.2)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.2))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,r,a
    # outs: w,_,k,s,_
    # outs: _,0,3
    # tags: entropy,raw,spreaddep
    EN_DistributionEntropy_raw_ks__03 = HCTSAOperation(
        'EN_DistributionEntropy_raw_ks__03',
        "EN_DistributionEntropy(x,'ks',[],0.3)",
        EN_DistributionEntropy(histOrKS='ks', numBins=array([], dtype=float64), olremp=0.3))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,5,_
    # outs: 0
    # tags: entropy
    EN_DistributionEntropy_hist_5_0 = HCTSAOperation(
        'EN_DistributionEntropy_hist_5_0',
        "EN_DistributionEntropy(y,'hist',5,0)",
        EN_DistributionEntropy(histOrKS='hist', numBins=5, olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,1,0
    # outs: _,0
    # tags: entropy
    EN_DistributionEntropy_hist_10_0 = HCTSAOperation(
        'EN_DistributionEntropy_hist_10_0',
        "EN_DistributionEntropy(y,'hist',10,0)",
        EN_DistributionEntropy(histOrKS='hist', numBins=10, olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,2,0
    # outs: _,0
    # tags: entropy
    EN_DistributionEntropy_hist_20_0 = HCTSAOperation(
        'EN_DistributionEntropy_hist_20_0',
        "EN_DistributionEntropy(y,'hist',20,0)",
        EN_DistributionEntropy(histOrKS='hist', numBins=20, olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,5,0
    # outs: _,0
    # tags: entropy
    EN_DistributionEntropy_hist_50_0 = HCTSAOperation(
        'EN_DistributionEntropy_hist_50_0',
        "EN_DistributionEntropy(y,'hist',50,0)",
        EN_DistributionEntropy(histOrKS='hist', numBins=50, olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,a,u
    # outs: t,o,_,0
    # tags: entropy
    EN_DistributionEntropy_hist_auto_0 = HCTSAOperation(
        'EN_DistributionEntropy_hist_auto_0',
        "EN_DistributionEntropy(y,'hist','auto',0)",
        EN_DistributionEntropy(histOrKS='hist', numBins='auto', olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,s,q
    # outs: r,t,_,0
    # tags: entropy
    EN_DistributionEntropy_hist_sqrt_0 = HCTSAOperation(
        'EN_DistributionEntropy_hist_sqrt_0',
        "EN_DistributionEntropy(y,'hist','sqrt',0)",
        EN_DistributionEntropy(histOrKS='hist', numBins='sqrt', olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,f,d
    # outs: _,0
    # tags: entropy
    EN_DistributionEntropy_hist_fd_0 = HCTSAOperation(
        'EN_DistributionEntropy_hist_fd_0',
        "EN_DistributionEntropy(y,'hist','fd',0)",
        EN_DistributionEntropy(histOrKS='hist', numBins='fd', olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,s,t
    # outs: u,r,g,e,s
    # outs: _,0
    # tags: entropy
    EN_DistributionEntropy_hist_sturges_0 = HCTSAOperation(
        'EN_DistributionEntropy_hist_sturges_0',
        "EN_DistributionEntropy(y,'hist','sturges',0)",
        EN_DistributionEntropy(histOrKS='hist', numBins='sturges', olremp=0))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,1,0
    # outs: _,0,0,1
    # tags: entropy
    EN_DistributionEntropy_hist_10_001 = HCTSAOperation(
        'EN_DistributionEntropy_hist_10_001',
        "EN_DistributionEntropy(y,'hist',10,0.01)",
        EN_DistributionEntropy(histOrKS='hist', numBins=10, olremp=0.01))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,1,0
    # outs: _,0,0,2
    # tags: entropy
    EN_DistributionEntropy_hist_10_002 = HCTSAOperation(
        'EN_DistributionEntropy_hist_10_002',
        "EN_DistributionEntropy(y,'hist',10,0.02)",
        EN_DistributionEntropy(histOrKS='hist', numBins=10, olremp=0.02))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,1,0
    # outs: _,0,0,5
    # tags: entropy
    EN_DistributionEntropy_hist_10_005 = HCTSAOperation(
        'EN_DistributionEntropy_hist_10_005',
        "EN_DistributionEntropy(y,'hist',10,0.05)",
        EN_DistributionEntropy(histOrKS='hist', numBins=10, olremp=0.05))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,1,0
    # outs: _,0,1
    # tags: entropy
    EN_DistributionEntropy_hist_10_01 = HCTSAOperation(
        'EN_DistributionEntropy_hist_10_01',
        "EN_DistributionEntropy(y,'hist',10,0.1)",
        EN_DistributionEntropy(histOrKS='hist', numBins=10, olremp=0.1))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,1,0
    # outs: _,0,2
    # tags: entropy
    EN_DistributionEntropy_hist_10_02 = HCTSAOperation(
        'EN_DistributionEntropy_hist_10_02',
        "EN_DistributionEntropy(y,'hist',10,0.2)",
        EN_DistributionEntropy(histOrKS='hist', numBins=10, olremp=0.2))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,1,0
    # outs: _,0,3
    # tags: entropy
    EN_DistributionEntropy_hist_10_03 = HCTSAOperation(
        'EN_DistributionEntropy_hist_10_03',
        "EN_DistributionEntropy(y,'hist',10,0.3)",
        EN_DistributionEntropy(histOrKS='hist', numBins=10, olremp=0.3))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,a,u
    # outs: t,o,_,0,0
    # outs: 1
    # tags: entropy
    EN_DistributionEntropy_hist_auto_001 = HCTSAOperation(
        'EN_DistributionEntropy_hist_auto_001',
        "EN_DistributionEntropy(y,'hist','auto',0.01)",
        EN_DistributionEntropy(histOrKS='hist', numBins='auto', olremp=0.01))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,a,u
    # outs: t,o,_,0,0
    # outs: 2
    # tags: entropy
    EN_DistributionEntropy_hist_auto_002 = HCTSAOperation(
        'EN_DistributionEntropy_hist_auto_002',
        "EN_DistributionEntropy(y,'hist','auto',0.02)",
        EN_DistributionEntropy(histOrKS='hist', numBins='auto', olremp=0.02))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,a,u
    # outs: t,o,_,0,0
    # outs: 5
    # tags: entropy
    EN_DistributionEntropy_hist_auto_005 = HCTSAOperation(
        'EN_DistributionEntropy_hist_auto_005',
        "EN_DistributionEntropy(y,'hist','auto',0.05)",
        EN_DistributionEntropy(histOrKS='hist', numBins='auto', olremp=0.05))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,a,u
    # outs: t,o,_,0,1
    # tags: entropy
    EN_DistributionEntropy_hist_auto_01 = HCTSAOperation(
        'EN_DistributionEntropy_hist_auto_01',
        "EN_DistributionEntropy(y,'hist','auto',0.1)",
        EN_DistributionEntropy(histOrKS='hist', numBins='auto', olremp=0.1))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,a,u
    # outs: t,o,_,0,2
    # tags: entropy
    EN_DistributionEntropy_hist_auto_02 = HCTSAOperation(
        'EN_DistributionEntropy_hist_auto_02',
        "EN_DistributionEntropy(y,'hist','auto',0.2)",
        EN_DistributionEntropy(histOrKS='hist', numBins='auto', olremp=0.2))

    # outs: E,N,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: E,n,t,r,o
    # outs: p,y,_,h,i
    # outs: s,t,_,a,u
    # outs: t,o,_,0,3
    # tags: entropy
    EN_DistributionEntropy_hist_auto_03 = HCTSAOperation(
        'EN_DistributionEntropy_hist_auto_03',
        "EN_DistributionEntropy(y,'hist','auto',0.3)",
        EN_DistributionEntropy(histOrKS='hist', numBins='auto', olremp=0.3))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,2
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_2 = HCTSAOperation(
        'EN_MS_LZcomplexity_2',
        'EN_MS_LZcomplexity(y,2,[])',
        EN_MS_LZcomplexity(n=2, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,3
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_3 = HCTSAOperation(
        'EN_MS_LZcomplexity_3',
        'EN_MS_LZcomplexity(y,3,[])',
        EN_MS_LZcomplexity(n=3, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,4
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_4 = HCTSAOperation(
        'EN_MS_LZcomplexity_4',
        'EN_MS_LZcomplexity(y,4,[])',
        EN_MS_LZcomplexity(n=4, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,5
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_5 = HCTSAOperation(
        'EN_MS_LZcomplexity_5',
        'EN_MS_LZcomplexity(y,5,[])',
        EN_MS_LZcomplexity(n=5, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,6
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_6 = HCTSAOperation(
        'EN_MS_LZcomplexity_6',
        'EN_MS_LZcomplexity(y,6,[])',
        EN_MS_LZcomplexity(n=6, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,7
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_7 = HCTSAOperation(
        'EN_MS_LZcomplexity_7',
        'EN_MS_LZcomplexity(y,7,[])',
        EN_MS_LZcomplexity(n=7, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,8
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_8 = HCTSAOperation(
        'EN_MS_LZcomplexity_8',
        'EN_MS_LZcomplexity(y,8,[])',
        EN_MS_LZcomplexity(n=8, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,9
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_9 = HCTSAOperation(
        'EN_MS_LZcomplexity_9',
        'EN_MS_LZcomplexity(y,9,[])',
        EN_MS_LZcomplexity(n=9, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,1
    # outs: 0
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_10 = HCTSAOperation(
        'EN_MS_LZcomplexity_10',
        'EN_MS_LZcomplexity(y,10,[])',
        EN_MS_LZcomplexity(n=10, preProc=array([], dtype=float64)))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,2
    # outs: _,d,i,f,f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_2_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_2_diff',
        "EN_MS_LZcomplexity(y,2,'diff')",
        EN_MS_LZcomplexity(n=2, preProc='diff'))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,3
    # outs: _,d,i,f,f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_3_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_3_diff',
        "EN_MS_LZcomplexity(y,3,'diff')",
        EN_MS_LZcomplexity(n=3, preProc='diff'))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,4
    # outs: _,d,i,f,f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_4_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_4_diff',
        "EN_MS_LZcomplexity(y,4,'diff')",
        EN_MS_LZcomplexity(n=4, preProc='diff'))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,5
    # outs: _,d,i,f,f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_5_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_5_diff',
        "EN_MS_LZcomplexity(y,5,'diff')",
        EN_MS_LZcomplexity(n=5, preProc='diff'))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,6
    # outs: _,d,i,f,f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_6_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_6_diff',
        "EN_MS_LZcomplexity(y,6,'diff')",
        EN_MS_LZcomplexity(n=6, preProc='diff'))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,7
    # outs: _,d,i,f,f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_7_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_7_diff',
        "EN_MS_LZcomplexity(y,7,'diff')",
        EN_MS_LZcomplexity(n=7, preProc='diff'))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,8
    # outs: _,d,i,f,f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_8_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_8_diff',
        "EN_MS_LZcomplexity(y,8,'diff')",
        EN_MS_LZcomplexity(n=8, preProc='diff'))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,9
    # outs: _,d,i,f,f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_9_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_9_diff',
        "EN_MS_LZcomplexity(y,9,'diff')",
        EN_MS_LZcomplexity(n=9, preProc='diff'))

    # outs: E,N,_,M,S
    # outs: _,L,Z,c,o
    # outs: m,p,l,e,x
    # outs: i,t,y,_,1
    # outs: 0,_,d,i,f
    # outs: f
    # tags: LempelZiv,MichaelSmall,complexity,mex
    EN_MS_LZcomplexity_10_diff = HCTSAOperation(
        'EN_MS_LZcomplexity_10_diff',
        "EN_MS_LZcomplexity(y,10,'diff')",
        EN_MS_LZcomplexity(n=10, preProc='diff'))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,2,_,2
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_2_2 = HCTSAOperation(
        'MS_shannon_2_2',
        'EN_MS_shannon(y,2,2)',
        EN_MS_shannon(nbin=2, depth=2))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,2,_,3
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_2_3 = HCTSAOperation(
        'MS_shannon_2_3',
        'EN_MS_shannon(y,2,3)',
        EN_MS_shannon(nbin=2, depth=3))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,3,_,2
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_3_2 = HCTSAOperation(
        'MS_shannon_3_2',
        'EN_MS_shannon(y,3,2)',
        EN_MS_shannon(nbin=3, depth=2))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,2,t,1,0
    # outs: _,2
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_2t10_2 = HCTSAOperation(
        'MS_shannon_2t10_2',
        'EN_MS_shannon(y,2:10,2)',
        EN_MS_shannon(nbin=MatlabSequence('2:10'), depth=2))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,2,t,1,0
    # outs: _,3
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_2t10_3 = HCTSAOperation(
        'MS_shannon_2t10_3',
        'EN_MS_shannon(y,2:10,3)',
        EN_MS_shannon(nbin=MatlabSequence('2:10'), depth=3))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,2,t,1,0
    # outs: _,4
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_2t10_4 = HCTSAOperation(
        'MS_shannon_2t10_4',
        'EN_MS_shannon(y,2:10,4)',
        EN_MS_shannon(nbin=MatlabSequence('2:10'), depth=4))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,2,_,1,t
    # outs: 1,0
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_2_1t10 = HCTSAOperation(
        'MS_shannon_2_1t10',
        'EN_MS_shannon(y,2,1:10)',
        EN_MS_shannon(nbin=2, depth=MatlabSequence('1:10')))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,3,_,1,t
    # outs: 1,0
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_3_1t10 = HCTSAOperation(
        'MS_shannon_3_1t10',
        'EN_MS_shannon(y,3,1:10)',
        EN_MS_shannon(nbin=3, depth=MatlabSequence('1:10')))

    # outs: M,S,_,s,h
    # outs: a,n,n,o,n
    # outs: _,4,_,1,t
    # outs: 1,0
    # tags: MichaelSmall,entropy,mex,shannon
    MS_shannon_4_1t10 = HCTSAOperation(
        'MS_shannon_4_1t10',
        'EN_MS_shannon(y,4,1:10)',
        EN_MS_shannon(nbin=4, depth=MatlabSequence('1:10')))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 2,_,1
    # tags: entropy
    EN_PermEn_2_1 = HCTSAOperation(
        'EN_PermEn_2_1',
        'EN_PermEn(y,2,1)',
        EN_PermEn(m=2, tau=1))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 3,_,1
    # tags: entropy
    EN_PermEn_3_1 = HCTSAOperation(
        'EN_PermEn_3_1',
        'EN_PermEn(y,3,1)',
        EN_PermEn(m=3, tau=1))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 4,_,1
    # tags: entropy
    EN_PermEn_4_1 = HCTSAOperation(
        'EN_PermEn_4_1',
        'EN_PermEn(y,4,1)',
        EN_PermEn(m=4, tau=1))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 5,_,1
    # tags: entropy
    EN_PermEn_5_1 = HCTSAOperation(
        'EN_PermEn_5_1',
        'EN_PermEn(y,5,1)',
        EN_PermEn(m=5, tau=1))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 2,_,2
    # tags: entropy
    EN_PermEn_2_2 = HCTSAOperation(
        'EN_PermEn_2_2',
        'EN_PermEn(y,2,2)',
        EN_PermEn(m=2, tau=2))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 3,_,2
    # tags: entropy
    EN_PermEn_3_2 = HCTSAOperation(
        'EN_PermEn_3_2',
        'EN_PermEn(y,3,2)',
        EN_PermEn(m=3, tau=2))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 4,_,2
    # tags: entropy
    EN_PermEn_4_2 = HCTSAOperation(
        'EN_PermEn_4_2',
        'EN_PermEn(y,4,2)',
        EN_PermEn(m=4, tau=2))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 5,_,2
    # tags: entropy
    EN_PermEn_5_2 = HCTSAOperation(
        'EN_PermEn_5_2',
        'EN_PermEn(y,5,2)',
        EN_PermEn(m=5, tau=2))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 2,_,a,c
    # tags: entropy
    EN_PermEn_2_ac = HCTSAOperation(
        'EN_PermEn_2_ac',
        "EN_PermEn(y,2,'ac')",
        EN_PermEn(m=2, tau='ac'))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 3,_,a,c
    # tags: entropy
    EN_PermEn_3_ac = HCTSAOperation(
        'EN_PermEn_3_ac',
        "EN_PermEn(y,3,'ac')",
        EN_PermEn(m=3, tau='ac'))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 4,_,a,c
    # tags: entropy
    EN_PermEn_4_ac = HCTSAOperation(
        'EN_PermEn_4_ac',
        "EN_PermEn(y,4,'ac')",
        EN_PermEn(m=4, tau='ac'))

    # outs: E,N,_,P,e
    # outs: r,m,E,n,_
    # outs: 5,_,a,c
    # tags: entropy
    EN_PermEn_5_ac = HCTSAOperation(
        'EN_PermEn_5_ac',
        "EN_PermEn(y,5,'ac')",
        EN_PermEn(m=5, tau='ac'))

    # outs: R,M,_,e,n
    # outs: t,r,o,p,y
    # tags: entropy
    RM_entropy = HCTSAOperation(
        'RM_entropy',
        'EN_RM_entropy(y)',
        EN_RM_entropy())

    # outs: E,N,_,R,a
    # outs: n,d,o,m,i
    # outs: z,e,_,s,t
    # outs: a,t,d,i,s
    # outs: t
    # tags: entropy,lengthdep,slow
    EN_Randomize_statdist = HCTSAOperation(
        'EN_Randomize_statdist',
        "EN_Randomize(y,'statdist','default')",
        EN_Randomize(randomizeHow='statdist', randomSeed='default'))

    # outs: E,N,_,R,a
    # outs: n,d,o,m,i
    # outs: z,e,_,d,y
    # outs: n,d,i,s,t
    # tags: entropy,lengthdep,slow
    EN_Randomize_dyndist = HCTSAOperation(
        'EN_Randomize_dyndist',
        "EN_Randomize(y,'dyndist','default')",
        EN_Randomize(randomizeHow='dyndist', randomSeed='default'))

    # outs: E,N,_,R,a
    # outs: n,d,o,m,i
    # outs: z,e,_,p,e
    # outs: r,m,u,t,e
    # tags: entropy,lengthdep,slow
    EN_Randomize_permute = HCTSAOperation(
        'EN_Randomize_permute',
        "EN_Randomize(y,'permute','default')",
        EN_Randomize(randomizeHow='permute', randomSeed='default'))

    # outs: E,N,_,S,a
    # outs: m,p,E,n,_
    # outs: 5,_,0,0,5
    # tags: entropy,sampen
    EN_SampEn_5_005 = HCTSAOperation(
        'EN_SampEn_5_005',
        'EN_SampEn(y,5,0.05)',
        EN_SampEn(M=5, r=0.05))

    # outs: E,N,_,S,a
    # outs: m,p,E,n,_
    # outs: 5,_,0,1
    # tags: entropy,sampen
    EN_SampEn_5_01 = HCTSAOperation(
        'EN_SampEn_5_01',
        'EN_SampEn(y,5,0.1)',
        EN_SampEn(M=5, r=0.1))

    # outs: E,N,_,S,a
    # outs: m,p,E,n,_
    # outs: 5,_,0,1,5
    # tags: entropy,sampen
    EN_SampEn_5_015 = HCTSAOperation(
        'EN_SampEn_5_015',
        'EN_SampEn(y,5,0.15)',
        EN_SampEn(M=5, r=0.15))

    # outs: E,N,_,S,a
    # outs: m,p,E,n,_
    # outs: 5,_,0,2
    # tags: entropy,sampen
    EN_SampEn_5_02 = HCTSAOperation(
        'EN_SampEn_5_02',
        'EN_SampEn(y,5,0.2)',
        EN_SampEn(M=5, r=0.2))

    # outs: E,N,_,S,a
    # outs: m,p,E,n,_
    # outs: 5,_,0,3
    # tags: entropy,sampen
    EN_SampEn_5_03 = HCTSAOperation(
        'EN_SampEn_5_03',
        'EN_SampEn(y,5,0.3)',
        EN_SampEn(M=5, r=0.3))

    # outs: E,N,_,S,a
    # outs: m,p,E,n,_
    # outs: 5,_,0,1,_
    # outs: d,i,f,f,1
    # tags: controlen,entropy,sampen
    EN_SampEn_5_01_diff1 = HCTSAOperation(
        'EN_SampEn_5_01_diff1',
        "EN_SampEn(y,5,0.1,'diff1')",
        EN_SampEn(M=5, r=0.1, preProcessHow='diff1'))

    # outs: E,N,_,S,a
    # outs: m,p,E,n,_
    # outs: 5,_,0,2,_
    # outs: d,i,f,f,1
    # tags: controlen,entropy,sampen
    EN_SampEn_5_02_diff1 = HCTSAOperation(
        'EN_SampEn_5_02_diff1',
        "EN_SampEn(y,5,0.2,'diff1')",
        EN_SampEn(M=5, r=0.2, preProcessHow='diff1'))

    # outs: E,N,_,m,s
    # outs: e,_,1,_,1
    # outs: 0,_,2,_,0
    # outs: 1,5
    # tags: entropy,mse,sampen
    EN_mse_1_10_2_015 = HCTSAOperation(
        'EN_mse_1_10_2_015',
        'EN_mse(y,1:10,2,0.15)',
        EN_mse(scaleRange=MatlabSequence('1:10'), m=2, r=0.15))

    # outs: E,N,_,m,s
    # outs: e,_,1,_,1
    # outs: 0,_,2,_,0
    # outs: 1,5,_,d,i
    # outs: f,f,1
    # tags: controlen,entropy,mse,sampen
    EN_mse_1_10_2_015_diff1 = HCTSAOperation(
        'EN_mse_1_10_2_015_diff1',
        "EN_mse(y,1:10,2,0.15,'diff1')",
        EN_mse(scaleRange=MatlabSequence('1:10'), m=2, r=0.15, preProcessHow='diff1'))

    # outs: E,N,_,m,s
    # outs: e,_,1,_,1
    # outs: 0,_,2,_,0
    # outs: 1,5,_,r,e
    # outs: s,c,a,l,e
    # outs: _,t,a,u
    # tags: controlen,entropy,mse,sampen
    EN_mse_1_10_2_015_rescale_tau = HCTSAOperation(
        'EN_mse_1_10_2_015_rescale_tau',
        "EN_mse(y,1:10,2,0.15,'rescale_tau')",
        EN_mse(scaleRange=MatlabSequence('1:10'), m=2, r=0.15, preProcessHow='rescale_tau'))

    # outs: E,N,_,r,p
    # outs: d,e,_,3,_
    # outs: 1
    # tags: entropy
    EN_rpde_3_1 = HCTSAOperation(
        'EN_rpde_3_1',
        'EN_rpde(y,3,1)',
        EN_rpde(m=3, tau=1))

    # outs: E,N,_,r,p
    # outs: d,e,_,3,_
    # outs: a,c
    # tags: entropy
    EN_rpde_3_ac = HCTSAOperation(
        'EN_rpde_3_ac',
        "EN_rpde(y,3,'ac')",
        EN_rpde(m=3, tau='ac'))

    # outs: E,N,_,w,e
    # outs: n,t,r,o,p
    # outs: y,_,s,h,a
    # outs: n,n,o,n
    # tags: entropy,shannon
    EN_wentropy_shannon = HCTSAOperation(
        'EN_wentropy_shannon',
        "EN_wentropy(y,'shannon')",
        EN_wentropy(whaten='shannon'))

    # outs: E,N,_,w,e
    # outs: n,t,r,o,p
    # outs: y,_,l,o,g
    # outs: e,n,e,n,t
    # tags: entropy
    EN_wentropy_logenent = HCTSAOperation(
        'EN_wentropy_logenent',
        "EN_wentropy(y,'logenergy')",
        EN_wentropy(whaten='logenergy'))

    # outs: E,X,_,M,o
    # outs: v,i,n,g,T
    # outs: h,r,e,s,h
    # outs: o,l,d,_,0
    # outs: 1,_,0,1
    # tags: outliers
    EX_MovingThreshold_01_01 = HCTSAOperation(
        'EX_MovingThreshold_01_01',
        'EX_MovingThreshold(y,0.1,0.1)',
        EX_MovingThreshold(a=0.1, b=0.1))

    # outs: E,X,_,M,o
    # outs: v,i,n,g,T
    # outs: h,r,e,s,h
    # outs: o,l,d,_,0
    # outs: 1,_,0,0,2
    # tags: outliers
    EX_MovingThreshold_01_002 = HCTSAOperation(
        'EX_MovingThreshold_01_002',
        'EX_MovingThreshold(y,0.1,0.02)',
        EX_MovingThreshold(a=0.1, b=0.02))

    # outs: E,X,_,M,o
    # outs: v,i,n,g,T
    # outs: h,r,e,s,h
    # outs: o,l,d,_,1
    # outs: _,0,1
    # tags: outliers
    EX_MovingThreshold_1_01 = HCTSAOperation(
        'EX_MovingThreshold_1_01',
        'EX_MovingThreshold(y,1,0.1)',
        EX_MovingThreshold(a=1, b=0.1))

    # outs: E,X,_,M,o
    # outs: v,i,n,g,T
    # outs: h,r,e,s,h
    # outs: o,l,d,_,1
    # outs: _,0,0,2
    # tags: outliers
    EX_MovingThreshold_1_002 = HCTSAOperation(
        'EX_MovingThreshold_1_002',
        'EX_MovingThreshold(y,1,0.02)',
        EX_MovingThreshold(a=1, b=0.02))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: m,e,a,n,1
    # tags: forecasting
    FC_LocalSimple_mean1 = HCTSAOperation(
        'FC_LocalSimple_mean1',
        "FC_LocalSimple(y,'mean',1)",
        FC_LocalSimple(forecastMeth='mean', trainLength=1))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: m,e,a,n,2
    # tags: forecasting
    FC_LocalSimple_mean2 = HCTSAOperation(
        'FC_LocalSimple_mean2',
        "FC_LocalSimple(y,'mean',2)",
        FC_LocalSimple(forecastMeth='mean', trainLength=2))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: m,e,a,n,3
    # tags: forecasting
    FC_LocalSimple_mean3 = HCTSAOperation(
        'FC_LocalSimple_mean3',
        "FC_LocalSimple(y,'mean',3)",
        FC_LocalSimple(forecastMeth='mean', trainLength=3))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: m,e,a,n,4
    # tags: forecasting
    FC_LocalSimple_mean4 = HCTSAOperation(
        'FC_LocalSimple_mean4',
        "FC_LocalSimple(y,'mean',4)",
        FC_LocalSimple(forecastMeth='mean', trainLength=4))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: m,e,a,n,t
    # outs: a,u
    # tags: forecasting
    FC_LocalSimple_meantau = HCTSAOperation(
        'FC_LocalSimple_meantau',
        "FC_LocalSimple(y,'mean','ac')",
        FC_LocalSimple(forecastMeth='mean', trainLength='ac'))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: m,e,d,i,a
    # outs: n,3
    # tags: forecasting
    FC_LocalSimple_median3 = HCTSAOperation(
        'FC_LocalSimple_median3',
        "FC_LocalSimple(y,'median',3)",
        FC_LocalSimple(forecastMeth='median', trainLength=3))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: m,e,d,i,a
    # outs: n,5
    # tags: forecasting
    FC_LocalSimple_median5 = HCTSAOperation(
        'FC_LocalSimple_median5',
        "FC_LocalSimple(y,'median',5)",
        FC_LocalSimple(forecastMeth='median', trainLength=5))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: m,e,d,i,a
    # outs: n,7
    # tags: forecasting
    FC_LocalSimple_median7 = HCTSAOperation(
        'FC_LocalSimple_median7',
        "FC_LocalSimple(y,'median',7)",
        FC_LocalSimple(forecastMeth='median', trainLength=7))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: l,f,i,t,2
    # tags: forecasting
    FC_LocalSimple_lfit2 = HCTSAOperation(
        'FC_LocalSimple_lfit2',
        "FC_LocalSimple(y,'lfit',2)",
        FC_LocalSimple(forecastMeth='lfit', trainLength=2))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: l,f,i,t,3
    # tags: forecasting
    FC_LocalSimple_lfit3 = HCTSAOperation(
        'FC_LocalSimple_lfit3',
        "FC_LocalSimple(y,'lfit',3)",
        FC_LocalSimple(forecastMeth='lfit', trainLength=3))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: l,f,i,t,4
    # tags: forecasting
    FC_LocalSimple_lfit4 = HCTSAOperation(
        'FC_LocalSimple_lfit4',
        "FC_LocalSimple(y,'lfit',4)",
        FC_LocalSimple(forecastMeth='lfit', trainLength=4))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: l,f,i,t,5
    # tags: forecasting
    FC_LocalSimple_lfit5 = HCTSAOperation(
        'FC_LocalSimple_lfit5',
        "FC_LocalSimple(y,'lfit',5)",
        FC_LocalSimple(forecastMeth='lfit', trainLength=5))

    # outs: F,C,_,L,o
    # outs: c,a,l,S,i
    # outs: m,p,l,e,_
    # outs: l,f,i,t,t
    # outs: a,u
    # tags: forecasting
    FC_LocalSimple_lfittau = HCTSAOperation(
        'FC_LocalSimple_lfittau',
        "FC_LocalSimple(y,'lfit','ac')",
        FC_LocalSimple(forecastMeth='lfit', trainLength='ac'))

    # outs: F,C,_,L,o
    # outs: o,p,L,o,c
    # outs: a,l,S,i,m
    # outs: p,l,e,_,m
    # outs: e,a,n
    # tags: forecasting
    FC_LoopLocalSimple_mean = HCTSAOperation(
        'FC_LoopLocalSimple_mean',
        "FC_LoopLocalSimple(y,'mean')",
        FC_LoopLocalSimple(forecastMeth='mean'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,5,_,2
    # outs: _,q,_,5,0
    # outs: 0
    # tags: information,symbolic
    FC_Surprise_dist_5_2_q_500 = HCTSAOperation(
        'FC_Surprise_dist_5_2_q_500',
        "FC_Surprise(y,'dist',5,2,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='dist', memory=5, numGroups=2, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,2,0,_
    # outs: 2,_,q,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_dist_20_2_q_500 = HCTSAOperation(
        'FC_Surprise_dist_20_2_q_500',
        "FC_Surprise(y,'dist',20,2,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='dist', memory=20, numGroups=2, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,5,0,_
    # outs: 3,_,q,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_dist_50_3_q_500 = HCTSAOperation(
        'FC_Surprise_dist_50_3_q_500',
        "FC_Surprise(y,'dist',50,3,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='dist', memory=50, numGroups=3, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,1,0,0
    # outs: _,4,_,q,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_100_4_q_500 = HCTSAOperation(
        'FC_Surprise_dist_100_4_q_500',
        "FC_Surprise(y,'dist',100,4,'quantile',500,'default')",
        FC_Surprise(whatPrior='dist', memory=100, numGroups=4, cgmeth='quantile', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,1,0,0
    # outs: _,5,_,q,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_100_5_q_500 = HCTSAOperation(
        'FC_Surprise_dist_100_5_q_500',
        "FC_Surprise(y,'dist',100,5,'quantile',500,'default')",
        FC_Surprise(whatPrior='dist', memory=100, numGroups=5, cgmeth='quantile', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 2,0,_,2,_
    # outs: q,_,5,0,0
    # tags: information,symbolic
    FC_Surprise_T1_20_2_q_500 = HCTSAOperation(
        'FC_Surprise_T1_20_2_q_500',
        "FC_Surprise(y,'T1',20,2,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='T1', memory=20, numGroups=2, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 5,0,_,3,_
    # outs: q,_,5,0,0
    # tags: information,symbolic
    FC_Surprise_T1_50_3_q_500 = HCTSAOperation(
        'FC_Surprise_T1_50_3_q_500',
        "FC_Surprise(y,'T1',50,3,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='T1', memory=50, numGroups=3, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 1,0,0,_,4
    # outs: _,q,_,5,0
    # outs: 0
    # tags: information,symbolic
    FC_Surprise_T1_100_4_q_500 = HCTSAOperation(
        'FC_Surprise_T1_100_4_q_500',
        "FC_Surprise(y,'T1',100,4,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='T1', memory=100, numGroups=4, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 1,0,0,_,5
    # outs: _,q,_,5,0
    # outs: 0
    # tags: information,symbolic
    FC_Surprise_T1_100_5_q_500 = HCTSAOperation(
        'FC_Surprise_T1_100_5_q_500',
        "FC_Surprise(y,'T1',100,5,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='T1', memory=100, numGroups=5, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 2,0,_,2,_
    # outs: q,_,5,0,0
    # tags: information,symbolic
    FC_Surprise_T2_20_2_q_500 = HCTSAOperation(
        'FC_Surprise_T2_20_2_q_500',
        "FC_Surprise(y,'T2',20,2,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='T2', memory=20, numGroups=2, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 5,0,_,3,_
    # outs: q,_,5,0,0
    # tags: information,symbolic
    FC_Surprise_T2_50_3_q_500 = HCTSAOperation(
        'FC_Surprise_T2_50_3_q_500',
        "FC_Surprise(y,'T2',50,3,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='T2', memory=50, numGroups=3, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 1,0,0,_,4
    # outs: _,q,_,5,0
    # outs: 0
    # tags: information,symbolic
    FC_Surprise_T2_100_4_q_500 = HCTSAOperation(
        'FC_Surprise_T2_100_4_q_500',
        "FC_Surprise(y,'T2',100,4,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='T2', memory=100, numGroups=4, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 1,0,0,_,5
    # outs: _,q,_,5,0
    # outs: 0
    # tags: information,symbolic
    FC_Surprise_T2_100_5_q_500 = HCTSAOperation(
        'FC_Surprise_T2_100_5_q_500',
        "FC_Surprise(y,'T2',100,5,'quantile',500,'default')",
    
                       FC_Surprise(whatPrior='T2', memory=100, numGroups=5, cgmeth='quantile', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,5,_,2
    # outs: _,u,d,q,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_5_2_udq_500 = HCTSAOperation(
        'FC_Surprise_dist_5_2_udq_500',
        "FC_Surprise(y,'dist',5,2,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='dist', memory=5, numGroups=2, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,2,0,_
    # outs: 2,_,u,d,q
    # outs: _,5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_20_2_udq_500 = HCTSAOperation(
        'FC_Surprise_dist_20_2_udq_500',
        "FC_Surprise(y,'dist',20,2,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='dist', memory=20, numGroups=2, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,5,0,_
    # outs: 3,_,u,d,q
    # outs: _,5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_50_3_udq_500 = HCTSAOperation(
        'FC_Surprise_dist_50_3_udq_500',
        "FC_Surprise(y,'dist',50,3,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='dist', memory=50, numGroups=3, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,1,0,0
    # outs: _,4,_,u,d
    # outs: q,_,5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_100_4_udq_500 = HCTSAOperation(
        'FC_Surprise_dist_100_4_udq_500',
        "FC_Surprise(y,'dist',100,4,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='dist', memory=100, numGroups=4, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,1,0,0
    # outs: _,5,_,u,d
    # outs: q,_,5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_100_5_udq_500 = HCTSAOperation(
        'FC_Surprise_dist_100_5_udq_500',
        "FC_Surprise(y,'dist',100,5,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='dist', memory=100, numGroups=5, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 2,0,_,2,_
    # outs: u,d,q,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_T1_20_2_udq_500 = HCTSAOperation(
        'FC_Surprise_T1_20_2_udq_500',
        "FC_Surprise(y,'T1',20,2,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='T1', memory=20, numGroups=2, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 5,0,_,3,_
    # outs: u,d,q,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_T1_50_3_udq_500 = HCTSAOperation(
        'FC_Surprise_T1_50_3_udq_500',
        "FC_Surprise(y,'T1',50,3,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='T1', memory=50, numGroups=3, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 1,0,0,_,4
    # outs: _,u,d,q,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_T1_100_4_udq_500 = HCTSAOperation(
        'FC_Surprise_T1_100_4_udq_500',
        "FC_Surprise(y,'T1',100,4,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='T1', memory=100, numGroups=4, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 1,0,0,_,5
    # outs: _,u,d,q,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_T1_100_5_udq_500 = HCTSAOperation(
        'FC_Surprise_T1_100_5_udq_500',
        "FC_Surprise(y,'T1',100,5,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='T1', memory=100, numGroups=5, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 2,0,_,2,_
    # outs: u,d,q,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_T2_20_2_udq_500 = HCTSAOperation(
        'FC_Surprise_T2_20_2_udq_500',
        "FC_Surprise(y,'T2',20,2,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='T2', memory=20, numGroups=2, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 5,0,_,3,_
    # outs: u,d,q,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_T2_50_3_udq_500 = HCTSAOperation(
        'FC_Surprise_T2_50_3_udq_500',
        "FC_Surprise(y,'T2',50,3,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='T2', memory=50, numGroups=3, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 1,0,0,_,4
    # outs: _,u,d,q,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_T2_100_4_udq_500 = HCTSAOperation(
        'FC_Surprise_T2_100_4_udq_500',
        "FC_Surprise(y,'T2',100,4,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='T2', memory=100, numGroups=4, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 1,0,0,_,5
    # outs: _,u,d,q,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_T2_100_5_udq_500 = HCTSAOperation(
        'FC_Surprise_T2_100_5_udq_500',
        "FC_Surprise(y,'T2',100,5,'updown',500,'default')",
    
                       FC_Surprise(whatPrior='T2', memory=100, numGroups=5, cgmeth='updown', numIters=500, randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,5,_,1
    # outs: _,m,2,q,u
    # outs: a,d,_,5,0
    # outs: 0
    # tags: information,symbolic
    FC_Surprise_dist_5_1_m2quad_500 = HCTSAOperation(
        'FC_Surprise_dist_5_1_m2quad_500',
        "FC_Surprise(y,'dist',5,1,'embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='dist', memory=5, numGroups=1, cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,5,_,t
    # outs: a,u,_,m,2
    # outs: q,u,a,d,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_5_tau_m2quad_500 = HCTSAOperation(
        'FC_Surprise_dist_5_tau_m2quad_500',
        "FC_Surprise(y,'dist',5,'tau','embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='dist', memory=5, numGroups='tau', cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,1,0,_
    # outs: t,a,u,_,m
    # outs: 2,q,u,a,d
    # outs: _,5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_10_tau_m2quad_500 = HCTSAOperation(
        'FC_Surprise_dist_10_tau_m2quad_500',
        "FC_Surprise(y,'dist',10,'tau','embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='dist', memory=10, numGroups='tau', cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,d,i,s
    # outs: t,_,2,0,_
    # outs: t,a,u,_,m
    # outs: 2,q,u,a,d
    # outs: _,5,0,0
    # tags: information,symbolic
    FC_Surprise_dist_20_tau_m2quad_500 = HCTSAOperation(
        'FC_Surprise_dist_20_tau_m2quad_500',
        "FC_Surprise(y,'dist',20,'tau','embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='dist', memory=20, numGroups='tau', cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 1,0,_,1,_
    # outs: m,2,q,u,a
    # outs: d,_,5,0,0
    # tags: information,symbolic
    FC_Surprise_T1_10_1_m2quad_500 = HCTSAOperation(
        'FC_Surprise_T1_10_1_m2quad_500',
        "FC_Surprise(y,'T1',10,1,'embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='T1', memory=10, numGroups=1, cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 1,0,_,t,a
    # outs: u,_,m,2,q
    # outs: u,a,d,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_T1_10_tau_m2quad_500 = HCTSAOperation(
        'FC_Surprise_T1_10_tau_m2quad_500',
        "FC_Surprise(y,'T1',10,'tau','embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='T1', memory=10, numGroups='tau', cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 2,0,_,t,a
    # outs: u,_,m,2,q
    # outs: u,a,d,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_T1_20_tau_m2quad_500 = HCTSAOperation(
        'FC_Surprise_T1_20_tau_m2quad_500',
        "FC_Surprise(y,'T1',20,'tau','embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='T1', memory=20, numGroups='tau', cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,1,_
    # outs: 5,0,_,t,a
    # outs: u,_,m,2,q
    # outs: u,a,d,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_T1_50_tau_m2quad_500 = HCTSAOperation(
        'FC_Surprise_T1_50_tau_m2quad_500',
        "FC_Surprise(y,'T1',50,'tau','embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='T1', memory=50, numGroups='tau', cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 5,0,_,t,a
    # outs: u,_,m,2,q
    # outs: u,a,d,_,5
    # outs: 0,0
    # tags: information,symbolic
    FC_Surprise_T2_50_tau_m2quad_500 = HCTSAOperation(
        'FC_Surprise_T2_50_tau_m2quad_500',
        "FC_Surprise(y,'T2',50,'tau','embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='T2', memory=50, numGroups='tau', cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: F,C,_,S,u
    # outs: r,p,r,i,s
    # outs: e,_,T,2,_
    # outs: 1,0,0,_,t
    # outs: a,u,_,m,2
    # outs: q,u,a,d,_
    # outs: 5,0,0
    # tags: information,symbolic
    FC_Surprise_T2_100_tau_m2quad_500 = HCTSAOperation(
        'FC_Surprise_T2_100_tau_m2quad_500',
        "FC_Surprise(y,'T2',100,'tau','embed2quadrants',500,'default')",
        FC_Surprise(whatPrior='T2', memory=100, numGroups='tau', cgmeth='embed2quadrants', numIters=500,
                    randomSeed='default'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,n
    # outs: o,r,m,5
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2norm5 = HCTSAOperation(
        'HT_DistributionTest_chi2norm5',
        "HT_DistributionTest(x,'chi2gof','norm',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='norm', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,n
    # outs: o,r,m,1,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2norm10 = HCTSAOperation(
        'HT_DistributionTest_chi2norm10',
        "HT_DistributionTest(x,'chi2gof','norm',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='norm', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,n
    # outs: o,r,m,2,5
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2norm25 = HCTSAOperation(
        'HT_DistributionTest_chi2norm25',
        "HT_DistributionTest(x,'chi2gof','norm',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='norm', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,n
    # outs: o,r,m,5,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2norm50 = HCTSAOperation(
        'HT_DistributionTest_chi2norm50',
        "HT_DistributionTest(x,'chi2gof','norm',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='norm', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,n
    # outs: o,r,m,1,0
    # outs: 0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2norm100 = HCTSAOperation(
        'HT_DistributionTest_chi2norm100',
        "HT_DistributionTest(x,'chi2gof','norm',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='norm', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: v,5
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2ev5 = HCTSAOperation(
        'HT_DistributionTest_chi2ev5',
        "HT_DistributionTest(x,'chi2gof','ev',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='ev', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: v,1,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2ev10 = HCTSAOperation(
        'HT_DistributionTest_chi2ev10',
        "HT_DistributionTest(x,'chi2gof','ev',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='ev', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: v,2,5
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2ev25 = HCTSAOperation(
        'HT_DistributionTest_chi2ev25',
        "HT_DistributionTest(x,'chi2gof','ev',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='ev', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: v,5,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2ev50 = HCTSAOperation(
        'HT_DistributionTest_chi2ev50',
        "HT_DistributionTest(x,'chi2gof','ev',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='ev', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: v,1,0,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2ev100 = HCTSAOperation(
        'HT_DistributionTest_chi2ev100',
        "HT_DistributionTest(x,'chi2gof','ev',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='ev', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,u
    # outs: n,i,5
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2uni5 = HCTSAOperation(
        'HT_DistributionTest_chi2uni5',
        "HT_DistributionTest(x,'chi2gof','uni',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='uni', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,u
    # outs: n,i,1,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2uni10 = HCTSAOperation(
        'HT_DistributionTest_chi2uni10',
        "HT_DistributionTest(x,'chi2gof','uni',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='uni', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,u
    # outs: n,i,2,5
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2uni25 = HCTSAOperation(
        'HT_DistributionTest_chi2uni25',
        "HT_DistributionTest(x,'chi2gof','uni',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='uni', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,u
    # outs: n,i,5,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2uni50 = HCTSAOperation(
        'HT_DistributionTest_chi2uni50',
        "HT_DistributionTest(x,'chi2gof','uni',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='uni', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,u
    # outs: n,i,1,0,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2uni100 = HCTSAOperation(
        'HT_DistributionTest_chi2uni100',
        "HT_DistributionTest(x,'chi2gof','uni',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='uni', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,b
    # outs: e,t,a,5
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2beta5 = HCTSAOperation(
        'HT_DistributionTest_chi2beta5',
        "HT_DistributionTest(x,'chi2gof','beta',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='beta', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,b
    # outs: e,t,a,1,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2beta10 = HCTSAOperation(
        'HT_DistributionTest_chi2beta10',
        "HT_DistributionTest(x,'chi2gof','beta',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='beta', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,b
    # outs: e,t,a,2,5
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2beta25 = HCTSAOperation(
        'HT_DistributionTest_chi2beta25',
        "HT_DistributionTest(x,'chi2gof','beta',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='beta', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,b
    # outs: e,t,a,5,0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2beta50 = HCTSAOperation(
        'HT_DistributionTest_chi2beta50',
        "HT_DistributionTest(x,'chi2gof','beta',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='beta', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,b
    # outs: e,t,a,1,0
    # outs: 0
    # tags: chi2gof,distribution,hypothesistest,raw
    HT_DistributionTest_chi2beta100 = HCTSAOperation(
        'HT_DistributionTest_chi2beta100',
        "HT_DistributionTest(x,'chi2gof','beta',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='beta', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,r
    # outs: a,y,l,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2rayl5 = HCTSAOperation(
        'HT_DistributionTest_chi2rayl5',
        "HT_DistributionTest(x,'chi2gof','rayleigh',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='rayleigh', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,r
    # outs: a,y,l,1,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2rayl10 = HCTSAOperation(
        'HT_DistributionTest_chi2rayl10',
        "HT_DistributionTest(x,'chi2gof','rayleigh',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='rayleigh', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,r
    # outs: a,y,l,2,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2rayl25 = HCTSAOperation(
        'HT_DistributionTest_chi2rayl25',
        "HT_DistributionTest(x,'chi2gof','rayleigh',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='rayleigh', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,r
    # outs: a,y,l,5,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2rayl50 = HCTSAOperation(
        'HT_DistributionTest_chi2rayl50',
        "HT_DistributionTest(x,'chi2gof','rayleigh',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='rayleigh', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,r
    # outs: a,y,l,1,0
    # outs: 0
    # tags: chi2gof,distribution,hypothesistest,posOnly,raw
    HT_DistributionTest_chi2rayl100 = HCTSAOperation(
        'HT_DistributionTest_chi2rayl100',
        "HT_DistributionTest(x,'chi2gof','rayleigh',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='rayleigh', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: x,p,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2exp5 = HCTSAOperation(
        'HT_DistributionTest_chi2exp5',
        "HT_DistributionTest(x,'chi2gof','exp',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='exp', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: x,p,1,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2exp10 = HCTSAOperation(
        'HT_DistributionTest_chi2exp10',
        "HT_DistributionTest(x,'chi2gof','exp',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='exp', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: x,p,2,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2exp25 = HCTSAOperation(
        'HT_DistributionTest_chi2exp25',
        "HT_DistributionTest(x,'chi2gof','exp',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='exp', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: x,p,5,0
    # tags: chi2gof,distribution,hypothesistest,posOnly,raw
    HT_DistributionTest_chi2exp50 = HCTSAOperation(
        'HT_DistributionTest_chi2exp50',
        "HT_DistributionTest(x,'chi2gof','exp',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='exp', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,e
    # outs: x,p,1,0,0
    # tags: chi2gof,distribution,hypothesistest,posOnly,raw
    HT_DistributionTest_chi2exp100 = HCTSAOperation(
        'HT_DistributionTest_chi2exp100',
        "HT_DistributionTest(x,'chi2gof','exp',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='exp', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,g
    # outs: a,m,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2gam5 = HCTSAOperation(
        'HT_DistributionTest_chi2gam5',
        "HT_DistributionTest(x,'chi2gof','gamma',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='gamma', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,g
    # outs: a,m,1,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2gam10 = HCTSAOperation(
        'HT_DistributionTest_chi2gam10',
        "HT_DistributionTest(x,'chi2gof','gamma',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='gamma', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,g
    # outs: a,m,2,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2gam25 = HCTSAOperation(
        'HT_DistributionTest_chi2gam25',
        "HT_DistributionTest(x,'chi2gof','gamma',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='gamma', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,g
    # outs: a,m,5,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2gam50 = HCTSAOperation(
        'HT_DistributionTest_chi2gam50',
        "HT_DistributionTest(x,'chi2gof','gamma',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='gamma', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,g
    # outs: a,m,1,0,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2gam100 = HCTSAOperation(
        'HT_DistributionTest_chi2gam100',
        "HT_DistributionTest(x,'chi2gof','gamma',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='gamma', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,l
    # outs: o,g,n,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2logn5 = HCTSAOperation(
        'HT_DistributionTest_chi2logn5',
        "HT_DistributionTest(x,'chi2gof','logn',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='logn', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,l
    # outs: o,g,n,1,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2logn10 = HCTSAOperation(
        'HT_DistributionTest_chi2logn10',
        "HT_DistributionTest(x,'chi2gof','logn',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='logn', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,l
    # outs: o,g,n,2,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2logn25 = HCTSAOperation(
        'HT_DistributionTest_chi2logn25',
        "HT_DistributionTest(x,'chi2gof','logn',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='logn', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,l
    # outs: o,g,n,5,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2logn50 = HCTSAOperation(
        'HT_DistributionTest_chi2logn50',
        "HT_DistributionTest(x,'chi2gof','logn',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='logn', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,l
    # outs: o,g,n,1,0
    # outs: 0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2logn100 = HCTSAOperation(
        'HT_DistributionTest_chi2logn100',
        "HT_DistributionTest(x,'chi2gof','logn',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='logn', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,w
    # outs: b,l,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2wbl5 = HCTSAOperation(
        'HT_DistributionTest_chi2wbl5',
        "HT_DistributionTest(x,'chi2gof','wbl',5)",
        HT_DistributionTest(theTest='chi2gof', theDistn='wbl', numBins=5))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,w
    # outs: b,l,1,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2wbl10 = HCTSAOperation(
        'HT_DistributionTest_chi2wbl10',
        "HT_DistributionTest(x,'chi2gof','wbl',10)",
        HT_DistributionTest(theTest='chi2gof', theDistn='wbl', numBins=10))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,w
    # outs: b,l,2,5
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2wbl25 = HCTSAOperation(
        'HT_DistributionTest_chi2wbl25',
        "HT_DistributionTest(x,'chi2gof','wbl',25)",
        HT_DistributionTest(theTest='chi2gof', theDistn='wbl', numBins=25))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,w
    # outs: b,l,5,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2wbl50 = HCTSAOperation(
        'HT_DistributionTest_chi2wbl50',
        "HT_DistributionTest(x,'chi2gof','wbl',50)",
        HT_DistributionTest(theTest='chi2gof', theDistn='wbl', numBins=50))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: c,h,i,2,w
    # outs: b,l,1,0,0
    # tags: chi2gof,distribution,hypothesistest,locdep,posOnly,raw
    HT_DistributionTest_chi2wbl100 = HCTSAOperation(
        'HT_DistributionTest_chi2wbl100',
        "HT_DistributionTest(x,'chi2gof','wbl',100)",
        HT_DistributionTest(theTest='chi2gof', theDistn='wbl', numBins=100))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,n,o
    # outs: r,m
    # tags: distribution,hypothesistest,ks,raw
    HT_DistributionTest_ks_norm = HCTSAOperation(
        'HT_DistributionTest_ks_norm',
        "HT_DistributionTest(x,'ks','norm')",
        HT_DistributionTest(theTest='ks', theDistn='norm'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,e,v
    # tags: distribution,hypothesistest,ks,raw
    HT_DistributionTest_ks_ev = HCTSAOperation(
        'HT_DistributionTest_ks_ev',
        "HT_DistributionTest(x,'ks','ev')",
        HT_DistributionTest(theTest='ks', theDistn='ev'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,u,n
    # outs: i
    # tags: distribution,hypothesistest,ks,raw
    HT_DistributionTest_ks_uni = HCTSAOperation(
        'HT_DistributionTest_ks_uni',
        "HT_DistributionTest(x,'ks','uni')",
        HT_DistributionTest(theTest='ks', theDistn='uni'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,b,e
    # outs: t,a
    # tags: distribution,hypothesistest,ks,raw
    HT_DistributionTest_ks_beta = HCTSAOperation(
        'HT_DistributionTest_ks_beta',
        "HT_DistributionTest(x,'ks','beta')",
        HT_DistributionTest(theTest='ks', theDistn='beta'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,r,a
    # outs: y,l
    # tags: distribution,hypothesistest,ks,locdep,posOnly,raw
    HT_DistributionTest_ks_rayl = HCTSAOperation(
        'HT_DistributionTest_ks_rayl',
        "HT_DistributionTest(x,'ks','rayleigh')",
        HT_DistributionTest(theTest='ks', theDistn='rayleigh'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,e,x
    # outs: p
    # tags: distribution,hypothesistest,ks,locdep,posOnly,raw
    HT_DistributionTest_ks_exp = HCTSAOperation(
        'HT_DistributionTest_ks_exp',
        "HT_DistributionTest(x,'ks','exp')",
        HT_DistributionTest(theTest='ks', theDistn='exp'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,g,a
    # outs: m
    # tags: distribution,hypothesistest,ks,locdep,posOnly,raw
    HT_DistributionTest_ks_gam = HCTSAOperation(
        'HT_DistributionTest_ks_gam',
        "HT_DistributionTest(x,'ks','gamma')",
        HT_DistributionTest(theTest='ks', theDistn='gamma'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,l,o
    # outs: g,n
    # tags: distribution,hypothesistest,ks,locdep,posOnly,raw
    HT_DistributionTest_ks_logn = HCTSAOperation(
        'HT_DistributionTest_ks_logn',
        "HT_DistributionTest(x,'ks','logn')",
        HT_DistributionTest(theTest='ks', theDistn='logn'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: k,s,_,w,b
    # outs: l
    # tags: distribution,hypothesistest,ks,locdep,posOnly,raw
    HT_DistributionTest_ks_wbl = HCTSAOperation(
        'HT_DistributionTest_ks_wbl',
        "HT_DistributionTest(x,'ks','wbl')",
        HT_DistributionTest(theTest='ks', theDistn='wbl'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: l,i,l,l,i
    # outs: e,_,n,o,r
    # outs: m
    # tags: distribution,hypothesistest,lillie,raw
    HT_DistributionTest_lillie_norm = HCTSAOperation(
        'HT_DistributionTest_lillie_norm',
        "HT_DistributionTest(x,'lillie','norm')",
        HT_DistributionTest(theTest='lillie', theDistn='norm'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: l,i,l,l,i
    # outs: e,_,e,v
    # tags: distribution,hypothesistest,lillie,raw
    HT_DistributionTest_lillie_ev = HCTSAOperation(
        'HT_DistributionTest_lillie_ev',
        "HT_DistributionTest(x,'lillie','ev')",
        HT_DistributionTest(theTest='lillie', theDistn='ev'))

    # outs: H,T,_,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: T,e,s,t,_
    # outs: l,i,l,l,i
    # outs: e,_,e,x,p
    # tags: distribution,hypothesistest,lillie,locdep,posOnly,raw
    HT_DistributionTest_lillie_exp = HCTSAOperation(
        'HT_DistributionTest_lillie_exp',
        "HT_DistributionTest(x,'lillie','exp')",
        HT_DistributionTest(theTest='lillie', theDistn='exp'))

    # outs: H,T,_,H,y
    # outs: p,o,t,h,e
    # outs: s,i,s,T,e
    # outs: s,t,_,s,i
    # outs: g,n,t,e,s
    # outs: t
    # tags: hypothesistest,signtest
    HT_HypothesisTest_signtest = HCTSAOperation(
        'HT_HypothesisTest_signtest',
        "HT_HypothesisTest(y,'signtest')",
        HT_HypothesisTest(theTest='signtest'))

    # outs: H,T,_,H,y
    # outs: p,o,t,h,e
    # outs: s,i,s,T,e
    # outs: s,t,_,r,u
    # outs: n,s,t,e,s
    # outs: t
    # tags: hypothesistest,randomness,raw
    HT_HypothesisTest_runstest = HCTSAOperation(
        'HT_HypothesisTest_runstest',
        "HT_HypothesisTest(x,'runstest')",
        HT_HypothesisTest(theTest='runstest'))

    # outs: H,T,_,H,y
    # outs: p,o,t,h,e
    # outs: s,i,s,T,e
    # outs: s,t,_,l,b
    # outs: q,t,e,s,t
    # tags: econometricstoolbox,hypothesistest,randomness
    HT_HypothesisTest_lbqtest = HCTSAOperation(
        'HT_HypothesisTest_lbqtest',
        "HT_HypothesisTest(y,'lbq')",
        HT_HypothesisTest(theTest='lbq'))

    # outs: H,T,_,H,y
    # outs: p,o,t,h,e
    # outs: s,i,s,T,e
    # outs: s,t,_,l,b
    # outs: q,t,e,s,t
    # outs: r,a,w
    # tags: econometricstoolbox,hypothesistest,randomness,raw
    HT_HypothesisTest_lbqtestraw = HCTSAOperation(
        'HT_HypothesisTest_lbqtestraw',
        "HT_HypothesisTest(x,'lbq')",
        HT_HypothesisTest(theTest='lbq'))

    # outs: H,T,_,H,y
    # outs: p,o,t,h,e
    # outs: s,i,s,T,e
    # outs: s,t,_,z,t
    # outs: e,s,t
    # tags: hypothesistest
    HT_HypothesisTest_ztest = HCTSAOperation(
        'HT_HypothesisTest_ztest',
        "HT_HypothesisTest(y,'ztest')",
        HT_HypothesisTest(theTest='ztest'))

    # outs: H,T,_,H,y
    # outs: p,o,t,h,e
    # outs: s,i,s,T,e
    # outs: s,t,_,s,i
    # outs: g,n,r,a,n
    # outs: k
    # tags: hypothesistest
    HT_HypothesisTest_signrank = HCTSAOperation(
        'HT_HypothesisTest_signrank',
        "HT_HypothesisTest(y,'signrank')",
        HT_HypothesisTest(theTest='signrank'))

    # outs: H,T,_,H,y
    # outs: p,o,t,h,e
    # outs: s,i,s,T,e
    # outs: s,t,_,j,b
    # outs: t,e,s,t
    # tags: hypothesistest,raw
    HT_HypothesisTest_jbtest = HCTSAOperation(
        'HT_HypothesisTest_jbtest',
        "HT_HypothesisTest(x,'jbtest')",
        HT_HypothesisTest(theTest='jbtest'))

    # outs: I,N,_,A,u
    # outs: t,o,M,u,t
    # outs: u,a,l,I,n
    # outs: f,o,S,t,a
    # outs: t,s,_,4,0
    # outs: _,g,a,u,s
    # outs: s,i,a,n
    # tags: AMI,correlation,information
    IN_AutoMutualInfoStats_40_gaussian = HCTSAOperation(
        'IN_AutoMutualInfoStats_40_gaussian',
        "IN_AutoMutualInfoStats(y,40,'gaussian')",
        IN_AutoMutualInfoStats(maxTau=40, estMethod='gaussian'))

    # outs: I,N,_,A,u
    # outs: t,o,M,u,t
    # outs: u,a,l,I,n
    # outs: f,o,S,t,a
    # outs: t,s,_,4,0
    # outs: _,k,r,a,s
    # outs: k,o,v,1,_
    # outs: 4
    # tags: AMI,correlation,information
    IN_AutoMutualInfoStats_40_kraskov1_4 = HCTSAOperation(
        'IN_AutoMutualInfoStats_40_kraskov1_4',
        "IN_AutoMutualInfoStats(y,40,'kraskov1','4')",
        IN_AutoMutualInfoStats(maxTau=40, estMethod='kraskov1', extraParam='4'))

    # outs: I,N,_,A,u
    # outs: t,o,M,u,t
    # outs: u,a,l,I,n
    # outs: f,o,S,t,a
    # outs: t,s,_,d,i
    # outs: f,f,_,2,0
    # outs: _,g,a,u,s
    # outs: s,i,a,n
    # tags: AMI,correlation,information
    IN_AutoMutualInfoStats_diff_20_gaussian = HCTSAOperation(
        'IN_AutoMutualInfoStats_diff_20_gaussian',
        "IN_AutoMutualInfoStats(diff(y),20,'gaussian')",
        IN_AutoMutualInfoStats(maxTau=20, estMethod='gaussian'))

    # outs: I,N,_,A,u
    # outs: t,o,M,u,t
    # outs: u,a,l,I,n
    # outs: f,o,S,t,a
    # outs: t,s,_,d,i
    # outs: f,f,_,2,0
    # outs: _,k,r,a,s
    # outs: k,o,v,1,_
    # outs: 4
    # tags: AMI,correlation,information
    IN_AutoMutualInfoStats_diff_20_kraskov1_4 = HCTSAOperation(
        'IN_AutoMutualInfoStats_diff_20_kraskov1_4',
        "IN_AutoMutualInfoStats(diff(y),20,'kraskov1','4')",
        IN_AutoMutualInfoStats(maxTau=20, estMethod='kraskov1', extraParam='4'))

    # outs: M,D,_,h,r
    # outs: v,_,c,l,a
    # outs: s,s,i,c
    # tags: medical
    MD_hrv_classic = HCTSAOperation(
        'MD_hrv_classic',
        'MD_hrv_classic(y)',
        MD_hrv_classic())

    # outs: M,D,_,p,N
    # outs: N,_,r,a,w
    # tags: medical,raw,spreaddep
    MD_pNN_raw = HCTSAOperation(
        'MD_pNN_raw',
        'MD_pNN(x)',
        MD_pNN())

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 1,_,3
    # tags: medical,symbolic
    MD_polvar_1_3 = HCTSAOperation(
        'MD_polvar_1_3',
        'MD_polvar(y,1,3)',
        MD_polvar(d=1, D=3))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 0,5,_,3
    # tags: medical,symbolic
    MD_polvar_05_3 = HCTSAOperation(
        'MD_polvar_05_3',
        'MD_polvar(y,0.5,3)',
        MD_polvar(d=0.5, D=3))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 0,1,_,3
    # tags: medical,symbolic
    MD_polvar_01_3 = HCTSAOperation(
        'MD_polvar_01_3',
        'MD_polvar(y,0.1,3)',
        MD_polvar(d=0.1, D=3))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 1,_,5
    # tags: medical,symbolic
    MD_polvar_1_5 = HCTSAOperation(
        'MD_polvar_1_5',
        'MD_polvar(y,1,5)',
        MD_polvar(d=1, D=5))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 0,5,_,5
    # tags: medical,symbolic
    MD_polvar_05_5 = HCTSAOperation(
        'MD_polvar_05_5',
        'MD_polvar(y,0.5,5)',
        MD_polvar(d=0.5, D=5))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 0,1,_,5
    # tags: medical,symbolic
    MD_polvar_01_5 = HCTSAOperation(
        'MD_polvar_01_5',
        'MD_polvar(y,0.1,5)',
        MD_polvar(d=0.1, D=5))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 1,_,4
    # tags: medical,symbolic
    MD_polvar_1_4 = HCTSAOperation(
        'MD_polvar_1_4',
        'MD_polvar(y,1,4)',
        MD_polvar(d=1, D=4))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 0,5,_,4
    # tags: medical,symbolic
    MD_polvar_05_4 = HCTSAOperation(
        'MD_polvar_05_4',
        'MD_polvar(y,0.5,4)',
        MD_polvar(d=0.5, D=4))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 0,1,_,4
    # tags: medical,symbolic
    MD_polvar_01_4 = HCTSAOperation(
        'MD_polvar_01_4',
        'MD_polvar(y,0.1,4)',
        MD_polvar(d=0.1, D=4))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 1,_,6
    # tags: medical,symbolic
    MD_polvar_1_6 = HCTSAOperation(
        'MD_polvar_1_6',
        'MD_polvar(y,1,6)',
        MD_polvar(d=1, D=6))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 0,5,_,6
    # tags: medical,symbolic
    MD_polvar_05_6 = HCTSAOperation(
        'MD_polvar_05_6',
        'MD_polvar(y,0.5,6)',
        MD_polvar(d=0.5, D=6))

    # outs: M,D,_,p,o
    # outs: l,v,a,r,_
    # outs: 0,1,_,6
    # tags: medical,symbolic
    MD_polvar_01_6 = HCTSAOperation(
        'MD_polvar_01_6',
        'MD_polvar(y,0.1,6)',
        MD_polvar(d=0.1, D=6))

    # outs: M,D,_,r,a
    # outs: w,H,R,V,m
    # outs: e,a,s
    # tags: medical,raw,spreaddep
    MD_rawHRVmeas = HCTSAOperation(
        'MD_rawHRVmeas',
        'MD_rawHRVmeas(x)',
        MD_rawHRVmeas())

    # outs: M,F,_,A,R
    # outs: M,A,_,o,r
    # outs: d,e,r,s,_
    # outs: 1,_,6,_,1
    # outs: _,4
    # tags: arma,model,systemidentificationtoolbox
    MF_ARMA_orders_1_6_1_4 = HCTSAOperation(
        'MF_ARMA_orders_1_6_1_4',
        'MF_ARMA_orders(y,1:6,1:4)',
        MF_ARMA_orders(pr=MatlabSequence('1:6'), qr=MatlabSequence('1:4')))

    # outs: M,F,_,A,R
    # outs: _,a,r,c,o
    # outs: v,_,1
    # tags: ar,gof,model
    MF_AR_arcov_1 = HCTSAOperation(
        'MF_AR_arcov_1',
        'MF_AR_arcov(y,1)',
        MF_AR_arcov(p=1))

    # outs: M,F,_,A,R
    # outs: _,a,r,c,o
    # outs: v,_,2
    # tags: ar,fit,gof,model
    MF_AR_arcov_2 = HCTSAOperation(
        'MF_AR_arcov_2',
        'MF_AR_arcov(y,2)',
        MF_AR_arcov(p=2))

    # outs: M,F,_,A,R
    # outs: _,a,r,c,o
    # outs: v,_,3
    # tags: ar,fit,gof,model
    MF_AR_arcov_3 = HCTSAOperation(
        'MF_AR_arcov_3',
        'MF_AR_arcov(y,3)',
        MF_AR_arcov(p=3))

    # outs: M,F,_,A,R
    # outs: _,a,r,c,o
    # outs: v,_,4
    # tags: ar,fit,gof,model
    MF_AR_arcov_4 = HCTSAOperation(
        'MF_AR_arcov_4',
        'MF_AR_arcov(y,4)',
        MF_AR_arcov(p=4))

    # outs: M,F,_,A,R
    # outs: _,a,r,c,o
    # outs: v,_,5
    # tags: ar,fit,gof,model
    MF_AR_arcov_5 = HCTSAOperation(
        'MF_AR_arcov_5',
        'MF_AR_arcov(y,5)',
        MF_AR_arcov(p=5))

    # outs: M,F,_,C,o
    # outs: m,p,a,r,e
    # outs: A,R,_,1,_
    # outs: 1,0,_,a,l
    # outs: l
    # tags: ar,model,systemidentificationtoolbox
    MF_CompareAR_1_10_all = HCTSAOperation(
        'MF_CompareAR_1_10_all',
        "MF_CompareAR(y,1:10,'all')",
        MF_CompareAR(orders=MatlabSequence('1:10'), testHow='all'))

    # outs: M,F,_,C,o
    # outs: m,p,a,r,e
    # outs: A,R,_,1,_
    # outs: 1,0,_,0,5
    # tags: ar,model,systemidentificationtoolbox
    MF_CompareAR_1_10_05 = HCTSAOperation(
        'MF_CompareAR_1_10_05',
        'MF_CompareAR(y,1:10,0.5)',
        MF_CompareAR(orders=MatlabSequence('1:10'), testHow=0.5))

    # outs: M,F,_,C,o
    # outs: m,p,a,r,e
    # outs: T,e,s,t,S
    # outs: e,t,s,_,y
    # outs: _,a,r,_,4
    # outs: _,r,a,n,d
    # outs: _,2,5,_,0
    # outs: 1,_,1
    # tags: model,prediction,systemidentificationtoolbox
    MF_CompareTestSets_y_ar_4_rand_25_01_1 = HCTSAOperation(
        'MF_CompareTestSets_y_ar_4_rand_25_01_1',
        "MF_CompareTestSets(y,'ar',4,'rand',[25,0.1],1,'default')",
        MF_CompareTestSets(theModel='ar', ordd=4, subsetHow='rand', samplep=array([25. ,  0.1]),
                           steps=1, randomSeed='default'))

    # outs: M,F,_,C,o
    # outs: m,p,a,r,e
    # outs: T,e,s,t,S
    # outs: e,t,s,_,y
    # outs: _,a,r,_,b
    # outs: e,s,t,_,u
    # outs: n,i,f,o,r
    # outs: m,_,2,5,_
    # outs: 0,1,_,1
    # tags: ar,arfit,model,prediction,systemidentificationtoolbox
    MF_CompareTestSets_y_ar_best_uniform_25_01_1 = HCTSAOperation(
        'MF_CompareTestSets_y_ar_best_uniform_25_01_1',
        "MF_CompareTestSets(y,'ar','best','uniform',[25,0.1],1)",
        MF_CompareTestSets(theModel='ar', ordd='best', subsetHow='uniform', samplep=array([25. ,
                            0.1]), steps=1))

    # outs: M,F,_,C,o
    # outs: m,p,a,r,e
    # outs: T,e,s,t,S
    # outs: e,t,s,_,y
    # outs: _,s,s,_,b
    # outs: e,s,t,_,u
    # outs: n,i,f,o,r
    # outs: m,_,2,5,_
    # outs: 0,1,_,1
    # tags: model,prediction,statespace,systemidentificationtoolbox
    MF_CompareTestSets_y_ss_best_uniform_25_01_1 = HCTSAOperation(
        'MF_CompareTestSets_y_ss_best_uniform_25_01_1',
        "MF_CompareTestSets(y,'ss','best','uniform',[25,0.1],1)",
        MF_CompareTestSets(theModel='ss', ordd='best', subsetHow='uniform', samplep=array([25. ,
                            0.1]), steps=1))

    # outs: M,F,_,C,o
    # outs: m,p,a,r,e
    # outs: T,e,s,t,S
    # outs: e,t,s,_,y
    # outs: _,s,s,_,2
    # outs: _,u,n,i,f
    # outs: o,r,m,_,2
    # outs: 5,_,0,1,_
    # outs: 1
    # tags: model,prediction,statespace,systemidentificationtoolbox
    MF_CompareTestSets_y_ss_2_uniform_25_01_1 = HCTSAOperation(
        'MF_CompareTestSets_y_ss_2_uniform_25_01_1',
        "MF_CompareTestSets(y,'ss',2,'uniform',[25,0.1],1)",
        MF_CompareTestSets(theModel='ss', ordd=2, subsetHow='uniform', samplep=array([25. ,  0.1]), steps=1))

    # outs: M,F,_,E,x
    # outs: p,S,m,o,o
    # outs: t,h,i,n,g
    # outs: _,0,5,_,b
    # outs: e,s,t
    # tags: expsmoothing,model
    MF_ExpSmoothing_05_best = HCTSAOperation(
        'MF_ExpSmoothing_05_best',
        "MF_ExpSmoothing(y,0.5,'best')",
        MF_ExpSmoothing(ntrain=0.5, alpha='best'))

    # outs: M,F,_,F,i
    # outs: t,S,u,b,s
    # outs: e,g,m,e,n
    # outs: t,s,_,a,r
    # outs: s,b,c,_,u
    # outs: n,i,f,o,r
    # outs: m,_,2,5,_
    # outs: 0,1
    # tags: ar,arfit,model,prediction,systemidentificationtoolbox
    MF_FitSubsegments_arsbc_uniform_25_01 = HCTSAOperation(
        'MF_FitSubsegments_arsbc_uniform_25_01',
        "MF_FitSubsegments(y,'arsbc',[],'uniform',[25,0.1])",
        MF_FitSubsegments(model='arsbc', order=array([], dtype=float64), subsetHow='uniform', samplep=array([25. ,
                           0.1])))

    # outs: M,F,_,F,i
    # outs: t,S,u,b,s
    # outs: e,g,m,e,n
    # outs: t,s,_,a,r
    # outs: s,b,c,_,r
    # outs: a,n,d,_,2
    # outs: 5,_,0,1
    # tags: ar,arfit,model,prediction,systemidentificationtoolbox
    MF_FitSubsegments_arsbc_rand_25_01 = HCTSAOperation(
        'MF_FitSubsegments_arsbc_rand_25_01',
        "MF_FitSubsegments(y,'arsbc',[],'rand',[25,0.1],'default')",
        MF_FitSubsegments(model='arsbc', order=array([], dtype=float64), subsetHow='rand', samplep=array([25. ,
                           0.1]), randomSeed='default'))

    # outs: M,F,_,F,i
    # outs: t,S,u,b,s
    # outs: e,g,m,e,n
    # outs: t,s,_,a,r
    # outs: _,2,_,u,n
    # outs: i,f,o,r,m
    # outs: _,2,5,_,0
    # outs: 1
    # tags: ar,model,prediction,systemidentificationtoolbox
    MF_FitSubsegments_ar_2_uniform_25_01 = HCTSAOperation(
        'MF_FitSubsegments_ar_2_uniform_25_01',
        "MF_FitSubsegments(y,'ar',2,'uniform',[25,0.1])",
        MF_FitSubsegments(model='ar', order=2, subsetHow='uniform', samplep=array([25. ,  0.1])))

    # outs: M,F,_,F,i
    # outs: t,S,u,b,s
    # outs: e,g,m,e,n
    # outs: t,s,_,s,s
    # outs: _,2,_,u,n
    # outs: i,f,o,r,m
    # outs: _,2,5,_,0
    # outs: 1
    # tags: model,prediction,statespace,systemidentificationtoolbox
    MF_FitSubsegments_ss_2_uniform_25_01 = HCTSAOperation(
        'MF_FitSubsegments_ss_2_uniform_25_01',
        "MF_FitSubsegments(y,'ss',2,'uniform',[25,0.1])",
        MF_FitSubsegments(model='ss', order=2, subsetHow='uniform', samplep=array([25. ,  0.1])))

    # outs: M,F,_,F,i
    # outs: t,S,u,b,s
    # outs: e,g,m,e,n
    # outs: t,s,_,a,r
    # outs: m,a,_,2,_
    # outs: 2,_,u,n,i
    # outs: f,o,r,m,_
    # outs: 2,5,_,0,1
    # tags: arma,model,prediction,systemidentificationtoolbox
    MF_FitSubsegments_arma_2_2_uniform_25_01 = HCTSAOperation(
        'MF_FitSubsegments_arma_2_2_uniform_25_01',
        "MF_FitSubsegments(y,'arma',[2,2],'uniform',[25,0.1])",
        MF_FitSubsegments(model='arma', order=array([2, 2]), subsetHow='uniform', samplep=array([25. ,  0.1])))

    # outs: M,F,_,c,o
    # outs: m,p,a,r,e
    # outs: _,G,A,R,C
    # outs: H,_,a,r,_
    # outs: 1,_,3,_,1
    # outs: _,3
    # tags: aic,bic,econometricstoolbox,garch,model
    MF_compare_GARCH_ar_1_3_1_3 = HCTSAOperation(
        'MF_compare_GARCH_ar_1_3_1_3',
        "MF_GARCHcompare(y,'ar',1:3,1:3)",
        MF_GARCHcompare(preProc='ar', pr=MatlabSequence('1:3'), qr=MatlabSequence('1:3')))

    # outs: M,F,_,G,A
    # outs: R,C,H,f,i
    # outs: t,_,a,r,_
    # outs: P,1,_,Q,1
    # tags: econometricstoolbox,garch,model
    MF_GARCHfit_ar_P1_Q1 = HCTSAOperation(
        'MF_GARCHfit_ar_P1_Q1',
        "MF_GARCHfit(y,'ar',1,1)",
        MF_GARCHfit(preproc='ar', P=1, Q=1))

    # outs: M,F,_,G,A
    # outs: R,C,H,f,i
    # outs: t,_,a,r,_
    # outs: P,1,_,Q,2
    # tags: econometricstoolbox,garch,model
    MF_GARCHfit_ar_P1_Q2 = HCTSAOperation(
        'MF_GARCHfit_ar_P1_Q2',
        "MF_GARCHfit(y,'ar',1,2)",
        MF_GARCHfit(preproc='ar', P=1, Q=2))

    # outs: M,F,_,G,P
    # outs: _,F,i,t,A
    # outs: c,r,o,s,s
    # outs: _,c,o,v,S
    # outs: E,i,s,o,_
    # outs: c,o,v,N,o
    # outs: i,s,e,_,2
    # outs: 0
    # tags: gaussianprocess
    MF_GP_FitAcross_covSEiso_covNoise_20 = HCTSAOperation(
        'MF_GP_FitAcross_covSEiso_covNoise_20',
        "MF_GP_FitAcross(y,{'covSum',{'covSEiso','covNoise'}},20)",
        MF_GP_FitAcross(covFunc=array(['covSum', array(['covSEiso', 'covNoise'], dtype=object)],
                              dtype=object), npoints=20))

    # outs: M,F,_,G,P
    # outs: _,L,o,c,a
    # outs: l,P,r,e,d
    # outs: i,c,t,i,o
    # outs: n,_,c,o,v
    # outs: S,E,i,s,o
    # outs: _,c,o,v,N
    # outs: o,i,s,e,_
    # outs: 5,_,3,_,1
    # outs: 0,_,b,e,f
    # outs: o,r,e,a,f
    # outs: t,e,r
    # tags: gaussianprocess
    MF_GP_LocalPrediction_covSEiso_covNoise_5_3_10_beforeafter = HCTSAOperation(
        'MF_GP_LocalPrediction_covSEiso_covNoise_5_3_10_beforeafter',
        "MF_GP_LocalPrediction(y,{'covSum',{'covSEiso','covNoise'}},5,3,10,'beforeafter')",
        MF_GP_LocalPrediction(covFunc=array(['covSum', array(['covSEiso', 'covNoise'], dtype=object)],
                                    dtype=object), numTrain=5, numTest=3, numPreds=10, pmode='beforeafter'))

    # outs: M,F,_,G,P
    # outs: _,L,o,c,a
    # outs: l,P,r,e,d
    # outs: i,c,t,i,o
    # outs: n,_,c,o,v
    # outs: S,E,i,s,o
    # outs: _,c,o,v,N
    # outs: o,i,s,e,_
    # outs: 1,0,_,3,_
    # outs: 2,0,_,f,r
    # outs: o,m,b,e,f
    # outs: o,r,e
    # tags: gaussianprocess
    MF_GP_LocalPrediction_covSEiso_covNoise_10_3_20_frombefore = HCTSAOperation(
        'MF_GP_LocalPrediction_covSEiso_covNoise_10_3_20_frombefore',
        "MF_GP_LocalPrediction(y,{'covSum',{'covSEiso','covNoise'}},10,3,20,'frombefore')",
        MF_GP_LocalPrediction(covFunc=array(['covSum', array(['covSEiso', 'covNoise'], dtype=object)],
                                    dtype=object), numTrain=10, numTest=3, numPreds=20, pmode='frombefore'))

    # outs: M,F,_,G,P
    # outs: _,L,o,c,a
    # outs: l,P,r,e,d
    # outs: i,c,t,i,o
    # outs: n,_,c,o,v
    # outs: S,E,i,s,o
    # outs: _,c,o,v,N
    # outs: o,i,s,e,_
    # outs: 1,0,_,3,_
    # outs: 2,0,_,r,a
    # outs: n,d,o,m,g
    # outs: a,p
    # tags: gaussianprocess
    MF_GP_LocalPrediction_covSEiso_covNoise_10_3_20_randomgap = HCTSAOperation(
        'MF_GP_LocalPrediction_covSEiso_covNoise_10_3_20_randomgap',
        "MF_GP_LocalPrediction(y,{'covSum',{'covSEiso','covNoise'}},10,3,20,'randomgap','default')",
        MF_GP_LocalPrediction(covFunc=array(['covSum', array(['covSEiso', 'covNoise'], dtype=object)],
                                    dtype=object), numTrain=10, numTest=3, numPreds=20, pmode='randomgap', randomSeed='default'))

    # outs: M,F,_,G,P
    # outs: _,h,y,p,e
    # outs: r,p,a,r,a
    # outs: m,e,t,e,r
    # outs: s,_,c,o,v
    # outs: S,E,i,s,o
    # outs: _,c,o,v,N
    # outs: o,i,s,e,_
    # outs: 1,_,2,0,0
    # outs: _,r,e,s,a
    # outs: m,p,l,e
    # tags: gaussianprocess
    MF_GP_hyperparameters_covSEiso_covNoise_1_200_resample = HCTSAOperation(
        'MF_GP_hyperparameters_covSEiso_covNoise_1_200_resample',
        "MF_GP_hyperparameters(y,{'covSum',{'covSEiso','covNoise'}},1,200,'resample')",
        MF_GP_hyperparameters(covFunc=array(['covSum', array(['covSEiso', 'covNoise'], dtype=object)],
                                    dtype=object), squishorsquash=1, maxN=200, resampleHow='resample'))

    # outs: M,F,_,G,P
    # outs: _,h,y,p,e
    # outs: r,p,a,r,a
    # outs: m,e,t,e,r
    # outs: s,_,c,o,v
    # outs: S,E,i,s,o
    # outs: _,c,o,v,P
    # outs: e,r,i,o,d
    # outs: i,c,_,c,o
    # outs: v,N,o,i,s
    # outs: e,_,1,_,2
    # outs: 0,0,_,r,e
    # outs: s,a,m,p,l
    # outs: e
    # tags: gaussianprocess
    MF_GP_hyperparameters_covSEiso_covPeriodic_covNoise_1_200_resample = HCTSAOperation(
        'MF_GP_hyperparameters_covSEiso_covPeriodic_covNoise_1_200_resample',
        "MF_GP_hyperparameters(y,{'covSum',{'covSEiso','covPeriodic','covNoise'}},1,200,'resample')",
        MF_GP_hyperparameters(covFunc=array(['covSum',
       array(['covSEiso', 'covPeriodic',
                              'covNoise'], dtype=object)],
      dtype=object), squishorsquash=1, maxN=200, resampleHow='resample'))

    # outs: M,F,_,G,P
    # outs: _,h,y,p,e
    # outs: r,p,a,r,a
    # outs: m,e,t,e,r
    # outs: s,_,c,o,v
    # outs: S,E,i,s,o
    # outs: _,c,o,v,N
    # outs: o,i,s,e,_
    # outs: 1,_,2,0,0
    # outs: _,f,i,r,s
    # outs: t
    # tags: gaussianprocess
    MF_GP_hyperparameters_covSEiso_covNoise_1_200_first = HCTSAOperation(
        'MF_GP_hyperparameters_covSEiso_covNoise_1_200_first',
        "MF_GP_hyperparameters(y,{'covSum',{'covSEiso','covNoise'}},1,200,'first')",
        MF_GP_hyperparameters(covFunc=array(['covSum', array(['covSEiso', 'covNoise'], dtype=object)],
                                    dtype=object), squishorsquash=1, maxN=200, resampleHow='first'))

    # outs: M,F,_,G,P
    # outs: _,h,y,p,e
    # outs: r,p,a,r,a
    # outs: m,e,t,e,r
    # outs: s,_,c,o,v
    # outs: S,E,i,s,o
    # outs: _,c,o,v,P
    # outs: e,r,i,o,d
    # outs: i,c,_,c,o
    # outs: v,N,o,i,s
    # outs: e,_,1,_,2
    # outs: 0,0,_,f,i
    # outs: r,s,t
    # tags: gaussianprocess
    MF_GP_hyperparameters_covSEiso_covPeriodic_covNoise_1_200_first = HCTSAOperation(
        'MF_GP_hyperparameters_covSEiso_covPeriodic_covNoise_1_200_first',
        "MF_GP_hyperparameters(y,{'covSum',{'covSEiso','covPeriodic','covNoise'}},1,200,'first')",
        MF_GP_hyperparameters(covFunc=array(['covSum',
       array(['covSEiso', 'covPeriodic',
                              'covNoise'], dtype=object)],
      dtype=object), squishorsquash=1, maxN=200, resampleHow='first'))

    # outs: M,F,_,G,P
    # outs: _,h,y,p,e
    # outs: r,p,a,r,a
    # outs: m,e,t,e,r
    # outs: s,_,c,o,v
    # outs: S,E,i,s,o
    # outs: _,c,o,v,N
    # outs: o,i,s,e,_
    # outs: 1,_,5,0,_
    # outs: r,a,n,d,o
    # outs: m,_,i
    # tags: gaussianprocess
    MF_GP_hyperparameters_covSEiso_covNoise_1_50_random_i = HCTSAOperation(
        'MF_GP_hyperparameters_covSEiso_covNoise_1_50_random_i',
        "MF_GP_hyperparameters(y,{'covSum',{'covSEiso','covNoise'}},1,50,'random_i','default')",
        MF_GP_hyperparameters(covFunc=array(['covSum', array(['covSEiso', 'covNoise'], dtype=object)],
                                    dtype=object), squishorsquash=1, maxN=50, resampleHow='random_i', randomSeed='default'))

    # outs: M,F,_,S,t
    # outs: a,t,e,S,p
    # outs: a,c,e,C,o
    # outs: m,p,O,r,d
    # outs: e,r,_,8
    # tags: model,statespace,systemidentificationtoolbox
    MF_StateSpaceCompOrder_8 = HCTSAOperation(
        'MF_StateSpaceCompOrder_8',
        'MF_StateSpaceCompOrder(y,8)',
        MF_StateSpaceCompOrder(maxOrder=8))

    # outs: M,F,_,S,t
    # outs: a,t,e,S,p
    # outs: a,c,e,_,n
    # outs: 4,s,i,d,_
    # outs: 1,_,0,5,_
    # outs: 1
    # tags: model,statespace,systemidentificationtoolbox
    MF_StateSpace_n4sid_1_05_1 = HCTSAOperation(
        'MF_StateSpace_n4sid_1_05_1',
        'MF_StateSpace_n4sid(y,1,0.5,1)',
        MF_StateSpace_n4sid(ordd=1, ptrain=0.5, steps=1))

    # outs: M,F,_,S,t
    # outs: a,t,e,S,p
    # outs: a,c,e,_,n
    # outs: 4,s,i,d,_
    # outs: 2,_,0,5,_
    # outs: 1
    # tags: model,statespace,systemidentificationtoolbox
    MF_StateSpace_n4sid_2_05_1 = HCTSAOperation(
        'MF_StateSpace_n4sid_2_05_1',
        'MF_StateSpace_n4sid(y,2,0.5,1)',
        MF_StateSpace_n4sid(ordd=2, ptrain=0.5, steps=1))

    # outs: M,F,_,S,t
    # outs: a,t,e,S,p
    # outs: a,c,e,_,n
    # outs: 4,s,i,d,_
    # outs: 3,_,0,5,_
    # outs: 1
    # tags: model,statespace,systemidentificationtoolbox
    MF_StateSpace_n4sid_3_05_1 = HCTSAOperation(
        'MF_StateSpace_n4sid_3_05_1',
        'MF_StateSpace_n4sid(y,3,0.5,1)',
        MF_StateSpace_n4sid(ordd=3, ptrain=0.5, steps=1))

    # outs: M,F,_,a,r
    # outs: f,i,t,_,1
    # outs: _,8,_,s,b
    # outs: c
    # tags: arfit,modelfit
    MF_arfit_1_8_sbc = HCTSAOperation(
        'MF_arfit_1_8_sbc',
        "MF_arfit(y,1,8,'sbc')",
        MF_arfit(pmin=1, pmax=8, selector='sbc'))

    # outs: M,F,_,a,r
    # outs: m,a,x,_,3
    # outs: _,1,_,0,5
    # outs: _,1
    # tags: arma,model,systemidentificationtoolbox
    MF_armax_3_1_05_1 = HCTSAOperation(
        'MF_armax_3_1_05_1',
        'MF_armax(y,[3,1],0.5,1)',
        MF_armax(orders=array([3, 1]), ptrain=0.5, numSteps=1))

    # outs: M,F,_,a,r
    # outs: m,a,x,_,1
    # outs: _,1,_,0,5
    # outs: _,1
    # tags: arma,model,systemidentificationtoolbox
    MF_armax_1_1_05_1 = HCTSAOperation(
        'MF_armax_1_1_05_1',
        'MF_armax(y,[1,1],0.5,1)',
        MF_armax(orders=array([1, 1]), ptrain=0.5, numSteps=1))

    # outs: M,F,_,a,r
    # outs: m,a,x,_,2
    # outs: _,2,_,0,5
    # outs: _,1
    # tags: arma,model,systemidentificationtoolbox
    MF_armax_2_2_05_1 = HCTSAOperation(
        'MF_armax_2_2_05_1',
        'MF_armax(y,[2,2],0.5,1)',
        MF_armax(orders=array([2, 2]), ptrain=0.5, numSteps=1))

    # outs: M,F,_,h,m
    # outs: m,_,C,o,m
    # outs: p,a,r,e,N
    # outs: S,t,a,t,e
    # outs: s,_,0,6,_
    # outs: 2,4
    # tags: gharamani,hmm,model
    MF_hmm_CompareNStates_06_24 = HCTSAOperation(
        'MF_hmm_CompareNStates_06_24',
        'MF_hmm_CompareNStates(y,0.6,2:4)',
        MF_hmm_CompareNStates(trainp=0.6, nstater=MatlabSequence('2:4')))

    # outs: M,F,_,h,m
    # outs: m,_,0,7,_
    # outs: 3
    # tags: gharamani,hmm,model
    MF_hmm_07_3 = HCTSAOperation(
        'MF_hmm_07_3',
        'MF_hmm_fit(y,0.7,3)',
        MF_hmm_fit(trainp=0.7, numStates=3))

    # outs: M,F,_,h,m
    # outs: m,_,0,8,_
    # outs: 2
    # tags: gharamani,hmm,model
    MF_hmm_08_2 = HCTSAOperation(
        'MF_hmm_08_2',
        'MF_hmm_fit(y,0.8,2)',
        MF_hmm_fit(trainp=0.8, numStates=2))

    # outs: M,F,_,s,t
    # outs: e,p,s,_,a
    # outs: h,e,a,d,_
    # outs: a,r,_,2,_
    # outs: 6
    # tags: ar,model,prediction,systemidentificationtoolbox
    MF_steps_ahead_ar_2_6 = HCTSAOperation(
        'MF_steps_ahead_ar_2_6',
        "MF_steps_ahead(y,'ar',2,6)",
        MF_steps_ahead(model='ar', order=2, maxSteps=6))

    # outs: M,F,_,s,t
    # outs: e,p,s,_,a
    # outs: h,e,a,d,_
    # outs: a,r,_,b,e
    # outs: s,t,_,6
    # tags: ar,arfit,model,prediction,systemidentificationtoolbox
    MF_steps_ahead_ar_best_6 = HCTSAOperation(
        'MF_steps_ahead_ar_best_6',
        "MF_steps_ahead(y,'ar','best',6)",
        MF_steps_ahead(model='ar', order='best', maxSteps=6))

    # outs: M,F,_,s,t
    # outs: e,p,s,_,a
    # outs: h,e,a,d,_
    # outs: s,s,_,b,e
    # outs: s,t,_,6
    # tags: model,prediction,statespace,systemidentificationtoolbox
    MF_steps_ahead_ss_best_6 = HCTSAOperation(
        'MF_steps_ahead_ss_best_6',
        "MF_steps_ahead(y,'ss','best',6)",
        MF_steps_ahead(model='ss', order='best', maxSteps=6))

    # outs: M,F,_,s,t
    # outs: e,p,s,_,a
    # outs: h,e,a,d,_
    # outs: a,r,m,a,_
    # outs: 3,_,1,_,6
    # tags: arma,model,prediction,systemidentificationtoolbox
    MF_steps_ahead_arma_3_1_6 = HCTSAOperation(
        'MF_steps_ahead_arma_3_1_6',
        "MF_steps_ahead(y,'arma',[3,1],6)",
        MF_steps_ahead(model='arma', order=array([3, 1]), maxSteps=6))

    # outs: N,L,_,B,o
    # outs: x,C,o,r,r
    # outs: D,i,m,_,5
    # outs: 0,_,a,c,_
    # outs: 5
    # tags: corrdim,correlation,nonlinear,tstool
    NL_BoxCorrDim_50_ac_5 = HCTSAOperation(
        'NL_BoxCorrDim_50_ac_5',
        "NL_BoxCorrDim(y,50,{'ac',5})",
        NL_BoxCorrDim(numBins=50, embedParams=array(['ac', 5], dtype=object)))

    # outs: N,L,_,D,V
    # outs: V,_,3,_,1
    # outs: 0,0,_,2,_
    # outs: 5,0,_,1,0
    # outs: _,d,e,f,a
    # outs: u,l,t
    # tags: delayVectorVariance
    NL_DVV_3_100_2_50_10_default = HCTSAOperation(
        'NL_DVV_3_100_2_50_10_default',
        "NL_DVV(y,3,100,2,50,10,'default')",
        NL_DVV(m=3, numDVs=100, nd=2, Ntv=50, numSurr=10, randomSeed='default'))

    # outs: N,L,_,M,S
    # outs: _,f,n,n,_
    # outs: 1,_,1,0,_
    # outs: m,i,_,5,_
    # outs: 1
    # tags: MichaelSmall,fnn,nonlinear,slow
    NL_MS_fnn_1_10_mi_5_1 = HCTSAOperation(
        'NL_MS_fnn_1_10_mi_5_1',
        "NL_MS_fnn(y,1:10,'mi',5,1)",
        NL_MS_fnn(de=MatlabSequence('1:10'), tau='mi', th=5, kth=1))

    # outs: N,L,_,M,S
    # outs: _,n,l,p,e
    # outs: _,2,_,m,i
    # tags: MichaelSmall,model,nlpe,nonlinear
    NL_MS_nlpe_2_mi = HCTSAOperation(
        'NL_MS_nlpe_2_mi',
        "NL_MS_nlpe(y,2,'mi')",
        NL_MS_nlpe(de=2, tau='mi'))

    # outs: N,L,_,M,S
    # outs: _,n,l,p,e
    # outs: _,3,_,a,c
    # tags: MichaelSmall,model,nlpe,nonlinear
    NL_MS_nlpe_3_ac = HCTSAOperation(
        'NL_MS_nlpe_3_ac',
        "NL_MS_nlpe(y,3,'ac')",
        NL_MS_nlpe(de=3, tau='ac'))

    # outs: N,L,_,M,S
    # outs: _,n,l,p,e
    # outs: _,f,n,n,_
    # outs: m,i
    # tags: MichaelSmall,model,nlpe,nonlinear,slow
    NL_MS_nlpe_fnn_mi = HCTSAOperation(
        'NL_MS_nlpe_fnn_mi',
        "NL_MS_nlpe(y,'fnn','mi')",
        NL_MS_nlpe(de='fnn', tau='mi'))

    # outs: N,L,_,T,I
    # outs: S,E,A,N,_
    # outs: c,1,_,1,_
    # outs: 1,_,7,_,0
    # outs: 0,2,_,0,5
    # tags: dimension,entropy,nonlinear,tisean
    NL_TISEAN_c1_1_1_7_002_05 = HCTSAOperation(
        'NL_TISEAN_c1_1_1_7_002_05',
        'NL_TISEAN_c1(y,1,[1,7],0.02,0.5)',
        NL_TISEAN_c1(tau=1, mmm=array([1, 7]), tsep=0.02, Nref=0.5))

    # outs: N,L,_,T,I
    # outs: S,E,A,N,_
    # outs: c,1,_,1,_
    # outs: 2,_,6,_,2
    # outs: 5,_,0,1
    # tags: dimension,entropy,nonlinear,tisean
    NL_TISEAN_c1_1_2_6_25_01 = HCTSAOperation(
        'NL_TISEAN_c1_1_2_6_25_01',
        'NL_TISEAN_c1(y,1,[2,6],25,0.1)',
        NL_TISEAN_c1(tau=1, mmm=array([2, 6]), tsep=25, Nref=0.1))

    # outs: N,L,_,T,I
    # outs: S,E,A,N,_
    # outs: d,2,_,1,_
    # outs: 1,0,_,0
    # tags: dimension,nonlinear,tisean
    NL_TISEAN_d2_1_10_0 = HCTSAOperation(
        'NL_TISEAN_d2_1_10_0',
        'NL_TISEAN_d2(y,1,10,0)',
        NL_TISEAN_d2(tau=1, maxm=10, theilerWin=0))

    # outs: N,L,_,T,I
    # outs: S,E,A,N,_
    # outs: d,2,_,a,c
    # outs: _,1,0,_,0
    # outs: 0,1
    # tags: dimension,nonlinear,tisean
    NL_TISEAN_d2_ac_10_001 = HCTSAOperation(
        'NL_TISEAN_d2_ac_10_001',
        "NL_TISEAN_d2(y,'ac',10,0.01)",
        NL_TISEAN_d2(tau='ac', maxm=10, theilerWin=0.01))

    # outs: N,L,_,T,I
    # outs: S,E,A,N,_
    # outs: f,n,n,_,a
    # outs: c,_,1,0,_
    # outs: 0,0,5
    # tags: dimension,nonlinear,tisean
    NL_TISEAN_fnn_ac_10_005 = HCTSAOperation(
        'NL_TISEAN_fnn_ac_10_005',
        "NL_TISEAN_fnn(y,'ac',10,0.05,0)",
        NL_TISEAN_fnn(tau='ac', maxm=10, theilerWin=0.05, justBest=0))

    # outs: N,L,_,T,I
    # outs: S,E,A,N,_
    # outs: f,n,n,_,m
    # outs: i,_,1,0,_
    # outs: 0,0,5
    # tags: dimension,nonlinear,tisean
    NL_TISEAN_fnn_mi_10_005 = HCTSAOperation(
        'NL_TISEAN_fnn_mi_10_005',
        "NL_TISEAN_fnn(y,'mi',10,0.05,0)",
        NL_TISEAN_fnn(tau='mi', maxm=10, theilerWin=0.05, justBest=0))

    # outs: N,L,_,T,I
    # outs: S,E,A,N,_
    # outs: f,n,n,_,1
    # outs: _,1,0,_,0
    # outs: 0,5
    # tags: dimension,nonlinear,tisean
    NL_TISEAN_fnn_1_10_005 = HCTSAOperation(
        'NL_TISEAN_fnn_1_10_005',
        'NL_TISEAN_fnn(y,1,10,0.05,0)',
        NL_TISEAN_fnn(tau=1, maxm=10, theilerWin=0.05, justBest=0))

    # outs: N,L,_,T,S
    # outs: T,L,_,F,r
    # outs: a,c,t,a,l
    # outs: D,i,m,e,n
    # outs: s,i,o,n,s
    # outs: _,5,_,2,0
    # outs: _,0,2,_,1
    # outs: _,1,0,_,0
    # outs: _,3,2,_,a
    # outs: c,_,f,n,n
    # outs: m,a,r
    # tags: correlation,dimension,nonlinear,stochastic,tstool
    NL_TSTL_FractalDimensions_5_20_02_1_10_0_32_ac_fnnmar = HCTSAOperation(
        'NL_TSTL_FractalDimensions_5_20_02_1_10_0_32_ac_fnnmar',
        "NL_TSTL_FractalDimensions(y,5,20,0.2,1,10,0,32,{'ac','fnnmar'})",
        NL_TSTL_FractalDimensions(kmin=5, kmax=20, Nref=0.2, gstart=1, gend=10, past=0, steps=32,
                                  embedParams=array(['ac', 'fnnmar'], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,F,r
    # outs: a,c,t,a,l
    # outs: D,i,m,e,n
    # outs: s,i,o,n,s
    # outs: _,5,_,1,0
    # outs: 0,_,0,2,_
    # outs: 1,_,1,0,_
    # outs: 0,_,3,2,_
    # outs: a,c,_,3
    # tags: correlation,dimension,nonlinear,stochastic,tstool
    NL_TSTL_FractalDimensions_5_100_02_1_10_0_32_ac_3 = HCTSAOperation(
        'NL_TSTL_FractalDimensions_5_100_02_1_10_0_32_ac_3',
        "NL_TSTL_FractalDimensions(y,5,100,0.2,1,10,0,32,{'ac',3})",
        NL_TSTL_FractalDimensions(kmin=5, kmax=100, Nref=0.2, gstart=1, gend=10, past=0, steps=32,
                                  embedParams=array(['ac', 3], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,F,r
    # outs: a,c,t,a,l
    # outs: D,i,m,e,n
    # outs: s,i,o,n,s
    # outs: _,2,_,1,0
    # outs: 0,_,0,2,_
    # outs: 1,_,5,_,1
    # outs: 0,_,3,2,_
    # outs: a,c,_,5
    # tags: correlation,dimension,nonlinear,stochastic,tstool
    NL_TSTL_FractalDimensions_2_100_02_1_5_10_32_ac_5 = HCTSAOperation(
        'NL_TSTL_FractalDimensions_2_100_02_1_5_10_32_ac_5',
        "NL_TSTL_FractalDimensions(y,2,100,0.2,1,5,10,32,{'ac',5})",
        NL_TSTL_FractalDimensions(kmin=2, kmax=100, Nref=0.2, gstart=1, gend=5, past=10, steps=32,
                                  embedParams=array(['ac', 5], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,F,r
    # outs: a,c,t,a,l
    # outs: D,i,m,e,n
    # outs: s,i,o,n,s
    # outs: _,2,_,1,0
    # outs: 0,_,0,2,_
    # outs: 1,_,5,_,1
    # outs: 0,_,3,2,_
    # outs: 1,_,5
    # tags: correlation,dimension,nonlinear,stochastic,tstool
    NL_TSTL_FractalDimensions_2_100_02_1_5_10_32_1_5 = HCTSAOperation(
        'NL_TSTL_FractalDimensions_2_100_02_1_5_10_32_1_5',
        'NL_TSTL_FractalDimensions(y,2,100,0.2,1,5,10,32,{1,5})',
        NL_TSTL_FractalDimensions(kmin=2, kmax=100, Nref=0.2, gstart=1, gend=5, past=10, steps=32,
                                  embedParams=array([1, 5], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,F,r
    # outs: a,c,t,a,l
    # outs: D,i,m,e,n
    # outs: s,i,o,n,s
    # outs: _,2,_,1,0
    # outs: _,n,1,_,1
    # outs: _,5,_,1,0
    # outs: _,3,2,_,1
    # outs: _,5
    # tags: correlation,dimension,nonlinear,tstool
    NL_TSTL_FractalDimensions_2_10_n1_1_5_10_32_1_5 = HCTSAOperation(
        'NL_TSTL_FractalDimensions_2_10_n1_1_5_10_32_1_5',
        'NL_TSTL_FractalDimensions(y,2,10,-1,1,5,10,32,{1,5})',
        NL_TSTL_FractalDimensions(kmin=2, kmax=10, Nref=-1, gstart=1, gend=5, past=10, steps=32,
                                  embedParams=array([1, 5], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,G,P
    # outs: C,o,r,r,S
    # outs: u,m,2,_,n
    # outs: 1,_,0,5,_
    # outs: 4,0,_,2,0
    # outs: _,a,c,_,f
    # outs: n,n,m,a,r
    # tags: correlation,corrsum,nonlinear,stochastic,tstool
    NL_TSTL_GPCorrSum2_n1_05_40_20_ac_fnnmar = HCTSAOperation(
        'NL_TSTL_GPCorrSum2_n1_05_40_20_ac_fnnmar',
        "NL_TSTL_GPCorrSum(y,-1,0.5,40,20,{'ac','fnnmar'},1)",
        HCTSA_NL_TSTL_GPCorrSum(Nref=-1, r=0.5, thwin=40, nbins=20, embedParams=array(['ac', 'fnnmar'],
                          dtype=object), doTwo=1))

    # outs: N,L,_,T,S
    # outs: T,L,_,G,P
    # outs: C,o,r,r,S
    # outs: u,m,2,_,n
    # outs: 1,_,0,5,_
    # outs: 1,0,0,_,2
    # outs: 0,_,a,c,_
    # outs: f,n,n,m,a
    # outs: r
    # tags: correlation,corrsum,nonlinear,stochastic,tstool
    NL_TSTL_GPCorrSum2_n1_05_100_20_ac_fnnmar = HCTSAOperation(
        'NL_TSTL_GPCorrSum2_n1_05_100_20_ac_fnnmar',
        "NL_TSTL_GPCorrSum(y,-1,0.5,100,20,{'ac','fnnmar'},1)",
        HCTSA_NL_TSTL_GPCorrSum(Nref=-1, r=0.5, thwin=100, nbins=20, embedParams=array(['ac', 'fnnmar'],
                          dtype=object), doTwo=1))

    # outs: N,L,_,T,S
    # outs: T,L,_,G,P
    # outs: C,o,r,r,S
    # outs: u,m,2,_,n
    # outs: 1,_,0,1,_
    # outs: 4,0,_,2,0
    # outs: _,a,c,_,f
    # outs: n,n,m,a,r
    # tags: correlation,corrsum,nonlinear,stochastic,tstool
    NL_TSTL_GPCorrSum2_n1_01_40_20_ac_fnnmar = HCTSAOperation(
        'NL_TSTL_GPCorrSum2_n1_01_40_20_ac_fnnmar',
        "NL_TSTL_GPCorrSum(y,-1,0.1,40,20,{'ac','fnnmar'},1)",
        HCTSA_NL_TSTL_GPCorrSum(Nref=-1, r=0.1, thwin=40, nbins=20, embedParams=array(['ac', 'fnnmar'],
                          dtype=object), doTwo=1))

    # outs: N,L,_,T,S
    # outs: T,L,_,G,P
    # outs: C,o,r,r,S
    # outs: u,m,2,_,n
    # outs: 1,_,0,1,_
    # outs: 4,0,_,4,0
    # outs: _,a,c,_,f
    # outs: n,n,m,a,r
    # tags: correlation,corrsum,nonlinear,stochastic,tstool
    NL_TSTL_GPCorrSum2_n1_01_40_40_ac_fnnmar = HCTSAOperation(
        'NL_TSTL_GPCorrSum2_n1_01_40_40_ac_fnnmar',
        "NL_TSTL_GPCorrSum(y,-1,0.1,40,40,{'ac','fnnmar'},1)",
        HCTSA_NL_TSTL_GPCorrSum(Nref=-1, r=0.1, thwin=40, nbins=40, embedParams=array(['ac', 'fnnmar'],
                          dtype=object), doTwo=1))

    # outs: N,L,_,T,S
    # outs: T,L,_,L,a
    # outs: r,g,e,s,t
    # outs: L,y,a,p,_
    # outs: n,1,_,0,1
    # outs: _,0,0,1,_
    # outs: 3,_,m,i,_
    # outs: f,n,n,m,a
    # outs: r
    # tags: largelyap,nonlinear,tstool
    NL_TSTL_LargestLyap_n1_01_001_3_mi_fnnmar = HCTSAOperation(
        'NL_TSTL_LargestLyap_n1_01_001_3_mi_fnnmar',
        "NL_TSTL_LargestLyap(y,-1,0.1,0.01,3,{'mi','fnnmar'})",
        NL_TSTL_LargestLyap(Nref=-1, maxtstep=0.1, past=0.01, NNR=3, embedParams=array(['mi', 'fnnmar'],
                            dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,L,a
    # outs: r,g,e,s,t
    # outs: L,y,a,p,_
    # outs: n,1,_,0,1
    # outs: _,0,0,1,_
    # outs: 3,_,1,_,4
    # tags: largelyap,nonlinear,tstool
    NL_TSTL_LargestLyap_n1_01_001_3_1_4 = HCTSAOperation(
        'NL_TSTL_LargestLyap_n1_01_001_3_1_4',
        'NL_TSTL_LargestLyap(y,-1,0.1,0.01,3,{1,4})',
        NL_TSTL_LargestLyap(Nref=-1, maxtstep=0.1, past=0.01, NNR=3, embedParams=array([1, 4], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,P,o
    # outs: i,n,c,a,r
    # outs: e,S,e,c,t
    # outs: i,o,n,_,m
    # outs: a,x,_,m,i
    # tags: nonlinear,poincare,tstool
    NL_TSTL_PoincareSection_max_mi = HCTSAOperation(
        'NL_TSTL_PoincareSection_max_mi',
        "NL_TSTL_PoincareSection(y,'max',{'mi',3})",
        NL_TSTL_PoincareSection(ref='max', embedParams=array(['mi', 3], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,P,o
    # outs: i,n,c,a,r
    # outs: e,S,e,c,t
    # outs: i,o,n,_,m
    # outs: a,x,_,a,c
    # tags: nonlinear,poincare,tstool
    NL_TSTL_PoincareSection_max_ac = HCTSAOperation(
        'NL_TSTL_PoincareSection_max_ac',
        "NL_TSTL_PoincareSection(y,'max',{'ac',3})",
        NL_TSTL_PoincareSection(ref='max', embedParams=array(['ac', 3], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,P,o
    # outs: i,n,c,a,r
    # outs: e,S,e,c,t
    # outs: i,o,n,_,m
    # outs: a,x,_,1
    # tags: nonlinear,poincare,tstool
    NL_TSTL_PoincareSection_max_1 = HCTSAOperation(
        'NL_TSTL_PoincareSection_max_1',
        "NL_TSTL_PoincareSection(y,'max',{1,3})",
        NL_TSTL_PoincareSection(ref='max', embedParams=array([1, 3], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,R,e
    # outs: t,u,r,n,T
    # outs: i,m,e,_,1
    # outs: 0,_,1,_,1
    # outs: _,n,1,_,a
    # outs: c,_,8
    # tags: nonlinear,returntime,tstool
    NL_TSTL_ReturnTime_10_1_1_n1_ac_8 = HCTSAOperation(
        'NL_TSTL_ReturnTime_10_1_1_n1_ac_8',
        "NL_TSTL_ReturnTime(y,10,1,1,-1,{'ac',8})",
        HCTSA_NL_TSTL_ReturnTime(NNR=10, maxT=1, past=1, Nref=-1, embedParams=array(['ac', 8], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,R,e
    # outs: t,u,r,n,T
    # outs: i,m,e,_,5
    # outs: _,1,_,4,0
    # outs: _,n,1,_,1
    # outs: _,8
    # tags: nonlinear,returntime,tstool
    NL_TSTL_ReturnTime_5_1_40_n1_1_8 = HCTSAOperation(
        'NL_TSTL_ReturnTime_5_1_40_n1_1_8',
        'NL_TSTL_ReturnTime(y,5,1,40,-1,{1,8})',
        HCTSA_NL_TSTL_ReturnTime(NNR=5, maxT=1, past=40, Nref=-1, embedParams=array([1, 8], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,R,e
    # outs: t,u,r,n,T
    # outs: i,m,e,_,0
    # outs: 0,5,_,1,_
    # outs: 0,0,5,_,n
    # outs: 1,_,1,_,3
    # tags: nonlinear,returntime,tstool
    NL_TSTL_ReturnTime_005_1_005_n1_1_3 = HCTSAOperation(
        'NL_TSTL_ReturnTime_005_1_005_n1_1_3',
        'NL_TSTL_ReturnTime(y,0.05,1,0.05,-1,{1,3})',
        HCTSA_NL_TSTL_ReturnTime(NNR=0.05, maxT=1, past=0.05, Nref=-1, embedParams=array([1, 3], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,T,a
    # outs: k,e,n,s,E
    # outs: s,t,i,m,a
    # outs: t,o,r,_,n
    # outs: 1,_,0,0,5
    # outs: _,0,0,5,_
    # outs: 1,_,8
    # tags: dimension,nonlinear,scaling,takens,tstool
    NL_TSTL_TakensEstimator_n1_005_005_1_8 = HCTSAOperation(
        'NL_TSTL_TakensEstimator_n1_005_005_1_8',
        'NL_TSTL_TakensEstimator(y,-1,0.05,0.05,{1,8})',
        NL_TSTL_TakensEstimator(Nref=-1, rad=0.05, past=0.05, embedParams=array([1, 8], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,T,a
    # outs: k,e,n,s,E
    # outs: s,t,i,m,a
    # outs: t,o,r,_,n
    # outs: 1,_,0,0,5
    # outs: _,0,0,5,_
    # outs: 1,_,3
    # tags: dimension,nonlinear,scaling,takens,tstool
    NL_TSTL_TakensEstimator_n1_005_005_1_3 = HCTSAOperation(
        'NL_TSTL_TakensEstimator_n1_005_005_1_3',
        'NL_TSTL_TakensEstimator(y,-1,0.05,0.05,{1,3})',
        NL_TSTL_TakensEstimator(Nref=-1, rad=0.05, past=0.05, embedParams=array([1, 3], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,T,a
    # outs: k,e,n,s,E
    # outs: s,t,i,m,a
    # outs: t,o,r,_,n
    # outs: 1,_,0,0,5
    # outs: _,0,0,5,_
    # outs: a,c,_,3
    # tags: dimension,nonlinear,scaling,takens,tstool
    NL_TSTL_TakensEstimator_n1_005_005_ac_3 = HCTSAOperation(
        'NL_TSTL_TakensEstimator_n1_005_005_ac_3',
        "NL_TSTL_TakensEstimator(y,-1,0.05,0.05,{'ac',3})",
        NL_TSTL_TakensEstimator(Nref=-1, rad=0.05, past=0.05, embedParams=array(['ac', 3], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,T,a
    # outs: k,e,n,s,E
    # outs: s,t,i,m,a
    # outs: t,o,r,_,n
    # outs: 1,_,0,0,5
    # outs: _,0,0,5,_
    # outs: m,i,_,3
    # tags: dimension,nonlinear,scaling,takens,tstool
    NL_TSTL_TakensEstimator_n1_005_005_mi_3 = HCTSAOperation(
        'NL_TSTL_TakensEstimator_n1_005_005_mi_3',
        "NL_TSTL_TakensEstimator(y,-1,0.05,0.05,{'mi',3})",
        NL_TSTL_TakensEstimator(Nref=-1, rad=0.05, past=0.05, embedParams=array(['mi', 3], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,T,a
    # outs: k,e,n,s,E
    # outs: s,t,i,m,a
    # outs: t,o,r,_,n
    # outs: 1,_,0,0,5
    # outs: _,0,0,5,_
    # outs: m,i,_,f,n
    # outs: n,m,a,r
    # tags: crptool,dimension,nonlinear,scaling,takens,tstool
    NL_TSTL_TakensEstimator_n1_005_005_mi_fnnmar = HCTSAOperation(
        'NL_TSTL_TakensEstimator_n1_005_005_mi_fnnmar',
        "NL_TSTL_TakensEstimator(y,-1,0.05,0.05,{'mi','fnnmar'},'default')",
        NL_TSTL_TakensEstimator(Nref=-1, rad=0.05, past=0.05, embedParams=array(['mi', 'fnnmar'],
                                dtype=object), randomSeed='default'))

    # outs: N,L,_,T,S
    # outs: T,L,_,T,a
    # outs: k,e,n,s,E
    # outs: s,t,i,m,a
    # outs: t,o,r,_,n
    # outs: 1,_,0,1,_
    # outs: 0,0,5,_,1
    # outs: _,1,0
    # tags: dimension,nonlinear,scaling,takens,tstool
    NL_TSTL_TakensEstimator_n1_01_005_1_10 = HCTSAOperation(
        'NL_TSTL_TakensEstimator_n1_01_005_1_10',
        'NL_TSTL_TakensEstimator(y,-1,0.1,0.05,{1,10})',
        NL_TSTL_TakensEstimator(Nref=-1, rad=0.1, past=0.05, embedParams=array([1, 10], dtype=object)))

    # outs: N,L,_,T,S
    # outs: T,L,_,a,c
    # outs: p,_,m,i,_
    # outs: 1,_,_,1,0
    # tags: acp,correlation,nonlinear
    NL_TSTL_acp_mi_1__10 = HCTSAOperation(
        'NL_TSTL_acp_mi_1__10',
        "NL_TSTL_acp(y,'mi',1,[],10,[])",
        NL_TSTL_acp(tau='mi', past=1, maxDelay=array([], dtype=float64), maxDim=10, Nref=array([],
                    dtype=float64)))

    # outs: N,L,_,T,S
    # outs: T,L,_,a,c
    # outs: p,_,1,_,0
    # outs: 0,1,_,0,2
    # outs: 5,_,1,0,_
    # outs: 0,5
    # tags: acp,correlation,nonlinear
    NL_TSTL_acp_1_001_025_10_05 = HCTSAOperation(
        'NL_TSTL_acp_1_001_025_10_05',
        'NL_TSTL_acp(y,1,0.01,0.25,10,0.5)',
        NL_TSTL_acp(tau=1, past=0.01, maxDelay=0.25, maxDim=10, Nref=0.5))

    # outs: N,L,_,T,S
    # outs: T,L,_,d,i
    # outs: m,e,n,s,i
    # outs: o,n,s,_,5
    # outs: 0,_,a,c,_
    # outs: f,n,n,m,a
    # outs: r
    # tags: dimension,nonlinear,scaling,tstool
    NL_TSTL_dimensions_50_ac_fnnmar = HCTSAOperation(
        'NL_TSTL_dimensions_50_ac_fnnmar',
        "NL_TSTL_dimensions(y,50,{'ac','fnnmar'})",
        NL_TSTL_dimensions(nbins=50, embedParams=array(['ac', 'fnnmar'], dtype=object)))

    # outs: N,L,_,c,r
    # outs: p,t,o,o,l
    # outs: _,f,n,n,_
    # outs: 1,0,_,2,_
    # outs: m,i
    # tags: crptool,dimension,nonlinear
    NL_crptool_fnn_10_2_mi = HCTSAOperation(
        'NL_crptool_fnn_10_2_mi',
        "NL_crptool_fnn(y,10,2,'mi',[],'default')",
        NL_crptool_fnn(maxm=10, r=2, taum='mi', th=array([], dtype=float64), randomSeed='default'))

    # outs: N,L,_,c,r
    # outs: p,t,o,o,l
    # outs: _,f,n,n,_
    # outs: 1,0,_,2,_
    # outs: a,c
    # tags: crptool,dimension,nonlinear
    NL_crptool_fnn_10_2_ac = HCTSAOperation(
        'NL_crptool_fnn_10_2_ac',
        "NL_crptool_fnn(y,10,2,'ac',[],'default')",
        NL_crptool_fnn(maxm=10, r=2, taum='ac', th=array([], dtype=float64), randomSeed='default'))

    # outs: N,L,_,c,r
    # outs: p,t,o,o,l
    # outs: _,f,n,n,_
    # outs: 1,0,_,2,_
    # outs: 1
    # tags: crptool,dimension,nonlinear
    NL_crptool_fnn_10_2_1 = HCTSAOperation(
        'NL_crptool_fnn_10_2_1',
        "NL_crptool_fnn(y,10,2,1,[],'default')",
        NL_crptool_fnn(maxm=10, r=2, taum=1, th=array([], dtype=float64), randomSeed='default'))

    # outs: N,L,_,e,m
    # outs: b,e,d,_,P
    # outs: C,A,_,m,i
    # outs: _,1,0
    # tags: pca,tdembedding
    NL_embed_PCA_mi_10 = HCTSAOperation(
        'NL_embed_PCA_mi_10',
        "NL_embed_PCA(y,'mi',10)",
        NL_embed_PCA(tau='mi', m=10))

    # outs: N,L,_,e,m
    # outs: b,e,d,_,P
    # outs: C,A,_,1,_
    # outs: 1,0
    # tags: pca,tdembedding
    NL_embed_PCA_1_10 = HCTSAOperation(
        'NL_embed_PCA_1_10',
        'NL_embed_PCA(y,1,10)',
        NL_embed_PCA(tau=1, m=10))

    # outs: N,L,_,e,m
    # outs: b,e,d,_,P
    # outs: C,A,_,a,c
    # outs: _,3
    # tags: pca,tdembedding
    NL_embed_PCA_ac_3 = HCTSAOperation(
        'NL_embed_PCA_ac_3',
        "NL_embed_PCA(y,'ac',3)",
        NL_embed_PCA(tau='ac', m=3))

    # outs: N,W,_,V,i
    # outs: s,i,b,i,l
    # outs: i,t,y,G,r
    # outs: a,p,h,_,h
    # outs: o,r,i,z
    # tags: lengthdep,network,visibilitygraph
    NW_VisibilityGraph_horiz = HCTSAOperation(
        'NW_VisibilityGraph_horiz',
        "NW_VisibilityGraph(y,'horiz')",
        NW_VisibilityGraph(meth='horiz'))

    # outs: N,W,_,V,i
    # outs: s,i,b,i,l
    # outs: i,t,y,G,r
    # outs: a,p,h,_,n
    # outs: o,r,m
    # tags: lengthdep,network,visibilitygraph
    NW_VisibilityGraph_norm = HCTSAOperation(
        'NW_VisibilityGraph_norm',
        "NW_VisibilityGraph(y,'norm')",
        NW_VisibilityGraph(meth='norm'))

    # outs: P,D,_,P,e
    # outs: r,i,o,d,i
    # outs: c,i,t,y,W
    # outs: a,n,g
    # tags: periodicity,spline
    PD_PeriodicityWang = HCTSAOperation(
        'PD_PeriodicityWang',
        'PD_PeriodicityWang(y)',
        PD_PeriodicityWang())

    # outs: P,H,_,F,o
    # outs: r,c,e,P,o
    # outs: t,e,n,t,i
    # outs: a,l,_,d,b
    # outs: l,w,e,l,l
    # outs: _,1,_,0,2
    # outs: _,0,1
    # tags: dblwell,dynsys
    PH_ForcePotential_dblwell_1_02_01 = HCTSAOperation(
        'PH_ForcePotential_dblwell_1_02_01',
        "PH_ForcePotential(y,'dblwell',[1,0.2,0.1])",
        PH_ForcePotential(whatPotential='dblwell', params=array([1. , 0.2, 0.1])))

    # outs: P,H,_,F,o
    # outs: r,c,e,P,o
    # outs: t,e,n,t,i
    # outs: a,l,_,d,b
    # outs: l,w,e,l,l
    # outs: _,1,_,0,5
    # outs: _,0,2
    # tags: dblwell,dynsys
    PH_ForcePotential_dblwell_1_05_02 = HCTSAOperation(
        'PH_ForcePotential_dblwell_1_05_02',
        "PH_ForcePotential(y,'dblwell',[1,0.5,0.2])",
        PH_ForcePotential(whatPotential='dblwell', params=array([1. , 0.5, 0.2])))

    # outs: P,H,_,F,o
    # outs: r,c,e,P,o
    # outs: t,e,n,t,i
    # outs: a,l,_,d,b
    # outs: l,w,e,l,l
    # outs: _,2,_,0,0
    # outs: 5,_,0,2
    # tags: dblwell,dynsys
    PH_ForcePotential_dblwell_2_005_02 = HCTSAOperation(
        'PH_ForcePotential_dblwell_2_005_02',
        "PH_ForcePotential(y,'dblwell',[2,0.05,0.2])",
        PH_ForcePotential(whatPotential='dblwell', params=array([2.  , 0.05, 0.2 ])))

    # outs: P,H,_,F,o
    # outs: r,c,e,P,o
    # outs: t,e,n,t,i
    # outs: a,l,_,d,b
    # outs: l,w,e,l,l
    # outs: _,3,_,0,0
    # outs: 1,_,0,1
    # tags: dblwell,dynsys
    PH_ForcePotential_dblwell_3_001_01 = HCTSAOperation(
        'PH_ForcePotential_dblwell_3_001_01',
        "PH_ForcePotential(y,'dblwell',[3,0.01,0.1])",
        PH_ForcePotential(whatPotential='dblwell', params=array([3.  , 0.01, 0.1 ])))

    # outs: P,H,_,F,o
    # outs: r,c,e,P,o
    # outs: t,e,n,t,i
    # outs: a,l,_,s,i
    # outs: n,e,_,3,_
    # outs: 0,5,_,1
    # tags: dynsys,sine
    PH_ForcePotential_sine_3_05_1 = HCTSAOperation(
        'PH_ForcePotential_sine_3_05_1',
        "PH_ForcePotential(y,'sine',[3,0.5,1])",
        PH_ForcePotential(whatPotential='sine', params=array([3. , 0.5, 1. ])))

    # outs: P,H,_,F,o
    # outs: r,c,e,P,o
    # outs: t,e,n,t,i
    # outs: a,l,_,s,i
    # outs: n,e,_,1,_
    # outs: 1,_,1
    # tags: dynsys,sine
    PH_ForcePotential_sine_1_1_1 = HCTSAOperation(
        'PH_ForcePotential_sine_1_1_1',
        "PH_ForcePotential(y,'sine',[1,1,1])",
        PH_ForcePotential(whatPotential='sine', params=array([1, 1, 1])))

    # outs: P,H,_,F,o
    # outs: r,c,e,P,o
    # outs: t,e,n,t,i
    # outs: a,l,_,s,i
    # outs: n,e,_,1,0
    # outs: _,0,0,4,_
    # outs: 1,0
    # tags: dynsys,sine
    PH_ForcePotential_sine_10_004_10 = HCTSAOperation(
        'PH_ForcePotential_sine_10_004_10',
        "PH_ForcePotential(y,'sine',[10,0.04,10])",
        PH_ForcePotential(whatPotential='sine', params=array([10.  ,  0.04, 10.  ])))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: p,r,o,p,_
    # outs: 0,1
    # tags: trend
    PH_Walker_prop_01 = HCTSAOperation(
        'PH_Walker_prop_01',
        "PH_Walker(y,'prop',0.1)",
        PH_Walker(walkerRule='prop', walkerParams=0.1))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: p,r,o,p,_
    # outs: 0,5
    # tags: trend
    PH_Walker_prop_05 = HCTSAOperation(
        'PH_Walker_prop_05',
        "PH_Walker(y,'prop',0.5)",
        PH_Walker(walkerRule='prop', walkerParams=0.5))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: p,r,o,p,_
    # outs: 0,9
    # tags: trend
    PH_Walker_prop_09 = HCTSAOperation(
        'PH_Walker_prop_09',
        "PH_Walker(y,'prop',0.9)",
        PH_Walker(walkerRule='prop', walkerParams=0.9))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: p,r,o,p,_
    # outs: 1,1
    # tags: trend
    PH_Walker_prop_11 = HCTSAOperation(
        'PH_Walker_prop_11',
        "PH_Walker(y,'prop',1.1)",
        PH_Walker(walkerRule='prop', walkerParams=1.1))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: b,i,a,s,p
    # outs: r,o,p,_,0
    # outs: 5,_,0,1
    # tags: trend
    PH_Walker_biasprop_05_01 = HCTSAOperation(
        'PH_Walker_biasprop_05_01',
        "PH_Walker(y,'biasprop',[0.5,0.1])",
        PH_Walker(walkerRule='biasprop', walkerParams=array([0.5, 0.1])))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: b,i,a,s,p
    # outs: r,o,p,_,0
    # outs: 1,_,0,5
    # tags: trend
    PH_Walker_biasprop_01_05 = HCTSAOperation(
        'PH_Walker_biasprop_01_05',
        "PH_Walker(y,'biasprop',[0.1,0.5])",
        PH_Walker(walkerRule='biasprop', walkerParams=array([0.1, 0.5])))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: m,o,m,e,n
    # outs: t,u,m,_,2
    # tags: trend
    PH_Walker_momentum_2 = HCTSAOperation(
        'PH_Walker_momentum_2',
        "PH_Walker(y,'momentum',2)",
        PH_Walker(walkerRule='momentum', walkerParams=2))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: m,o,m,e,n
    # outs: t,u,m,_,5
    # tags: trend
    PH_Walker_momentum_5 = HCTSAOperation(
        'PH_Walker_momentum_5',
        "PH_Walker(y,'momentum',5)",
        PH_Walker(walkerRule='momentum', walkerParams=5))

    # outs: P,H,_,W,a
    # outs: l,k,e,r,_
    # outs: r,u,n,n,i
    # outs: n,g,v,a,r
    # outs: _,1,5,_,5
    # outs: 0
    # tags: trend
    PH_Walker_runningvar_15_50 = HCTSAOperation(
        'PH_Walker_runningvar_15_50',
        "PH_Walker(y,'runningvar',[1.5,50])",
        PH_Walker(walkerRule='runningvar', walkerParams=array([ 1.5, 50. ])))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,p,o,l,y
    # outs: 1
    # tags: preprocessing,raw
    PP_Compare_poly1 = HCTSAOperation(
        'PP_Compare_poly1',
        "PP_Compare(x,'poly1')",
        PP_Compare(detrndmeth='poly1'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,p,o,l,y
    # outs: 2
    # tags: preprocessing,raw
    PP_Compare_poly2 = HCTSAOperation(
        'PP_Compare_poly2',
        "PP_Compare(x,'poly2')",
        PP_Compare(detrndmeth='poly2'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,s,i,n,1
    # tags: locdep,preprocessing,raw
    PP_Compare_sin1 = HCTSAOperation(
        'PP_Compare_sin1',
        "PP_Compare(x,'sin1')",
        PP_Compare(detrndmeth='sin1'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,s,i,n,2
    # tags: locdep,preprocessing,raw
    PP_Compare_sin2 = HCTSAOperation(
        'PP_Compare_sin2',
        "PP_Compare(x,'sin2')",
        PP_Compare(detrndmeth='sin2'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,s,p,l,i
    # outs: n,e,2,4
    # tags: preprocessing,raw,spline
    PP_Compare_spline24 = HCTSAOperation(
        'PP_Compare_spline24',
        "PP_Compare(x,'spline24')",
        PP_Compare(detrndmeth='spline24'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,s,p,l,i
    # outs: n,e,4,4
    # tags: preprocessing,raw,spline
    PP_Compare_spline44 = HCTSAOperation(
        'PP_Compare_spline44',
        "PP_Compare(x,'spline44')",
        PP_Compare(detrndmeth='spline44'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,s,p,l,i
    # outs: n,e,6,4
    # tags: preprocessing,raw,spline
    PP_Compare_spline64 = HCTSAOperation(
        'PP_Compare_spline64',
        "PP_Compare(x,'spline64')",
        PP_Compare(detrndmeth='spline64'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,d,i,f,f
    # outs: 1
    # tags: preprocessing,raw
    PP_Compare_diff1 = HCTSAOperation(
        'PP_Compare_diff1',
        "PP_Compare(x,'diff1')",
        PP_Compare(detrndmeth='diff1'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,d,i,f,f
    # outs: 2
    # tags: preprocessing,raw
    PP_Compare_diff2 = HCTSAOperation(
        'PP_Compare_diff2',
        "PP_Compare(x,'diff2')",
        PP_Compare(detrndmeth='diff2'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,m,e,d,i
    # outs: a,n,f,3
    # tags: locdep,preprocessing,raw
    PP_Compare_medianf3 = HCTSAOperation(
        'PP_Compare_medianf3',
        "PP_Compare(x,'medianf3')",
        PP_Compare(detrndmeth='medianf3'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,m,e,d,i
    # outs: a,n,f,4
    # tags: locdep,preprocessing,raw
    PP_Compare_medianf4 = HCTSAOperation(
        'PP_Compare_medianf4',
        "PP_Compare(x,'medianf4')",
        PP_Compare(detrndmeth='medianf4'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,m,e,d,i
    # outs: a,n,f,1,0
    # tags: locdep,preprocessing,raw
    PP_Compare_medianf10 = HCTSAOperation(
        'PP_Compare_medianf10',
        "PP_Compare(x,'medianf10')",
        PP_Compare(detrndmeth='medianf10'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,r,a,v,2
    # tags: locdep,preprocessing,raw
    PP_Compare_rav2 = HCTSAOperation(
        'PP_Compare_rav2',
        "PP_Compare(x,'rav2')",
        PP_Compare(detrndmeth='rav2'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,r,a,v,3
    # tags: locdep,preprocessing,raw
    PP_Compare_rav3 = HCTSAOperation(
        'PP_Compare_rav3',
        "PP_Compare(x,'rav3')",
        PP_Compare(detrndmeth='rav3'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,r,a,v,4
    # tags: locdep,preprocessing,raw
    PP_Compare_rav4 = HCTSAOperation(
        'PP_Compare_rav4',
        "PP_Compare(x,'rav4')",
        PP_Compare(detrndmeth='rav4'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,r,a,v,1
    # outs: 0
    # tags: locdep,preprocessing,raw
    PP_Compare_rav10 = HCTSAOperation(
        'PP_Compare_rav10',
        "PP_Compare(x,'rav10')",
        PP_Compare(detrndmeth='rav10'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,r,e,s,a
    # outs: m,p,l,e,_
    # outs: 1,_,2
    # tags: locdep,preprocessing,raw
    PP_Compare_resample_1_2 = HCTSAOperation(
        'PP_Compare_resample_1_2',
        "PP_Compare(x,'resample_1_2')",
        PP_Compare(detrndmeth='resample_1_2'))

    # outs: P,P,_,C,o
    # outs: m,p,a,r,e
    # outs: _,r,e,s,a
    # outs: m,p,l,e,_
    # outs: 2,_,1
    # tags: locdep,preprocessing,raw
    PP_Compare_resample_2_1 = HCTSAOperation(
        'PP_Compare_resample_2_1',
        "PP_Compare(x,'resample_2_1')",
        PP_Compare(detrndmeth='resample_2_1'))

    # outs: P,P,_,I,t
    # outs: e,r,a,t,e
    # outs: _,d,i,f,f
    # tags: preprocessing,raw
    PP_Iterate_diff = HCTSAOperation(
        'PP_Iterate_diff',
        "PP_Iterate(x,'diff')",
        PP_Iterate(dtMeth='diff'))

    # outs: P,P,_,M,o
    # outs: d,e,l,F,i
    # outs: t,_,a,r,_
    # outs: 2
    # tags: preprocessing,trend
    PP_ModelFit_ar_2 = HCTSAOperation(
        'PP_ModelFit_ar_2',
        "PP_ModelFit(y,'ar',2,'default')",
        PP_ModelFit(model='ar', order=2, randomSeed='default'))

    # outs: S,B,_,B,i
    # outs: n,a,r,y,S
    # outs: t,a,t,s,_
    # outs: d,i,f,f
    # tags: distribution,stationarity
    SB_BinaryStats_diff = HCTSAOperation(
        'SB_BinaryStats_diff',
        "SB_BinaryStats(y,'diff')",
        SB_BinaryStats(binaryMethod='diff'))

    # outs: S,B,_,B,i
    # outs: n,a,r,y,S
    # outs: t,a,t,s,_
    # outs: m,e,a,n
    # tags: distribution,stationarity
    SB_BinaryStats_mean = HCTSAOperation(
        'SB_BinaryStats_mean',
        "SB_BinaryStats(y,'mean')",
        SB_BinaryStats(binaryMethod='mean'))

    # outs: S,B,_,B,i
    # outs: n,a,r,y,S
    # outs: t,a,t,s,_
    # outs: i,q,r
    # tags: distribution,stationarity
    SB_BinaryStats_iqr = HCTSAOperation(
        'SB_BinaryStats_iqr',
        "SB_BinaryStats(y,'iqr')",
        SB_BinaryStats(binaryMethod='iqr'))

    # outs: S,B,_,B,i
    # outs: n,a,r,y,S
    # outs: t,r,e,t,c
    # outs: h,_,l,s,e
    # outs: q,1
    # tags: binary
    SB_BinaryStretch_lseq1 = HCTSAOperation(
        'SB_BinaryStretch_lseq1',
        "SB_BinaryStretch(y,'lseq1')",
        SB_BinaryStretch(stretchWhat='lseq1'))

    # outs: S,B,_,B,i
    # outs: n,a,r,y,S
    # outs: t,r,e,t,c
    # outs: h,_,l,s,e
    # outs: q,0
    # tags: binary
    SB_BinaryStretch_lseq0 = HCTSAOperation(
        'SB_BinaryStretch_lseq0',
        "SB_BinaryStretch(y,'lseq0')",
        SB_BinaryStretch(stretchWhat='lseq0'))

    # outs: S,B,_,M,o
    # outs: t,i,f,T,h
    # outs: r,e,e,_,q
    # outs: u,a,n,t,i
    # outs: l,e
    # tags: motifs
    SB_MotifThree_quantile = HCTSAOperation(
        'SB_MotifThree_quantile',
        "SB_MotifThree(y,'quantile')",
        SB_MotifThree(cgHow='quantile'))

    # outs: S,B,_,M,o
    # outs: t,i,f,T,h
    # outs: r,e,e,_,d
    # outs: i,f,f,q,u
    # outs: a,n,t
    # tags: motifs
    SB_MotifThree_diffquant = HCTSAOperation(
        'SB_MotifThree_diffquant',
        "SB_MotifThree(y,'diffquant')",
        SB_MotifThree(cgHow='diffquant'))

    # outs: S,B,_,M,o
    # outs: t,i,f,T,w
    # outs: o,_,d,i,f
    # outs: f
    # tags: motifs
    SB_MotifTwo_diff = HCTSAOperation(
        'SB_MotifTwo_diff',
        "SB_MotifTwo(y,'diff')",
        SB_MotifTwo(binarizeHow='diff'))

    # outs: S,B,_,M,o
    # outs: t,i,f,T,w
    # outs: o,_,m,e,a
    # outs: n
    # tags: motifs
    SB_MotifTwo_mean = HCTSAOperation(
        'SB_MotifTwo_mean',
        "SB_MotifTwo(y,'mean')",
        SB_MotifTwo(binarizeHow='mean'))

    # outs: S,B,_,M,o
    # outs: t,i,f,T,w
    # outs: o,_,m,e,d
    # outs: i,a,n
    # tags: motifs
    SB_MotifTwo_median = HCTSAOperation(
        'SB_MotifTwo_median',
        "SB_MotifTwo(y,'median')",
        SB_MotifTwo(binarizeHow='median'))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,M,a
    # outs: t,r,i,x,_
    # outs: 2,1
    # tags: transitionmat
    SB_TransitionMatrix_21 = HCTSAOperation(
        'SB_TransitionMatrix_21',
        "SB_TransitionMatrix(y,'quantile',2,1)",
        SB_TransitionMatrix(howtocg='quantile', numGroups=2, tau=1))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,M,a
    # outs: t,r,i,x,_
    # outs: 3,1
    # tags: transitionmat
    SB_TransitionMatrix_31 = HCTSAOperation(
        'SB_TransitionMatrix_31',
        "SB_TransitionMatrix(y,'quantile',3,1)",
        SB_TransitionMatrix(howtocg='quantile', numGroups=3, tau=1))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,M,a
    # outs: t,r,i,x,_
    # outs: 4,1
    # tags: transitionmat
    SB_TransitionMatrix_41 = HCTSAOperation(
        'SB_TransitionMatrix_41',
        "SB_TransitionMatrix(y,'quantile',4,1)",
        SB_TransitionMatrix(howtocg='quantile', numGroups=4, tau=1))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,M,a
    # outs: t,r,i,x,_
    # outs: 5,1
    # tags: transitionmat
    SB_TransitionMatrix_51 = HCTSAOperation(
        'SB_TransitionMatrix_51',
        "SB_TransitionMatrix(y,'quantile',5,1)",
        SB_TransitionMatrix(howtocg='quantile', numGroups=5, tau=1))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,M,a
    # outs: t,r,i,x,_
    # outs: 2,a,c
    # tags: transitionmat
    SB_TransitionMatrix_2ac = HCTSAOperation(
        'SB_TransitionMatrix_2ac',
        "SB_TransitionMatrix(y,'quantile',2,'ac')",
        SB_TransitionMatrix(howtocg='quantile', numGroups=2, tau='ac'))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,M,a
    # outs: t,r,i,x,_
    # outs: 3,a,c
    # tags: transitionmat
    SB_TransitionMatrix_3ac = HCTSAOperation(
        'SB_TransitionMatrix_3ac',
        "SB_TransitionMatrix(y,'quantile',3,'ac')",
        SB_TransitionMatrix(howtocg='quantile', numGroups=3, tau='ac'))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,M,a
    # outs: t,r,i,x,_
    # outs: 4,a,c
    # tags: transitionmat
    SB_TransitionMatrix_4ac = HCTSAOperation(
        'SB_TransitionMatrix_4ac',
        "SB_TransitionMatrix(y,'quantile',4,'ac')",
        SB_TransitionMatrix(howtocg='quantile', numGroups=4, tau='ac'))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,M,a
    # outs: t,r,i,x,_
    # outs: 5,a,c
    # tags: transitionmat
    SB_TransitionMatrix_5ac = HCTSAOperation(
        'SB_TransitionMatrix_5ac',
        "SB_TransitionMatrix(y,'quantile',5,'ac')",
        SB_TransitionMatrix(howtocg='quantile', numGroups=5, tau='ac'))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,p,A
    # outs: l,p,h,a,b
    # outs: e,t,_,4,0
    # outs: _,1
    # tags: transitionmat
    SB_TransitionpAlphabet_40_1 = HCTSAOperation(
        'SB_TransitionpAlphabet_40_1',
        'SB_TransitionpAlphabet(y,2:40,1)',
        SB_TransitionpAlphabet(numGroups=MatlabSequence('2:40'), tau=1))

    # outs: S,B,_,T,r
    # outs: a,n,s,i,t
    # outs: i,o,n,p,A
    # outs: l,p,h,a,b
    # outs: e,t,_,2,0
    # outs: _,a,c
    # tags: transitionmat
    SB_TransitionpAlphabet_20_ac = HCTSAOperation(
        'SB_TransitionpAlphabet_20_ac',
        "SB_TransitionpAlphabet(y,2:20,'ac')",
        SB_TransitionpAlphabet(numGroups=MatlabSequence('2:20'), tau='ac'))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: n,o,t,h,i
    # outs: n,g,_,5,0
    # outs: _,l,o,g,i
    # tags: fa,scaling
    SC_FluctAnal_2_nothing_50_logi = HCTSAOperation(
        'SC_FluctAnal_2_nothing_50_logi',
        "SC_FluctAnal(y,2,'nothing',50,[],[],1)",
        SC_FluctAnal(q=2, wtf='nothing', tauStep=50, k=array([], dtype=float64), lag=array([], dtype=float64),
                     logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: e,n,d,p,t
    # outs: d,i,f,f,_
    # outs: 5,0,_,l,o
    # outs: g,i
    # tags: fa,scaling
    SC_FluctAnal_2_endptdiff_50_logi = HCTSAOperation(
        'SC_FluctAnal_2_endptdiff_50_logi',
        "SC_FluctAnal(y,2,'endptdiff',50,[],[],1)",
        SC_FluctAnal(q=2, wtf='endptdiff', tauStep=50, k=array([], dtype=float64), lag=array([],
                     dtype=float64), logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: r,a,n,g,e
    # outs: _,5,0,_,l
    # outs: o,g,i
    # tags: fa,scaling
    SC_FluctAnal_2_range_50_logi = HCTSAOperation(
        'SC_FluctAnal_2_range_50_logi',
        "SC_FluctAnal(y,2,'range',50,[],[],1)",
        SC_FluctAnal(q=2, wtf='range', tauStep=50, k=array([], dtype=float64), lag=array([], dtype=float64),
                     logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: s,t,d,_,5
    # outs: 0,_,l,o,g
    # outs: i
    # tags: fa,scaling
    SC_FluctAnal_2_std_50_logi = HCTSAOperation(
        'SC_FluctAnal_2_std_50_logi',
        "SC_FluctAnal(y,2,'std',50,[],[],1)",
        SC_FluctAnal(q=2, wtf='std', tauStep=50, k=array([], dtype=float64), lag=array([], dtype=float64),
                     logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: i,q,r,_,5
    # outs: 0,_,l,o,g
    # outs: i
    # tags: fa,scaling
    SC_FluctAnal_2_iqr_50_logi = HCTSAOperation(
        'SC_FluctAnal_2_iqr_50_logi',
        "SC_FluctAnal(y,2,'iqr',50,[],[],1)",
        SC_FluctAnal(q=2, wtf='iqr', tauStep=50, k=array([], dtype=float64), lag=array([], dtype=float64),
                     logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: r,s,r,a,n
    # outs: g,e,_,5,0
    # outs: _,l,o,g,i
    # tags: rsrange,scaling
    SC_FluctAnal_2_rsrange_50_logi = HCTSAOperation(
        'SC_FluctAnal_2_rsrange_50_logi',
        "SC_FluctAnal(y,2,'rsrange',50,[],[],1)",
        SC_FluctAnal(q=2, wtf='rsrange', tauStep=50, k=array([], dtype=float64), lag=array([], dtype=float64),
                     logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: r,s,r,a,n
    # outs: g,e,f,i,t
    # outs: _,5,0,_,1
    # outs: _,l,o,g,i
    # tags: rsrange,scaling
    SC_FluctAnal_2_rsrangefit_50_1_logi = HCTSAOperation(
        'SC_FluctAnal_2_rsrangefit_50_1_logi',
        "SC_FluctAnal(y,2,'rsrangefit',50,1,[],1)",
        SC_FluctAnal(q=2, wtf='rsrangefit', tauStep=50, k=1, lag=array([], dtype=float64), logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: d,f,a,_,5
    # outs: 0,_,0,_,l
    # outs: o,g,i
    # tags: dfa,scaling
    SC_FluctAnal_2_dfa_50_0_logi = HCTSAOperation(
        'SC_FluctAnal_2_dfa_50_0_logi',
        "SC_FluctAnal(y,2,'dfa',50,0,[],1)",
        SC_FluctAnal(q=2, wtf='dfa', tauStep=50, k=0, lag=array([], dtype=float64), logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: d,f,a,_,5
    # outs: 0,_,1,_,l
    # outs: o,g,i
    # tags: dfa,scaling
    SC_FluctAnal_2_dfa_50_1_logi = HCTSAOperation(
        'SC_FluctAnal_2_dfa_50_1_logi',
        "SC_FluctAnal(y,2,'dfa',50,1,[],1)",
        SC_FluctAnal(q=2, wtf='dfa', tauStep=50, k=1, lag=array([], dtype=float64), logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: d,f,a,_,5
    # outs: 0,_,2,_,l
    # outs: o,g,i
    # tags: dfa,scaling
    SC_FluctAnal_2_dfa_50_2_logi = HCTSAOperation(
        'SC_FluctAnal_2_dfa_50_2_logi',
        "SC_FluctAnal(y,2,'dfa',50,2,[],1)",
        SC_FluctAnal(q=2, wtf='dfa', tauStep=50, k=2, lag=array([], dtype=float64), logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: d,f,a,_,5
    # outs: 0,_,3,_,l
    # outs: o,g,i
    # tags: dfa,scaling
    SC_FluctAnal_2_dfa_50_3_logi = HCTSAOperation(
        'SC_FluctAnal_2_dfa_50_3_logi',
        "SC_FluctAnal(y,2,'dfa',50,3,[],1)",
        SC_FluctAnal(q=2, wtf='dfa', tauStep=50, k=3, lag=array([], dtype=float64), logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: d,f,a,_,5
    # outs: 0,_,1,_,2
    # outs: _,l,o,g,i
    # tags: dfa,scaling
    SC_FluctAnal_2_dfa_50_1_2_logi = HCTSAOperation(
        'SC_FluctAnal_2_dfa_50_1_2_logi',
        "SC_FluctAnal(y,2,'dfa',50,1,2,1)",
        SC_FluctAnal(q=2, wtf='dfa', tauStep=50, k=1, lag=2, logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,2,_
    # outs: d,f,a,_,5
    # outs: 0,_,1,_,3
    # outs: _,l,o,g,i
    # tags: dfa,scaling
    SC_FluctAnal_2_dfa_50_1_3_logi = HCTSAOperation(
        'SC_FluctAnal_2_dfa_50_1_3_logi',
        "SC_FluctAnal(y,2,'dfa',50,1,3,1)",
        SC_FluctAnal(q=2, wtf='dfa', tauStep=50, k=1, lag=3, logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,m,a
    # outs: g,_,2,_,d
    # outs: f,a,_,5,0
    # outs: _,2,_,l,o
    # outs: g,i
    # tags: dfa,scaling
    SC_FluctAnal_mag_2_dfa_50_2_logi = HCTSAOperation(
        'SC_FluctAnal_mag_2_dfa_50_2_logi',
        "SC_FluctAnal(zscore(abs(y)),2,'dfa',50,2,[],1)",
        SC_FluctAnal(q=2, wtf='dfa', tauStep=50, k=2, lag=array([], dtype=float64), logInc=1))

    # outs: S,C,_,F,l
    # outs: u,c,t,A,n
    # outs: a,l,_,s,i
    # outs: g,n,_,2,_
    # outs: d,f,a,_,5
    # outs: 0,_,2,_,l
    # outs: o,g,i
    # tags: dfa,scaling
    SC_FluctAnal_sign_2_dfa_50_2_logi = HCTSAOperation(
        'SC_FluctAnal_sign_2_dfa_50_2_logi',
        "SC_FluctAnal(zscore(sign(y)),2,'dfa',50,2,[],1)",
        SC_FluctAnal(q=2, wtf='dfa', tauStep=50, k=2, lag=array([], dtype=float64), logInc=1))

    # outs: S,C,_,M,M
    # outs: A,_,0,_,_
    # outs: n,5,_,5
    # tags: fractal,scaling
    SC_MMA_0__n5_5 = HCTSAOperation(
        'SC_MMA_0__n5_5',
        'SC_MMA(y,0,[],[-5,5])',
        SC_MMA(doOverlap=0, scaleRange=array([], dtype=float64), qRange=array([-5,  5])))

    # outs: S,C,_,f,a
    # outs: s,t,d,f,a
    # tags: dfa,mex,scaling
    SC_fastdfa = HCTSAOperation(
        'SC_fastdfa',
        'SC_fastdfa(y)',
        SC_fastdfa())

    # outs: S,D,_,S,u
    # outs: r,r,o,g,a
    # outs: t,e,T,e,s
    # outs: t,_,R,P,_
    # outs: 9,9
    # tags: nonlinearity,surrogatedata
    SD_SurrogateTest_RP_99 = HCTSAOperation(
        'SD_SurrogateTest_RP_99',
        "SD_SurrogateTest(y,'RP',99,[],{'ami1','fmmi','o3','tc3'},'default')",
        SD_SurrogateTest(surrMeth='RP', numSurrs=99, extrap=array([], dtype=float64), theTestStat=array(['ami1',
                         'fmmi', 'o3', 'tc3'], dtype=object), randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,1
    # outs: _,1,0,0,_
    # outs: 1,_,t,c,3
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_1_100_1_tc3 = HCTSAOperation(
        'SD_TSTL_surrogates_1_100_1_tc3',
        "SD_TSTL_surrogates(y,1,100,1,'tc3','default')",
        SD_TSTL_surrogates(tau=1, nsurr=100, surrMethod=1, surrfn='tc3', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,m
    # outs: i,_,1,0,0
    # outs: _,1,_,t,c
    # outs: 3
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_mi_100_1_tc3 = HCTSAOperation(
        'SD_TSTL_surrogates_mi_100_1_tc3',
        "SD_TSTL_surrogates(y,'mi',100,1,'tc3','default')",
        SD_TSTL_surrogates(tau='mi', nsurr=100, surrMethod=1, surrfn='tc3', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,1
    # outs: _,1,0,0,_
    # outs: 2,_,t,c,3
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_1_100_2_tc3 = HCTSAOperation(
        'SD_TSTL_surrogates_1_100_2_tc3',
        "SD_TSTL_surrogates(y,1,100,2,'tc3','default')",
        SD_TSTL_surrogates(tau=1, nsurr=100, surrMethod=2, surrfn='tc3', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,m
    # outs: i,_,1,0,0
    # outs: _,2,_,t,c
    # outs: 3
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_mi_100_2_tc3 = HCTSAOperation(
        'SD_TSTL_surrogates_mi_100_2_tc3',
        "SD_TSTL_surrogates(y,'mi',100,2,'tc3','default')",
        SD_TSTL_surrogates(tau='mi', nsurr=100, surrMethod=2, surrfn='tc3', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,1
    # outs: _,1,0,0,_
    # outs: 3,_,t,c,3
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_1_100_3_tc3 = HCTSAOperation(
        'SD_TSTL_surrogates_1_100_3_tc3',
        "SD_TSTL_surrogates(y,1,100,3,'tc3','default')",
        SD_TSTL_surrogates(tau=1, nsurr=100, surrMethod=3, surrfn='tc3', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,m
    # outs: i,_,1,0,0
    # outs: _,3,_,t,c
    # outs: 3
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_mi_100_3_tc3 = HCTSAOperation(
        'SD_TSTL_surrogates_mi_100_3_tc3',
        "SD_TSTL_surrogates(y,'mi',100,3,'tc3','default')",
        SD_TSTL_surrogates(tau='mi', nsurr=100, surrMethod=3, surrfn='tc3', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,1
    # outs: _,1,0,0,_
    # outs: 1,_,t,r,e
    # outs: v
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_1_100_1_trev = HCTSAOperation(
        'SD_TSTL_surrogates_1_100_1_trev',
        "SD_TSTL_surrogates(y,1,100,1,'trev','default')",
        SD_TSTL_surrogates(tau=1, nsurr=100, surrMethod=1, surrfn='trev', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,m
    # outs: i,_,1,0,0
    # outs: _,1,_,t,r
    # outs: e,v
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_mi_100_1_trev = HCTSAOperation(
        'SD_TSTL_surrogates_mi_100_1_trev',
        "SD_TSTL_surrogates(y,'mi',100,1,'trev','default')",
        SD_TSTL_surrogates(tau='mi', nsurr=100, surrMethod=1, surrfn='trev', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,1
    # outs: _,1,0,0,_
    # outs: 2,_,t,r,e
    # outs: v
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_1_100_2_trev = HCTSAOperation(
        'SD_TSTL_surrogates_1_100_2_trev',
        "SD_TSTL_surrogates(y,1,100,2,'trev','default')",
        SD_TSTL_surrogates(tau=1, nsurr=100, surrMethod=2, surrfn='trev', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,m
    # outs: i,_,1,0,0
    # outs: _,2,_,t,r
    # outs: e,v
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_mi_100_2_trev = HCTSAOperation(
        'SD_TSTL_surrogates_mi_100_2_trev',
        "SD_TSTL_surrogates(y,'mi',100,2,'trev','default')",
        SD_TSTL_surrogates(tau='mi', nsurr=100, surrMethod=2, surrfn='trev', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,1
    # outs: _,1,0,0,_
    # outs: 3,_,t,r,e
    # outs: v
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_1_100_3_trev = HCTSAOperation(
        'SD_TSTL_surrogates_1_100_3_trev',
        "SD_TSTL_surrogates(y,1,100,3,'trev','default')",
        SD_TSTL_surrogates(tau=1, nsurr=100, surrMethod=3, surrfn='trev', randomSeed='default'))

    # outs: S,D,_,T,S
    # outs: T,L,_,s,u
    # outs: r,r,o,g,a
    # outs: t,e,s,_,m
    # outs: i,_,1,0,0
    # outs: _,3,_,t,r
    # outs: e,v
    # tags: correlation,nonlinear,surrogate,tstool
    SD_TSTL_surrogates_mi_100_3_trev = HCTSAOperation(
        'SD_TSTL_surrogates_mi_100_3_trev',
        "SD_TSTL_surrogates(y,'mi',100,3,'trev','default')",
        SD_TSTL_surrogates(tau='mi', nsurr=100, surrMethod=3, surrfn='trev', randomSeed='default'))

    # outs: S,P,_,S,u
    # outs: m,m,a,r,i
    # outs: e,s,_,p,g
    # outs: r,a,m,_,h
    # outs: a,m,m
    # tags: spectral
    SP_Summaries_pgram_hamm = HCTSAOperation(
        'SP_Summaries_pgram_hamm',
        "SP_Summaries(y,'periodogram','hamming',[],0)",
        SP_Summaries(psdmeth='periodogram', wmeth='hamming', nf=array([], dtype=float64), dologabs=0))

    # outs: S,P,_,S,u
    # outs: m,m,a,r,i
    # outs: e,s,_,w,e
    # outs: l,c,h,_,r
    # outs: e,c,t
    # tags: spectral
    SP_Summaries_welch_rect = HCTSAOperation(
        'SP_Summaries_welch_rect',
        "SP_Summaries(y,'welch','rect',[],0)",
        SP_Summaries(psdmeth='welch', wmeth='rect', nf=array([], dtype=float64), dologabs=0))

    # outs: S,P,_,S,u
    # outs: m,m,a,r,i
    # outs: e,s,_,f,f
    # outs: t,_,l,o,g
    # outs: d,e,v
    # tags: spectral
    SP_Summaries_fft_logdev = HCTSAOperation(
        'SP_Summaries_fft_logdev',
        "SP_Summaries(y,'fft',[],[],1)",
        SP_Summaries(psdmeth='fft', wmeth=array([], dtype=float64), nf=array([], dtype=float64), dologabs=1))

    # outs: S,T,_,F,i
    # outs: t,P,o,l,y
    # outs: n,o,m,i,a
    # outs: l,_,1
    # tags: raw,spreaddep,trend
    ST_FitPolynomial_1 = HCTSAOperation(
        'ST_FitPolynomial_1',
        'ST_FitPolynomial(x,1)',
        ST_FitPolynomial(k=1))

    # outs: S,T,_,F,i
    # outs: t,P,o,l,y
    # outs: n,o,m,i,a
    # outs: l,_,2
    # tags: raw,spreaddep,trend
    ST_FitPolynomial_2 = HCTSAOperation(
        'ST_FitPolynomial_2',
        'ST_FitPolynomial(x,2)',
        ST_FitPolynomial(k=2))

    # outs: S,T,_,F,i
    # outs: t,P,o,l,y
    # outs: n,o,m,i,a
    # outs: l,_,3
    # tags: raw,spreaddep,trend
    ST_FitPolynomial_3 = HCTSAOperation(
        'ST_FitPolynomial_3',
        'ST_FitPolynomial(x,3)',
        ST_FitPolynomial(k=3))

    # outs: S,T,_,F,i
    # outs: t,P,o,l,y
    # outs: n,o,m,i,a
    # outs: l,_,4
    # tags: raw,spreaddep,trend
    ST_FitPolynomial_4 = HCTSAOperation(
        'ST_FitPolynomial_4',
        'ST_FitPolynomial(x,4)',
        ST_FitPolynomial(k=4))

    # outs: S,T,_,l,e
    # outs: n,g,t,h
    # tags: lengthdep,misc,raw
    ST_length = HCTSAOperation(
        'ST_length',
        'ST_Length(x)',
        ST_Length())

    # outs: S,T,_,L,o
    # outs: c,a,l,E,x
    # outs: t,r,e,m,a
    # outs: _,l,5,0
    # tags: distribution,stationarity
    ST_LocalExtrema_l50 = HCTSAOperation(
        'ST_LocalExtrema_l50',
        "ST_LocalExtrema(y,'l',50)",
        ST_LocalExtrema(lorf='l', n=50))

    # outs: S,T,_,L,o
    # outs: c,a,l,E,x
    # outs: t,r,e,m,a
    # outs: _,l,1,0,0
    # tags: distribution,stationarity
    ST_LocalExtrema_l100 = HCTSAOperation(
        'ST_LocalExtrema_l100',
        "ST_LocalExtrema(y,'l',100)",
        ST_LocalExtrema(lorf='l', n=100))

    # outs: S,T,_,L,o
    # outs: c,a,l,E,x
    # outs: t,r,e,m,a
    # outs: _,n,2,5
    # tags: distribution,stationarity
    ST_LocalExtrema_n25 = HCTSAOperation(
        'ST_LocalExtrema_n25',
        "ST_LocalExtrema(y,'n',25)",
        ST_LocalExtrema(lorf='n', n=25))

    # outs: S,T,_,L,o
    # outs: c,a,l,E,x
    # outs: t,r,e,m,a
    # outs: _,n,5,0
    # tags: distribution,stationarity
    ST_LocalExtrema_n50 = HCTSAOperation(
        'ST_LocalExtrema_n50',
        "ST_LocalExtrema(y,'n',50)",
        ST_LocalExtrema(lorf='n', n=50))

    # outs: S,T,_,L,o
    # outs: c,a,l,E,x
    # outs: t,r,e,m,a
    # outs: _,n,1,0,0
    # tags: distribution,stationarity
    ST_LocalExtrema_n100 = HCTSAOperation(
        'ST_LocalExtrema_n100',
        "ST_LocalExtrema(y,'n',100)",
        ST_LocalExtrema(lorf='n', n=100))

    # outs: S,T,_,M,o
    # outs: m,e,n,t,C
    # outs: o,r,r,_,0
    # outs: 0,2,_,0,2
    # outs: _,m,e,a,n
    # outs: _,s,t,d,_
    # outs: n,o,n,e
    # tags: statistics
    ST_MomentCorr_002_02_mean_std_none = HCTSAOperation(
        'ST_MomentCorr_002_02_mean_std_none',
        "ST_MomentCorr(y,0.02,0.2,'mean','std','none')",
        ST_MomentCorr(windowLength=0.02, wOverlap=0.2, mom1='mean', mom2='std', whatTransform='none'))

    # outs: S,T,_,M,o
    # outs: m,e,n,t,C
    # outs: o,r,r,_,0
    # outs: 0,2,_,0,2
    # outs: _,m,e,d,i
    # outs: a,n,_,i,q
    # outs: r,_,n,o,n
    # outs: e
    # tags: statistics
    ST_MomentCorr_002_02_median_iqr_none = HCTSAOperation(
        'ST_MomentCorr_002_02_median_iqr_none',
        "ST_MomentCorr(y,0.02,0.2,'median','iqr','none')",
        ST_MomentCorr(windowLength=0.02, wOverlap=0.2, mom1='median', mom2='iqr', whatTransform='none'))

    # outs: S,T,_,M,o
    # outs: m,e,n,t,C
    # outs: o,r,r,_,0
    # outs: 0,2,_,0,2
    # outs: _,m,e,a,n
    # outs: _,s,t,d,_
    # outs: a,b,s
    # tags: statistics
    ST_MomentCorr_002_02_mean_std_abs = HCTSAOperation(
        'ST_MomentCorr_002_02_mean_std_abs',
        "ST_MomentCorr(y,0.02,0.2,'mean','std','abs')",
        ST_MomentCorr(windowLength=0.02, wOverlap=0.2, mom1='mean', mom2='std', whatTransform='abs'))

    # outs: S,T,_,M,o
    # outs: m,e,n,t,C
    # outs: o,r,r,_,0
    # outs: 0,2,_,0,2
    # outs: _,m,e,d,i
    # outs: a,n,_,i,q
    # outs: r,_,a,b,s
    # tags: statistics
    ST_MomentCorr_002_02_median_iqr_abs = HCTSAOperation(
        'ST_MomentCorr_002_02_median_iqr_abs',
        "ST_MomentCorr(y,0.02,0.2,'median','iqr','abs')",
        ST_MomentCorr(windowLength=0.02, wOverlap=0.2, mom1='median', mom2='iqr', whatTransform='abs'))

    # outs: S,T,_,M,o
    # outs: m,e,n,t,C
    # outs: o,r,r,_,0
    # outs: 0,2,_,0,2
    # outs: _,m,e,a,n
    # outs: _,s,t,d,_
    # outs: s,q,r,t
    # tags: statistics
    ST_MomentCorr_002_02_mean_std_sqrt = HCTSAOperation(
        'ST_MomentCorr_002_02_mean_std_sqrt',
        "ST_MomentCorr(y,0.02,0.2,'mean','std','sqrt')",
        ST_MomentCorr(windowLength=0.02, wOverlap=0.2, mom1='mean', mom2='std', whatTransform='sqrt'))

    # outs: S,T,_,M,o
    # outs: m,e,n,t,C
    # outs: o,r,r,_,0
    # outs: 0,2,_,0,2
    # outs: _,m,e,d,i
    # outs: a,n,_,i,q
    # outs: r,_,s,q,r
    # outs: t
    # tags: statistics
    ST_MomentCorr_002_02_median_iqr_sqrt = HCTSAOperation(
        'ST_MomentCorr_002_02_median_iqr_sqrt',
        "ST_MomentCorr(y,0.02,0.2,'median','iqr','sqrt')",
        ST_MomentCorr(windowLength=0.02, wOverlap=0.2, mom1='median', mom2='iqr', whatTransform='sqrt'))

    # outs: S,T,_,S,i
    # outs: m,p,l,e,S
    # outs: t,a,t,s,_
    # outs: z,c,r,o,s
    # outs: s
    # tags: noisiness
    ST_SimpleStats_zcross = HCTSAOperation(
        'ST_SimpleStats_zcross',
        "ST_SimpleStats(y,'zcross')",
        ST_SimpleStats(whatStat='zcross'))

    # outs: S,T,_,S,i
    # outs: m,p,l,e,S
    # outs: t,a,t,s,_
    # outs: p,m,c,r,o
    # outs: s,s
    # tags: distribution
    ST_SimpleStats_pmcross = HCTSAOperation(
        'ST_SimpleStats_pmcross',
        "ST_SimpleStats(y,'pmcross')",
        ST_SimpleStats(whatStat='pmcross'))

    # outs: S,Y,_,D,r
    # outs: i,f,t,i,n
    # outs: g,M,e,a,n
    # outs: 2,0
    # tags: stationarity
    SY_DriftingMean20 = HCTSAOperation(
        'SY_DriftingMean20',
        "SY_DriftingMean(y,'fix',20)",
        SY_DriftingMean(howl='fix', l=20))

    # outs: S,Y,_,D,r
    # outs: i,f,t,i,n
    # outs: g,M,e,a,n
    # outs: 5,0
    # tags: stationarity
    SY_DriftingMean50 = HCTSAOperation(
        'SY_DriftingMean50',
        "SY_DriftingMean(y,'fix',50)",
        SY_DriftingMean(howl='fix', l=50))

    # outs: S,Y,_,D,r
    # outs: i,f,t,i,n
    # outs: g,M,e,a,n
    # outs: 1,0,0
    # tags: stationarity
    SY_DriftingMean100 = HCTSAOperation(
        'SY_DriftingMean100',
        "SY_DriftingMean(y,'fix',100)",
        SY_DriftingMean(howl='fix', l=100))

    # outs: S,Y,_,D,r
    # outs: i,f,t,i,n
    # outs: g,M,e,a,n
    # outs: n,5
    # tags: stationarity
    SY_DriftingMeann5 = HCTSAOperation(
        'SY_DriftingMeann5',
        "SY_DriftingMean(y,'num',5)",
        SY_DriftingMean(howl='num', l=5))

    # outs: S,Y,_,D,r
    # outs: i,f,t,i,n
    # outs: g,M,e,a,n
    # outs: n,1,0
    # tags: stationarity
    SY_DriftingMeann10 = HCTSAOperation(
        'SY_DriftingMeann10',
        "SY_DriftingMean(y,'num',10)",
        SY_DriftingMean(howl='num', l=10))

    # outs: S,Y,_,D,y
    # outs: n,W,i,n,1
    # outs: 0
    # tags: stationarity
    SY_DynWin10 = HCTSAOperation(
        'SY_DynWin10',
        'SY_DynWin(y,10)',
        SY_DynWin(maxnseg=10))

    # outs: S,Y,_,K,P
    # outs: S,S,t,e,s
    # outs: t,_,0
    # tags: econometricstoolbox,hypothesistest,kpsstest,pvalue,stationarity
    SY_KPSStest_0 = HCTSAOperation(
        'SY_KPSStest_0',
        'SY_KPSStest(y,0)',
        SY_KPSStest(lags=0))

    # outs: S,Y,_,K,P
    # outs: S,S,t,e,s
    # outs: t,_,1
    # tags: econometricstoolbox,hypothesistest,kpsstest,pvalue,stationarity
    SY_KPSStest_1 = HCTSAOperation(
        'SY_KPSStest_1',
        'SY_KPSStest(y,1)',
        SY_KPSStest(lags=1))

    # outs: S,Y,_,K,P
    # outs: S,S,t,e,s
    # outs: t,_,2
    # tags: econometricstoolbox,hypothesistest,kpsstest,pvalue,stationarity
    SY_KPSStest_2 = HCTSAOperation(
        'SY_KPSStest_2',
        'SY_KPSStest(y,2)',
        SY_KPSStest(lags=2))

    # outs: S,Y,_,K,P
    # outs: S,S,t,e,s
    # outs: t,_,0,_,1
    # outs: 0
    # tags: econometricstoolbox,hypothesistest,kpsstest,pvalue,stationarity
    SY_KPSStest_0_10 = HCTSAOperation(
        'SY_KPSStest_0_10',
        'SY_KPSStest(y,0:10)',
        SY_KPSStest(lags=MatlabSequence('0:10')))

    # outs: S,Y,_,L,o
    # outs: c,a,l,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: s,_,2,_,e
    # outs: a,c,h
    # tags: stationarity
    SY_LocalDistributions_2_each = HCTSAOperation(
        'SY_LocalDistributions_2_each',
        "SY_LocalDistributions(y,2,'each')",
        SY_LocalDistributions(nseg=2, eachOrPar='each'))

    # outs: S,Y,_,L,o
    # outs: c,a,l,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: s,_,2,_,p
    # outs: a,r
    # tags: stationarity
    SY_LocalDistributions_2_par = HCTSAOperation(
        'SY_LocalDistributions_2_par',
        "SY_LocalDistributions(y,2,'par')",
        SY_LocalDistributions(nseg=2, eachOrPar='par'))

    # outs: S,Y,_,L,o
    # outs: c,a,l,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: s,_,3,_,p
    # outs: a,r
    # tags: stationarity
    SY_LocalDistributions_3_par = HCTSAOperation(
        'SY_LocalDistributions_3_par',
        "SY_LocalDistributions(y,3,'par')",
        SY_LocalDistributions(nseg=3, eachOrPar='par'))

    # outs: S,Y,_,L,o
    # outs: c,a,l,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: s,_,4,_,p
    # outs: a,r
    # tags: stationarity
    SY_LocalDistributions_4_par = HCTSAOperation(
        'SY_LocalDistributions_4_par',
        "SY_LocalDistributions(y,4,'par')",
        SY_LocalDistributions(nseg=4, eachOrPar='par'))

    # outs: S,Y,_,L,o
    # outs: c,a,l,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: s,_,5,_,p
    # outs: a,r
    # tags: stationarity
    SY_LocalDistributions_5_par = HCTSAOperation(
        'SY_LocalDistributions_5_par',
        "SY_LocalDistributions(y,5,'par')",
        SY_LocalDistributions(nseg=5, eachOrPar='par'))

    # outs: S,Y,_,L,o
    # outs: c,a,l,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: s,_,3,_,e
    # outs: a,c,h
    # tags: stationarity
    SY_LocalDistributions_3_each = HCTSAOperation(
        'SY_LocalDistributions_3_each',
        "SY_LocalDistributions(y,3,'each')",
        SY_LocalDistributions(nseg=3, eachOrPar='each'))

    # outs: S,Y,_,L,o
    # outs: c,a,l,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: s,_,4,_,e
    # outs: a,c,h
    # tags: stationarity
    SY_LocalDistributions_4_each = HCTSAOperation(
        'SY_LocalDistributions_4_each',
        "SY_LocalDistributions(y,4,'each')",
        SY_LocalDistributions(nseg=4, eachOrPar='each'))

    # outs: S,Y,_,L,o
    # outs: c,a,l,D,i
    # outs: s,t,r,i,b
    # outs: u,t,i,o,n
    # outs: s,_,5,_,e
    # outs: a,c,h
    # tags: stationarity
    SY_LocalDistributions_5_each = HCTSAOperation(
        'SY_LocalDistributions_5_each',
        "SY_LocalDistributions(y,5,'each')",
        SY_LocalDistributions(nseg=5, eachOrPar='each'))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: l,1,0
    # tags: stationarity
    SY_LocalGlobal_l10 = HCTSAOperation(
        'SY_LocalGlobal_l10',
        "SY_LocalGlobal(y,'l',10)",
        SY_LocalGlobal(subsetHow='l', n=10))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: l,2,0
    # tags: stationarity
    SY_LocalGlobal_l20 = HCTSAOperation(
        'SY_LocalGlobal_l20',
        "SY_LocalGlobal(y,'l',20)",
        SY_LocalGlobal(subsetHow='l', n=20))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: l,5,0
    # tags: stationarity
    SY_LocalGlobal_l50 = HCTSAOperation(
        'SY_LocalGlobal_l50',
        "SY_LocalGlobal(y,'l',50)",
        SY_LocalGlobal(subsetHow='l', n=50))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: l,1,0,0
    # tags: stationarity
    SY_LocalGlobal_l100 = HCTSAOperation(
        'SY_LocalGlobal_l100',
        "SY_LocalGlobal(y,'l',100)",
        SY_LocalGlobal(subsetHow='l', n=100))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: l,5,0,0
    # tags: stationarity
    SY_LocalGlobal_l500 = HCTSAOperation(
        'SY_LocalGlobal_l500',
        "SY_LocalGlobal(y,'l',500)",
        SY_LocalGlobal(subsetHow='l', n=500))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: p,0,1
    # tags: stationarity
    SY_LocalGlobal_p01 = HCTSAOperation(
        'SY_LocalGlobal_p01',
        "SY_LocalGlobal(y,'p',0.01)",
        SY_LocalGlobal(subsetHow='p', n=0.01))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: p,0,5
    # tags: stationarity
    SY_LocalGlobal_p05 = HCTSAOperation(
        'SY_LocalGlobal_p05',
        "SY_LocalGlobal(y,'p',0.05)",
        SY_LocalGlobal(subsetHow='p', n=0.05))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: p,1
    # tags: stationarity
    SY_LocalGlobal_p1 = HCTSAOperation(
        'SY_LocalGlobal_p1',
        "SY_LocalGlobal(y,'p',0.1)",
        SY_LocalGlobal(subsetHow='p', n=0.1))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: p,5
    # tags: stationarity
    SY_LocalGlobal_p5 = HCTSAOperation(
        'SY_LocalGlobal_p5',
        "SY_LocalGlobal(y,'p',0.5)",
        SY_LocalGlobal(subsetHow='p', n=0.5))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: u,n,i,c,g
    # outs: 1,0
    # tags: stationarity
    SY_LocalGlobal_unicg10 = HCTSAOperation(
        'SY_LocalGlobal_unicg10',
        "SY_LocalGlobal(y,'unicg',10)",
        SY_LocalGlobal(subsetHow='unicg', n=10))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: u,n,i,c,g
    # outs: 2,0
    # tags: stationarity
    SY_LocalGlobal_unicg20 = HCTSAOperation(
        'SY_LocalGlobal_unicg20',
        "SY_LocalGlobal(y,'unicg',20)",
        SY_LocalGlobal(subsetHow='unicg', n=20))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: u,n,i,c,g
    # outs: 5,0
    # tags: stationarity
    SY_LocalGlobal_unicg50 = HCTSAOperation(
        'SY_LocalGlobal_unicg50',
        "SY_LocalGlobal(y,'unicg',50)",
        SY_LocalGlobal(subsetHow='unicg', n=50))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: u,n,i,c,g
    # outs: 1,0,0
    # tags: stationarity
    SY_LocalGlobal_unicg100 = HCTSAOperation(
        'SY_LocalGlobal_unicg100',
        "SY_LocalGlobal(y,'unicg',100)",
        SY_LocalGlobal(subsetHow='unicg', n=100))

    # outs: S,Y,_,L,o
    # outs: c,a,l,G,l
    # outs: o,b,a,l,_
    # outs: u,n,i,c,g
    # outs: 5,0,0
    # tags: stationarity
    SY_LocalGlobal_unicg500 = HCTSAOperation(
        'SY_LocalGlobal_unicg500',
        "SY_LocalGlobal(y,'unicg',500)",
        SY_LocalGlobal(subsetHow='unicg', n=500))

    # outs: S,Y,_,P,P
    # outs: t,e,s,t,_
    # outs: 0,_,5,_,a
    # outs: r,_,t,1
    # tags: bic,econometricstoolbox,pptest,pvalue,rmse,unitroot
    SY_PPtest_0_5_ar_t1 = HCTSAOperation(
        'SY_PPtest_0_5_ar_t1',
        "SY_PPtest(y,0:5,'ar','t1')",
        SY_PPtest(lags=MatlabSequence('0:5'), model='ar', testStatistic='t1'))

    # outs: S,Y,_,R,a
    # outs: n,g,e,E,v
    # outs: o,l,v,e
    # tags: stationarity
    SY_RangeEvolve = HCTSAOperation(
        'SY_RangeEvolve',
        'SY_RangeEvolve(y)',
        SY_RangeEvolve())

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,2,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_2_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_2_1',
        "SY_SlidingWindow(y,'mean','std',2,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,3,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_3_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_3_1',
        "SY_SlidingWindow(y,'mean','std',3,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=3, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,4,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_4_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_4_1',
        "SY_SlidingWindow(y,'mean','std',4,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=4, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,5,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_5_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_5_1',
        "SY_SlidingWindow(y,'mean','std',5,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,6,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_6_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_6_1',
        "SY_SlidingWindow(y,'mean','std',6,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=6, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,7,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_7_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_7_1',
        "SY_SlidingWindow(y,'mean','std',7,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=7, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,8,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_8_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_8_1',
        "SY_SlidingWindow(y,'mean','std',8,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=8, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,9,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_9_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_9_1',
        "SY_SlidingWindow(y,'mean','std',9,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=9, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: _,1,0,_,1
    # tags: StatAv,slidingwin,stationarity
    SY_SlidingWindow_m_s_10_1 = HCTSAOperation(
        'SY_SlidingWindow_m_s_10_1',
        "SY_SlidingWindow(y,'mean','std',10,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: 2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_s2_2 = HCTSAOperation(
        'SY_SlidingWindow_m_s2_2',
        "SY_SlidingWindow(y,'mean','std',2,2)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: 2,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_s2_10 = HCTSAOperation(
        'SY_SlidingWindow_m_s2_10',
        "SY_SlidingWindow(y,'mean','std',2,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: 5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_s5_2 = HCTSAOperation(
        'SY_SlidingWindow_m_s5_2',
        "SY_SlidingWindow(y,'mean','std',5,2)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: 5,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_s5_10 = HCTSAOperation(
        'SY_SlidingWindow_m_s5_10',
        "SY_SlidingWindow(y,'mean','std',5,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: 1,0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_s10_2 = HCTSAOperation(
        'SY_SlidingWindow_m_s10_2',
        "SY_SlidingWindow(y,'mean','std',10,2)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: 1,0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_s10_10 = HCTSAOperation(
        'SY_SlidingWindow_m_s10_10',
        "SY_SlidingWindow(y,'mean','std',10,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='std', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: a,m,p,e,n
    # outs: 2,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_sampen2_10 = HCTSAOperation(
        'SY_SlidingWindow_m_sampen2_10',
        "SY_SlidingWindow(y,'mean','sampen',2,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='sampen', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: a,m,p,e,n
    # outs: 5,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_sampen5_10 = HCTSAOperation(
        'SY_SlidingWindow_m_sampen5_10',
        "SY_SlidingWindow(y,'mean','sampen',5,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='sampen', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_sampen10_1 = HCTSAOperation(
        'SY_SlidingWindow_m_sampen10_1',
        "SY_SlidingWindow(y,'mean','sampen',10,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='sampen', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_sampen10_2 = HCTSAOperation(
        'SY_SlidingWindow_m_sampen10_2',
        "SY_SlidingWindow(y,'mean','sampen',10,2)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='sampen', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_sampen10_10 = HCTSAOperation(
        'SY_SlidingWindow_m_sampen10_10',
        "SY_SlidingWindow(y,'mean','sampen',10,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='sampen', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent2_1 = HCTSAOperation(
        'SY_SlidingWindow_m_ent2_1',
        "SY_SlidingWindow(y,'mean','ent',2,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent2_2 = HCTSAOperation(
        'SY_SlidingWindow_m_ent2_2',
        "SY_SlidingWindow(y,'mean','ent',2,2)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent2_10 = HCTSAOperation(
        'SY_SlidingWindow_m_ent2_10',
        "SY_SlidingWindow(y,'mean','ent',2,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent5_1 = HCTSAOperation(
        'SY_SlidingWindow_m_ent5_1',
        "SY_SlidingWindow(y,'mean','ent',5,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent5_2 = HCTSAOperation(
        'SY_SlidingWindow_m_ent5_2',
        "SY_SlidingWindow(y,'mean','ent',5,2)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent5_10 = HCTSAOperation(
        'SY_SlidingWindow_m_ent5_10',
        "SY_SlidingWindow(y,'mean','ent',5,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,1,0,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent10_1 = HCTSAOperation(
        'SY_SlidingWindow_m_ent10_1',
        "SY_SlidingWindow(y,'mean','ent',10,1)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent10_2 = HCTSAOperation(
        'SY_SlidingWindow_m_ent10_2',
        "SY_SlidingWindow(y,'mean','ent',10,2)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,_,e
    # outs: n,t,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_m_ent10_10 = HCTSAOperation(
        'SY_SlidingWindow_m_ent10_10',
        "SY_SlidingWindow(y,'mean','ent',10,10)",
        SY_SlidingWindow(windowStat='mean', acrossWinStat='ent', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s2_1 = HCTSAOperation(
        'SY_SlidingWindow_s_s2_1',
        "SY_SlidingWindow(y,'std','std',2,1)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s2_2 = HCTSAOperation(
        'SY_SlidingWindow_s_s2_2',
        "SY_SlidingWindow(y,'std','std',2,2)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 2,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s2_10 = HCTSAOperation(
        'SY_SlidingWindow_s_s2_10',
        "SY_SlidingWindow(y,'std','std',2,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s5_1 = HCTSAOperation(
        'SY_SlidingWindow_s_s5_1',
        "SY_SlidingWindow(y,'std','std',5,1)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s5_2 = HCTSAOperation(
        'SY_SlidingWindow_s_s5_2',
        "SY_SlidingWindow(y,'std','std',5,2)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 5,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s5_10 = HCTSAOperation(
        'SY_SlidingWindow_s_s5_10',
        "SY_SlidingWindow(y,'std','std',5,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 1,0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s10_1 = HCTSAOperation(
        'SY_SlidingWindow_s_s10_1',
        "SY_SlidingWindow(y,'std','std',10,1)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 1,0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s10_2 = HCTSAOperation(
        'SY_SlidingWindow_s_s10_2',
        "SY_SlidingWindow(y,'std','std',10,2)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: 1,0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_s10_10 = HCTSAOperation(
        'SY_SlidingWindow_s_s10_10',
        "SY_SlidingWindow(y,'std','std',10,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='std', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: a,m,p,e,n
    # outs: 2,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_sampen2_10 = HCTSAOperation(
        'SY_SlidingWindow_s_sampen2_10',
        "SY_SlidingWindow(y,'std','sampen',2,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='sampen', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: a,m,p,e,n
    # outs: 5,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_sampen5_10 = HCTSAOperation(
        'SY_SlidingWindow_s_sampen5_10',
        "SY_SlidingWindow(y,'std','sampen',5,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='sampen', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_sampen10_1 = HCTSAOperation(
        'SY_SlidingWindow_s_sampen10_1',
        "SY_SlidingWindow(y,'std','sampen',10,1)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='sampen', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_sampen10_2 = HCTSAOperation(
        'SY_SlidingWindow_s_sampen10_2',
        "SY_SlidingWindow(y,'std','sampen',10,2)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='sampen', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_sampen10_10 = HCTSAOperation(
        'SY_SlidingWindow_s_sampen10_10',
        "SY_SlidingWindow(y,'std','sampen',10,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='sampen', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent2_1 = HCTSAOperation(
        'SY_SlidingWindow_s_ent2_1',
        "SY_SlidingWindow(y,'std','ent',2,1)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent2_2 = HCTSAOperation(
        'SY_SlidingWindow_s_ent2_2',
        "SY_SlidingWindow(y,'std','ent',2,2)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent2_10 = HCTSAOperation(
        'SY_SlidingWindow_s_ent2_10',
        "SY_SlidingWindow(y,'std','ent',2,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent5_1 = HCTSAOperation(
        'SY_SlidingWindow_s_ent5_1',
        "SY_SlidingWindow(y,'std','ent',5,1)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent5_2 = HCTSAOperation(
        'SY_SlidingWindow_s_ent5_2',
        "SY_SlidingWindow(y,'std','ent',5,2)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent5_10 = HCTSAOperation(
        'SY_SlidingWindow_s_ent5_10',
        "SY_SlidingWindow(y,'std','ent',5,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,1,0,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent10_1 = HCTSAOperation(
        'SY_SlidingWindow_s_ent10_1',
        "SY_SlidingWindow(y,'std','ent',10,1)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent10_2 = HCTSAOperation(
        'SY_SlidingWindow_s_ent10_2',
        "SY_SlidingWindow(y,'std','ent',10,2)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,_,e
    # outs: n,t,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_s_ent10_10 = HCTSAOperation(
        'SY_SlidingWindow_s_ent10_10',
        "SY_SlidingWindow(y,'std','ent',10,10)",
        SY_SlidingWindow(windowStat='std', acrossWinStat='ent', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s2_1 = HCTSAOperation(
        'SY_SlidingWindow_ent_s2_1',
        "SY_SlidingWindow(y,'ent','std',2,1)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s2_2 = HCTSAOperation(
        'SY_SlidingWindow_ent_s2_2',
        "SY_SlidingWindow(y,'ent','std',2,2)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s2_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_s2_10',
        "SY_SlidingWindow(y,'ent','std',2,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s5_1 = HCTSAOperation(
        'SY_SlidingWindow_ent_s5_1',
        "SY_SlidingWindow(y,'ent','std',5,1)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s5_2 = HCTSAOperation(
        'SY_SlidingWindow_ent_s5_2',
        "SY_SlidingWindow(y,'ent','std',5,2)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s5_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_s5_10',
        "SY_SlidingWindow(y,'ent','std',5,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,1,0,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s10_1 = HCTSAOperation(
        'SY_SlidingWindow_ent_s10_1',
        "SY_SlidingWindow(y,'ent','std',10,1)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s10_2 = HCTSAOperation(
        'SY_SlidingWindow_ent_s10_2',
        "SY_SlidingWindow(y,'ent','std',10,2)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_s10_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_s10_10',
        "SY_SlidingWindow(y,'ent','std',10,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='std', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,a,m,p
    # outs: e,n,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_sampen2_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_sampen2_10',
        "SY_SlidingWindow(y,'ent','sampen',2,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='sampen', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,a,m,p
    # outs: e,n,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_sampen5_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_sampen5_10',
        "SY_SlidingWindow(y,'ent','sampen',5,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='sampen', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,a,m,p
    # outs: e,n,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_sampen10_2 = HCTSAOperation(
        'SY_SlidingWindow_ent_sampen10_2',
        "SY_SlidingWindow(y,'ent','sampen',10,2)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='sampen', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,s,a,m,p
    # outs: e,n,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_sampen10_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_sampen10_10',
        "SY_SlidingWindow(y,'ent','sampen',10,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='sampen', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,2
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent2_1 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent2_1',
        "SY_SlidingWindow(y,'ent','ent',2,1)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,2
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent2_2 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent2_2',
        "SY_SlidingWindow(y,'ent','ent',2,2)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,2
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent2_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent2_10',
        "SY_SlidingWindow(y,'ent','ent',2,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,5
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent5_1 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent5_1',
        "SY_SlidingWindow(y,'ent','ent',5,1)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,5
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent5_2 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent5_2',
        "SY_SlidingWindow(y,'ent','ent',5,2)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,5
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent5_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent5_10',
        "SY_SlidingWindow(y,'ent','ent',5,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,1
    # outs: 0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent10_1 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent10_1',
        "SY_SlidingWindow(y,'ent','ent',10,1)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,1
    # outs: 0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent10_2 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent10_2',
        "SY_SlidingWindow(y,'ent','ent',10,2)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,e,n,t
    # outs: _,e,n,t,1
    # outs: 0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ent_ent10_10 = HCTSAOperation(
        'SY_SlidingWindow_ent_ent10_10',
        "SY_SlidingWindow(y,'ent','ent',10,10)",
        SY_SlidingWindow(windowStat='ent', acrossWinStat='ent', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s2_1 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s2_1',
        "SY_SlidingWindow(y,'sampen','std',2,1)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s2_2 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s2_2',
        "SY_SlidingWindow(y,'sampen','std',2,2)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 2,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s2_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s2_10',
        "SY_SlidingWindow(y,'sampen','std',2,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s5_1 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s5_1',
        "SY_SlidingWindow(y,'sampen','std',5,1)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s5_2 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s5_2',
        "SY_SlidingWindow(y,'sampen','std',5,2)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 5,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s5_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s5_10',
        "SY_SlidingWindow(y,'sampen','std',5,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 1,0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s10_1 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s10_1',
        "SY_SlidingWindow(y,'sampen','std',10,1)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 1,0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s10_2 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s10_2',
        "SY_SlidingWindow(y,'sampen','std',10,2)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: 1,0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_s10_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_s10_10',
        "SY_SlidingWindow(y,'sampen','std',10,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='std', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: a,m,p,e,n
    # outs: 2,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_sampen2_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_sampen2_10',
        "SY_SlidingWindow(y,'sampen','sampen',2,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='sampen', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: a,m,p,e,n
    # outs: 5,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_sampen5_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_sampen5_10',
        "SY_SlidingWindow(y,'sampen','sampen',5,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='sampen', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_sampen10_1 = HCTSAOperation(
        'SY_SlidingWindow_sampen_sampen10_1',
        "SY_SlidingWindow(y,'sampen','sampen',10,1)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='sampen', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_sampen10_2 = HCTSAOperation(
        'SY_SlidingWindow_sampen_sampen10_2',
        "SY_SlidingWindow(y,'sampen','sampen',10,2)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='sampen', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,s
    # outs: a,m,p,e,n
    # outs: 1,0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_sampen10_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_sampen10_10',
        "SY_SlidingWindow(y,'sampen','sampen',10,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='sampen', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent2_1 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent2_1',
        "SY_SlidingWindow(y,'sampen','ent',2,1)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent2_2 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent2_2',
        "SY_SlidingWindow(y,'sampen','ent',2,2)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent2_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent2_10',
        "SY_SlidingWindow(y,'sampen','ent',2,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent5_1 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent5_1',
        "SY_SlidingWindow(y,'sampen','ent',5,1)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent5_2 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent5_2',
        "SY_SlidingWindow(y,'sampen','ent',5,2)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent5_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent5_10',
        "SY_SlidingWindow(y,'sampen','ent',5,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,1,0,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent10_1 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent10_1',
        "SY_SlidingWindow(y,'sampen','ent',10,1)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent10_2 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent10_2',
        "SY_SlidingWindow(y,'sampen','ent',10,2)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,s,a,m
    # outs: p,e,n,_,e
    # outs: n,t,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_sampen_ent10_10 = HCTSAOperation(
        'SY_SlidingWindow_sampen_ent10_10',
        "SY_SlidingWindow(y,'sampen','ent',10,10)",
        SY_SlidingWindow(windowStat='sampen', acrossWinStat='ent', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,2,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s2_1 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s2_1',
        "SY_SlidingWindow(y,'mom3','std',2,1)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,2,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s2_2 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s2_2',
        "SY_SlidingWindow(y,'mom3','std',2,2)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,2,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s2_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s2_10',
        "SY_SlidingWindow(y,'mom3','std',2,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,5,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s5_1 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s5_1',
        "SY_SlidingWindow(y,'mom3','std',5,1)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,5,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s5_2 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s5_2',
        "SY_SlidingWindow(y,'mom3','std',5,2)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,5,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s5_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s5_10',
        "SY_SlidingWindow(y,'mom3','std',5,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,1,0
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s10_1 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s10_1',
        "SY_SlidingWindow(y,'mom3','std',10,1)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,1,0
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s10_2 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s10_2',
        "SY_SlidingWindow(y,'mom3','std',10,2)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,1,0
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_s10_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_s10_10',
        "SY_SlidingWindow(y,'mom3','std',10,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='std', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,a,m
    # outs: p,e,n,2,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_sampen2_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_sampen2_10',
        "SY_SlidingWindow(y,'mom3','sampen',2,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='sampen', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,a,m
    # outs: p,e,n,5,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_sampen5_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_sampen5_10',
        "SY_SlidingWindow(y,'mom3','sampen',5,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='sampen', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,a,m
    # outs: p,e,n,1,0
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_sampen10_1 = HCTSAOperation(
        'SY_SlidingWindow_mom3_sampen10_1',
        "SY_SlidingWindow(y,'mom3','sampen',10,1)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='sampen', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,a,m
    # outs: p,e,n,1,0
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_sampen10_2 = HCTSAOperation(
        'SY_SlidingWindow_mom3_sampen10_2',
        "SY_SlidingWindow(y,'mom3','sampen',10,2)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='sampen', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,s,a,m
    # outs: p,e,n,1,0
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_sampen10_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_sampen10_10',
        "SY_SlidingWindow(y,'mom3','sampen',10,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='sampen', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent2_1 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent2_1',
        "SY_SlidingWindow(y,'mom3','ent',2,1)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent2_2 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent2_2',
        "SY_SlidingWindow(y,'mom3','ent',2,2)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 2,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent2_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent2_10',
        "SY_SlidingWindow(y,'mom3','ent',2,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent5_1 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent5_1',
        "SY_SlidingWindow(y,'mom3','ent',5,1)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent5_2 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent5_2',
        "SY_SlidingWindow(y,'mom3','ent',5,2)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 5,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent5_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent5_10',
        "SY_SlidingWindow(y,'mom3','ent',5,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 1,0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent10_1 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent10_1',
        "SY_SlidingWindow(y,'mom3','ent',10,1)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 1,0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent10_2 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent10_2',
        "SY_SlidingWindow(y,'mom3','ent',10,2)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 3,_,e,n,t
    # outs: 1,0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom3_ent10_10 = HCTSAOperation(
        'SY_SlidingWindow_mom3_ent10_10',
        "SY_SlidingWindow(y,'mom3','ent',10,10)",
        SY_SlidingWindow(windowStat='mom3', acrossWinStat='ent', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,2,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s2_1 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s2_1',
        "SY_SlidingWindow(y,'mom4','std',2,1)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,2,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s2_2 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s2_2',
        "SY_SlidingWindow(y,'mom4','std',2,2)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,2,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s2_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s2_10',
        "SY_SlidingWindow(y,'mom4','std',2,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,5,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s5_1 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s5_1',
        "SY_SlidingWindow(y,'mom4','std',5,1)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,5,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s5_2 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s5_2',
        "SY_SlidingWindow(y,'mom4','std',5,2)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,5,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s5_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s5_10',
        "SY_SlidingWindow(y,'mom4','std',5,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,1,0
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s10_1 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s10_1',
        "SY_SlidingWindow(y,'mom4','std',10,1)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,1,0
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s10_2 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s10_2',
        "SY_SlidingWindow(y,'mom4','std',10,2)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,1,0
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_s10_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_s10_10',
        "SY_SlidingWindow(y,'mom4','std',10,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='std', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,a,m
    # outs: p,e,n,2,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_sampen2_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_sampen2_10',
        "SY_SlidingWindow(y,'mom4','sampen',2,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='sampen', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,a,m
    # outs: p,e,n,5,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_sampen5_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_sampen5_10',
        "SY_SlidingWindow(y,'mom4','sampen',5,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='sampen', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,a,m
    # outs: p,e,n,1,0
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_sampen10_1 = HCTSAOperation(
        'SY_SlidingWindow_mom4_sampen10_1',
        "SY_SlidingWindow(y,'mom4','sampen',10,1)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='sampen', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,a,m
    # outs: p,e,n,1,0
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_sampen10_2 = HCTSAOperation(
        'SY_SlidingWindow_mom4_sampen10_2',
        "SY_SlidingWindow(y,'mom4','sampen',10,2)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='sampen', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,s,a,m
    # outs: p,e,n,1,0
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_sampen10_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_sampen10_10',
        "SY_SlidingWindow(y,'mom4','sampen',10,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='sampen', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent2_1 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent2_1',
        "SY_SlidingWindow(y,'mom4','ent',2,1)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent2_2 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent2_2',
        "SY_SlidingWindow(y,'mom4','ent',2,2)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 2,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent2_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent2_10',
        "SY_SlidingWindow(y,'mom4','ent',2,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent5_1 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent5_1',
        "SY_SlidingWindow(y,'mom4','ent',5,1)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent5_2 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent5_2',
        "SY_SlidingWindow(y,'mom4','ent',5,2)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 5,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent5_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent5_10',
        "SY_SlidingWindow(y,'mom4','ent',5,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 1,0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent10_1 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent10_1',
        "SY_SlidingWindow(y,'mom4','ent',10,1)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 1,0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent10_2 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent10_2',
        "SY_SlidingWindow(y,'mom4','ent',10,2)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,m,o,m
    # outs: 4,_,e,n,t
    # outs: 1,0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_mom4_ent10_10 = HCTSAOperation(
        'SY_SlidingWindow_mom4_ent10_10',
        "SY_SlidingWindow(y,'mom4','ent',10,10)",
        SY_SlidingWindow(windowStat='mom4', acrossWinStat='ent', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s2_1 = HCTSAOperation(
        'SY_SlidingWindow_lil_s2_1',
        "SY_SlidingWindow(y,'lillie','std',2,1)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s2_2 = HCTSAOperation(
        'SY_SlidingWindow_lil_s2_2',
        "SY_SlidingWindow(y,'lillie','std',2,2)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s2_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_s2_10',
        "SY_SlidingWindow(y,'lillie','std',2,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s5_1 = HCTSAOperation(
        'SY_SlidingWindow_lil_s5_1',
        "SY_SlidingWindow(y,'lillie','std',5,1)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s5_2 = HCTSAOperation(
        'SY_SlidingWindow_lil_s5_2',
        "SY_SlidingWindow(y,'lillie','std',5,2)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s5_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_s5_10',
        "SY_SlidingWindow(y,'lillie','std',5,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,1,0,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s10_1 = HCTSAOperation(
        'SY_SlidingWindow_lil_s10_1',
        "SY_SlidingWindow(y,'lillie','std',10,1)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s10_2 = HCTSAOperation(
        'SY_SlidingWindow_lil_s10_2',
        "SY_SlidingWindow(y,'lillie','std',10,2)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_s10_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_s10_10',
        "SY_SlidingWindow(y,'lillie','std',10,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='std', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,a,m,p
    # outs: e,n,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_sampen2_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_sampen2_10',
        "SY_SlidingWindow(y,'lillie','sampen',2,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='sampen', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,a,m,p
    # outs: e,n,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_sampen5_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_sampen5_10',
        "SY_SlidingWindow(y,'lillie','sampen',5,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='sampen', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,a,m,p
    # outs: e,n,1,0,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_sampen10_1 = HCTSAOperation(
        'SY_SlidingWindow_lil_sampen10_1',
        "SY_SlidingWindow(y,'lillie','sampen',10,1)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='sampen', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,a,m,p
    # outs: e,n,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_sampen10_2 = HCTSAOperation(
        'SY_SlidingWindow_lil_sampen10_2',
        "SY_SlidingWindow(y,'lillie','sampen',10,2)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='sampen', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,s,a,m,p
    # outs: e,n,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_sampen10_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_sampen10_10',
        "SY_SlidingWindow(y,'lillie','sampen',10,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='sampen', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,2
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent2_1 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent2_1',
        "SY_SlidingWindow(y,'lillie','ent',2,1)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,2
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent2_2 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent2_2',
        "SY_SlidingWindow(y,'lillie','ent',2,2)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,2
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent2_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent2_10',
        "SY_SlidingWindow(y,'lillie','ent',2,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,5
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent5_1 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent5_1',
        "SY_SlidingWindow(y,'lillie','ent',5,1)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,5
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent5_2 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent5_2',
        "SY_SlidingWindow(y,'lillie','ent',5,2)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,5
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent5_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent5_10',
        "SY_SlidingWindow(y,'lillie','ent',5,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,1
    # outs: 0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent10_1 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent10_1',
        "SY_SlidingWindow(y,'lillie','ent',10,1)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,1
    # outs: 0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent10_2 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent10_2',
        "SY_SlidingWindow(y,'lillie','ent',10,2)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,l,i,l
    # outs: _,e,n,t,1
    # outs: 0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_lil_ent10_10 = HCTSAOperation(
        'SY_SlidingWindow_lil_ent10_10',
        "SY_SlidingWindow(y,'lillie','ent',10,10)",
        SY_SlidingWindow(windowStat='lillie', acrossWinStat='ent', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,2,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s2_1 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s2_1',
        "SY_SlidingWindow(y,'AC1','std',2,1)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,2,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s2_2 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s2_2',
        "SY_SlidingWindow(y,'AC1','std',2,2)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s2_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s2_10',
        "SY_SlidingWindow(y,'AC1','std',2,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,5,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s5_1 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s5_1',
        "SY_SlidingWindow(y,'AC1','std',5,1)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,5,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s5_2 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s5_2',
        "SY_SlidingWindow(y,'AC1','std',5,2)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s5_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s5_10',
        "SY_SlidingWindow(y,'AC1','std',5,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,1,0,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s10_1 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s10_1',
        "SY_SlidingWindow(y,'AC1','std',10,1)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s10_2 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s10_2',
        "SY_SlidingWindow(y,'AC1','std',10,2)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_s10_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_s10_10',
        "SY_SlidingWindow(y,'AC1','std',10,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='std', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,a,m,p
    # outs: e,n,2,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_sampen2_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_sampen2_10',
        "SY_SlidingWindow(y,'AC1','sampen',2,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='sampen', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,a,m,p
    # outs: e,n,5,_,1
    # outs: 0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_sampen5_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_sampen5_10',
        "SY_SlidingWindow(y,'AC1','sampen',5,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='sampen', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,a,m,p
    # outs: e,n,1,0,_
    # outs: 1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_sampen10_1 = HCTSAOperation(
        'SY_SlidingWindow_ac1_sampen10_1',
        "SY_SlidingWindow(y,'AC1','sampen',10,1)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='sampen', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,a,m,p
    # outs: e,n,1,0,_
    # outs: 2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_sampen10_2 = HCTSAOperation(
        'SY_SlidingWindow_ac1_sampen10_2',
        "SY_SlidingWindow(y,'AC1','sampen',10,2)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='sampen', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,s,a,m,p
    # outs: e,n,1,0,_
    # outs: 1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_sampen10_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_sampen10_10',
        "SY_SlidingWindow(y,'AC1','sampen',10,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='sampen', numSeg=10, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,2
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent2_1 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent2_1',
        "SY_SlidingWindow(y,'AC1','ent',2,1)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=2, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,2
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent2_2 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent2_2',
        "SY_SlidingWindow(y,'AC1','ent',2,2)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=2, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,2
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent2_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent2_10',
        "SY_SlidingWindow(y,'AC1','ent',2,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=2, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,5
    # outs: _,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent5_1 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent5_1',
        "SY_SlidingWindow(y,'AC1','ent',5,1)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=5, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,5
    # outs: _,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent5_2 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent5_2',
        "SY_SlidingWindow(y,'AC1','ent',5,2)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=5, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,5
    # outs: _,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent5_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent5_10',
        "SY_SlidingWindow(y,'AC1','ent',5,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=5, incMove=10))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,1
    # outs: 0,_,1
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent10_1 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent10_1',
        "SY_SlidingWindow(y,'AC1','ent',10,1)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=10, incMove=1))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,1
    # outs: 0,_,2
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent10_2 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent10_2',
        "SY_SlidingWindow(y,'AC1','ent',10,2)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=10, incMove=2))

    # outs: S,Y,_,S,l
    # outs: i,d,i,n,g
    # outs: W,i,n,d,o
    # outs: w,_,a,c,1
    # outs: _,e,n,t,1
    # outs: 0,_,1,0
    # tags: slidingwin,stationarity
    SY_SlidingWindow_ac1_ent10_10 = HCTSAOperation(
        'SY_SlidingWindow_ac1_ent10_10',
        "SY_SlidingWindow(y,'AC1','ent',10,10)",
        SY_SlidingWindow(windowStat='AC1', acrossWinStat='ent', numSeg=10, incMove=10))

    # outs: S,Y,_,S,p
    # outs: r,e,a,d,R
    # outs: a,n,d,o,m
    # outs: L,o,c,a,l
    # outs: _,a,c,2,_
    # outs: 1,0,0
    # tags: stationarity
    SY_SpreadRandomLocal_ac2_100 = HCTSAOperation(
        'SY_SpreadRandomLocal_ac2_100',
        "SY_SpreadRandomLocal(y,'ac2',100,'default')",
        SY_SpreadRandomLocal(l='ac2', numSegs=100, randomSeed='default'))

    # outs: S,Y,_,S,p
    # outs: r,e,a,d,R
    # outs: a,n,d,o,m
    # outs: L,o,c,a,l
    # outs: _,a,c,5,_
    # outs: 1,0,0
    # tags: stationarity
    SY_SpreadRandomLocal_ac5_100 = HCTSAOperation(
        'SY_SpreadRandomLocal_ac5_100',
        "SY_SpreadRandomLocal(y,'ac5',100,'default')",
        SY_SpreadRandomLocal(l='ac5', numSegs=100, randomSeed='default'))

    # outs: S,Y,_,S,p
    # outs: r,e,a,d,R
    # outs: a,n,d,o,m
    # outs: L,o,c,a,l
    # outs: _,5,0,_,1
    # outs: 0,0
    # tags: stationarity
    SY_SpreadRandomLocal_50_100 = HCTSAOperation(
        'SY_SpreadRandomLocal_50_100',
        "SY_SpreadRandomLocal(y,50,100,'default')",
        SY_SpreadRandomLocal(l=50, numSegs=100, randomSeed='default'))

    # outs: S,Y,_,S,p
    # outs: r,e,a,d,R
    # outs: a,n,d,o,m
    # outs: L,o,c,a,l
    # outs: _,1,0,0,_
    # outs: 1,0,0
    # tags: stationarity
    SY_SpreadRandomLocal_100_100 = HCTSAOperation(
        'SY_SpreadRandomLocal_100_100',
        "SY_SpreadRandomLocal(y,100,100,'default')",
        SY_SpreadRandomLocal(l=100, numSegs=100, randomSeed='default'))

    # outs: S,Y,_,S,p
    # outs: r,e,a,d,R
    # outs: a,n,d,o,m
    # outs: L,o,c,a,l
    # outs: _,2,0,0,_
    # outs: 1,0,0
    # tags: stationarity
    SY_SpreadRandomLocal_200_100 = HCTSAOperation(
        'SY_SpreadRandomLocal_200_100',
        "SY_SpreadRandomLocal(y,200,100,'default')",
        SY_SpreadRandomLocal(l=200, numSegs=100, randomSeed='default'))

    # outs: S,t,a,t,A
    # outs: v,l,2,5
    # tags: StatAv,stationarity
    StatAvl25 = HCTSAOperation(
        'StatAvl25',
        "SY_StatAv(y,'len',25)",
        SY_StatAv(whatType='len', n=25))

    # outs: S,t,a,t,A
    # outs: v,l,5,0
    # tags: StatAv,stationarity
    StatAvl50 = HCTSAOperation(
        'StatAvl50',
        "SY_StatAv(y,'len',50)",
        SY_StatAv(whatType='len', n=50))

    # outs: S,t,a,t,A
    # outs: v,l,1,0,0
    # tags: StatAv,stationarity
    StatAvl100 = HCTSAOperation(
        'StatAvl100',
        "SY_StatAv(y,'len',100)",
        SY_StatAv(whatType='len', n=100))

    # outs: S,t,a,t,A
    # outs: v,l,1,5,0
    # tags: StatAv,stationarity
    StatAvl150 = HCTSAOperation(
        'StatAvl150',
        "SY_StatAv(y,'len',150)",
        SY_StatAv(whatType='len', n=150))

    # outs: S,t,a,t,A
    # outs: v,l,2,0,0
    # tags: StatAv,stationarity
    StatAvl200 = HCTSAOperation(
        'StatAvl200',
        "SY_StatAv(y,'len',200)",
        SY_StatAv(whatType='len', n=200))

    # outs: S,t,a,t,A
    # outs: v,l,2,5,0
    # tags: StatAv,stationarity
    StatAvl250 = HCTSAOperation(
        'StatAvl250',
        "SY_StatAv(y,'len',250)",
        SY_StatAv(whatType='len', n=250))

    # outs: S,t,a,t,A
    # outs: v,l,5,0,0
    # tags: StatAv,stationarity
    StatAvl500 = HCTSAOperation(
        'StatAvl500',
        "SY_StatAv(y,'len',500)",
        SY_StatAv(whatType='len', n=500))

    # outs: S,t,a,t,A
    # outs: v,l,1,0,0
    # outs: 0
    # tags: StatAv,stationarity
    StatAvl1000 = HCTSAOperation(
        'StatAvl1000',
        "SY_StatAv(y,'len',1000)",
        SY_StatAv(whatType='len', n=1000))

    # outs: S,Y,_,S,t
    # outs: d,N,t,h,D
    # outs: e,r,_,1
    # tags: entropy
    SY_StdNthDer_1 = HCTSAOperation(
        'SY_StdNthDer_1',
        'SY_StdNthDer(y,1)',
        SY_StdNthDer(n=1))

    # outs: S,Y,_,S,t
    # outs: d,N,t,h,D
    # outs: e,r,_,2
    # tags: entropy
    SY_StdNthDer_2 = HCTSAOperation(
        'SY_StdNthDer_2',
        'SY_StdNthDer(y,2)',
        SY_StdNthDer(n=2))

    # outs: S,Y,_,S,t
    # outs: d,N,t,h,D
    # outs: e,r,_,3
    # tags: entropy
    SY_StdNthDer_3 = HCTSAOperation(
        'SY_StdNthDer_3',
        'SY_StdNthDer(y,3)',
        SY_StdNthDer(n=3))

    # outs: S,Y,_,S,t
    # outs: d,N,t,h,D
    # outs: e,r,_,4
    # tags: entropy
    SY_StdNthDer_4 = HCTSAOperation(
        'SY_StdNthDer_4',
        'SY_StdNthDer(y,4)',
        SY_StdNthDer(n=4))

    # outs: S,Y,_,S,t
    # outs: d,N,t,h,D
    # outs: e,r,_,5
    # tags: entropy
    SY_StdNthDer_5 = HCTSAOperation(
        'SY_StdNthDer_5',
        'SY_StdNthDer(y,5)',
        SY_StdNthDer(n=5))

    # outs: S,Y,_,S,t
    # outs: d,N,t,h,D
    # outs: e,r,_,1,0
    # tags: entropy
    SY_StdNthDer_10 = HCTSAOperation(
        'SY_StdNthDer_10',
        'SY_StdNthDer(y,10)',
        SY_StdNthDer(n=10))

    # outs: S,Y,_,S,t
    # outs: d,N,t,h,D
    # outs: e,r,C,h,a
    # outs: n,g,e
    # tags: entropy
    SY_StdNthDerChange = HCTSAOperation(
        'SY_StdNthDerChange',
        'SY_StdNthDerChange(y)',
        SY_StdNthDerChange())

    # outs: S,Y,_,T,I
    # outs: S,E,A,N,_
    # outs: n,s,t,a,t
    # outs: _,z,_,4,_
    # outs: 1,_,3
    # tags: model,nonlinear,stationarity,tisean
    SY_TISEAN_nstat_z_4_1_3 = HCTSAOperation(
        'SY_TISEAN_nstat_z_4_1_3',
        'SY_TISEAN_nstat_z(y,4,{1,3})',
        SY_TISEAN_nstat_z(numSeg=4, embedParams=array([1, 3], dtype=object)))

    # outs: S,Y,_,T,I
    # outs: S,E,A,N,_
    # outs: n,s,t,a,t
    # outs: _,z,_,5,_
    # outs: 1,_,3
    # tags: model,nonlinear,stationarity,tisean
    SY_TISEAN_nstat_z_5_1_3 = HCTSAOperation(
        'SY_TISEAN_nstat_z_5_1_3',
        'SY_TISEAN_nstat_z(y,5,{1,3})',
        SY_TISEAN_nstat_z(numSeg=5, embedParams=array([1, 3], dtype=object)))

    # outs: S,Y,_,T,r
    # outs: e,n,d
    # tags: stationarity
    SY_Trend = HCTSAOperation(
        'SY_Trend',
        'SY_Trend(y)',
        SY_Trend())

    # outs: S,Y,_,V,a
    # outs: r,R,a,t,i
    # outs: o,T,e,s,t
    # outs: _,2,_,0
    # tags: econometricstoolbox,pvalue,vratiotest
    SY_VarRatioTest_2_0 = HCTSAOperation(
        'SY_VarRatioTest_2_0',
        'SY_VarRatioTest(y,2,0)',
        SY_VarRatioTest(periods=2, IIDs=0))

    # outs: S,Y,_,V,a
    # outs: r,R,a,t,i
    # outs: o,T,e,s,t
    # outs: _,2,_,1
    # tags: econometricstoolbox,pvalue,vratiotest
    SY_VarRatioTest_2_1 = HCTSAOperation(
        'SY_VarRatioTest_2_1',
        'SY_VarRatioTest(y,2,1)',
        SY_VarRatioTest(periods=2, IIDs=1))

    # outs: S,Y,_,V,a
    # outs: r,R,a,t,i
    # outs: o,T,e,s,t
    # outs: _,4,_,0
    # tags: econometricstoolbox,pvalue,vratiotest
    SY_VarRatioTest_4_0 = HCTSAOperation(
        'SY_VarRatioTest_4_0',
        'SY_VarRatioTest(y,4,0)',
        SY_VarRatioTest(periods=4, IIDs=0))

    # outs: S,Y,_,V,a
    # outs: r,R,a,t,i
    # outs: o,T,e,s,t
    # outs: _,4,_,1
    # tags: econometricstoolbox,pvalue,vratiotest
    SY_VarRatioTest_4_1 = HCTSAOperation(
        'SY_VarRatioTest_4_1',
        'SY_VarRatioTest(y,4,1)',
        SY_VarRatioTest(periods=4, IIDs=1))

    # outs: S,Y,_,V,a
    # outs: r,R,a,t,i
    # outs: o,T,e,s,t
    # outs: _,2,4,6,8
    # outs: 2,4,6,8,_
    # outs: 0,0,0,0,1
    # outs: 1,1,1
    # tags: econometricstoolbox,pvalue,vratiotest
    SY_VarRatioTest_24682468_00001111 = HCTSAOperation(
        'SY_VarRatioTest_24682468_00001111',
        'SY_VarRatioTest(y,[2,4,6,8,2,4,6,8],[0,0,0,0,1,1,1,1])',
        SY_VarRatioTest(periods=array([2, 4, 6, 8, 2, 4, 6, 8]), IIDs=array([0, 0, 0, 0, 1, 1, 1, 1])))

    # outs: T,S,T,L,_
    # outs: d,e,l,a,y
    # outs: t,i,m,e,_
    # outs: 0,1,_,1
    # tags: correlation,nonlinear,tau,tstool
    TSTL_delaytime_01_1 = HCTSAOperation(
        'TSTL_delaytime_01_1',
        "TSTL_delaytime(y,0.1,1,'default')",
        TSTL_delaytime(maxDelay=0.1, past=1, randomSeed='default'))

    # outs: T,S,T,L,_
    # outs: l,o,c,a,l
    # outs: d,e,n,s,i
    # outs: t,y,_,5,_
    # outs: 4,0,_,a,c
    # outs: _,f,n,n,m
    # outs: a,r
    # tags: localdensity,nonlinear,tstool
    TSTL_localdensity_5_40_ac_fnnmar = HCTSAOperation(
        'TSTL_localdensity_5_40_ac_fnnmar',
        "TSTL_localdensity(y,5,40,{'ac','fnnmar'})",
        TSTL_localdensity(NNR=5, past=40, embedParams=array(['ac', 'fnnmar'], dtype=object)))

    # outs: T,S,T,L,_
    # outs: l,o,c,a,l
    # outs: d,e,n,s,i
    # outs: t,y,_,5,_
    # outs: 4,0,_,a,c
    # outs: _,2
    # tags: localdensity,nonlinear,tstool
    TSTL_localdensity_5_40_ac_2 = HCTSAOperation(
        'TSTL_localdensity_5_40_ac_2',
        "TSTL_localdensity(y,5,40,{'ac',2})",
        TSTL_localdensity(NNR=5, past=40, embedParams=array(['ac', 2], dtype=object)))

    # outs: T,S,T,L,_
    # outs: l,o,c,a,l
    # outs: d,e,n,s,i
    # outs: t,y,_,5,_
    # outs: 4,0,_,1,_
    # outs: 3
    # tags: localdensity,nonlinear,tstool
    TSTL_localdensity_5_40_1_3 = HCTSAOperation(
        'TSTL_localdensity_5_40_1_3',
        'TSTL_localdensity(y,5,40,{1,3})',
        TSTL_localdensity(NNR=5, past=40, embedParams=array([1, 3], dtype=object)))

    # outs: W,L,_,D,e
    # outs: t,a,i,l,C
    # outs: o,e,f,f,s
    # outs: _,d,b,3,_
    # outs: m,a,x
    # tags: wavelet,waveletTB
    WL_DetailCoeffs_db3_max = HCTSAOperation(
        'WL_DetailCoeffs_db3_max',
        "WL_DetailCoeffs(y,'db3','max')",
        WL_DetailCoeffs(wname='db3', maxlevel='max'))

    # outs: W,L,_,c,o
    # outs: e,f,f,s,_
    # outs: d,b,3,_,1
    # tags: lengthdep,wavelet,waveletTB
    WL_coeffs_db3_1 = HCTSAOperation(
        'WL_coeffs_db3_1',
        "WL_coeffs(y,'db3',1)",
        WL_coeffs(wname='db3', level=1))

    # outs: W,L,_,c,o
    # outs: e,f,f,s,_
    # outs: d,b,3,_,2
    # tags: lengthdep,wavelet,waveletTB
    WL_coeffs_db3_2 = HCTSAOperation(
        'WL_coeffs_db3_2',
        "WL_coeffs(y,'db3',2)",
        WL_coeffs(wname='db3', level=2))

    # outs: W,L,_,c,o
    # outs: e,f,f,s,_
    # outs: d,b,3,_,3
    # tags: lengthdep,wavelet,waveletTB
    WL_coeffs_db3_3 = HCTSAOperation(
        'WL_coeffs_db3_3',
        "WL_coeffs(y,'db3',3)",
        WL_coeffs(wname='db3', level=3))

    # outs: W,L,_,c,o
    # outs: e,f,f,s,_
    # outs: d,b,3,_,4
    # tags: lengthdep,wavelet,waveletTB
    WL_coeffs_db3_4 = HCTSAOperation(
        'WL_coeffs_db3_4',
        "WL_coeffs(y,'db3',4)",
        WL_coeffs(wname='db3', level=4))

    # outs: W,L,_,c,o
    # outs: e,f,f,s,_
    # outs: d,b,3,_,5
    # tags: lengthdep,wavelet,waveletTB
    WL_coeffs_db3_5 = HCTSAOperation(
        'WL_coeffs_db3_5',
        "WL_coeffs(y,'db3',5)",
        WL_coeffs(wname='db3', level=5))

    # outs: W,L,_,c,o
    # outs: e,f,f,s,_
    # outs: d,b,3,_,m
    # outs: a,x
    # tags: lengthdep,wavelet,waveletTB
    WL_coeffs_db3_max = HCTSAOperation(
        'WL_coeffs_db3_max',
        "WL_coeffs(y,'db3','max')",
        WL_coeffs(wname='db3', level='max'))

    # outs: W,L,_,c,w
    # outs: t,_,d,b,3
    # outs: _,3,2
    # tags: cwt,statTB,wavelet,waveletTB
    WL_cwt_db3_32 = HCTSAOperation(
        'WL_cwt_db3_32',
        "WL_cwt(y,'db3',32)",
        WL_cwt(wname='db3', maxScale=32))

    # outs: W,L,_,c,w
    # outs: t,_,s,y,m
    # outs: 2,_,3,2
    # tags: cwt,statTB,wavelet,waveletTB
    WL_cwt_sym2_32 = HCTSAOperation(
        'WL_cwt_sym2_32',
        "WL_cwt(y,'sym2',32)",
        WL_cwt(wname='sym2', maxScale=32))

    # outs: W,L,_,d,w
    # outs: t,c,o,e,f
    # outs: f,_,d,b,3
    # outs: _,5
    # tags: dwt,wavelet,waveletTB
    WL_dwtcoeff_db3_5 = HCTSAOperation(
        'WL_dwtcoeff_db3_5',
        "WL_dwtcoeff(y,'db3',5)",
        WL_dwtcoeff(wname='db3', level=5))

    # outs: W,L,_,d,w
    # outs: t,c,o,e,f
    # outs: f,_,s,y,m
    # outs: 2,_,5
    # tags: dwt,wavelet,waveletTB
    WL_dwtcoeff_sym2_5 = HCTSAOperation(
        'WL_dwtcoeff_sym2_5',
        "WL_dwtcoeff(y,'sym2',5)",
        WL_dwtcoeff(wname='sym2', level=5))

    # outs: W,L,_,f,B
    # outs: M
    # tags: wavelet,waveletTB
    WL_fBM = HCTSAOperation(
        'WL_fBM',
        'WL_fBM(y)',
        WL_fBM())

    # outs: W,L,_,s,c
    # outs: a,l,2,f,r
    # outs: q,_,d,b,3
    # outs: _,m,a,x,_
    # outs: 1
    # tags: wavelet,waveletTB
    WL_scal2frq_db3_max_1 = HCTSAOperation(
        'WL_scal2frq_db3_max_1',
        "WL_scal2frq(y,'db3','max',1)",
        WL_scal2frq(wname='db3', amax='max', delta=1))
