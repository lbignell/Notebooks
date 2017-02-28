# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:36:45 2017

@author: lbignell
"""

import pymc
import numpy as np
import matplotlib.pyplot as plt

def CalcGP(sampledata, newpoints, ntrials=500, nburn=200, nthin=1):
    """
    Return estimates of the function sampled at sampledata as a Gaussian Process, at newpoints.

    The sampledata should be Nsamples by Ndim + 1 (values as last column).
    The newpoints should be Npoints by Ndim.
    """
    nsamples, ndims = np.shape(sampledata)
    ndims -= 1 #There's an extra dimension corresponding to the function value.
    y = sampledata[:,-1]
    samplepoints = sampledata[:, :-1]
    #Generate containers for the theta and exponent values...
    #The parameters are random variables.
    thetas = np.empty(ndims, dtype=object)
    exponents = np.empty(ndims, dtype=object)

    for i in range(ndims):
      #I'll arbitrarily assume the thetas are in the range [-10, 10].
      thetas[i] = pymc.Uniform('theta_{0}'.format(i), lower=-10, upper=10, value=1)
      exponents[i] = pymc.Uniform('exponent_{0}'.format(i), lower=0, upper=100, value=2)

#==============================================================================
#     @pymc.deterministic(plot=False, dtype=float)
#     def distance(point1, point2, thetas=thetas, exponents=exponents):
#       """
#       Return the non-Euclidean distance metric.
#       """
#       absvector = np.abs(point1 - point2)
#       return sum(thetas*absvector**exponents)
#==============================================================================
    def f(x):
      return np.exp(float(x))
    vf = np.vectorize(f)

    @pymc.deterministic(plot=False)
    def corr(samples=samplepoints, nsamples=nsamples, thetas=thetas, exponents=exponents):
      """
      Return the correlation matrix.
      """
      #The correlation matrix is nsamples x nsamples
      #corrmatrix = np.empty([nsamples, nsamples])

      #tmp = -np.sum(thetas[:,None]*np.abs(samples.T - samples[:,:,None])**exponents[:,None], axis=1)
      return vf(-np.sum(thetas[:,None]*np.abs(samples.T - samples[:,:,None])**exponents[:,None], axis=1))
#==============================================================================
#       for row in range(nsamples):
#         for col in range(nsamples):
#           absvector = np.abs(samplepoints[row] - samplepoints[col])
#           corrmatrix[row, col] = np.exp(float(-np.sum(thetas*absvector**exponents)))
#
#       return corrmatrix + 1e-10*np.eye(nsamples)
#==============================================================================

    @pymc.deterministic(plot=False, dtype=float)
    def mean(corrmat=corr, y=y, nsamples=nsamples):
      """
      The maximum likelihood mean is entirely defined by the correlation matrix.
      """
      I_vec = np.ones(nsamples)
      try:
        inv = np.linalg.inv(corrmat)
      except np.linalg.LinAlgError:
        print("I'm having trouble inverting the correlation matrix; conditioning...")
        inv = np.linalg.inv(corrmat + 1e-12*np.eye(nsamples))

      return I_vec * np.dot(I_vec, np.dot(inv, y)) / np.dot(I_vec, np.dot(inv, I_vec))

    @pymc.deterministic(plot=False, dtype=float)
    def variance(corrmat=corr, y=y, themean=mean, nsamples=nsamples):
      """
      The maximum likelihood variance is entirely defined by the correlation matrix.
      """
      try:
        inv = np.linalg.inv(corrmat)
      except np.linalg.LinAlgError:
        print("I'm having trouble inverting the correlation matrix; conditioning...")
        inv = np.linalg.inv(corrmat + 1e-12*np.eye(nsamples))

      return np.dot(y - themean, np.dot(inv, y - themean))/nsamples

    Data = pymc.MvNormalCov('Data', mu=mean, C=variance*corr, value=y, observed=True)

    model = pymc.Model([thetas, exponents, corr, mean, variance, Data])
    thegraph = pymc.graph.graph(model, prog=r'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe', format='png', path=r'.')
    imdata = plt.imread('.\\container.png')
    plt.title("A schematic of the model.")

    mcmc = pymc.MCMC([thetas, exponents, corr, mean, variance, Data])
    mcmc.sample(iter=ntrials, burn=nburn, thin=nthin)
    thegraph = pymc.graph.graph(mcmc, prog=r'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe', format='png', path=r'.')
    imdata = plt.imread('.\\container.png')
    plt.title("A schematic of the model.")
    pymc.Matplot.plot(mcmc)

    mapmodel = pymc.MAP([thetas, exponents, corr, mean, variance, Data])
    mapmodel.fit()
    print('log-probability: {0} (probability: {1})'.format(mapmodel.logp, np.exp(mapmodel.logp)))

    print('Making predictions...')

#==============================================================================
#     #TODO: Make a proper estimate of the fit using all trials.
#     meancorr = np.mean(mcmc.trace('corr')[:], axis=0)
#     meanmean = np.mean(mcmc.trace('mean')[:])
#     meanthetas = np.array([np.mean(mcmc.trace('theta_{0}'.format(i))[:]) for i in range(3)])
#     meanexps = np.array([np.mean(mcmc.trace('exponent_{0}'.format(i))[:]) for i in range(3)])
#     predictions = np.empty(len(newpoints), dtype=np.float64)
#     for i, point in enumerate(newpoints):
#       rvec = np.exp(-np.sum(meanthetas*np.abs(point - samplepoints)**meanexps, axis=1))
#       predictions[i] = meanmean + np.dot(rvec.T, np.dot(np.linalg.inv(meancorr), (y - np.ones(len(y))*meanmean)))
#==============================================================================

    def do_predictions(mcmc, samplepoints, newpoints, y):
      """
      The interpolated values as a VaR trial x MCMC trial array.
      """
      #Need to do it vectorised to get accurate calculations!
      #the line below makes the r-vector over [MCMC trials, rvec(len(sobols)), VaR shock]
      rvecs = \
        np.exp(-(mcmc.trace('theta_0')[:, None, None]*
                 np.abs(newpoints[:,0] - samplepoints[:,0,None])**
                 mcmc.trace('exponent_0')[:, None, None] +
                 mcmc.trace('theta_1')[:, None, None]*
                 np.abs(newpoints[:,1] - samplepoints[:,1,None])**
                 mcmc.trace('exponent_1')[:, None, None] +
                 mcmc.trace('theta_2')[:, None, None]*
                 np.abs(newpoints[:,2] - samplepoints[:,2,None])**
                 mcmc.trace('exponent_2')[:, None, None])).astype(float)

      #the line below returns [MCMCtrials, len(sobols)xlen(sobols) corr matrix]
      corrs = mcmc.trace('corr')[:].astype(float)
#==============================================================================
#       #make the interpolation predictions. Note that mean is [MCMC trials, len(sobols)]
#       # but the values are constant along the second dimension (it's just needed
#       # to make the np.MvNormal call work)
#       N_MCMC = np.shape(rvecs)[0]
#       N_VaR = np.shape(rvecs)[2]
#       predictions = np.empty([N_MCMC, N_VaR], dtype=float)
#==============================================================================

      #Broadcasting is the most efficient (and unreadable) way to go:
      endbit = y[None, :] - mcmc.trace('mean')[:][:,0, None]
      #Left-most dot-product
      firstdp = np.sum(np.linalg.inv(corrs)*endbit[:, :, None], axis=1)

      #Note that this operation makes the output [VaRtrials, MCMCtrials]
      secondterm = np.sum(np.rollaxis(rvecs, 2)*firstdp[:,:], axis=2)
      #Finally, add the mean. Add the VaR dimension as the first one.
      #Note that the last dimension of mean is redundant -- just used for
      #making MvNormal work.
      return mcmc.trace('mean')[:][None, :, 0] + secondterm

      #To avoid a memoryerror, let's loop over the MCMC trials and use einsum:
      #Actually, this code is WAY too slow.
#==============================================================================
#       for MCMCtrial in range(N_MCMC):
#         for VaRtrial in range(N_VaR):
#           predictions[MCMCtrial, VaRtrial] = mcmc.trace('mean')[:][MCMCtrial, 0] + \
#             np.dot(rvecs[MCMCtrial,:,VaRtrial].T,
#                    np.dot(np.linalg.inv(corrs[MCMCtrial,:,:]),
#                           rvecs[MCMCtrial,:,VaRtrial]))
#==============================================================================
#==============================================================================
#           predictions[MCMCtrial,:] = mcmc.trace('mean')[:][MCMCtrial,0,None] + \
#             np.einsum('ij,ij->i', rvecs[MCMCtrial,:,:].T,
#                       np.dot(np.linalg.inv(corrs[MCMCtrial,:,:]), rvecs[MCMCtrial,:,:]).T)
#==============================================================================

    return mcmc, mapmodel, do_predictions(mcmc, samplepoints, newpoints, y)