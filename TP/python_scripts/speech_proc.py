# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 16:42:18 2020

@author: fontaine
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import matplotlib.widgets as wid
import randproc as rp
import wave
from scipy import linalg as la


#%% Implementation of a analysis/synthesis system for speech signal


#%% Read sound data from file
filename = '../phrasezoe8k.wav'

wavObject = wave.open(filename)
nchannels, sampwidth, framerate, nframes, comptype, compname  = wavObject.getparams()

s = wavObject.readframes(nframes)
samples =  np.frombuffer(s, dtype='int16')/32768.
 
# Pre-emphasis
x = sig.lfilter([1, -0.98], [1], samples)


#%%
# Analysis parameters
frameLen = 0.04 # frame length in seconds
p        =   12 # AR parameter model
overlap  =   50 # frame overlap in %
Fmax     =  200 # maximum voicing frequency
Fmin     =   60 # minimum voicing frequency

Nx       = int(frameLen*framerate) # length of the frames (number of samples)
lag      = int(Nx * overlap / 100)
framesToProc  = int((nframes-Nx)/lag - 0.5) # floor
minT     = int(framerate/Fmax - 0.5)       # floor
maxT     = int(framerate/Fmin + 0.5)       # ceil
                   
synth    = np.zeros_like(samples)
pitch    = np.zeros(framesToProc)
coeff    = np.zeros([p,framesToProc])
sigma2   = np.zeros(framesToProc)


#%% Create a figure for interactive display
plt.close('all')
class btnAction(object):
    stopnow = 0
    def stop(self,event):      
        self.stopnow = 1

plt.figure(1,[10, 8])
plt.grid()
axstop = plt.axes([0.75, 0.01, 0.12, 0.05])
btn = wid.Button(axstop, 'Stop refresh')
callback = btnAction()
btn.on_clicked(callback.stop)


#%% Loop over frames
position = 0
for frameIndex in np.arange(framesToProc):
    startSample = frameIndex*lag
    stopSample  = startSample + Nx
    frame       = x[startSample:stopSample]
    
    ### ANALYSIS
    ## Pitch detection
    minFreq = 80 # Hz
    maxFreq = 400 # Hz
    minT    = int(framerate/maxFreq) # samples
    maxT    = int(framerate/minFreq) # samples
    pitch[frameIndex] = rp.detectPitch(frame, minT, maxT)
    
    
    ## AR estimation    
    ## estimate the coefficients for the current frame
    ## note that the leading "1" does not need to be stored <-- WRONG ?
    
    gamma =  rp.acovb(frame)
    Gamma_hat = la.toeplitz(gamma[0:p])
    v1 = np.zeros(p)
    v1[0] = 1
    v = la.inv(Gamma_hat) @ v1
    
    # estimating the variance
    sigma2Est = 1/v[0]
    
    # estimating the AR coefficients
    estimated_coeff = sigma2Est * v
    estimated_coeff[0] = 1
    
    coeff[:,frameIndex] = estimated_coeff # 1D array of size p
    sigma2[frameIndex]  = sigma2Est # scalar
    
    c1 = estimated_coeff
    
    # Graphical check
    if callback.stopnow == 0:
        plt.figure(1)
        ax1 = plt.subplot(1, 1, 1)
        plt.ylim([-80, 0])
        nPoints = 2**9
        FRAME = np.fft.fft(frame, 2*nPoints)
        XF    = 10*np.log10((abs(FRAME[0:nPoints])**2)/Nx)
        XAR   = 1/abs(np.fft.fft(c1,1024))
        XAR   = 20*np.log10(np.sqrt(sigma2Est)*XAR[0:nPoints])
        nu = np.linspace(0, 0.5 - 1/nPoints, nPoints)  
        
        if frameIndex == 0: # Create the line at iteration 0
            line1, = plt.plot(nu,XF)
            line2, = plt.plot(nu,XAR)
        else: # Update the sample covariance plot
            ax1.tick_params(labelleft = False,
                labelbottom = False)        
            line1, = plt.plot(nu, XF)
            line2, = plt.plot(nu, XAR)
        
        plt.legend(['Empirical PSD', 'PSD from AR model'])
        plt.title('Analysis of frame %4d'%frameIndex)
        plt.pause(0.01)
        ax1.clear()        
        
    
    ### SYNTHESIS
    if pitch[frameIndex]: # Voiced sound
        T = 0.5*pitch[frameIndex]
        pT = np.zeros(Nx+p)
        pT[position::int(T)]=1 # "Dirac comb" or rather pulse train            
        position = int(T)
        xs = sig.lfilter([1], c1, pT)            
    else: # non voiced sound
        Z = np.random.normal(0, 1, Nx+p) # White noise          
        xs = sig.lfilter([1], c1, Z)
        
    xs = xs[p::]    
    normFact = np.sqrt(sigma2[frameIndex]/np.var(xs))
    xs = normFact * xs    
    synth[startSample:stopSample] = synth[startSample:stopSample] + xs * rp.myhann(Nx)
    
# De-emphasis filter
synth = sig.lfilter([1], [1, -0.98], synth)


#%% Write the WAV file
data = synth*(2**15)
wavObject2 = wave.open('synth.wav','wb')

wavObject2.setparams((nchannels, sampwidth, framerate, nframes, "NONE", "Uncompressed"))
wavObject2.writeframes(data.astype('int16').tobytes())
wavObject2.close()


#%% Plots
plt.figure(2)
plt.subplot(311)
t = np.arange(nframes, dtype=float)/framerate
plt.plot(t, samples)

tp = np.arange(framesToProc, dtype=float)*lag/framerate
plt.subplot(312)
plt.plot(tp, pitch)
plt.subplot(313)
plt.plot(tp, sigma2)
