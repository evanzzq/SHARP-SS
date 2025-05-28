import numpy as np
from scipy.signal import windows
from sharp_ss.model import Model, Prior

def create_G_from_model(model: Model, prior: Prior):
    """
    Create G in the time domain (as a time series).
    """

    # Initialize G_model
    tlen_sp = int(round(prior.tlen/prior.dt)) # in sample points
    G = np.zeros(tlen_sp * 2 - 1)

    # Build center Gaussian
    camp = 1.0
    cwid = 3  # in sample points (minimum)
    gauss_0 = camp * windows.gaussian(cwid, std=cwid/6)  # MATLAB's gausswin ~ Gaussian with std ~ width/6

    # Insert center Gaussian
    G[tlen_sp-2:tlen_sp+1] = gauss_0

    # Return if there's no phase in model
    if model.Nphase == 0:
        return G

    # Build other Gaussians in model
    for iphase in range(model.Nphase):
        
        # Build the Gaussian
        loc = int(model.loc[iphase] / prior.dt) # in sample points
        amp = model.amp[iphase]
        wid = int(model.wid[iphase] / prior.dt) # in sample points
        gauss = amp * windows.gaussian(wid, std=wid/6)

        # Insert the Gaussian
        try:
            start_idx = int(tlen_sp+loc-wid/2)
            G[start_idx:start_idx+wid] = gauss
            start_idx_neg = int(tlen_sp-loc-wid/2)
            G[start_idx_neg:start_idx_neg+wid] = -gauss
        except:
            pass
        
    return G

def convolve_P_G(P, G):
    """
    Convolve P with a single G trace.
    
    Args:
        P: (n,) array
        G: (nG,) array (single trace)

    Returns:
        D: (n,) array, the result of the convolution
    """
    npts_fft = 2 ** (1 + int(np.ceil(np.log2(len(P)))))
    fftP = np.fft.fft(P, n=npts_fft)

    nG = len(G)

    # Padding G symmetrically
    pad_left = int(np.ceil((npts_fft - nG) / 2))
    pad_right = int(np.floor((npts_fft - nG) / 2))
    G_padded = np.pad(G, (pad_left, pad_right), mode='constant')

    # FFT of G
    fftG = np.fft.fft(G_padded)

    # Multiply and inverse FFT
    Dtmp = np.fft.ifft(fftP * fftG)
    Dtmp = np.fft.fftshift(Dtmp)

    D = np.real(Dtmp[:len(P)])

    return D

def align_D(D_model, D, align):
    """
    Align the D_model and D traces based on the max amplitude of D_model.
    Aligns all traces in D if D is 2D.
    
    Args:
        D_model: (n,) array, model data trace
        D: (n, m) array, where n is the number of sample points and m is the number of traces
        align: max shift in sample points; note that prior.align is in seconds (should be divided by prior.dt when passing in)
    
    Returns:
        D_model_aligned: (n,) or (n, m) array, aligned model data trace
        D_aligned: (n, m) array, aligned data
    """
        
    # Cut length in sample points
    cut_pts = int(np.ceil(len(D_model) / 2 - align))
    
    # Find max index of D_model
    maxind = np.argmax(D_model)
    center_time = len(D_model) / 2
    
    # If mismatch is larger than align, abort
    if abs(maxind - center_time) >= align:
        return D_model, D      
    
    # Midpoint index for D (for 2D D)
    midind = D.shape[0] // 2
    
    # Align D_model
    D_model_aligned = D_model[maxind - cut_pts : maxind + cut_pts + 1]
    
    # Align D (apply cut to each trace)
    D_aligned = D[midind - cut_pts : midind + cut_pts + 1, :]
    
    return D_model_aligned, D_aligned