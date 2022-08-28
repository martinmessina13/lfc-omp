import numpy as np
from scipy import signal, integrate
from scipy.optimize import brentq

def next_multiple(number, dividend):
    '''
    Returns the next multiple of a given number, given a dividend.
    
    Parameters
    number: The number to obtain the closest multiple.
    dividend: The number used to obtain the next multiple.
    '''
    
    return int(np.ceil(number / dividend) * dividend)

def hardclip(audio, clippingLevel):
    '''
    Hardclips the input signal in a symmetrical fashion, given a clipping threshold.
     
    Parameters
    audio: The input audio signal.
    clippingLevel: The clipping threshold.
    '''

    clipped = np.copy(audio)
    
    masks = {
        'r': np.zeros(audio.size, dtype=bool),
        'c+': np.zeros(audio.size, dtype=bool),
        'c-': np.zeros(audio.size, dtype=bool)
    }

    masks['r'][:] = (audio < clippingLevel) == (audio > -clippingLevel)
    masks['c+'][:] = audio > clippingLevel
    masks['c-'][:] = audio < -clippingLevel

    clipped[masks['c+']] = clippingLevel
    clipped[masks['c-']] = -clippingLevel

    return clipped, masks

def SDR(x, x_hat):
    '''
    Obtains SDR of an estimation signal by comparing to ground truth.

    audio: Original audio signal, ground truth.
    x_hat: Estimated audio signal.
    '''
    return np.round(20 * np.log10(np.linalg.norm(x) / np.linalg.norm(x - x_hat)), 3)

def clip_SDR(audio, desiredSDR):
    '''
    Obtains the clipping threshold from a desired input SDR.

    audio: Input audio signal.
    desiredSDR: Desired SDR of the output clipped signal.
    '''

    # eps value is initialized
    eps = np.finfo(np.float64).eps

    # we want to minimize the difference between the desired SDR and a function used to obtain SDR dependent of variable audio, namely the clipping threshold
    diffSDR = lambda audio: SDR(audio, hardclip(audio, audio)[0]) - desiredSDR

    # # unpacking solution audio, and diffSDR value in solution f(audio)
    eps = np.finfo(np.float64).eps
    clipping_threshold = brentq(diffSDR, eps, 0.99 * np.max(np.abs(audio)))

    diff_from_desired_SDR = diffSDR(clipping_threshold)

    # the true SDR is computing adding the desired SDR to the diffSDR function evaluated at the found root
    trueSDR = desiredSDR + diff_from_desired_SDR

    # next step is to clip the signal using the recently found clipping threshold
    [clipped, masks] = hardclip(audio, clipping_threshold)

    # the percentage of clipped samples is also obtained
    percentage = np.round((np.sum(masks['c+'])+np.sum(masks['c-'])) / audio.size * 100, 3)

    return clipped, masks, clipping_threshold, trueSDR, percentage

def fourier_fixed(audio, n):
    '''
    Computes Fourier series for a given fixed samples input signal.

    audio: Input audio signal.
    n: Amount of harmonics. 
    '''
    l = audio.size/2
    ts = np.arange(audio.size)
    # Constant term
    a0=1/l*integrate.simpson(audio, ts)
    # Cosine coefficents
    A = np.zeros((n))
    # Sine coefficents
    B = np.zeros((n))

    series = a0/2.0
     
    for i in np.arange(1,n+1):
        A[i-1]=1/l* integrate.simpson(audio * np.cos(i * np.pi * ts / l), ts)
        B[i-1]=1/l* integrate.simpson(audio * np.sin(i * np.pi * ts / l), ts)

        series += A[i-1] * np.cos(i * np.pi * ts / l) + B[i-1] * np.sin(i * np.pi * ts / l)
    
    return series

def fejer_averaging(audio, n):
    '''
    Computes Fourier series for a given fixed samples input signal.

    audio: Input audio signal.
    n: Amount of harmonics. 
    '''
    s = np.zeros((audio.size, n))
    l = audio.size / 2
    ts = np.arange(audio.size)
    for i in np.arange(1, n + 1):
        if i == 1:
            s[:, i-1] = fourier_fixed(audio, i)
        else: 
            A = 1/l* integrate.simpson(audio * np.cos(i * np.pi * ts / l), ts)
            B = 1/l* integrate.simpson(audio * np.sin(i * np.pi * ts / l), ts)
            s[:, i-1] = s[:, i-2] +  A * np.cos(i * np.pi * ts / l) + B * np.sin(i * np.pi * ts / l) 
    return np.sum(s, axis=1) / (n + 1)

def fejer_averaging_ola(audio, n, N=1024):
    '''
    Computes Fejer averaging by means of the overlap-add method.

    audio: Input audio signal.
    n: Amount of harmonics. 
    N: Frame's length.
    '''

    oL = audio.size
    L = next_multiple(oL, N)
    audio = np.pad(audio, (0, L - oL))

    # In order to achieve 75% overlap, the offset distance (hop size) must be of N / 4.
    R = N // 4
    R_support = np.arange(0, L - N + 1, R)

    # The total number of frames is obtained.
    frames = R_support.size

    # The memory allocation for the matrix that will contain each of the frames is performed.
    Y = np.empty((N, frames))

    # Fejer averaging is computed for each frame.
    for i, r in enumerate(R_support):
        Y[:, i] = fejer_averaging(audio[r:r+N], n)

    # Memory allocation of a list that will contain the processed signal and another that will contain a sum of windows.
    proc_audio = np.zeros(L)
    proc_w_sin = np.zeros(L)
    
    # A sine window is used as the synthesis window.
    w_sin = signal.windows.cosine(N)

    # The signal is restored using the overlap-add method by definition.
    for i, r in enumerate(R_support):
        proc_audio[r:r+N] = proc_audio[r:r+N] + Y[:, i] * w_sin
        proc_w_sin[r:r+N] = proc_w_sin[r:r+N] + w_sin

    return (proc_audio / proc_w_sin)[:oL]