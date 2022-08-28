import numpy as np
from scipy import signal, integrate, optimize, interpolate

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
    clipping_threshold = optimize.brentq(diffSDR, eps, 0.99 * np.max(np.abs(audio)))

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

def interpolate_linearly(audio, fs, sampling_factor):
    '''
    Linear interpolation by a given upsampling factor.

    audio: Input audio signal.
    sampling_factor: Factor to which increase the signal's sample frequency rate.
    '''
    interpolator = interpolate.interp1d(np.arange(audio.size), audio)
    extended_audio = np.linspace(0, audio.size-1, int(audio.size * sampling_factor))
    upsampled_fs = fs * sampling_factor
    return interpolator(extended_audio), upsampled_fs

def apply_antialiasing(audio, fs):
    '''
    Applies 1st order Bessel low pass antialiasing filter to an audio signal.

    audio: Input audio signal.
    '''
    fc = 22050 # Cutoff frequency

    rs = 100 # Attenuation level in dB at cutoff frequency.

    sos = signal.iirfilter(30, fc, btype='lp', rs=rs,

                analog=False, ftype='cheby2',

                output='sos', fs=fs)
    
    return signal.sosfilt(sos, audio)

def get_clipping_threshold(analog_original, upsampled_audio, upsampled_fs, desiredSDR):
    '''
    Obtains the clipping threshold given an input desired SDR.

    analog_original: Input audio signal.
    upsampled_audio: Oversampled audio signal.
    upsampled_audio: Oversampled audio signal sample frequency.
    desiredSDR: Desired SDR of the output audio signal, after being clipped by the clipping threshold.
    '''

    p = lambda x: SDR(analog_original, apply_antialiasing(hardclip(upsampled_audio, x)[0], upsampled_fs)) - desiredSDR  

    bracket = [np.finfo(np.float64).eps, 0.99 * np.max(np.abs(analog_original))]

    rootresult = optimize.root_scalar(p, bracket=bracket, xtol=1e-5)
    
    return rootresult.root

def decimate(audio, sampling_factor):
    '''
    Decimates an input audio signal by a given downsampling factor.

    audio: Input audio signal.
    sampling_factor: Factor by which apply the downsampling.
    '''
    return signal.resample_poly(audio, 1, sampling_factor, window='rectangular')

def analog_hardclip(filtered_no_clipping, upsampled_audio, upsampled_fs, sampling_factor, desiredSDR):
    '''
    Simulates hardclipping in the analog signal domain.

    Parameters
    filtered_no_clipping: Low-pass filtered interpolated input audio without clipping.
    upsampled_audio: Interpolated input audio.
    upsampled_fs: Sample rate as incremented by the oversampling factor.
    desiredSDR: Goal SDR between the analog clipped and analog original signal.
    sampling_factor: Sampling factor to increase the sample frequency rate.

    Returns
    analog_clipped: Analog clipped resulting audio signal.
    masks: Reliable and clipped samples indexes.
    '''
    optimal_clipping_threshold = get_clipping_threshold(filtered_no_clipping, upsampled_audio, upsampled_fs, desiredSDR)

    clipped_audio = hardclip(upsampled_audio, optimal_clipping_threshold)[0]

    filtered_audio = apply_antialiasing(clipped_audio, upsampled_fs)

    analog_clipped = decimate(filtered_audio, sampling_factor)
    
    return analog_clipped

def get_fejer_threshold(analog_clipped, n=500):
    fejer_threshold = np.max(np.abs(fejer_averaging_ola(analog_clipped, n)))

    masks = {
        'r': np.zeros(analog_clipped.size, dtype=bool),
        'c+': np.zeros(analog_clipped.size, dtype=bool),
        'c-': np.zeros(analog_clipped.size, dtype=bool)
    }

    masks['r'][:] = (analog_clipped < fejer_threshold) == (analog_clipped > -fejer_threshold)
    masks['c+'][:] = analog_clipped > fejer_threshold
    masks['c-'][:] = analog_clipped < -fejer_threshold
    
    return fejer_threshold, masks, n