import numpy as np
from scipy import signal, optimize, interpolate
from helpers import fejer_averaging_ola, hardclip, SDR
import librosa
import os
from time import time
import soundfile as sf
from invoke import invoke_OMP
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

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

def analog_hardclip(filtered_no_clipping, upsampled_audio, upsampled_fs, desiredSDR):
    '''
    Simulates hardclipping in the analog signal domain.

    Parameters
    filtered_no_clipping: Low-pass filtered interpolated input audio without clipping.
    upsampled_audio: Interpolated input audio.
    upsampled_fs: Sample rate as incremented by the oversampling factor.
    desiredSDR: Goal SDR between the analog clipped and analog original signal.

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
    tic = time()
    # fejer_threshold = np.max(np.abs(fejer_averaging_ola(analog_clipped, n))) * (1 - 0.0486)
    fejer_threshold = np.max(np.abs(fejer_averaging_ola(analog_clipped, n)))
    toc = time()
    print(f'Tiempo transcurrido realizando promediado de Fejer: {int((toc - tic))} s, n: {n}')

    masks = {
        'r': np.zeros(analog_clipped.size, dtype=bool),
        'c+': np.zeros(analog_clipped.size, dtype=bool),
        'c-': np.zeros(analog_clipped.size, dtype=bool)
    }

    masks['r'][:] = (analog_clipped < fejer_threshold) == (analog_clipped > -fejer_threshold)
    masks['c+'][:] = analog_clipped > fejer_threshold
    masks['c-'][:] = analog_clipped < -fejer_threshold
    
    return fejer_threshold, masks

path = './sounds/'
proposed_fs = 5e6
inputSDRs = [20, 15, 10, 7, 5, 3, 1];
params = {}

TIC = time()
for desiredSDR in inputSDRs:
    for file in os.listdir(path):
            if file[-4:] == '.wav':
                    print(f'\nSeñal: {file}.')
                    print('\nComenzando preparación de la señal original, verdad fundamental...')

                    [audio, fs] = librosa.load(path+file, sr=None)

                    sampling_factor = int(np.ceil(proposed_fs/fs)) 

                    tic = time()
                    [upsampled_audio, upsampled_fs] = interpolate_linearly(audio, fs, sampling_factor)
       
                    filtered_no_clipping = apply_antialiasing(upsampled_audio, upsampled_fs)

                    analog_original = decimate(filtered_no_clipping, sampling_factor)
                    toc = time()

                    print(f'Tiempo de preparación transcurrido: {int((toc - tic)*1000)} ms. '+
                    f'\nFrecuencia de muestreo luego de interpolar: {upsampled_fs / 1e6} MHz.')

                    # Simulate analog clipping.
                    print('\nComenzando simulación de recorte analógico, señal a restaurar...\n' +
                    f'Nivel de recorte deseado: {desiredSDR} dB.')  
                    tic = time()
                    analog_clipped = analog_hardclip(filtered_no_clipping, upsampled_audio, upsampled_fs, desiredSDR)
                    toc = time()
                    print(f'Tiempo de simulación transcurrido: {int((toc - tic)*1000)} ms.')

                    finalSDR = SDR(analog_original, analog_clipped)
                    # print(f'Verdadero SDR de la señal a restaurar: {finalSDR} dB.')

                    assert np.allclose(finalSDR, desiredSDR, atol=0.1), 'La tolerancia entre el SDR deseado y el final es de más de 0.1 dB.'

                    #Parameters
                    params['N'] = 1024
                    params['dictionary'] = 'G'
                    params['constrained'] = True
                    params['skip_clean_frames'] = True
                    params['analog_clipping'] = True
                    params['verbose'] = 1

                    # Fejer averaging
                    print('\nObteniendo umbral de recorte por medio del promediado de Fejer...')
                    params['fejer_threshold'], masks = get_fejer_threshold(analog_clipped)

                    ### Restoration
                    print('\nComenzando restauración...')
                    tic = time()
                    y = invoke_OMP(analog_clipped, params)
                    toc = time()

                    lapsed_time = np.round(toc - tic, 1)
                    
                    sdr = SDR(analog_original, y) # estimation sdr
                    csdr = SDR(analog_original, analog_clipped) # clipped signal sdr
                    delta_sdr = np.round(sdr - csdr, 3)

                    sdr_c = SDR(analog_original[~masks['r']], y[~masks['r']]) # sdr on clipped samples
                    csdr_c = SDR(analog_original[~masks['r']], analog_clipped[~masks['r']]) # sdr on clipped samples
                    delta_sdr_c = np.round(sdr_c - csdr_c, 3)

                    print(f'LFC-OMP tardó {lapsed_time} s en completar la restauración.\n' +
                    f'SDRc de la señal recortada: {csdr_c} dB.\nSDRc de la señal restaurada: {sdr_c} dB.\n' + f'La mejora del SDRc es de {delta_sdr_c} dB.\n' +
                    f'\nSDR de la señal recortada: {csdr} dB.\nSDR de la señal restaurada: {sdr} dB.\nLa mejora del SDRc es de {delta_sdr} dB.\n')
                          
                    results = np.array([desiredSDR, file, sdr_c, csdr_c, delta_sdr_c, sdr, csdr, delta_sdr, lapsed_time], dtype=str)

                    sf.write(f'./signal/lfcomp/{desiredSDR}_recovered_{file}', y, samplerate=fs) # estimation
                    # sf.write(f'./restored/original/analog_{file}', analog_original, samplerate=fs) # analog original
                    sf.write(f'./signal/clipped/{desiredSDR}_analog_clipped_{file}', analog_clipped, samplerate=fs) # analog clipped
                    break
    break
TOC = time()
print(f'Tiempo total transcurrido: {np.round(TOC - TIC, 1)} s')

np.save('results.npy', results)