import numpy as np
from helpers import SDR, interpolate_linearly, apply_antialiasing, decimate, analog_hardclip, get_fejer_threshold
import librosa
import os
from time import time
import soundfile as sf
from invoke import invoke_OMP
import warnings
import csv

warnings.filterwarnings("ignore", category=RuntimeWarning)

''' Analog clipped samples preparation script '''

# Parameters for the OMP algorithm
params = {}
params['N'] = 1024
params['epsilon'] = 1e-6
params['K_omp'] = params['N'] // 4
params['dictionary'] = 'G'
params['constrained'] = True
params['skip_clean_frames'] = True
params['analog_clipping'] = True
params['verbose'] = 1

path = './sounds/'
desired_fs = 5e6
inputSDRs = [20, 15, 10, 7, 5, 3, 1];

TIC = time()
for desiredSDR in inputSDRs:
    for file in os.listdir(path):
        with open('./results/lfcompresults.csv', 'a+', newline='') as csvfile:
            csvwrite = csv.writer(csvfile, delimiter=',')
            if file[-4:] == '.wav':
                    print(f'\nSeñal: {file}.')
                    print('\nComenzando preparación de la señal original (verdad fundamental)...')

                    [audio, fs] = librosa.load(path+file, sr=None)

                    sampling_factor = int(np.ceil(desired_fs/fs)) 

                    tic = time()
                    [upsampled_audio, upsampled_fs] = interpolate_linearly(audio, fs, sampling_factor)
       
                    filtered_no_clipping = apply_antialiasing(upsampled_audio, upsampled_fs)

                    analog_original = decimate(filtered_no_clipping, sampling_factor)
                    toc = time()

                    print(f'Tiempo de preparación transcurrido: {int((toc - tic)*1000)} ms. '+
                    f'\nFrecuencia de muestreo luego de interpolar: {upsampled_fs / 1e6} MHz.')

                    # Simulate analog clipping.
                    print('\nComenzando simulación de recorte analógico (preparado de señal a restaurar)...\n' +
                    f'Nivel de recorte deseado: {desiredSDR} dB.')  
                    tic = time()
                    analog_clipped = analog_hardclip(filtered_no_clipping, upsampled_audio, upsampled_fs, sampling_factor, desiredSDR)
                    toc = time()
                    print(f'Tiempo de simulación transcurrido: {int((toc - tic)*1000)} ms.')

                    finalSDR = SDR(analog_original, analog_clipped)
                    # print(f'Verdadero SDR de la señal a restaurar: {finalSDR} dB.')

                    assert np.allclose(finalSDR, desiredSDR, atol=0.1), 'La tolerancia entre el SDR deseado y el final es de más de 0.1 dB.'

                    # Fejer averaging
                    print('\nObteniendo umbral de recorte por medio del promediado de Fejer...')

                    tic = time()
                    params['fejer_threshold'], masks, n = get_fejer_threshold(analog_clipped)
                    toc = time()
                    print(f'Tiempo transcurrido realizando promediado de Fejer: {int((toc - tic))} s, n: {n}')

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
                    csvwrite.writerow(results)

                    sf.write(f'./signal/lfcomp/{desiredSDR}_recovered_{file}', y, samplerate=fs) # estimation
                    # sf.write(f'./restored/original/analog_{file}', analog_original, samplerate=fs) # analog original
                    sf.write(f'./signal/clipped/{desiredSDR}_analog_clipped_{file}', analog_clipped, samplerate=fs) # analog clipped
                    # break
    # break
TOC = time()
print(f'Tiempo total transcurrido: {np.round(TOC - TIC, 1)} s')

# np.save('results.npy', results)