import numpy as np
import scipy as sp
from omp import OMP
from helpers import next_multiple, hardclip

def invoke_OMP(audio, params):
    '''
    Utilitary function to call the Orthogonal Matching Pursuit (OMP) algorithm 
    and return the audio signal estimation by means of the overlapp-add method.
    
    Parameters
    audio: Signal to restore.
    params.N: Length of the frame.
    params.dictionary: Toggles the DCT or DGT dictionary method. Takes 'D' or 'G' as values, respectively.
    params.constrained: Whether to enforce the unconstrained ('omp') or the constrained ('lfcomp') problem.
    params.analog_clipping: Toggles analog clipping.
    params.skip_clean_frames: Whether to skip undistorted frames or not.
    params.verbose: Toggles verbosity. Available levels are 0, 1 and 2.
    params.fejer_threshold: Clipping threshold obtained by means of the Fejer averaging.
    '''

    # Error handling
    if 'N' not in params:
        raise ValueError('Longitud del segmento no especificada.')
    if 'dictionary' not in params:
        raise ValueError('Diccionario no especificado.')
    else:
        if params['dictionary'] not in ['C', 'G']:
            raise ValueError('El diccionario no se encuentra dentro de las opciones disponibles (\'C\', \'G\').')
    if 'constrained' not in params:
        raise ValueError('Modo del algoritmo (\'lfcomp\',\'omp\') no especificado.')
    if 'analog_clipping' not in params:
        raise ValueError('Recorte analógico no especificado.')
    if 'fejer_threshold' not in params:
        raise ValueError('Umbral de recorte brindado por promediado de Fejer no especificado.')
    if 'epsilon' not in params:
        raise ValueError('Error tolerable (\'epsilon\') no especificado.')
    if 'K_omp' not in params:
        raise ValueError('Máximo nivel de escasez (\'K omp\') no especificado.')
    if 'skip_clean_frames' not in params:
        raise ValueError('Booleano indicador de restauración en segmentos no distorsionados no especificado.')
    if 'verbose' not in params:
        raise ValueError('Booleano de verbosidad no especificado.')
    
    # Parameters
    N = params['N']
    dictionary = params['dictionary']
    constrained = params['constrained']
    skip_clean_frames = params['skip_clean_frames']
    analog_clipping = params['analog_clipping']
    verbose = params['verbose']
    fejer_threshold = params['fejer_threshold']
    epsilon = params['epsilon']
    K_omp = params['K_omp']

    # An iterator object is obtained from the frame's length
    N_support = np.arange(N)

    # In order to make frame-based processing possible we need to pad with zeros the original audio vector. 
    oL = audio.size # Original length
    L = next_multiple(oL, N) # New padded vector length
    audio = np.pad(audio, (0, L - oL))

    if analog_clipping:
        # If the context is analog clipping, Fejer averaging is employed to estimate the clipping threshold    
        CLIP_LEVEL = fejer_threshold
        # Since the saturation does not have constant amplitude, the input audio signal is hardclipped.
        audio = hardclip(audio, CLIP_LEVEL)[0]
    else:
        # If the context is digital clipping, the clipping level is obtained from considering the infinity norm (defined by the maximum amplitude level) of the audio vector. 
        CLIP_LEVEL = np.max(audio)
    
    # We assign a different value to the frequency bins K variable depending on what dictionary we are working on.
    if dictionary =='C':
        K = N * 2
        K_support = np.arange(K)

    elif dictionary =='G':
        K = N
        K_support = np.arange(K)

    ''' Dictionary '''
    
    # We will use a rectangular window for the dictionary and as the analysis window per frame.
    w = np.ones(N)

    # DCT (Discrete Cosine Transform)
    if dictionary =='C':
        # Memory allocation for the dictionary.
        D = np.empty((N, K))
        
        # Now we gradually complete each column of the dictionary through iteration for each case.
        for j in K_support:
            D[:, j] = w * np.cos((np.pi / K) * (N_support + 1/2) * (j + 1/2))

    # DGT (Discrete Gabor Transform)
    elif dictionary =='G':
        D = np.empty((N, 2 * K))
        
        for j in K_support:
            D[:, j] = w * np.cos((np.pi / K) * (N_support + 1/2) * (j + 1/2))
            D[:, j + K] = w * np.sin((np.pi / K) * (N_support + 1/2) * (j + 1/2))

    # The dictionary name and matrix values are stored. 
    D = {
        'name':dictionary,
        'matrix':D
    }

    ''' Frame-based processing '''

    # In order to achieve 75% overlap, the offset distance (hop size) must be of N / 4.
    R = N // 4
    R_support = np.arange(0, L - N + 1, R)

    # The total number of frames is obtained.
    frames = R_support.size

    # The memory allocation for the matrix that will contain each of the frames is performed.
    Y = np.empty((N, frames))

    # The rectangular window is applied and each frame is assigned to the matrix Y.
    for i, r in enumerate(R_support):
        Y[:, i] = w * audio[r:r+N]

    # We will also generate the measurement matrix M from an identity matrix of dimension N x N.
    M = np.identity(N)

    # Memory allocation for the matrix that will contain the processed frames.
    if dictionary =='C':
        # DCT.
        X_k = np.zeros((K, frames))
    else:
        # Gabor.
        X_k = np.zeros((2 * K, frames))
    
    # From here on, we will process each frame independently and put it in each column of the matrix X_k.
    for i in np.arange(frames):

            # Frame assignment.
            y = Y[:, i]

            # The support of the reliable samples I_r. 
            # It is obtained through finding the samples where the clipping level is not reached.
            I_r = hardclip(y, CLIP_LEVEL)[1]['r']
            
            # The reliable samples support vector is used to obtain the reliable samples vector.
            y_r = y[I_r]

            if y_r.size == y.size and skip_clean_frames: # no clipped samples
                continue
            if verbose > 0: print(f'Frame {i}/{frames-1}')

            # if analog_clipping:
            #     I_mp, I_mn = np.where(y >= CLIP_LEVEL)[0], np.where(y <= - CLIP_LEVEL)[0]
            # else:
            I_mp, I_mn = np.where(y == CLIP_LEVEL)[0], np.where(y == - CLIP_LEVEL)[0]
            
            # Measurement matrices are built upon the support vectors also.
            M_r = M[I_r]
            M_mp = M[I_mp]
            M_mn = M[I_mn]

            # Further, we multiply this value for the amount of elements in the reliable samples support vector.
            epsilon_omp = epsilon * I_r.size

            # # The OMP algorithm is called.
            x_k = OMP(y_r, M_r, M_mp, M_mn, D, K_omp, epsilon_omp, CLIP_LEVEL, constrained, verbose == 2)

            # After execution, we store the sparse representation into the previously allocated in memory matrix X_k.
            X_k[:, i] = x_k

    ''' Overlap-Add Method '''
    
    # Memory allocation of a list that will contain the processed signal and another that will contain a sum of windows.
    proc_audio = np.zeros(L)
    proc_w_sin = np.zeros(L)
    
    # A sine window is used as the synthesis window.
    w_sin = sp.signal.windows.cosine(N)

    # The signal is restored using the overlap-add method by definition.
    for i, (z, r) in enumerate(zip(X_k.T, R_support)):
        if np.any(z):
            # if there any non-zero values use the estimated frame for the result
            proc_audio[r:r+N] = proc_audio[r:r+N] + D['matrix'].dot(z) * w_sin
        else:
            # if all frame values are zero use the input signal frame
            proc_audio[r:r+N] = proc_audio[r:r+N] + Y[:, i] * w_sin
        proc_w_sin[r:r+N] = proc_w_sin[r:r+N] + w_sin

    proc_audio = (proc_audio / proc_w_sin)[:oL] # changing back to the length of the input audio vector
    
    return proc_audio