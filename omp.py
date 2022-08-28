import numpy as np
from qpsolvers import solve_ls

def OMP(y_r, M_r, M_mp, M_mn, D, K_omp, epsilon_omp, CLIP_LEVEL, /, constrained, verbose):
    '''
    Orthogonal Matching Pursuit (OMP) declipping algorithm.
    Returns the sparsest representation of a given vector, provided a dictionary.
    
    Parameters
    y_r: The reliable (not clipped) samples of the audio signal.
    M_r: The measurement matrix such that y_r = M_r · s, where s is the reconstructed audio signal.
    D.name: Dictionary (\'C\' for DCT or \'G\' DGT). Matrix comprised by elementary signals, or atoms.
    D.matrix: The dictionary values. 
    K_omp: The number of discrete frequency bins.
    epsilon_omp: The tolerance threshold.
    CLIP_LEVEL: The maximum absolute value of the clipped audio signal.
    constrained: Whether to enforce the unconstrained ('omp') or the constrained ('lfcomp') problem.
    verbose: Toggles verbosity.
    '''
    # Initialization of data structures.
    name = D['name']
    D = D['matrix']

    # Frequency bins quantity.
    if(name == 'C'):
        K = D.shape[1]
        W = np.identity(K)
    else:
        K = D.shape[1] // 2
        W = np.identity(2 * K)

    # Normalization matrix.
    W *= 1/np.linalg.norm(M_r.dot(D), axis=0)

    # The dictionary is multiplied with the measurement matrix M_r and its columns normalized.        
    D_ = M_r.dot(D).dot(W)
    
    # DCT and DST dictionaries. Their concatenation results in a dictionary that considers the phase component of audio signals.
    if(name == 'G'):
        D_c_, D_s_ = D_[:, :K], D_[:, K:]
    
    # Frequency counter.
    k = 0
    
    # Support set.
    support = set()
    
    # Residual vector.
    r = y_r
    
    # Sparse vector memory allocation.
    if(name == 'C'):
        x_k = np.zeros(K)
    else:
        x_k = np.zeros(2 * K)
    
    # The length of the missing samples vectors is stored for later use.
    L_Imp, L_Imn = M_mp.shape[0], M_mn.shape[0]
    
    # The loop stops when the error threshold is reached or when all columns of the dictionary are visited.
    while k < K_omp and r.dot(r) > epsilon_omp:

        # The frequency counter k is increased by one.
        k += 1
        
        # The residual is reshaped as a multidimensional array to take advantage of numpy's matrix multiplication capabilities.
        r = r.reshape(-1, 1)

        if(name == 'C'):
            # We compute the dot product between all the atoms and the current residual. This gives us the correlation between the vectors.
            correlation = np.abs(np.sum(r * D_, axis=0))
            # The atom index related to the highest correlation is saved in memory.
            atom = np.argmax(correlation)   
        else:
            # Gabor's dictionary atom selection step.
            x_c_ = np.array([(np.sum((D_c_ * r), axis=0) - np.sum(D_c_ * D_s_, axis=0) * np.sum(D_s_ * r, axis=0))/(1 - np.sum(D_c_ * D_s_, axis=0) ** 2)])
            x_s_ = np.array([(np.sum(D_s_ * r, axis=0) - np.sum(D_c_ * D_s_, axis=0) * np.sum(D_c_ * r, axis=0))/(1 - np.sum(D_c_ * D_s_, axis=0) ** 2)])
        
            # Objective function that we want to minimize.
            result = np.linalg.norm(r - D_c_ * x_c_ - D_s_ * x_s_, axis=0) ** 2
            
            # In DGT, the atom is the vector with the shortest norm of the previous step's objective function.
            atom = np.argmin(result)
        
        # The atom index is stored inside a set.
        support.add(atom)

        # In DGT, a DST matrix is concatenated with a DCT matrix.
        if(name == 'G'):
            support.add(atom + K)
        
        # Then, the support set is casted into a list for later use.
        support_list = list(support)
        
        # The length of the support and the reliable samples vector is obtained.
        L_sup = len(support)
        L_r = y_r.size
        
        # Here we perform the memory allocation of a new dictionary built from the chosen atoms.
        D__ = np.empty((L_r, L_sup))
        
        # The chosen atoms are assigned to the empty dictionary.
        D__[:, :L_sup] = D_[:, support_list]
        
        # The sparse vector is obtained. As we are dealing with an underdetermined system (infinite solutions), 
        # we will obtain the solution with the minimum orthogonal error by means of a least-squares projection.
        # The projection is computed using the pseudoinverse. The linalg numpy module is leveraged for this task.
        x_k[support_list] = np.linalg.pinv(D__).dot(y_r)

        # And we update the residual using the recent result.
        r = y_r - D__.dot(x_k[support_list])

        # A final update on x_k is made in order to improve the solution for the declipping problem.
        if constrained:
            MAX_LEVEL = np.inf
            
            # The following conditional checks if the maximum sparsity level or the minimum residual error has been reached. 
            # It also considers if there are positive, negative or both types of clipped samples in the current frame.
            if (k == K_omp or r.dot(r) < epsilon_omp) and (L_Imp != 0 or L_Imn != 0):
                if verbose: print('Entrando al optimizador convexo...')
                
                # The explanation of the following least squares problem is in https://scaron.info/doc/qpsolvers/least-squares.html
                if L_Imp != 0 and L_Imn != 0:
                    G0 = (-1) * M_mp.dot(D).dot(W[:, support_list])
                    G1 = M_mp.dot(D).dot(W[:, support_list])
                    G2 = M_mn.dot(D).dot(W[:, support_list])
                    G3 = (-1) * M_mn.dot(D).dot(W[:, support_list])
                    G = np.vstack((G0, G1, G2, G3))

                    h0 = np.ones(M_mp.shape[0]) * (- CLIP_LEVEL)
                    h1 = np.ones(M_mp.shape[0]) * MAX_LEVEL
                    h2 = np.ones(M_mn.shape[0]) * (- CLIP_LEVEL)
                    h3 = np.ones(M_mn.shape[0]) * MAX_LEVEL
                    h = np.hstack((h0, h1, h2, h3))

                elif L_Imp != 0: 
                    G0 = (-1) * M_mp.dot(D).dot(W[:, support_list])
                    G1 = M_mp.dot(D).dot(W[:, support_list])
                    G = np.vstack((G0, G1))
                    
                    h0 = np.ones(M_mp.shape[0]) * (- CLIP_LEVEL)
                    h1 = np.ones(M_mp.shape[0]) * MAX_LEVEL
                    h = np.hstack((h0, h1))
                    
                elif L_Imn != 0: 
                    G2 = M_mn.dot(D).dot(W[:, support_list])
                    G3 = (-1) * M_mn.dot(D).dot(W[:, support_list])
                    
                    G = np.vstack((G2, G3))
                    h2 = np.ones(M_mn.shape[0]) * (- CLIP_LEVEL)
                    h3 = np.ones(M_mn.shape[0]) * MAX_LEVEL
                    h = np.hstack((h2, h3))

                # We implement a convex optimization solver for the constrained least squares (CLS) problem.
                try:
                    x_k[support_list] = solve_ls(D__, y_r, G, h, solver='cvxpy', verbose=verbose)

                    # If there is no solution to the convex optimization problem, or the solution exceeds 
                    # the tolerable error use the unconstrained solution.
                    s = y_r - D__.dot(x_k[support_list]) 
                    if np.all(np.isnan(x_k[support_list])) or s.dot(s) > 1000 * epsilon_omp:
                        x_k[support_list] = np.linalg.pinv(D__).dot(y_r)
                        if s.dot(s) > epsilon_omp and verbose: 
                            print(f'Superado el error tolerable de LFC-OMP. Se utiliza la solución sin restricciones de OMP.')
                    # If there is an solver exception raised use the unconstrained solution.
                except:
                    if verbose:
                        print('Excepción en solucionador convexo. Se utiliza la solución sin restricciones de OMP.')
                    x_k[support_list] = np.linalg.pinv(D__).dot(y_r)

                if verbose: print('...saliendo del optimizador convexo.')    
     
    return W.dot(x_k)