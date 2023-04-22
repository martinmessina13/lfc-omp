
# Low Frequency Constrained Orthogonal Matching Pursuit

Code repository for my thesis "Restauración de señales de audio digitales recortadas con aliasing reducido". The proposed algorithm restores aliasing-free clipped signals (e.g. those coming from a recording studio). The solution arises in terms of a successive approximation method based on the sparse signals framework. Signal restitution in the presented approach is possible due to the clipping threshold estimation's by means of the Fejér's averaging method. 

This procedure, coined Low Frequency Constrained Orthogonal Matching Pursuit (LFC-OMP), is a variation of the traditional algorithm Constrained Orthogonal Matching Pursuit or C-OMP to the use case of the aliasing-free clipped signals.

The results state a positive linear correlation between both algorithms after studying the relevant metrics obtained from restoring aliasing-free clipped signals with LFC-OMP and digitally clipped signals with C-OMP. Also, the absolute differences between each of these metrics are insignicant. These results imply that LFC-OMP can be used to restore clipped aliasing-free signals, in the same way that C-OMP is used to restore signals which were clipped in the digital domain.

invoke.py provides an script with both the call for the clipping threshold estimation method and the OMP algorithm while omp.py contains the code for the OMP algorithm, with comments explaining each line.
