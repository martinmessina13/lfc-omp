# Low Frequency Constrained Orthogonal Matching Pursuit

Algorithm to restore aliasing-free clipped signals (e.g. those coming from a recording studio). The solution arises in terms of a successive approximation method based on the sparse signals framework. Signal restitution in the presented approach is possible due to the clipping threshold estimation's by means of the Fejér's averaging method. This procedure, coined Low Frequency Constrained Orthogonal Matching Pursuit (LFC-OMP), is a variation of the traditional algorithm Constrained Orthogonal Matching Pursuit or C-OMP to the use case of the aliasing-free clipped signals.

omp.py contains the code for the algorithm, with comments explaining each line.

The results state a positive linear correlation between both algorithms after studying the metrics $\Delta  \text{SDR}_c$, $\text{SDR}_c$, $\text{PEAQ ODG}$ y $R_\text{nonlin}$ obtained from restoring aliasing-free clipped signals with LFC-OMP and digitally clipped signals with C-OMP. Also, the absolute differences between each of these metrics are insignicant. These results imply that LFC-OMP can be used to restore clipped aliasing-free signals, in the same way that C-OMP is used to restore signals which were clipped in the digital domain.

<script type="text/javascript" async
  src="./js/mathjax.js?config=TeX-MML-AM_CHTML">
</script>
