from cylinderModule import *

# shift value
sigma = 0.1+0.8j
# number of eigenmodes to compute
k = 5

cylinder = LIAproblem2D(50,"./Results/Re50/")
cylinder.steady_state_trueDensity()
cylinder.eigenvalues(sigma=sigma, k=k)
