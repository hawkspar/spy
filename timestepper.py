from cylinderModule import *

dt = 0.1
tend = 50.

cylinder = LIAproblem2D(100,"./Results/Re100/")
cylinder.lowmachTimestepper_CN_Newton_trueDensity(dt, tend)
