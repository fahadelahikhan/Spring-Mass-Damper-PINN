# Spring-Mass-Damper-PINN
A physics-informed neural network (pure PyTorch) that solves m·x'' + c·x' + k·x = 0, x(0)=1, x'(0)=0 with m=1, c=0.8, k=20 (underdamped), using "no measured data", only the equation and initial conditions.
