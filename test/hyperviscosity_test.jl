### This test implements the Poisson problem from Tominec's code 
# This will also be the basis of the overall regression test
#   for determining breaking changes in the RBF-FD library

# Declaring Dependencies
using Test
using StaticArrays
using DelimitedFiles

# Main Library
using RadialBasisFiniteDifferences

# Import Fitted Grid from Tominec
x = readdlm("data/x_nodes_fitted.csv", ',', Float64)
X = copy(vec(reinterpret(SVector{2,Float64}, x')))

# Set PHS Order and Polynomial Power
# For this test we will use the parameters
#   that are defined in Tominec's code
#bf = Basis.basisF('PHS', '2d'); % Choose basis functions. To be implemented
rbfdeg = 5; # PHS power (r^p).
polydeg = 5; # Augmented polynomial degree.
n = 2 * binomial(polydeg + 2, 2) # Stencil size.

### Generate Global Operator Matrices from Local RBF Operator Matrices
E, Dx, Dy, Dxx, Dyy, Dxy = generate_operator(X, X, rbfdeg, n, polydeg)

### Generate Hyperviscosity Operator
k = 1 # 2nd Derivative
Dxk, Dyk = hyperviscosity_operator(2 * k, X, X, rbfdeg, n, polydeg)
# Compare to Dxx
@test Dxk ≈ Dxx
@test Dyk ≈ Dyy
