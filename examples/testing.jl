#import DynamicPolynomials: @polyvar
#import DynamicPolynomials: monomials
import DynamicPolynomials: differentiate
using StaticPolynomials
using LinearAlgebra
using StaticArrays
#using NearestNeighbors
using HNSW
#using Symbolics
using SparseArrays
using DelimitedFiles
#using Meshes
using Plots
#using MultivariateBases
#including scaling 
include("scalestencil.jl")
include("interpolationmatrix.jl")
include("polynomialBasis.jl")
include("polynomialBlock.jl")
include("distanceMatrix.jl")
#include("rhslinearoperator.jl")
include("polylinearoperator.jl")
include("generateOperator.jl")
# RBF Operator
include("rbfdx.jl")
include("rbfdy.jl")
include("rbfdxx.jl")
include("rbfdyy.jl")
include("rbfdxy.jl")

### Import Fitted Grid from Tominec
x = readdlm("x_nodes_fitted.csv",',', Float64)
y = readdlm("y_nodes_fitted.csv",',', Float64)
X = copy(vec(reinterpret(SVector{2, Float64},x')))
Y = copy(vec(reinterpret(SVector{2, Float64},y')))

### Decide the proper schema for "Mesh" Definition
### Options include using Meshes.jl, FiniteMeshes.jl, HDF5.jl, etc.
# Generate Square Cartesian Grid
#=
N = 50
q = 3
x_start = 0
x_end = 10
x = range(x_start, x_end, N)
y = range(x_start, x_end, N)
X = vec(collect(Iterators.product(x, y)))
X = [tup[k] for tup in X, k in 1:2]
X = vec(reinterpret(SVector{2, Float64},X'))
#X = collect.(vec(X))
M = Int(ceil(N * sqrt(q)))
x = range(x_start, x_end, M)
y = range(x_start, x_end, M)
Y = vec(collect(Iterators.product(x, y)))
Y = [tup[k] for tup in Y, k in 1:2]
Y = vec(reinterpret(SVector{2, Float64},Y'))
=#

### Overwrite Y closest to X to X value
# Generate KNN Tree Using HNSW 
#Intialize HNSW struct
hnsw_y = HierarchicalNSW(Y)
#Add all data points into the graph
#Optionally pass a subset of the indices in data to partially construct the graph
add_to_graph!(hnsw_y)
# Find single nearest neighbor for each Y point
#idxs_y_x, dists_y_x = knn_search(hnsw_x, collect(Y[1]), 1)
idxs_y, dists_y = knn_search(hnsw_y, X, 1)
idxs_y = [convert.(Int,idxs_y[x]) for x=1:length(idxs_y)]
#Y[idxs_y] = X
# Overwrite Y closest to X to X value
for i = 1:length(X)
    Y[idxs_y[i][1]] = X[i]
end

# Set PHS Order and Polynomial Power
#bf = Basis.basisF('PHS', '2d'); % Choose basis functions. To be implemented
p = 3; # PHS power (r^p).
polydeg = 3; # Augmented polynomial degree.
n = 2 * binomial(polydeg+2,2) # Stencil size.

### Generate Global Operator Matrices from Local RBF Operator Matrices
E, Dx, Dy, Dxx, Dyy, Dxy = generateOperator(X, Y, p, n, polydeg)
spy(E)
spy(Dx)
spy(Dy)
spy(Dxx)
spy(Dyy)
spy(Dxy)

### Lots of type warnings to fix later
#@code_warntype generateOperator(X, Y, p, n, polydeg)

# Quick test nested neighbor address 
#scatter(Tuple(Y[2000]), label="Eval")
#scatter!(Tuple.(X[idxs_x[idxs_y_x[2000]][1]]), label="Interp field")
#scatter!(Tuple(X[idxs_y_x[2000]][1]), label="Interp point")

