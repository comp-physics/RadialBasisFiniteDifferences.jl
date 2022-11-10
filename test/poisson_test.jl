### This test implements the Poisson problem from Tominec's code 
# This will also be the basis of the overall regression test
#   for determining breaking changes in the RBF-FD library

# Declaring Dependencies
using Test
using StaticArrays
# using HNSW
using SparseArrays
using DelimitedFiles
using Statistics
using LinearAlgebra
using NearestNeighbors

# Main Library
using RadialBasisFiniteDifferences

# Import Fitted Grid from Tominec
x = readdlm("data/x_nodes_fitted.csv", ',', Float64)
y = readdlm("data/y_nodes_fitted.csv", ',', Float64)
X = copy(vec(reinterpret(SVector{2,Float64}, x')))
Y = copy(vec(reinterpret(SVector{2,Float64}, y')))
N = length(X)
M = length(Y)

# Import interior indices, boundary indices, 
#   and corresponding normals
Y_idx_in = vec(readdlm("data/Y_idx_in.csv", ',', Int64))
Y_idx_dirichlet = vec(readdlm("data/Y_idx_dirichlet.csv", ',', Int64))
Y_idx_neumann = vec(readdlm("data/Y_idx_neumann.csv", ',', Int64))
x_normals = readdlm("data/x_normals.csv", ',', Float64)
y_normals = readdlm("data/y_normals.csv", ',', Float64)

### Overwrite Y closest to X to X value
# Generate KNN Tree Using HNSW 
hnsw_y = KDTree(Y)
#Add all data points into the graph
#add_to_graph!(hnsw_y)
# Find single nearest neighbor for each Y point
idxs_y, dists_y = knn(hnsw_y, X, 1, true)
idxs_y = [convert.(Int, idxs_y[x]) for x in 1:length(idxs_y)]
# Proceed with overwrite
for i in 1:length(X)
    Y[idxs_y[i][1]] = X[i]
end

# Set PHS Order and Polynomial Power
# For this test we will use the parameters
#   that are defined in Tominec's code
#bf = Basis.basisF('PHS', '2d'); % Choose basis functions. To be implemented
p = 3; # PHS power (r^p).
polydeg = 3; # Augmented polynomial degree.
n = 2 * binomial(polydeg + 2, 2) # Stencil size.

### Generate Global Operator Matrices from Local RBF Operator Matrices
E, Dx, Dy, Dxx, Dyy, Dxy = generate_operator(X, Y, p, n, polydeg)

### Construct global discretized PDE system

# Decide for an exact solution.
u_exact(x, y) = sin(2 * pi * x * y)

# The corresponding right-hand-sides.
function f2(x, y)
    return x^2 * pi^2 * sin(x * y * pi * 2.0) * (-4.0) -
           y^2 * pi^2 * sin(x * y * pi * 2.0) * 4.0 # Interior RHS.
end # Interior RHS.
function f1(n1, n2, x, y)
    return n2 * x * pi * cos(x * y * pi * 2.0) * 2.0 +
           n1 * y * pi * cos(x * y * pi * 2.0) * 2.0 # Neumann RHS.
end # Neumann RHS.
f0(x, y) = u_exact(x, y) # Dirichlet RHS.

# Assemble the PDE operator.
D = zeros(Float64, M, N)
D[Y_idx_in, :] .= Dxx[Y_idx_in, :] + Dyy[Y_idx_in, :]
D[Y_idx_neumann, :] .= x_normals[:, 3] .* Dx[Y_idx_neumann, :] +
                       y_normals[:, 3] .* Dy[Y_idx_neumann, :]
D[Y_idx_dirichlet, :] .= E[Y_idx_dirichlet, :]

# Assemble the RHS.
f = zeros(M)
for i in 1:length(Y_idx_in)
    f[Y_idx_in[i]] = f2(Y[Y_idx_in[i]][1], Y[Y_idx_in[i]][2]) # Interior.
end
for i in 1:length(Y_idx_neumann)
    f[Y_idx_neumann[i]] = f1(x_normals[i, 3], y_normals[i, 3], Y[Y_idx_neumann[i]][1],
                             Y[Y_idx_neumann[i]][2]) # Neumann.
end
for i in 1:length(Y_idx_dirichlet)
    f[Y_idx_dirichlet[i]] = f0(Y[Y_idx_dirichlet[i]][1], Y[Y_idx_dirichlet[i]][2]) # Dirichlet.
end

# Scale the PDE operator and the RHS.
# Generate KNN Tree Using HNSW 
hnsw_x = KDTree(X)
#Add all data points into the graph
# add_to_graph!(hnsw_x)
# Find nearest neighbor for each X point
idxs_x, dists_x = knn(hnsw_x, X, 2, true)
idxs_x = [convert.(Int, idxs_x[x]) for x in 1:length(idxs_x)]
# Mean distance
h = mean(dists_x)[2]

M0 = length(Y_idx_dirichlet)
M1 = length(Y_idx_neumann)
M2 = length(Y_idx_in)

D[Y_idx_in, :] .= 1 / sqrt(M2) * D[Y_idx_in, :]
D[Y_idx_neumann, :] .= 1 / sqrt(M1) * D[Y_idx_neumann, :]
D[Y_idx_dirichlet, :] .= 1 / h * 1 / sqrt(M0) * D[Y_idx_dirichlet, :]

f[Y_idx_in] .= 1 / sqrt(M2) * f[Y_idx_in]
f[Y_idx_neumann] .= 1 / sqrt(M1) * f[Y_idx_neumann]
f[Y_idx_dirichlet] .= 1 / h * 1 / sqrt(M0) * f[Y_idx_dirichlet]

# Convert to Sparse
D = sparse(D)

# Solve.
u_Y = E * (D \ f)

# Compute error.
# This metric will be used to evaluate regression test
#   pass/fail condition
err = zeros(M)
err = norm(u_Y - u_exact.([Y[x][1] for x in 1:length(Y)], [Y[x][2] for x in 1:length(Y)])) /
      norm(u_exact.([Y[x][1] for x in 1:length(Y)], [Y[x][2] for x in 1:length(Y)]));

#@assert err < 0.0027
#display(err)
@test err < 0.0027