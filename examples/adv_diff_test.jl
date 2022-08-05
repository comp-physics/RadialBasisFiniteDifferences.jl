### This test implements an Advection-Diffusion problem with Hyperviscosity 
# This will also be the basis of the overall regression test
#   for determining breaking changes in the RBF-FD library

# Declaring Dependencies
using StaticArrays
using HNSW
using SparseArrays
using DelimitedFiles
using Statistics
#using Plots
using LinearAlgebra
using DifferentialEquations
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# Main Library
using RadialBasisFiniteDifferences
using WriteVTK

# Import meshes via CGNS/HDF5
using HDF5
# Import X field
meshname = "rect_0_05.cgns"
markernames = ["left", "right", "top", "bottom"]
X, x_point_mat, X_idx_in, X_idx_bc, X_idx_bc_g,
cells, bc_normals, bc_tangents = processmesh(meshname, markernames)
N = length(X)

# Import Y field
Y, y_point_mat, Y_idx_in, Y_idx_bc, Y_idx_bc_g,
cells, bc_normals, bc_tangents = processmesh(meshname, markernames)

### Overwrite Y closest to X to X value
# Generate KNN Tree Using HNSW 
hnsw_y = HierarchicalNSW(Y)
#Add all data points into the graph
add_to_graph!(hnsw_y)
# Find single nearest neighbor for each Y point
idxs_y, dists_y = knn_search(hnsw_y, X, 1)
idxs_y = [convert.(Int, idxs_y[x]) for x = 1:length(idxs_y)]
# Proceed with overwrite
for i = 1:length(X)
    Y[idxs_y[i][1]] = X[i]
end

# Set PHS Order and Polynomial Power
# For this test we will use the parameters
#   that are defined in Tominec's code
#bf = Basis.basisF('PHS', '2d'); % Choose basis functions. To be implemented
rbfdeg = 5; # PHS power (r^p).
polydeg = 5; # Augmented polynomial degree.
n = 2 * binomial(polydeg + 2, 2) # Stencil size.

### Generate Global Operator Matrices from Local RBF Operator Matrices
E, Dx, Dy, Dxx, Dyy, Dxy = generateOperator(X, Y, rbfdeg, n, polydeg)

### Construct global discretized PDE system
# Importing syntax from Tominec then porting to Julia
x₀ = zeros(length(X))
f_init(x, y) = ((x - 0.5)^2 + (y - 0.5)^2 <= (0.2)^2) ? 10 : 1
for i = 1:length(X)
    x₀[i] = f_init(X[i][1], X[i][2]) # Initial Condition
end
y₀ = zeros(length(Y))
for i = 1:length(Y)
    y₀[i] = f_init(Y[i][1], Y[i][2]) # Initial Condition
end

### Generate PDE Operator 

# Scale the PDE operator and the RHS.
# Generate KNN Tree Using HNSW 
hnsw_y = HierarchicalNSW(Y)
#Add all data points into the graph
add_to_graph!(hnsw_y)
# Find nearest neighbor for each X point
idxs_y, dists_y = knn_search(hnsw_y, Y, 2)
idxs_y = [convert.(Int, idxs_y[x]) for x = 1:length(idxs_y)]
# Mean distance
h_y = mean(dists_y)[2]

# Adding constant advection to system
α = 0.001 # Diffusion Coefficient 
u_x = 0.5
u_y = 0.0
CFL = 1
delta_t = CFL * h_y / u_x
### Generate Hyperviscosity Operator
k = 2
Dxk, Dyk = hyperviscosity_operator(2 * k, X, Y, rbfdeg, n, polydeg)
#Δᵏ = (Dxx + Dyy)^k
Δᵏ = (Dxk + Dyk)
# Compare to Dxx
#Dxk ≈ Dxx
#Dyk ≈ Dyy

# Precalculate ghost node weights
Y_idx_bc_int = Array{eltype(Y_idx_bc)}(undef, length(markernames))
w_bc = Array{typeof(Dx)}(undef, length(markernames))
w_bc_g = Array{typeof(Dx)}(undef, length(markernames))
w_bc_inv = deepcopy(w_bc)
w_bc_int = deepcopy(w_bc)
w_normal_x = [[bc_normals[x][y][1] for y = 1:length(bc_normals[x])] for x = 1:length(bc_normals)]
w_normal_y = [[bc_normals[x][y][2] for y = 1:length(bc_normals[x])] for x = 1:length(bc_normals)]
# Inlet
#w_bc[1] = Dx[Y_idx_bc[1], Y_idx_bc[1]]
w_bc_g[1] = Dx[Y_idx_bc[1], Y_idx_bc_g[1]]
w_bc_inv[1] = inv(Array(w_bc_g[1]))
w_bc_int[1] = Dx[Y_idx_bc[1], setdiff(1:end, Y_idx_bc_g[1])]
Y_idx_bc_int[1] = setdiff(1:length(Y), Y_idx_bc_g[1])
# Outlet
#w_bc[2] = Dx[Y_idx_bc[2], Y_idx_bc[2]]
w_bc_g[2] = Dx[Y_idx_bc[2], Y_idx_bc_g[2]]
w_bc_inv[2] = inv(Array(w_bc_g[2]))
w_bc_int[2] = Dx[Y_idx_bc[2], setdiff(1:end, Y_idx_bc_g[2])]
Y_idx_bc_int[2] = setdiff(1:length(Y), Y_idx_bc_g[2])
# Top
#w_bc[3] = Dy[Y_idx_bc[3], Y_idx_bc[3]]
w_bc_g[3] = Dy[Y_idx_bc[3], Y_idx_bc_g[3]]
w_bc_inv[3] = inv(Array(w_bc_g[3]))
w_bc_int[3] = Dy[Y_idx_bc[3], setdiff(1:end, Y_idx_bc_g[3])]
Y_idx_bc_int[3] = setdiff(1:length(Y), Y_idx_bc_g[3])
# Bottom
#w_bc[4] = Dy[Y_idx_bc[4], Y_idx_bc[4]]
w_bc_g[4] = Dy[Y_idx_bc[4], Y_idx_bc_g[4]]
w_bc_inv[4] = inv(Array(w_bc_g[4]))
w_bc_int[4] = Dy[Y_idx_bc[4], setdiff(1:end, Y_idx_bc_g[4])]
Y_idx_bc_int[4] = setdiff(1:length(Y), Y_idx_bc_g[4])

uy_idx_offset = length(Y)
function cons_sys(du, u, p, t)
    α, h_y, u_x, u_y,
    D_x, D_y, D_xx, D_yy, E, Δᵏ, k,
    Y_idx_in, Y_idx_bc, Y_idx_bc_g, Y_idx_bc_int,
    w_bc, w_bc_g, w_bc_inv, w_bc_int = p

    # Interior
    # du .= E' * (α * D_xx * u + α * D_yy * u - D_x * u_x * u - D_y * u_y * u) -
    #         1.0 * h_y^(2*k) * Δᵏ * u

    # Right Boundary
    # ux - indeces
    u[Y_idx_bc_g[2]] .= -w_bc_inv[2] * (w_bc_int[2] * u[Y_idx_bc_int[2]])

    # Left Boundary
    # ux - indeces
    u[Y_idx_bc[1]] .= 1.0
    u[Y_idx_bc_g[1]] .= 1.0

    # Top Boundary
    # ux - indeces
    u[Y_idx_bc_g[3]] .= -w_bc_inv[3] * (w_bc_int[3] * u[Y_idx_bc_int[3]])

    # Bottom Boundary
    # ux - indeces
    u[Y_idx_bc_g[4]] .= -w_bc_inv[4] * (w_bc_int[4] * u[Y_idx_bc_int[4]])

    # Interior
    du .= E' * (α * D_xx * u + α * D_yy * u - D_x * u_x * u - D_y * u_y * u) -
          1.0 * h_y^(2 * k) * Δᵏ * u
    # Zero out ghosts
    du[Y_idx_bc[1]] .= 0.0
    du[Y_idx_bc_g[1]] .= 0.0
    du[Y_idx_bc_g[2]] .= 0.0
    du[Y_idx_bc_g[3]] .= 0.0
    du[Y_idx_bc_g[4]] .= 0.0

end

M_mass = E' * E

p = (α, h_y, u_x, u_y,
    Dx, Dy, Dxx, Dyy, E, Δᵏ, k,
    Y_idx_in, Y_idx_bc, Y_idx_bc_g, Y_idx_bc_int,
    w_bc, w_bc_g, w_bc_inv, w_bc_int)
tspan = [0.0, 2.5]
f = ODEFunction(cons_sys)
probl = ODEProblem(f, x₀, tspan, p)
sol = solve(probl, SSPRK43(), progress=true, progress_steps=1)

x_final = sol(tspan[2])
x_final = sol(1.0)
using GLMakie
# Plot including Ghost Nodes
Makie.scatter(Tuple.(X), color=x_final, axis=(aspect=DataAspect(),))
# Plot Removing Ghost Nodes
idx_real = 1:minimum(minimum(Y_idx_bc_g))-1
Makie.scatter(Tuple.(X[idx_real]), color=x_final[idx_real], axis=(aspect=DataAspect(),))

# Testing BC Operators
using Plots
u = sol(1.0)
u_right = -w_bc_inv[2] * (w_bc_int[2] * u[Y_idx_bc_int[2]])
Plots.plot(u_right)
u_bottom = -w_bc_inv[4] * (w_bc_int[4] * u[Y_idx_bc_int[4]])
Plots.plot(u_bottom)

# Generate VTK Mesh Data Output
#vtk_grid("fields", x_point_mat, cells) do vtk
#    vtk["C"] = x_final[1:length(cells)]
#end

