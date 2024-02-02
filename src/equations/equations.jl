####################################################################################################
# Include files with actual implementations for different systems of equations.

# Numerical flux formulations that are independent of the specific system of equations
# include("numerical_fluxes.jl")

# Linear scalar advection
# abstract type AbstractLinearScalarAdvectionEquation{NDIMS, NVARS} <:
#                 AbstractEquations{NDIMS, NVARS} end
# include("linear_scalar_advection_1d.jl")
# include("linear_scalar_advection_2d.jl")
# include("linear_scalar_advection_3d.jl")

# Inviscid Burgers
# abstract type AbstractInviscidBurgersEquation{NDIMS, NVARS} <:
#                 AbstractEquations{NDIMS, NVARS} end
# include("inviscid_burgers_1d.jl")

# Shallow water equations
# abstract type AbstractShallowWaterEquations{NDIMS, NVARS} <:
#                 AbstractEquations{NDIMS, NVARS} end
# include("shallow_water_1d.jl")
# include("shallow_water_2d.jl")
# include("shallow_water_two_layer_1d.jl")
# include("shallow_water_two_layer_2d.jl")
# include("shallow_water_quasi_1d.jl")

# CompressibleEulerEquations
# abstract type AbstractCompressibleEulerEquations{NDIMS, NVARS} <:
#                 AbstractEquations{NDIMS, NVARS} end
# include("compressible_euler_1d.jl")
# include("compressible_euler_2d.jl")
# include("compressible_euler_3d.jl")
# include("compressible_euler_quasi_1d.jl")

# CompressibleEulerMulticomponentEquations
# abstract type AbstractCompressibleEulerMulticomponentEquations{NDIMS, NVARS, NCOMP} <:
#                 AbstractEquations{NDIMS, NVARS} end
# include("compressible_euler_multicomponent_1d.jl")
# include("compressible_euler_multicomponent_2d.jl")

# PolytropicEulerEquations
# abstract type AbstractPolytropicEulerEquations{NDIMS, NVARS} <:
#                 AbstractEquations{NDIMS, NVARS} end
# include("polytropic_euler_2d.jl")


