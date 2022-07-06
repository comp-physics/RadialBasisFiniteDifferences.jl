module RadialBasisFiniteDifferences

# Write your package code here.

# Dependencies
import DynamicPolynomials: @polyvar
import DynamicPolynomials: monomials
import DynamicPolynomials: differentiate
using StaticPolynomials
using LinearAlgebra
using StaticArrays
using HNSW
using SparseArrays
using DelimitedFiles
using HDF5
using Statistics
using WriteVTK

# Local Files 
# Global Linear Operator Matrices
include("scalestencil.jl")
export scalestencil
include("interpolationmatrix.jl")
export interpolationmatrix
include("polynomialBasis.jl")
export polynomialBasis
include("polynomialBlock.jl")
export polynomialBlock
include("distanceMatrix.jl")
export distanceMatrix
#include("rhslinearoperator.jl")
include("polylinearoperator.jl")
export polylinearoperator
include("generateOperator.jl")
# RBF Operator
include("rbfdx.jl")
include("rbfdy.jl")
include("rbfdxx.jl")
include("rbfdyy.jl")
include("rbfdxy.jl")

# Export Methods 
export generateOperator

# Local Files
# Mesh Pre-processing 
include("extractcoordinates.jl")
export extractcoordinates
include("extractelements.jl")
export extractelements
include("calculatenormal.jl")
export calculatenormal
include("genghostnodes.jl")
export genghostnodes
include("processmesh.jl")
export processmesh

end
