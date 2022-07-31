module RadialBasisFiniteDifferences

# Write your package code here.

# Dependencies
import DynamicPolynomials: @polyvar
import DynamicPolynomials: monomials
import DynamicPolynomials: differentiate
using FixedPolynomials
using LinearAlgebra
using StaticArrays
using HNSW
using SparseArrays
using DelimitedFiles
using HDF5
using Statistics
using WriteVTK
using Symbolics
using RuntimeGeneratedFunctions

# RBF and Polynomial Interpolation
include("scalestencil.jl")
export scalestencil
include("interpolationmatrix.jl")
export interpolationmatrix
include("polynomialBasis.jl")
export polynomialBasis
include("polynomialBlock.jl")
export polynomialBlock
include("rbfbasis.jl")
export rbfbasis
include("rbfblock.jl")
export rbfblock
#include("distanceMatrix.jl")
#export distanceMatrix
#include("rhslinearoperator.jl")
include("polylinearoperator.jl")
export polylinearoperator
# RBF Operator
#include("rbfdx.jl")
#include("rbfdy.jl")
#include("rbfdxx.jl")
#include("rbfdyy.jl")
#include("rbfdxy.jl")

# Hyperviscosity Methods 
include("rbfbasis_k.jl")
export rbfbasis_k
include("polynomialbasis_k.jl")
export polynomialbasis_k
include("hyperviscosity_operator.jl")
export hyperviscosity_operator

# Main Method
include("generateOperator.jl")
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
