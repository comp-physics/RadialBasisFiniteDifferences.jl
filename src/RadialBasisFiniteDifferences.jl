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

# Local Files 
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

end
