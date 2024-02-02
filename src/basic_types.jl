# Basic Abstract Types
# Based on Trixi.jl

"""
    AbstractSemidiscretization

Abstract supertype of specific semidiscretizations such as
 - SemidiscretizationHyperbolic for hyperbolic conservation laws
 - SemidiscretizationEulerGravity for Euler with self-gravity
"""
abstract type AbstractSemidiscretization end

"""
    AbstractEquations{NDIMS, NVARS}

An abstract supertype of specific equations such as the compressible Euler equations.
The type parameters encode the number of spatial dimensions (`NDIMS`) and the
number of primary variables (`NVARS`) of the physics model.
"""
abstract type AbstractEquations{NDIMS,NVARS} end