using SafeTestsets

#@testset "RadialBasisFiniteDifferences.jl" begin
# Write your tests here.
#    include("poisson_test.jl")
#end

# Original Poisson Test
@safetestset "Poisson Test" begin
    include("poisson_test.jl")
end

# Hyperviscosity Operator Test
@safetestset "Hyperviscosity Test" begin
    include("hyperviscosity_test.jl")
end