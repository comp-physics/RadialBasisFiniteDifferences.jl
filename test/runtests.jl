using SafeTestsets

#@testset "RadialBasisFiniteDifferences.jl" begin
# Write your tests here.
#    include("poisson_test.jl")
#end

using SafeTestsets
@safetestset "Poisson Test" begin
    include("poisson_test.jl")
end