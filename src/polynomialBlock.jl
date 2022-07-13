function polynomialBlock(F, X)
# Generate the polynomial basis block for one
#  interpolation point
#
# Inputs:   F - StaticPolynomial Array
#           X - Input Point Set
#
# Outputs:  P - Monomial Basis Block

# Determine size for Polynomial Block
n = length(F)
m = length(X)
# Determine size for Distance Block
#X_shift = Array{SVector,1}(undef, m)
# Pre-shift Values
# Assuming first value is the current interpolation point
#x_shift = X[1]
# X_shift
#for i = 1:m
#    X_shift[i] = X[i] - x_shift
#end

P = zeros(Float64, m, n)

# Evaluate Polynomial System at Data Point
for i=1:m
    #P[i, :] = StaticPolynomials.evaluate(F, X[i])
    # Switch to Fixed Polynomials
    P[i, :] = FixedPolynomials.evaluate(F, X[i])
end

return P

end