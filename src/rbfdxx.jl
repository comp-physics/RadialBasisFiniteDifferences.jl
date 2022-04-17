function rbfdxx(p, X)
# Determine size for Polynomial Block
m = length(X)
# Determine size for Dₓ RBF Block
r_xx = Vector{Float64}(undef, m)

#r_xx[1] = 0.0
for i = 2:m
    r_xx[i] = p * (sqrt(X[i][1]^2 + X[i][2]^2)^-1) * (sqrt(X[i][1]^2 + X[i][2]^2)^(p - 1)) + 
                p * (p - 1) * (X[i][1]^2) * (sqrt(X[i][1]^2 + X[i][2]^2)^-2) * (sqrt(X[i][1]^2 + X[i][2]^2)^(p - 2)) -
                p * (X[i][1]^2) * (sqrt(X[i][1]^2 + X[i][2]^2)^-3) * (sqrt(X[i][1]^2 + X[i][2]^2)^(p - 1))
end

return r_xx

end