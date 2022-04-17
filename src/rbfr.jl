function rbfdx(p, X)
# Determine size for Polynomial Block
m = length(X)
# Determine size for Dₓ RBF Block
r_x = Vector{Float64}(undef, m)

#r_x[1] = 0.0
for i = 2:m
    r_x[i] = p * X[i][1] * (sqrt(X[i][1]^2 + X[i][2]^2)^-1) * (sqrt(X[i][1]^2 + X[i][2]^2)^(p - 1))
end

return r_x

end