function rbfdxy(p, X)
# Determine size for Polynomial Block
m = length(X)
# Determine size for Dₓ RBF Block
r_xy = Vector{Float64}(undef, m)

#r_x[1] = 0.0
for i = 2:m
    r_xy[i] = p * X[i][1] * X[i][2] * (p - 1) * (sqrt(X[i][1]^2 + X[i][2]^2)^-2) * (sqrt(X[i][1]^2 + X[i][2]^2)^(p - 2)) 
       - p * X[i][1] * X[i][2] * (sqrt(X[i][1]^2 + X[i][2]^2)^-3) * (sqrt(X[i][1]^2 + X[i][2]^2)^(p - 1))
end

return r_xy

end