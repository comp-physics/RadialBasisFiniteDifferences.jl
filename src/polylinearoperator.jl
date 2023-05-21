function polylinearoperator(X, F, F_x, F_y, F_xx, F_yy, F_xy)
    #include("rbfdx.jl")
    #include("rbfdy.jl")
    #include("rbfdxx.jl")
    #include("rbfdyy.jl")
    #include("rbfdxy.jl")
    # Generate RHS Corresponding to Linear Operators on RBF System

    # Determine size for arrays
    n_p = length(F)
    m = lastindex(X)
    r_F = Matrix{Float64}(undef, m, n_p)
    r_Fx = Matrix{Float64}(undef, m, n_p)
    r_Fy = Matrix{Float64}(undef, m, n_p)
    r_Fxx = Matrix{Float64}(undef, m, n_p)
    r_Fyy = Matrix{Float64}(undef, m, n_p)
    r_Fxy = Matrix{Float64}(undef, m, n_p)
    # Preallocate Shifted X_vector
    #X_shift = Array{SVector,1}(undef, m)
    # Preallocate Operator RHS
    #RHS = Array{Matrix,2}(undef, (n_p+m, 6))
    #RHS = Array{Matrix,2}(undef, (n_p, 6))

    ### Generate RBF righthand side
    # R_int
    #r_int = norm.(X).^3
    # R_x
    #r_x_eval = rbfdx(p, X)
    # R_y
    #r_y_eval = rbfdy(p, X)
    #r_xx_eval = rbfdxx(p, X)
    #r_yy_eval = rbfdyy(p, X)
    #r_xy_eval = rbfdxy(p, X)

    ### Generate Poly righthand side
    for i in 1:m
        # Evaluate with Fixed Polynomials
        r_F[i, :] = StaticPolynomials.evaluate(F, X[i])
        r_Fx[i, :] = StaticPolynomials.evaluate(F_x, X[i])
        r_Fy[i, :] = StaticPolynomials.evaluate(F_y, X[i])
        r_Fxx[i, :] = StaticPolynomials.evaluate(F_xx, X[i])
        r_Fyy[i, :] = StaticPolynomials.evaluate(F_yy, X[i])
        r_Fxy[i, :] = StaticPolynomials.evaluate(F_xy, X[i])
    end

    ### Concatinate RHSs 
    #polyRHS = [r_F r_Fx r_Fy r_Fxx r_Fyy r_Fxy]

    return r_F, r_Fx, r_Fy, r_Fxx, r_Fyy, r_Fxy
end

function polylinearoperator(X, F_xk, F_yk)
    # Generate RHS Corresponding to Linear Operators on RBF System
    # Case for Hyperviscosity operator 

    # Determine size for arrays
    n_p = length(F_xk)
    m = lastindex(X)
    r_Fxk = Matrix{Float64}(undef, m, n_p)
    r_Fyk = Matrix{Float64}(undef, m, n_p)

    ### Generate Poly righthand side
    for i in 1:m
        # Evaluate with Fixed Polynomials
        r_Fxk[i, :] = StaticPolynomials.evaluate(F_xk, X[i])
        r_Fyk[i, :] = StaticPolynomials.evaluate(F_yk, X[i])
    end

    return r_Fxk, r_Fyk
end
