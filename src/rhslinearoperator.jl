function rhslinearoperator(p, X, F, F_x, F_y, F_xx, F_yy, F_xy)
    #include("rbfdx.jl")
    #include("rbfdy.jl")
    #include("rbfdxx.jl")
    #include("rbfdyy.jl")
    #include("rbfdxy.jl")
    # Generate RHS Corresponding to Linear Operators on RBF System

    # Determine size for arrays
    n_p = length(F)
    m = length(X)
    # Preallocate Shifted X_vector
    #X_shift = Array{SVector,1}(undef, m)
    # Preallocate Operator RHS
    RHS = Array{Matrix,2}(undef, (n_p + m, 6))

    ### Generate RBF righthand side
    # R_int
    r_int = norm.(X) .^ 3
    # R_x
    r_x_eval = rbfdx(p, X)
    # R_y
    r_y_eval = rbfdy(p, X)
    r_xx_eval = rbfdxx(p, X)
    r_yy_eval = rbfdyy(p, X)
    r_xy_eval = rbfdxy(p, X)

    ### Generate Poly righthand side
    r_F = StaticPolynomials.evaluate(F, X_shift[1])
    r_Fx = StaticPolynomials.evaluate(F_x, X_shift[1])
    r_Fy = StaticPolynomials.evaluate(F_y, X_shift[1])
    r_Fxx = StaticPolynomials.evaluate(F_xx, X_shift[1])
    r_Fyy = StaticPolynomials.evaluate(F_yy, X_shift[1])
    r_Fxy = StaticPolynomials.evaluate(F_xy, X_shift[1])

    ### Concatinate RHSs 
    RHS = [r_int r_x_eval r_y_eval r_xx_eval r_yy_eval r_xy_eval
           r_F r_Fx r_Fy r_Fxx r_Fyy r_Fxy]

    return RHS
end
