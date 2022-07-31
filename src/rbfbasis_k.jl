function rbfbasis_k(rbfdeg, k)

    # RBF Interpolation System
    @variables x y
    # Fixed p 
    p = rbfdeg

    # Generate Polyharmonic Spline RBF
    rbf = sqrt(x^2 + y^2)^p
    rbf_expr = build_function(rbf, [x, y], expression=Val{false})

    # Generate Derivatives
    # Differential Operators
    Dxk = Differential(x)^k
    Dyk = Differential(y)^k
    # Allocate Derivatives
    rbf_xk = simplify(expand_derivatives(Dxk(rbf)))
    rbf_yk = simplify(expand_derivatives(Dyk(rbf)))
    # Create functions
    rbf_xk_expr = build_function(rbf_xk, [x, y], expression=Val{false})
    rbf_yk_expr = build_function(rbf_yk, [x, y], expression=Val{false})

    return rbf_expr, rbf_xk_expr, rbf_yk_expr

end