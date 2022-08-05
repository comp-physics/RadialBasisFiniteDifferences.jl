function interpolationmatrix(X, rbf_expr, F)
    ### Generate Interpolation Matrix
    P_block = polynomialblock(F, X)
    Φ = rbfblock(rbf_expr, X)
    A = hvcat((2, 2), Φ, P_block, P_block', zeros(size(P_block)[2], size(P_block)[2]))

    M = A
    M_inv = inv(A)

    return M, M_inv
end