function interpolationmatrix(X, F)
    ### Generate Interpolation Matrix
    P_block = polynomialBlock(F, X)
    D_block = distanceMatrix(X)
    R = norm.(D_block)
    Φ = R.^3
    A = hvcat((2, 2), Φ, P_block, P_block', zeros(size(P_block)[2], size(P_block)[2]))
    
    M = A
    M_inv = inv(A)

    return M, M_inv
    
end