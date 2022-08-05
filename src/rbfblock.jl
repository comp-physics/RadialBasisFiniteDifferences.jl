function rbfblock(rbf_expr, X)
    # Generate RBF Matrix for one interpolation point
    #
    # Inputs:   rbf_expr - RBF Function
    #           X - Input Point Set
    #
    # Outputs:  Φ - RBF Matrix Block

    # Determine size for Distance Block
    m = length(X)
    D = Array{SVector,2}(undef, (m, m))

    # Generate Distance Matrix
    for j in 1:m
        for i in 1:m
            D[i, j] = X[i] - X[j]
        end
    end

    Φ = rbf_expr.(D)

    return Φ
end