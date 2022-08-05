function distanceMatrix(X)
    # Generate Distance Matrix for one interpolation point
    #
    # Inputs:   X - Input Point Set
    #
    # Outputs:  D - Distance Block

    # Determine size for Distance Block
    m = length(X)
    X_shift = deepcopy(X)
    D = Array{SVector,2}(undef, (m, m))

    # Pre-shift Values
    # Assuming first value is the current interpolation point
    #x_shift = X[1]

    # X_shift
    #for i = 1:m
    #    X_shift[i] = X[i] - x_shift
    #end

    # Generate Distance Matrix
    for j in 1:m
        for i in 1:m
            D[i, j] = X[i] - X[j]
        end
    end

    return D
end