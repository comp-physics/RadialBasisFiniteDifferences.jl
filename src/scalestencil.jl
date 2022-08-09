function scalestencil(X)
    ### Add Shifting and Scaling of Local X Matrices
    X_local = X
    # Shift to origin
    # Determine size for Distance Block
    X_shift = Array{MVector,1}(undef, length(X_local)) # Rework to use SVector
    # Pre-shift Values
    # Assuming first value is the current interpolation point
    x_shift = X_local[1]
    # X_shift
    for j in eachindex(X_local)
        X_shift[j] = X_local[j] - x_shift
    end

    # Normalize vectors according to largest entry
    scale_x = 1.0 / max([abs(X_shift[x][1]) for x in eachindex(X_shift)]...)
    scale_y = 1.0 / max([abs(X_shift[x][2]) for x in eachindex(X_shift)]...)
    [X_shift[x][1] = X_shift[x][1] * scale_x for x in eachindex(X_local)]
    [X_shift[x][2] = X_shift[x][2] * scale_y for x in eachindex(X_local)]

    return X_shift, scale_x, scale_y
end