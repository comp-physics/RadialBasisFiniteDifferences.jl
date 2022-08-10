function scalestencil(X)
    ### Add Shifting and Scaling of Local X Matrices
    X_local = X
    # Shift to origin
    # Determine size for Distance Block
    X_shift = Array{SVector{2},1}(undef, lastindex(X_local)) # Rework to use SVector
    X_ = Array{SVector{2},1}(undef, lastindex(X_local)) # Rework to use SVector
    # Pre-shift Values
    # Assuming first value is the current interpolation point
    x_shift = X_local[1]
    # X_shift
    for j in eachindex(X_local)
        X_[j] = X_local[j] - x_shift
    end

    # Normalize vectors according to largest entry
    scale_x = 1.0 / max([abs(X_[x][1]) for x in eachindex(X_)]...)
    scale_y = 1.0 / max([abs(X_[x][2]) for x in eachindex(X_)]...)
    for i in eachindex(X_local)
        X_shift[i] = [X_[i][1] * scale_x, X_[i][2] * scale_y]
    end

    return X_shift, scale_x, scale_y
end
