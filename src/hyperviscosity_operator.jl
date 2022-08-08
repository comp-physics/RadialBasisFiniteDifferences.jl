function hyperviscosity_operator(k_deriv, X, Y, p, n, polydeg)
    # Generate Global Hyperviscosity Operator Matrix
    # of order k 

    # RBF Interpolation system
    # Generate RBFs 
    rbf, rbf_xk, rbf_yk = rbfbasis_k(p, k_deriv)

    # Polynomial Interpolation System
    # Generate Polynomial Basis Functions
    F, F_xk, F_yk = polynomialbasis_k(polydeg, k_deriv)

    # Generate KNN Tree Using HNSW 
    #Intialize HNSW struct
    hnsw_x = HierarchicalNSW(X)
    #Add all data points into the graph
    #Optionally pass a subset of the indices in data to partially construct the graph
    add_to_graph!(hnsw_x)
    # Find k (approximate) nearest neighbors for each of the queries
    idxs_x, dists_x = knn_search(hnsw_x, X, n)
    idxs_x = [convert.(Int, idxs_x[x]) for x in eachindex(idxs_x)] # Convert to readable
    # Find single nearest neighbor for each Y point
    idxs_y_x, dists_y_x = knn_search(hnsw_x, Y, 1)
    idxs_y_x = [convert.(Int, idxs_y_x[x]) for x in eachindex(idxs_y_x)] # Convert to readable

    ### Storing matrices and scale factors
    m = lastindex(X)
    M_int = Array{Matrix,2}(undef, (m, 2))
    scale = Array{SVector{2},1}(undef, m)

    ### Testing looping through all indeces of X 
    for i in eachindex(X)

        ### Add Shifting and Scaling of Local X Matrices
        X_shift, scale_x, scale_y = scalestencil(X[idxs_x[i]])
        scale[i] = [scale_x, scale_y]

        ### Generate Interpolation Matrix
        # Pre-invert
        M, M_inv = interpolationmatrix(X_shift, rbf, F)
        M_int[i, 1] = M
        M_int[i, 2] = M_inv
    end

    ### Shift and Scale the Y points
    Y_shift = Array{SVector{2},1}(undef, lastindex(Y))
    for i in eachindex(Y)
        #ad1 = [idxs_x[idxs_y_x[i]][1]][1][1]
        #ad2 = [idxs_x[idxs_y_x[i]][1]][1][2]
        Y_shift[i] = [(Y[i][1] - X[idxs_x[idxs_y_x[i]][1][1]][1]) *
                      scale[idxs_x[idxs_y_x[i]][1][1]][1],
                      (Y[i][2] - X[idxs_x[idxs_y_x[i]][1][1]][2]) *
                      scale[idxs_x[idxs_y_x[i]][1][1]][2]]
    end

    # Generate Poly RHS at every Y point 
    cxk, cyk = polylinearoperator(Y_shift, F_xk, F_yk)

    ### Need to perform several shifts in order to get the correct values for PHS
    # Evaluate over domain of Y
    Dxk_loc = zeros(lastindex(Y), n)
    Dyk_loc = zeros(lastindex(Y), n)
    for k in eachindex(Y)
        # Take interpolation matrix from x in X which is closest to the current y in Y
        idx_inv = idxs_y_x[k]
        idx_inv_local = idxs_x[idx_inv][1][1]

        # Take the neighbors around the X-center
        idx = idxs_x[idx_inv][1]

        # The Y-center
        idx_c = k

        # Shift and scale the stencil nodes to a circle
        Xscaled = Array{SVector{2},1}(undef, lastindex(idx))
        Yscaled = Vector(undef, 2)

        # Shift the stencil nodes so that the center node is the origin
        for i in eachindex(idx)
            Xscaled[i] = X[idx[i]] - X[idx_inv][1]
        end

        # Repeat with Y-center
        Yscaled = [Y[idx_c][1] - X[idx_inv][1][1], Y[idx_c][2] - X[idx_inv][1][2]]

        # Scale the points to [0,1]x[0,1]x...x[0,1]
        scale_x = scale[idx_inv][1][1]
        scale_y = scale[idx_inv][1][2]

        # Unroll scaling
        for i in eachindex(idx)
            Xscaled[i] = [Xscaled[i][1] * scale_x, Xscaled[i][2] * scale_y]
        end
        Yscaled = [Yscaled[1] * scale_x, Yscaled[2] * scale_y]

        # Stencil distance matrices. Center point is in the origin (0)
        stencil_coord = Array{SVector{2},1}(undef, lastindex(idx))
        for i in eachindex(idx)
            dx_stencil = Yscaled[1] - Xscaled[i][1]
            dy_stencil = Yscaled[2] - Xscaled[i][2]
            if dx_stencil == 0
                dx_stencil = eps()
            end
            if dy_stencil == 0
                dy_stencil = eps()
            end
            stencil_coord[i] = [dx_stencil, dy_stencil]
        end

        # Compute distance matrix
        # Here the distance matrix is actually not scaled
        # Oth derivative 
        #b = r_stencil.^p
        b = rbf.(stencil_coord)

        # kth derivatives
        bxk = rbf_xk.(stencil_coord)
        byk = rbf_yk.(stencil_coord)

        # Compute all stencil at once
        RHS = [bxk byk; cxk[k, :] cyk[k, :]]
        stenc = M_int[idx_inv_local, 2] * RHS

        # Extract RBF Stencil Weights
        Dxk_loc[k, :] = scale_x^k_deriv * stenc[1:n, 1]
        Dyk_loc[k, :] = scale_y^k_deriv * stenc[1:n, 2]
    end

    ### Generate Sparse Matrices from Local Operator Matrices
    # From MATLAB Implementation
    idx_rows = repeat((eachindex(Y))', n)'
    idx_columns = Array{Int64}(undef, lastindex(Y), n) # Change to eltype of existing indices
    for i in eachindex(Y)
        idx_columns[i, :] = idxs_x[idxs_y_x[i]][1]
    end
    ### Convert to Generate Sparse Matrix 
    Dxk = sparse(vec(idx_rows), vec(idx_columns), vec(Dxk_loc))
    Dyk = sparse(vec(idx_rows), vec(idx_columns), vec(Dyk_loc))

    return Dxk, Dyk
end