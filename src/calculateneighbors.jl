function calculateneighbors(X, Y, n, X_idx_in, X_idx_bc, X_idx_bc_g, Y_idx_in, Y_idx_bc,
                            Y_idx_bc_g)
    # Generate KNN Tree Using NearestNeighbors 
    ### Note: Set up such that each boundary is independent of each other.
    ### Ghost nodes for a particular boundaries can depend on each other but not
    ### from other boundaries

    # Create empty arrays for solution
    idxs_x = Array{SVector{n}}(undef, lastindex(X))
    dists_x = Array{SVector{n}}(undef, lastindex(X))
    idxs_y_x = Array{SVector{1}}(undef, lastindex(Y))
    dists_y_x = Array{SVector{1}}(undef, lastindex(Y))

    # Initialize tree for calculating each boundary 
    # Operate on subset of points to get desired separation
    for i in eachindex(X_idx_bc)
        # Construct Temp Array of Indices
        # X_idx_in_bc = Array(X_idx_in)
        # append!(X_idx_in_bc, X_idx_bc[i])
        # append!(X_idx_in_bc, X_idx_bc_g[i])
        # Construct Temp Array of Indices v2
        # X_in_bc = Array{SVector{2}}(undef, lastindex(X))
        a = @SVector [Inf, Inf]
        X_in_bc = fill(a, lastindex(X))
        X_in_bc[X_idx_in] .= X[X_idx_in]
        X_in_bc[X_idx_bc[i]] .= X[X_idx_bc[i]]
        X_in_bc[X_idx_bc_g[i]] .= X[X_idx_bc_g[i]]
        # Create Subset Tree
        hnsw_x = KDTree(X_in_bc)
        # Calculate Neighbors
        idxs_local, dists_local = knn(hnsw_x, X[X_idx_bc[i]], n, true)
        idxs_local_g, dists_local_g = knn(hnsw_x, X[X_idx_bc_g[i]], n, true)
        # Store values
        # Need to offset everything
        idxs_x[X_idx_bc[i]] = idxs_local
        dists_x[X_idx_bc[i]] = dists_local
        idxs_x[X_idx_bc_g[i]] = idxs_local_g
        dists_x[X_idx_bc_g[i]] = dists_local_g
    end

    # hnsw_x = HierarchicalNSW(X)
    # hnsw_x = KDTree(X)
    #Optionally pass a subset of the indices in data to partially construct the graph
    # X_idx_in_bc = Array(X_idx_in)
    # for i in eachindex(X_idx_bc)
    #     append!(X_idx_in_bc, X_idx_bc[i])
    #     append!(X_idx_in_bc, X_idx_bc_g[i])

    # end
    # Graph only contains interior points
    # add_to_graph!(hnsw_x, X_idx_in)
    # hnsw_x = KDTree(X[X_idx_in])
    #add_to_graph!(hnsw_x, X_idx_in_bc)
    # hnsw_x = KDTree(X[X_idx_in_bc])
    # Separate according to element type
    # Calculate NN for each BC point but without
    # including other BCs
    # idxs_x = Array{SVector{n}}(undef, lastindex(X))
    # dists_x = Array{SVector{n}}(undef, lastindex(X))
    # idxs_y_x = Array{SVector{1}}(undef, lastindex(Y))
    # dists_y_x = Array{SVector{1}}(undef, lastindex(Y))
    # Calculate BC and ghost at the same time
    # for i in eachindex(X_idx_bc)
    #     for j in eachindex(X_idx_bc[i])
    #         idxs_local, dists_local = knn(hnsw_x, X[X_idx_bc[i][j]], n - 2, true)
    #         idxs_x[X_idx_bc[i][j]] = [X_idx_bc[i][j]; idxs_local; X_idx_bc_g[i][j]]
    #         idxs_x[X_idx_bc_g[i][j]] = [X_idx_bc_g[i][j]; X_idx_bc[i][j]; idxs_local]
    #         dists_x[X_idx_bc[i][j]] = [0.0; dists_local; 0.0] #Note: 0.0 at end should be ghost offset
    #         dists_x[X_idx_bc_g[i][j]] = [0.0; 0.0; dists_local] # Second 0.0 should be ghost offset
    #     end
    # end

    # Calculate weights for interior
    # X_idx_in_bc = Array(X_idx_in)
    # for i in eachindex(X_idx_bc)
    # append!(X_idx_in_bc, X_idx_bc[i])
    # end
    # Add BC points to graph
    # hnsw_x = KDTree(X[X_idx_in_bc])
    hnsw_x = KDTree(X)
    # Calculate weights for interior 
    for i in eachindex(X_idx_in)
        idxs_x[X_idx_in[i]], dists_x[X_idx_in[i]] = knn(hnsw_x, X[X_idx_in[i]], n, true)
    end
    # Add all points to graph
    # Calculate Y Nearest Neighbor
    hnsw_x = KDTree(X)
    # add_to_graph!(hnsw_x)
    for i in eachindex(Y)
        idxs_y_x[i], dists_y_x[i] = knn(hnsw_x, Y[i], 1, true)
    end

    return idxs_x, idxs_y_x, dists_x, dists_y_x
end

# function calculateneighbors(X, Y, X_idx_in, X_idx_bc, X_idx_bc_g, Y_idx_in, Y_idx_bc,
#     Y_idx_bc_g)
# # Generate KNN Tree Using NearestNeighbors 
# ### Note: we can make it so that the influence of other boundaries 
# ### is not present in BCs by evaluating NN using only interior points
# ### then at the end add BCs to tree and eval NN of interior points

# #Intialize HNSW struct
# # hnsw_x = HierarchicalNSW(X)
# # hnsw_x = KDTree(X)
# #Optionally pass a subset of the indices in data to partially construct the graph
# X_idx_in_bc = Array(X_idx_in)
# for i in eachindex(X_idx_bc)
# append!(X_idx_in_bc, X_idx_bc[i])
# end
# # Graph only contains interior points
# # add_to_graph!(hnsw_x, X_idx_in)
# hnsw_x = KDTree(X[X_idx_in])
# #add_to_graph!(hnsw_x, X_idx_in_bc)
# # hnsw_x = KDTree(X[X_idx_in_bc])
# # Separate according to element type
# # Calculate NN for each BC point but without
# # including other BCs
# idxs_x = Array{SVector{n}}(undef, lastindex(X))
# dists_x = Array{SVector{n}}(undef, lastindex(X))
# idxs_y_x = Array{SVector{1}}(undef, lastindex(Y))
# dists_y_x = Array{SVector{1}}(undef, lastindex(Y))
# # Calculate BC and ghost at the same time
# for i in eachindex(X_idx_bc)
# for j in eachindex(X_idx_bc[i])
# idxs_local, dists_local = knn(hnsw_x, X[X_idx_bc[i][j]], n - 2, true)
# idxs_x[X_idx_bc[i][j]] = [X_idx_bc[i][j]; idxs_local; X_idx_bc_g[i][j]]
# idxs_x[X_idx_bc_g[i][j]] = [X_idx_bc_g[i][j]; X_idx_bc[i][j]; idxs_local]
# dists_x[X_idx_bc[i][j]] = [0.0; dists_local; 0.0] #Note: 0.0 at end should be ghost offset
# dists_x[X_idx_bc_g[i][j]] = [0.0; 0.0; dists_local] # Second 0.0 should be ghost offset
# end
# end
# # Remove 1 neighbor for ghost
# # for i in eachindex(X_idx_bc)
# #     for j in eachindex(X_idx_bc[i])
# #         idxs_local, dists_local = knn(hnsw_x, X[X_idx_bc[i][j]], n - 1, true)
# #         idxs_x[X_idx_bc[i][j]] = [idxs_local; X_idx_bc_g[i][j]]
# #         idxs_x[X_idx_bc_g[i][j]] = [X_idx_bc_g[i][j]; idxs_local]
# #         dists_x[X_idx_bc[i][j]] = [dists_local; 0.0] #Note: 0.0 at end should be ghost offset
# #         dists_x[X_idx_bc_g[i][j]] = [0.0; dists_local] # Second 0.0 should be ghost offset
# #     end
# # end
# # Add BC points to graph
# # add_to_graph!(hnsw_x, X_idx_in_bc)
# hnsw_x = KDTree(X[X_idx_in_bc])
# #add_to_graph!(hnsw_x)
# # hnsw_x = KDTree(X)
# # Calculate weights for interior 
# for i in X_idx_in
# idxs_x[i], dists_x[i] = knn(hnsw_x, X[X_idx_in[i]], n, true)
# end
# # Add all points to graph
# # Calculate Y Nearest Neighbor
# hnsw_x = KDTree(X)
# # add_to_graph!(hnsw_x)
# for i in eachindex(Y)
# idxs_y_x[i], dists_y_x[i] = knn(hnsw_x, Y[i], 1, true)
# end

# return idxs_x, idxs_y_x, dists_x, dists_y_x
# end