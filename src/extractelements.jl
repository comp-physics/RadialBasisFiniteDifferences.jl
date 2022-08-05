function extractelements(fid, X, field::String, conn::Integer)
    # Function for extracting elements from HDF5 based CGNS file

    element_conn = read(fid[field]) # Read connectivity
    element_conn_ = reshape(element_conn, conn, :) # Reshape according to number of vertices per element
    element_connectivity = copy(vec(reinterpret(SVector{conn,Int64}, element_conn_))) # Cast as vector of static vectors for each element
    element_center = X[element_connectivity[1]] # Extract vertices according to connectivity 
    #element_center = Array{MVector,1}(undef, length(element_connectivity)) # Initialize empty array
    # Save element centroid 
    #for i = 1:length(element_connectivity)
    #    element_center[i] = mean(X[element_connectivity[i]])
    #end

    #element_center = []

    element_center = [mean(X[element_connectivity[x]])
                      for x in 1:length(element_connectivity)]

    return element_center, element_connectivity
end