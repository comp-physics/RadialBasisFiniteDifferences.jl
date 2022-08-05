function extractcoordinates(fid, field_x::String, field_y::String)
    # Function for extracting grid coordinates from HDF5 based CGNS file

    x_data = fid[field_x]
    x = read(x_data)[" data"]
    y_data = fid[field_y]
    y = read(y_data)[" data"]

    # Generating X
    x_ = [x y]
    X = copy(vec(reinterpret(SVector{2,Float64}, x_')))

    return X, x_'
end