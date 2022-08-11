function calculatenormal(X, BCelements)
    BCnormals = Array{SVector{2}}(undef, length(BCelements))
    BCtangents = Array{SVector{2}}(undef, length(BCelements))
    x_normals = zeros(length(BCelements))
    y_normals = zeros(length(BCelements))
    for i in eachindex(BCelements)
        dx = X[BCelements[i][2]][1] - X[BCelements[i][1]][1]
        dy = X[BCelements[i][2]][2] - X[BCelements[i][1]][2]
        len = norm([-dy, dx])
        BCnormals[i] = [-dy, dx] / len
        BCtangents[i] = [dx, dy] / len
        x_normals[i] = -dy / len
        y_normals[i] = dx / len
    end

    return BCnormals, BCtangents, x_normals, y_normals
end