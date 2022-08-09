function processmesh(meshname::String, markernames)
    # Function to automate processing of meshes
    # Includes functionality to import meshes, calculate normals,
    # and assemble system 

    # Testing 
    # Outward Normal Mesh
    #meshname = "rect_0_10.cgns"
    #markernames = ["left","right","top","bottom"]
    # Inward Normal Mesh
    #meshname = "naca_0012_8_X.cgns"
    #markernames = ["inlet","outlet","top","bottom","wall"]

    fid = h5open(meshname, "r")
    marker_regions = length(markernames)

    # Import Data
    x_field = "Base/dom-1/GridCoordinates/CoordinateX"
    y_field = "Base/dom-1/GridCoordinates/CoordinateY"
    Y_points, y_point_mat = extractcoordinates(fid, x_field, y_field)

    # Import Interior Data
    tri_field = "Base/dom-1/TriElements/ElementConnectivity/ data"
    tri_conns = 3
    (tri_faces, tri_elements) = extractelements(fid, Y_points, tri_field, tri_conns)
    int_range = read(fid["Base/dom-1/TriElements/ElementRange/ data"])
    Y_idx_in = int_range[1]:int_range[2]

    # Generate mesh for VTK export
    tri_cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, tri_elements[x])
                 for x in 1:length(tri_elements)]

    # Generate KNN Tree Using HNSW 
    hnsw_y = HierarchicalNSW(Y_points)
    #Add all data points into the graph
    add_to_graph!(hnsw_y)
    # Find nearest neighbor for each X point
    idxs_y, dists_y = knn_search(hnsw_y, Y_points, 2)
    idxs_y = [convert.(Int, idxs_y[x]) for x in 1:length(idxs_y)]
    # Mean distance
    h_y = mean(dists_y)[2]

    # Ghost Node Offset
    offset = abs(h_y)

    # Initialize empty vectors
    #Y_idx_bc = bc_range[1]:bc_range[2]
    Y = tri_faces
    int_cells = tri_cells
    int_range = read(fid["Base/dom-1/TriElements/ElementRange/ data"])
    bc_range = Array{Vector}(undef, marker_regions)
    ### Generate looping structure for processing through all boundary conditions at once
    # Determine BC indices per marker
    for i in 1:marker_regions
        bc_region = markernames[i]
        #bc_range_ = read(fid["Base/dom-1/"*bc_region*"/ElementRange/ data"])
        #append!(bc_range, [bc_range_])
        bc_range[i] = read(fid["Base/dom-1/" * bc_region * "/ElementRange/ data"])
    end
    Y_idx_bc_min = minimum(minimum.(bc_range))
    Y_idx_bc_max = maximum(maximum.(bc_range))
    # Process each BC and sort in CGNS order
    Y = Array{eltype(tri_faces)}(undef, Y_idx_bc_max)
    Y[Y_idx_in] = tri_faces
    #cells = Array{eltype(tri_cells)}(undef, Y_idx_bc_max)
    #cells[Y_idx_in] = tri_cells
    #cells = tri_cells
    bc_field = []
    bc_lines = Array{Vector{eltype(tri_faces)}}(undef, marker_regions)
    bc_normals = Array{Vector{eltype(tri_faces)}}(undef, marker_regions)
    bc_tangents = Array{Vector{eltype(tri_faces)}}(undef, marker_regions)
    bc_cell_type = typeof(MeshCell(VTKCellTypes.VTK_LINE, @SVector zeros(Int64, 2)))
    bc_cells = Array{bc_cell_type}(undef, Y_idx_bc_max - Y_idx_bc_min + 1)
    bc_ghost = Array{Vector{eltype(tri_faces)}}(undef, marker_regions)
    Y_idx_bc = Array{eltype(bc_range)}(undef, marker_regions)
    for i in 1:marker_regions
        # Perform Calculation
        bc_range_i = bc_range[i] # Indices where these properties will be inserted
        bc_region = markernames[i]
        bc_field = "Base/dom-1/" * bc_region * "/ElementConnectivity/ data"
        bc_conns = 2

        # Data Ranges 
        #bc_range = read(fid["Base/dom-1/"*bc_region*"/ElementRange/ data"])
        #Y_idx_bc = bc_range[1]:bc_range[2]
        #append!(Y_idx_bc, [bc_range[1]:bc_range[2]])
        Y_idx_bc[i] = bc_range[i][1]:bc_range[i][2]

        (bc_lines[i], bc_elements) = extractelements(fid, Y_points, bc_field, bc_conns)
        #scatter!(Tuple.(left_lines), label="left")

        bc_normal, bc_tangent, bc_x_normal, bc_y_normal = calculatenormal(Y_points,
                                                                          bc_elements)
        #quiver!(Tuple.(left_lines), quiver=(Tuple.(left_normals)), label="left normal")
        #quiver!(Tuple.(inlet_lines), quiver=(Tuple.(inlet_tangents)), label="inlet tangent")
        #bc_normals[Y_idx_bc[i] .- int_range[2]] = bc_normal
        #bc_tangents[Y_idx_bc[i] .- int_range[2]] = bc_tangent
        bc_normals[i] = bc_normal
        bc_tangents[i] = bc_tangent

        # Generate and store ghost cells 
        #bc_ghost[i] = genghostnodes(bc_lines[i], bc_normal, offset)
        #bc_ghost = genghostnodes(bc_lines, bc_normals, offset)
        #append!(bc_ghost, genghostnodes(bc_lines, bc_normals, offset))
        #scatter!(Tuple.(left_ghost), label="left ghost")

        # Generate BC cells
        bc_cell = [MeshCell(VTKCellTypes.VTK_LINE, bc_elements[x])
                   for x in 1:length(bc_elements)]
        #append!(cells, [MeshCell(VTKCellTypes.VTK_LINE, bc_elements[x]) for x = 1:length(bc_elements)])

        # Add to Y
        Y[Y_idx_bc[i]] = bc_lines[i]
        #Y = [tri_faces; bottom_lines; left_lines; right_lines; top_lines;
        #bottom_ghost; left_ghost; right_ghost; top_ghost]
        #append!(Y, bc_lines)

        #Add to BCcells
        #cells[Y_idx_bc[i]] = bc_cells
        bc_cells[Y_idx_bc[i] .- int_range[2]] = bc_cell
    end

    ### Check normal vector directions
    # Generate KNN Tree Using HNSW 
    hnsw_y = HierarchicalNSW(Y[Y_idx_in])
    #Add all data points into the graph
    add_to_graph!(hnsw_y)
    # Find nearest neighbor for each X point
    idxs_y, dists_y = knn_search(hnsw_y, Y[Y_idx_bc[1][1]], 1)
    idxs_y = convert.(Int, idxs_y)
    normal_check = bc_normals[1][1]
    Y[idxs_y[1]]
    Y[Y_idx_bc[1][1]]
    # Orient normals such that they point out of domain
    # Usually only applied to meshes with holes inside
    inward_direction = Y[Y_idx_bc[1][1]] - Y[idxs_y[1]]
    normal_orient = sign(dot(inward_direction, normal_check))
    bc_normals = normal_orient .* bc_normals
    bc_tangents = normal_orient .* bc_tangents

    # Generate ghost nodes 
    Y_idx_bc_g = [Y_idx_bc[x] .+ Y_idx_bc_max .- int_range[2] for x in 1:length(Y_idx_bc)]
    for i in 1:marker_regions
        bc_ghost[i] = genghostnodes(bc_lines[i], bc_normals[i], offset)
    end
    Y_bc_g = Array{eltype(bc_ghost[1])}(undef, Y_idx_bc_max - Y_idx_bc_min + 1)
    for i in 1:marker_regions
        Y_bc_g[Y_idx_bc[i] .- int_range[2]] = bc_ghost[i]
    end
    # Reconstruct Y Dataset and set indices
    Y_idx_in = int_range[1]:int_range[2]
    Y_idx_bc
    Y_idx_bc_g
    append!(Y, Y_bc_g) # Append ghost nodes to Y 

    # Concatenate Cells
    cells = [int_cells; bc_cells]
    #append!(cells, bc_cells)

    return Y, y_point_mat, Y_idx_in, Y_idx_bc, Y_idx_bc_g, cells, bc_normals, bc_tangents

    #int_range = read(fid["Base/dom-1/TriElements/ElementRange/ data"])
    #left_range = read(fid["Base/dom-1/left/ElementRange/ data"])
    #right_range = read(fid["Base/dom-1/right/ElementRange/ data"])
    #top_range = read(fid["Base/dom-1/top/ElementRange/ data"])
    #bottom_range = read(fid["Base/dom-1/bottom/ElementRange/ data"])
    #Y_idx_in = int_range[1]:int_range[2]
    #Y_idx_bottom = bottom_range[1]:bottom_range[2]
    #Y_idx_left = left_range[1]:left_range[2]
    #Y_idx_right = right_range[1]:right_range[2]
    #Y_idx_top = top_range[1]:top_range[2]

    # Add ghost node indices (to the end )
    #Y_idx_bottom_g = Y_idx_bottom .+ top_range[2] .- int_range[2]
    #Y_idx_left_g = Y_idx_left .+ top_range[2] .- int_range[2]
    #Y_idx_right_g = Y_idx_right .+ top_range[2] .- int_range[2]
    #Y_idx_top_g = Y_idx_top .+ top_range[2] .- int_range[2]
    # Ghost Node Indices 
    #Y_idx_bc_g = Y_idx_bc .+ bc_range[6][2] .- int_range[2]
    # Concatenate all 
    #Y = [tri_faces; bottom_lines; left_lines; right_lines; top_lines;
    #    bottom_ghost; left_ghost; right_ghost; top_ghost]
    #M = length(Y)
    # Generate Cells vector for plotting
    #cells = [tri_cells; bottom_cells; left_cells; right_cells; top_cells]

end