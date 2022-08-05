function genghostnodes(element, normal, offset)
    # Function for generating ghost node elements from boundary elements

    ghost_node = element .+ normal .* offset

    return ghost_node
end