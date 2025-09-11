# ToDo: make data structure that can handle 1d,2d,3d, meshes for a number of element types
module Meshes 

using Gmsh, LinearAlgebra, Combinatorics, Base.Threads, Base.Iterators, StaticArrays

include("Polynomials.jl")

include("SearchTrees.jl")
using .Polynomials

export readGmsh, Mesh, getElem, getElem2, getReferenceElem, getElementType_Info, getOutwardNormal, getNormal, getJacobian, getDeterminant, SearchTree, get_localElementList, build_cube_mesh

###############################################################################################################
#                                               1D Mesh
###############################################################################################################

# The Mesh Structure
mutable struct Mesh1D
    xL::Float64      # Left Endpoint 
    xR::Float64      # Right Endpoint

    N::Int64                 # Number of Elements
    grid::Vector{Float64}    # The mesh points
    dx::Vector{Float64}      # Vector with the elementsizes

    Neval::Int64            # Evaluation Points per element
    xEval::Vector{Float64}   # Evaluation Points on the mesh
end

# Returns the Element index in which a given point resides in
# Returns 0 for evaluation outside of the Domain
function getElem(x::Float64, mesh::Mesh1D)

    # x     Evaluation Point
    # mesh  The mesh

    E = 0

    for elem=1:mesh.N
        if (x >= mesh.grid[elem])&&(x <= mesh.grid[elem+1])
            E = elem
            break
        end
    end

    if E == 0
        @warn "Evaluation Point not in the Domain of the mesh! Returns 0."
    end

    return E

end

# Generates a 1D Mesh 
function GenerateMesh1D(N::Int64, xL::Float64=0.0, xR::Float64=1.0; Neval::Int=15, type="Uniform", c=1.0)

    # N         number of Elements
    # Neval     Number of Evaluation Points per element, optional
    # xL        Left Interval Point, optional
    # xR        Right Interval Point, optional


    # The mesh points
    grid = collect(LinRange(xL,xR,N+1))


    # For Mesh with central Element as the largest
    # Requires at least 3 Elements
    if type == "Centered"

        N2 = ceil(Int,N/2)
        Left = ones(N2)
        Right = ones(N2)

        for l=2:N2
            Left[l] = c*Left[l-1]
            Right[N2-(l-1)] = c*Right[N2-(l-2)]
        end
        dx = vcat(Left,Right)

        grid = zeros(N+1)
        grid[1]= xL
        for l=2:N+1
            grid[l] = grid[l-1]+dx[l-1]
        end
        sort!(grid)

        grid = grid .- grid[1]              # Move left point to zero
        grid = (xR-xL)*(grid / grid[end])   # Scale to length 
        grid = grid .+ xL                    # Move Left point to xL
    end

     # For Smoothly Varying mesh
     if type == "SmoothVarying"

        dx = grid[2]-grid[1]
        for l=2:N+1
            grid[l] = grid[l] + c*dx*sin.( abs(grid[l]/xR)*pi)
        end
        sort!(grid)
        grid = grid .- grid[1]              # Move left point to zero
        grid = (xR-xL)*(grid / grid[end])   # Scale to length 
        grid = grid .+ xL                    # Move Left point to xL
    end

    # For the Random mesh
    if type == "Random"
        for l=2:N+1
            grid[l] = grid[l] + (c*(rand()*2 -1))*(1/N)
        end
        sort!(grid)
        grid = grid .- grid[1]              # Move left point to zero
        grid = (xR-xL)*(grid / grid[end])   # Scale to length 
        grid = grid .+ xL                    # Move Left point to xL
    end

    # The element sizes
    dx = zeros(N)
    for k=1:N
        dx[k] = grid[k+1] - grid[k]
    end

    # The Evaluation Points
    Eval_Ref = collect(LinRange(-1,1,Neval)) 
    xEval = zeros(N*Neval)
    for j=1:N
        xEval[Neval*(j-1)+1:Neval*j] = grid[j] .+ (grid[j+1]-grid[j])/2 .* (Eval_Ref .+ 1) 
    end

    # The mesh Structure
    mesh = Mesh(xL,xR,N,grid,dx,Neval,xEval)

    return mesh

end

# Generates a Mesh from a sequence of Gridpoints
function GenerateMesh1D(Pts::Array)

    grid = sort(Pts)
    N = length(grid)-1
    xL = minimum(grid)
    xR = maximum(grid)

    # The element sizes
    dx = zeros(N)
    for k=1:N
        dx[k] = grid[k+1] - grid[k]
    end

    # The Evaluation Points
    Neval = 15
    Eval_Ref = collect(LinRange(-1,1,Neval)) 
    xEval = zeros(N*Neval)
    for j=1:N
        xEval[Neval*(j-1)+1:Neval*j] = grid[j] .+ (grid[j+1]-grid[j])/2 .* (Eval_Ref .+ 1) 
    end

    # The mesh Structure
    mesh = Mesh(xL,xR,N,grid,dx,Neval,xEval)

    return mesh

end

# Globally refines a mesh. I.e takes a mesh and subdivides each element in k equaly sized pieces
function RefineGlobal(mesh::Mesh1D; k::Int=2)

    # mesh      The initial Mesh to be refined
    # k         Number of sections each element is refined to, optional


    grid = zeros(k*mesh.N+1)

    # The refined Grid
    for elem=1:mesh.N
        for l=1:k+1
            grid[(elem-1)*k+l] = mesh.grid[elem] + (l-1)*(1/k)*mesh.dx[elem]
        end
    end

    # The refined Elemetsizes
    dx = zeros(k*mesh.N)
    for l=1:k*mesh.N
        dx[l] = grid[l+1] - grid[l]
    end

    # The Evaluation Points
    Eval_Ref = collect(LinRange(-1,1,mesh.Neval)) 
    xEval = zeros(k*mesh.N*mesh.Neval)
    for j=1:k*mesh.N
        xEval[mesh.Neval*(j-1)+1:mesh.Neval*j] = grid[j] .+ (grid[j+1]-grid[j])/2 .* (Eval_Ref .+ 1) 
    end

    # The mesh Structure
    mesh_refined = Mesh(mesh.xL,mesh.xR,k*mesh.N,grid,dx,mesh.Neval,xEval)

    return mesh_refined

end



###############################################################################################################
#                                               2D Mesh
###############################################################################################################

mutable struct Edge
    vertices     # vertices composing the edge
    tangent                # oriented Tangent
    he::Float64                                 # edge length
end

mutable struct Face
    vertices     # vertices composing the face
    normal                # oriented normal
    hf::Float64                                 # face length
end

# mutable struct SearchTree
#     Nodes::Matrix{Float64}
#     Childs::Union{Vector{SearchTree},Nothing}
#     ElemList::Array{Int}
#     numElem::Int
# end



# The Mesh Structure
mutable struct Mesh
    #------------------------------------------------------------------------------------------------#
    #                           Basic Mesh information 
    #------------------------------------------------------------------------------------------------#
    NC::Base.RefValue{Int64}    # Number of Cells
    NV::Base.RefValue{Int64}    # Number of Vertices
    NE::Base.RefValue{Int64}    # Number of Edges
    NF::Base.RefValue{Int64}    # Number of Faces
    element_types::Array{Int64}    # Types of elements


    #------------------------------------------------------------------------------------------------#
    #                           Main Data Structure
    #------------------------------------------------------------------------------------------------#
    vertices::Matrix{Float64}   # Array of Vertices
    edges::Array{Edge}          # Array of Edges
    faces::Array{Face}          # Array of Faces


    #------------------------------------------------------------------------------------------------#
    #                           Connectivity Data Structures
    #------------------------------------------------------------------------------------------------#
    j_cv::Matrix{Int64}        # Vertex connectivity matrix
    j_cf::Matrix{Int64}        # Face Connectivity matrix
    j_ce::Matrix{Int64}        # Edge connectivity matrix
    # the m-th vertex of the n-th cell has column-index j_cv[n,m] in the matrix vertices
    # the m-th face of the n-th cell has index j_cf[n,m] in the array faces
    # the m-th edge of the n-th cell has index j_ce[n,m] in the array edges

    face2elem::Dict{Int, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}} # returns element indices and locla face indices, of a given face
    # the face f in the array faces returns: (e1,fl1), (e2,fl2) = face2elem[f]
    # e1,e2 are the elements that have the face f, with local indices fl1, fl2


    #------------------------------------------------------------------------------------------------#
    #                           Interior Component Data
    #------------------------------------------------------------------------------------------------#
    NinV::Base.RefValue{Int64}  # Number of vertices in the interior
    NinE::Base.RefValue{Int64}  # Number of Edges in the interior
    NinF::Base.RefValue{Int64}  # Number of Faces in the interior

    in_cv::Array{Int64}     # Array of Vertex indices for the interior
    in_cf::Array{Int64}     # Array of Face indices for the interior
    in_ce::Array{Int64}     # Array of Edge indices for the interior
    # in_cv contains the column indices for the matrix vertices, of the vertices in the interior
    # in_cf contains the indices for the array edges, of the egdes in the interior
    # in_ce contains the indices for the array faces, of the faces in the interior

    #------------------------------------------------------------------------------------------------#
    #                           Boundary Component Data
    #------------------------------------------------------------------------------------------------#
    Nbd::Base.RefValue{Int64}   # Number of boundary components
    NbdV::Array{Int64}          # Array with number of vertices per boundary component
    NbdE::Array{Int64}          # Array with number of Edges per boundary component
    NbdF::Array{Int64}          # Array with number of Faces per boundary component

    bd_cv::Array{Array{Int64}}     # Arrays of Vertex indices for boundaries
    bd_cf::Array{Array{Int64}}     # Arrays of Face indices for boundaries
    bd_ce::Array{Array{Int64}}     # Arrays of Edge indices for boundaries
    # bd_cv[i] contains the column indices for the matrix vertices, of the vertices in the boundary component i
    # bd_ce[i] contains the indices for the array edges, of the egdes in the boundary component i
    # bd_cf[i] contains the indices for the array faces, of the faces in the boundary component i


    #------------------------------------------------------------------------------------------------#
    #                                          SearchTree
    #------------------------------------------------------------------------------------------------#
    ST::SearchTree  #A SearchTree to accelerate evaluation

end


function readGmsh(elemTypes, NodeCoords, NodeTags, NodeTags_elem, EdgeTags, EdgeNodes, FaceTags, FaceNodes,  NodeTags_bd, EdgeTags_bd, FaceTags_bd, dim)

    #------------------------------------------------------------------------------------------------#
    #       Initializing Basic Mesh information 
    #------------------------------------------------------------------------------------------------#
    eT = elemTypes[1]
    nVertex, ~, ~, nVertEdge, nVertFace, ~, ~, ~ = getElementType_Info(eT)


    # Getting the vertices
    vertices = reshape(NodeCoords, 3, div(length(NodeCoords),3))
    node_ids = NodeTags
    NV = Ref(size(vertices,2))

    # Getting the node lists for the elements
    element_node_ids = reshape(NodeTags_elem[1], nVertex, div(length(NodeTags_elem[1]),nVertex))
    NC = Ref(size(element_node_ids,2))

    # Getting basic information about the edges
    edge_node_ids = reshape(EdgeNodes, nVertEdge[1], div(length(EdgeNodes),nVertEdge[1]))
    NE = Ref(size(edge_node_ids,2))

     # Getting basic information about the faces
    NF = 0
    face_node_ids = reshape(FaceNodes, nVertFace[1], div(length(FaceNodes),nVertFace[1]))
    if dim==1 || dim==2
        face_node_ids = reshape(EdgeNodes, nVertFace[1], div(length(EdgeNodes),nVertFace[1]))
        NF = Ref(length(EdgeTags))
    else
        NF =Ref(length(FaceTags))
    end

    
    element_types = eT*ones(Int,NC[])
    #------------------------------------------------------------------------------------------------#
    #       Assigning mesh information 
    #------------------------------------------------------------------------------------------------#

    j_cv = makeNodeConnectivity(eT, NC, element_node_ids, node_ids, NV)
    j_ce = makeEdgeConnectivity(eT, NC, element_node_ids, edge_node_ids, NF)
    j_cf = makeFaceConnectivity(eT, NC, element_node_ids, face_node_ids, NF)

    face2elem = Dict{Int, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}()
    for (e, faces) in enumerate(eachrow(j_cf))
        for (lf, f) in enumerate(faces)
            if f > 0
                if haskey(face2elem, f)
                    face2elem[f] = (face2elem[f][1], (e, lf))
                else
                    face2elem[f] = ((e, lf), (-1, -1))  # will be filled later
                end
            end
        end
    end

    edges = makeEdges(NE, edge_node_ids, vertices, eT)
    faces = makeFaces(NF, face_node_ids, vertices, eT)

    #------------------------------------------------------------------------------------------------#
    #       Assigning Boundary information 
    #------------------------------------------------------------------------------------------------#

    Nbd = Ref(length(NodeTags_bd))
    NbdV = Array{Int64}(undef,Nbd[])
    NbdE = Array{Int64}(undef,Nbd[])
    NbdF = Array{Int64}(undef,Nbd[])

    bd_cv = Array{Array{Int64}}(undef,Nbd[])
    bd_ce = Array{Array{Int64}}(undef,Nbd[])
    bd_cf = Array{Array{Int64}}(undef,Nbd[])


    # Loop over the boundary components
    for i=1:Nbd[]

        # Updating vertex information for component i
        NbdV[i] = length(NodeTags_bd[i])
        bd_cv[i] = makeEntityConnectivity_bd(0, NbdV[i], NodeTags, NodeTags_bd[i], NV)

        # Updating edge information for component i
        NbdE[i] = length(EdgeTags_bd[i])
        bd_ce[i] = makeEntityConnectivity_bd(1, NbdE[i], EdgeTags, EdgeTags_bd[i], NE)

        # Updating face information for component i
        NbdF[i] = length(FaceTags_bd[i])
        bd_cf[i] = makeEntityConnectivity_bd(1, NbdF[i], FaceTags, FaceTags_bd[i], NF)
    end


    #------------------------------------------------------------------------------------------------#
    #       Assigning Interior information 
    #------------------------------------------------------------------------------------------------#

    in_cv = collect(1:NV[])
    in_cf = collect(1:NF[])
    in_ce = collect(1:NE[])

    # Loop over all boundary components
    for i=1:Nbd[]
        # Update vertex information
        for v = 1:NbdV[i]
            deleteat!(in_cv, findall(in_cv .== bd_cv[i][v]))
        end

        # Update Edge information
        for e = 1:NbdE[i]
            deleteat!(in_ce, findall(in_ce .== bd_ce[i][e]))
        end

        # Update Face information
        for f = 1:NbdF[i]
            deleteat!(in_cf, findall(in_cf .== bd_cf[i][f]))
        end
    end

    NinV = Ref(length(in_cv))
    NinE = Ref(length(in_ce))
    NinF = Ref(length(in_cf))


    #------------------------------------------------------------------------------------------------#
    #       Building the SearchTree 
    #------------------------------------------------------------------------------------------------#


    # ST = InitQuadtree(vertices, NC[])
    # BuildQuadtree(ST, vertices, j_cv; maxLvls=6, ElemPerCell=50)

    ST = InitSearchtree(dim, vertices, NC[])
    BuildSearchtree(ST, vertices, j_cv; maxLvls=6, ElemPerCell=50)

    #------------------------------------------------------------------------------------------------#
    #       Creating the mesh
    #------------------------------------------------------------------------------------------------#

    mesh =  Mesh(NC,NV,NE,NF,element_types,
                   vertices,edges,faces,
                   j_cv,j_cf,j_ce,face2elem,
                   NinV,NinE,NinF,
                   in_cv,in_cf,in_ce,
                   Nbd,NbdV,NbdE,NbdF,
                   bd_cv,bd_cf,bd_ce,
                   ST)
    return mesh
end

function build_cube_mesh(mesh_vertices::Matrix{Float64}, ElemsPerDim::Vector{Int}, dim::Int)



    NC = Ref(prod(ElemsPerDim))

    if dim==1

        #----------------------------------------------------------------------#
        #       Initializing Basic Mesh information 
        #----------------------------------------------------------------------#

        NF = Ref(NC[]+1)
        NE = Ref(NC[]+1)
        NV = Ref(NC[]+1)

        dx = (mesh_vertices[2]-mesh_vertices[1])/ElemsPerDim[1]
        vertices = reshape(collect(mesh_vertices[1]:dx:mesh_vertices[2]),1,NV[])

        eT = 1
        element_types = eT*ones(Int,NC[])

        #-----------------------------------------------------------------------#
        #       Assigning mesh information 
        #-----------------------------------------------------------------------#

        node_ids = reshape(collect(1:NV[]),1,NV[])
        edge_node_ids = reshape(collect(1:NE[]),1,NE[])
        face_node_ids = reshape(collect(1:NF[]),1,NF[])

        element_node_ids = zeros(Int64,2,NC[])
        for i=1:NC[]
            element_node_ids[:,i] = [i;i+1] 
        end

        j_cv = makeNodeConnectivity(eT, NC, element_node_ids, node_ids, NV)
        j_ce = makeEdgeConnectivity(eT, NC, element_node_ids, edge_node_ids, NF)
        j_cf = makeFaceConnectivity(eT, NC, element_node_ids, face_node_ids, NF)

        face2elem = Dict{Int, Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}()
        for (e, faces) in enumerate(eachrow(j_cf))
            for (lf, f) in enumerate(faces)
                if f > 0
                    if haskey(face2elem, f)
                        face2elem[f] = (face2elem[f][1], (e, lf))
                    else
                        face2elem[f] = ((e, lf), (-1, -1))  # will be filled later
                    end
                end
            end
        end

        edges = makeEdges(NE, edge_node_ids, vertices, eT)
        faces = makeFaces(NF, face_node_ids, vertices, eT)

        #-----------------------------------------------------------------------#
        #       Assigning Boundary information 
        #-----------------------------------------------------------------------#


        Nbd = Ref(2)
        NodeTags_bd = [node_ids[1]; node_ids[end]]
        EdgeTags_bd = [edge_node_ids[1]; edge_node_ids[end]]
        FaceTags_bd = [face_node_ids[1]; face_node_ids[end]]

        NbdV = Array{Int64}(undef,Nbd[])
        NbdE = Array{Int64}(undef,Nbd[])
        NbdF = Array{Int64}(undef,Nbd[])

        bd_cv = Array{Array{Int64}}(undef,Nbd[])
        bd_ce = Array{Array{Int64}}(undef,Nbd[])
        bd_cf = Array{Array{Int64}}(undef,Nbd[])


        # Loop over the boundary components
        for i=1:Nbd[]

            # Updating vertex information for component i
            NbdV[i] = length(NodeTags_bd[i])
            bd_cv[i] = makeEntityConnectivity_bd(0, 1, node_ids, NodeTags_bd[i], NV)

            # Updating edge information for component i
            NbdE[i] = length(EdgeTags_bd[i])
            bd_ce[i] = makeEntityConnectivity_bd(1, 1, edge_node_ids, EdgeTags_bd[i], NE)

            # Updating face information for component i
            NbdF[i] = length(FaceTags_bd[i])
            bd_cf[i] = makeEntityConnectivity_bd(1, 1, face_node_ids, FaceTags_bd[i], NF)
        end


        #-----------------------------------------------------------------------#
        #       Assigning Interior information 
        #-----------------------------------------------------------------------#

        in_cv = collect(1:NV[])
        in_cf = collect(1:NF[])
        in_ce = collect(1:NE[])

        # Loop over all boundary components
        for i=1:Nbd[]
            # Update vertex information
            for v = 1:NbdV[i]
                deleteat!(in_cv, findall(in_cv .== bd_cv[i][v]))
            end

            # Update Edge information
            for e = 1:NbdE[i]
                deleteat!(in_ce, findall(in_ce .== bd_ce[i][e]))
            end

            # Update Face information
            for f = 1:NbdF[i]
                deleteat!(in_cf, findall(in_cf .== bd_cf[i][f]))
            end
        end

        NinV = Ref(length(in_cv))
        NinE = Ref(length(in_ce))
        NinF = Ref(length(in_cf))


        #-----------------------------------------------------------------------#
        #       Building the SearchTree 
        #-----------------------------------------------------------------------#


        # ST = InitQuadtree(vertices, NC[])
        # BuildQuadtree(ST, vertices, j_cv; maxLvls=6, ElemPerCell=50)

        ST = InitSearchtree(dim, vertices, NC[])
        BuildSearchtree(ST, vertices, j_cv; maxLvls=6, ElemPerCell=20)

        #-----------------------------------------------------------------------#
        #       Creating the mesh
        #-----------------------------------------------------------------------#

        mesh =  Mesh(NC,NV,NE,NF,element_types,
                    vertices,edges,faces,
                    j_cv,j_cf,j_ce,face2elem,
                    NinV,NinE,NinF,
                    in_cv,in_cf,in_ce,
                    Nbd,NbdV,NbdE,NbdF,
                    bd_cv,bd_cf,bd_ce,
                    ST)
        return mesh

    elseif dim==2
    end

end


###############################################################################################################
#                                Element and Transformation Information
###############################################################################################################
# Getting info about an element type
# Implemented:
#   * Line, Triangle
# TODO: 
#   * Tetrahedron, Cube, Qudrangle, Tetrahedron 
function getElementType_Info(type::Integer)

    # nVertEdge and nVertFace are vectors to account for higher order elements,
    # or elements with differnt types of faces (Pyramids and Prisms)

    if type==1 # 2 Node Line
        # reference line:
        #  (v1=e1=f1)-----(v2=e2=f2)

        nVertex = 2     # number Vetices in element
        nEdges = 2      # number Edges in element
        nFaces = 2      # number Faces in element

        nVertEdge = [1 1]     # number of vertices of each edge in element
        nVertFace = [1 1]     # number of vertices of each face in element
        fTs = [Int(0) Int(0)]   # Face types

        Jacobian = x -> 0.5  # Element Jacobian
        Face_Jacobian = [x-> 1.0; x->1.0]    # Face Jacobians

        return nVertex, nEdges, nFaces, nVertEdge, nVertFace, fTs, Jacobian, Face_Jacobian

    elseif type==2 # 3 Node Triangle
        # reference triangle:
        #       (v3)
        #        |  \
        # (e2=f2)|    \ (e1=f1)
        #        |      \
        #       (v1)-----(v2)
        #          (e3=f3)

        nVertex = 3     # number Vetices in element
        nEdges = 3      # number Edges in element
        nFaces = 3      # number Faces in element

        nVertEdge = [2 2 2]     # number of vertices of each edge in element
        nVertFace = [2 2 2]     # number of vertices of each face in element
        fTs = [Int(1) Int(1) Int(1)]   # Face types

        # Actually the determinans
        Jacobian = x -> 0.5     # Element Jacobian

        Face_Jacobian = [x -> 1/sqrt(2);   # Quadrature length / Unit Face-length 
                         x -> 1/2;
                         x -> 1/2]        
        

        return nVertex, nEdges, nFaces, nVertEdge, nVertFace, fTs, Jacobian, Face_Jacobian


    elseif type==3 # 4 Node Quadrangle
        # reference Quadrangle:
        #           (e3=f3)
        #       (v3)-------(v4)
        #        |          |
        # (e2=f2)|          | (e4=f4)
        #        |          |
        #       (v1)-------(v2)
        #           (e1=f1)

        nVertex = 4     # number Vetices in element
        nEdges = 4      # number Edges in element
        nFaces = 4      # number Faces in element

        nVertEdge = [2 2 2 2]     # number of vertices of each edge in element
        nVertFace = [2 2 2 2]     # number of vertices of each face in element
        fTs = [Int(1) Int(1) Int(1) Int(1)]   # Face types

        # Actually the determinans
        Jacobian = x -> 0.5     # Element Jacobian

        Face_Jacobian = [x -> 1/sqrt(2);   # Quadrature length / Unit Face-length 
                         x -> 1/2;
                         x -> 1/2;
                         x -> 1/2]        
        

        return nVertex, nEdges, nFaces, nVertEdge, nVertFace, fTs, Jacobian, Face_Jacobian
    elseif type==4 # 4 Node tetrahedron
        # TODO: Redo ASCII graphics, check ordering
        # reference tetrahedron: 
        #             (v4)                              
        #            ,/|`\                            
        #          ,/  |  `\                            
        #  (e4)  ,/    '.   `\(e6)                         
        #      ,/  (f2) |(f3) `\                     
        #    ,/    (f4) | (e5)  `\                 
        # (v1)-----------'.------(v3)
        #   `\.  (e3)   | (f1) ,/    
        #      `\.      |    ,/  (e2)                
        #    (e1) `\.   '. ,/             
        #             `\. |/               
        #                (v2)                        
      

        nVertex = 4     # number Vetices in element
        nEdges = 6      # number Edges in element
        nFaces = 4      # number Faces in element

        nVertEdge = [3 3 3 3 3 3]     # number of vertices of each edge in element
        nVertFace = [3 3 3 3]     # number of vertices of each face in element

        return nVertex, nEdges, nFaces, nVertEdge, nVertFace
    end
    # elseif type==5 # 8 Node Hexaedron/Cube
end

function getJacobian(vertices, eT::Integer)

    if eT==0
        J = x -> 1.0
        JmT = x -> 1.0
        detJ = x-> 1.0

        return J, JmT, detJ

    elseif eT==1 # 2-node Line
        J =   x -> ((vertices[1,2]-vertices[1,1])/2)   # Jacobian
        JmT = x -> (2/(vertices[1,2]-vertices[1,1]))  # Inverse Transpose Jacobian
        detJ = x -> abs((vertices[1,2]-vertices[1,1])/2)

        return J, JmT, detJ

    elseif eT==2
        A = [vertices[1,2]-vertices[1,1]  vertices[1,3]-vertices[1,1];
             vertices[2,2]-vertices[2,1]  vertices[2,3]-vertices[2,1]]

        J =  x -> A
        JmT = x -> transpose(inv(A))
        detJ = x -> abs(det(A))

        return J, JmT, detJ

    elseif eT==3 
        v1, v2, v3, v4 = eachcol(vertices)

        J = x -> [-(1-x[2])*v1[1]+(1-x[2])*v2[1]-x[2]*v3[1]+x[2]*v4[1]  -(1-x[1])*v1[1]-x[1]*v2[1]+(1-x[1])*v3[1]+x[1]*v4[1];
                  -(1-x[2])*v1[2]+(1-x[2])*v2[2]-x[2]*v3[2]+x[2]*v4[2]  -(1-x[1])*v1[2]-x[1]*v2[2]+(1-x[1])*v3[2]+x[1]*v4[2]]
        JmT = x -> transpose(inv( [-(1-x[2])*v1[1]+(1-x[2])*v2[1]-x[2]*v3[1]+x[2]*v4[1]  -(1-x[1])*v1[1]-x[1]*v2[1]+(1-x[1])*v3[1]+x[1]*v4[1];
                                   -(1-x[2])*v1[2]+(1-x[2])*v2[2]-x[2]*v3[2]+x[2]*v4[2]  -(1-x[1])*v1[2]-x[1]*v2[2]+(1-x[1])*v3[2]+x[1]*v4[2]] ))
        detJ = x -> abs(det(J(x)))

        return J, JmT, detJ

    elseif eT==4

        # Points
        c1 = .25*(sqrt(5)-1)
        c2 = .25*(sqrt(5)+1)

        s1 =  .25*sqrt(10 + 2*sqrt(5))
        s2 =  .25*sqrt(10 - 2*sqrt(5))

        # triangles
        t1 = [0.0 2*s2 s2;
            0.0 0.0  c2]
        t2 = [2*s2 s1+s2 s2;
            0.0  c1+c2 c2]
        t3 = [s1+s2 s2   s2;
            c1+c2 c2+1 c2]
        t4 = [s2   s2-s1 s2;
            c2+1 c1+c2 c2]
        t5 = [s2-s1 0 s2;
            c1+c2 0 c2]
        Triangles = [[t1];[t2];[t3];[t4];[t5]]

        barycenter = (1/5)*sum(vertices,dims=2)

        v_edge = hcat(vertices,vertices[:,1])

        J = Array{Function}(undef,5)
        JmT = Array{Function}(undef,5)
        detJ = Array{Function}(undef,5)

        for i=1:5
            J[i], JmT[i], detJ[i] = getJacobian(hcat(v_edge[:,i:i+1],barycenter),2)
        end

        Triangles_trafo = []
        for i=1:5
            push!(Triangles_trafo,hcat(v_edge[:,i:i+1],barycenter))
        end
        dJ = t -> det([t[1,2]-t[1,1]  t[1,3]-t[1,1];
                       t[2,2]-t[2,1]  t[2,3]-t[2,1]])
        vol = sum(abs(dJ(Triangles_trafo[i])) for i=1:5)

        return y->J[Tri_ind(y,Triangles)](y), y->JmT[Tri_ind(y,Triangles)](y), y->detJ[Tri_ind(y,Triangles)](y)*vol


    elseif eT==5

        # Points
        c = sin(pi/3)
        
        # triangles
        t1 = [0.0 1.0 0.5;
              0.0 0.0 c]
        t2 = [1.0 1.5 0.5;
              0.0 c   c]
        t3 = [1.5 1   0.5;
              c   2c  c]
        t4 = [1.0 0.0 0.5;
              2c  2c  c]
        t5 = [0.0 -0.5 0.5;
              2c  c    c]
        t6 = [-0.5 0.0 0.5;
              c    0.0 c]
        Triangles = [[t1];[t2];[t3];[t4];[t5];[t6]]
      
        barycenter = (1/6)*sum(vertices,dims=2)

        v_edge = hcat(vertices,vertices[:,1])

        J = Array{Function}(undef,6)
        JmT = Array{Function}(undef,6)
        detJ = Array{Function}(undef,6)
        
        Triangles_trafo = []
        for i=1:6
            push!(Triangles_trafo,hcat(v_edge[:,i:i+1],barycenter))
        end
        dJ = t -> det([t[1,2]-t[1,1]  t[1,3]-t[1,1];
                       t[2,2]-t[2,1]  t[2,3]-t[2,1]])
        vol = sum(abs(dJ(Triangles_trafo[i])) for i=1:6)

        for i=1:6
            J[i], JmT[i], detJ[i] = getJacobian(hcat(v_edge[:,i:i+1],barycenter),2)
        end

        return y->J[Tri_ind(y,Triangles)](y), y->JmT[Tri_ind(y,Triangles)](y), y->detJ[Tri_ind(y,Triangles)](y)*vol


    elseif eT==6
        # The reference Heptagon
        al = (2/7)pi
        R = [cos(al) -sin(al);
            sin(al)  cos(al)]
        R2 = [cos(-al/2) -sin(-al/2);
            sin(-al/2)  cos(-al/2)]
        hepta = R2*[0.0; -1.0]
        for i=1:6
            p = R*hepta[:,i]
            hepta = hcat(hepta,p)
        end
        hepta .-= hepta[:,1]
        hepta_edges = hcat(hepta,hepta[:,1])
 
        Triangles = []
        bc = (1/7)*sum(hepta,dims=2)
        for i=1:7
            push!(Triangles,hcat(hepta_edges[:,i:i+1],bc))
        end
      
        barycenter = (1/7)*sum(vertices,dims=2)

        v_edge = hcat(vertices,vertices[:,1])

        J = Array{Function}(undef,7)
        JmT = Array{Function}(undef,7)
        detJ = Array{Function}(undef,7)
        
        for i=1:7
            J[i], JmT[i], detJ[i] = getJacobian(hcat(v_edge[:,i:i+1],barycenter),2)
        end

        Triangles_trafo = []
        for i=1:7
            push!(Triangles_trafo,hcat(v_edge[:,i:i+1],barycenter))
        end
        dJ = t -> det([t[1,2]-t[1,1]  t[1,3]-t[1,1];
                       t[2,2]-t[2,1]  t[2,3]-t[2,1]])
        vol = sum(abs(dJ(Triangles_trafo[i])) for i=1:7)

        return y->J[Tri_ind(y,Triangles)](y), y->JmT[Tri_ind(y,Triangles)](y), y->detJ[Tri_ind(y,Triangles)](y)*vol


    end

end

# Returns the (ordered) vertices of the refernece element and its faces
function getReferenceElem(eT::Integer)

    if eT==1
        vertices = [0.0; 1.0]
        faces = [[0.0]; [1.0]]

        return vertices,faces
    elseif eT==2
        vertices = [0.0  1.0  0.0;
                    0.0  0.0  1.0;
                    0.0  0.0  0.0]
        faces = [[vertices[:,2], vertices[:,3]], 
                 [vertices[:,1], vertices[:,3]],
                 [vertices[:,1], vertices[:,2]]]

        return vertices,faces
    elseif eT==3
        vertices = [0.0  1.0  0.0 1.0;
                    0.0  0.0  1.0 1.0;
                    0.0  0.0  0.0 0.0]
        faces = [[vertices[:,1], vertices[:,2]],
                 [vertices[:,1], vertices[:,3]], 
                 [vertices[:,3], vertices[:,4]], 
                 [vertices[:,2], vertices[:,4]]]

        return vertices,faces
    end
end




function Tri_ind(x, Triangles) 
    for i in eachindex(Triangles)
        if isinside(x,Triangles[i],inds)
            return i
        end
    end
end

function getDeterminant(z, vertices, eT::Integer)::Float64
    if eT==0 # Point
        return 1.0

    elseif eT==1 # 2-node Line
        return (vertices[1,2]-vertices[1,1])/2

    elseif eT==2 # 3-node Triangle
        A = [vertices[1,2]-vertices[1,1]  vertices[1,3]-vertices[1,1];
             vertices[2,2]-vertices[2,1]  vertices[2,3]-vertices[2,1]]

        return det(A)

    elseif eT==3 # 4-node Quadrangle
        A = [-(1-x[2])*vertices[1,1]+(1-x[2])*vertices[1,2]-x[2]*vertices[1,3]+x[2]*vertices[1,4]  -(1-x[1])*vertices[1,1]-x[1]*vertices[1,2]+(1-x[1])*vertices[1,3]+x[1]*vertices[1,4];
            -(1-x[2])*vertices[2,1]+(1-x[2])*vertices[2,2]-x[2]*vertices[2,3]+x[2]*vertices[2,4]  -(1-x[1])*vertices[2,1]-x[1]*vertices[2,2]+(1-x[1])*vertices[2,3]+x[1]*vertices[2,4]]

        return det(A)
    end
end

# fl    local face index
# TODO: 
#   * Tetrahedron, Cube, Qudrangle, Tetrahedron 
function getOutwardNormal(vertices, fl::Integer, eT::Integer)

    if eT==1
        if fl==1
            out = vertices[1]-vertices[2]
            return (1/norm(out))*out
        elseif fl==2
            out = vertices[2]-vertices[1]
            return (1/norm(out))*out
        end

    elseif eT==2
        R = [cos(pi/2)  -sin(pi/2)   0;
                 sin(pi/2)   cos(pi/2)   0; 
                 0           0           1]

        if fl==1
            out = vertices[:,2]-vertices[:,3]
            return (1/norm(out))*(R*out)
        elseif fl==2
            out = vertices[:,3]-vertices[:,1]
            return (1/norm(out))*(R*out)
        elseif fl==3
            out = vertices[:,1]-vertices[:,2]
            return (1/norm(out))*(R*out)
        end

    elseif eT==3
        R = [cos(pi/2)  -sin(pi/2)   0;
                 sin(pi/2)   cos(pi/2)   0; 
                 0           0           1]

        if fl==1
            out = vertices[:,1]-vertices[:,2]
            return (1/norm(out))*(R*out)
        elseif fl==2
            out = vertices[:,3]-vertices[:,1]
            return (1/norm(out))*(R*out)
        elseif fl==3
            out = vertices[:,4]-vertices[:,3]
            return (1/norm(out))*(R*out)
        elseif fl==4
            out = vertices[:,2]-vertices[:,4]
            return (1/norm(out))*(R*out)
        end

    end
end

function getNormal(vertices, fl::Integer, eT::Integer)

    if eT==1
        if fl==1
            out = vertices[:,1]-vertices[:,2]
            return (1/norm(out))*out
        elseif fl==2
            out = vertices[:,2]-vertices[:,1]
            return (1/norm(out))*out
        end

    elseif eT==2
        R = [cos(pi/2)  -sin(pi/2)   0;
                 sin(pi/2)   cos(pi/2)   0; 
                 0           0           1]

        if fl==1
            out = vertices[:,2]-vertices[:,3]
            return -(1/norm(out))*(R*out)
        elseif fl==2
            out = vertices[:,3]-vertices[:,1]
            return (1/norm(out))*(R*out)
        elseif fl==3
            out = vertices[:,1]-vertices[:,2]
            return -(1/norm(out))*(R*out)
        end

    elseif eT==3
        R = [cos(pi/2)  -sin(pi/2)   0;
                sin(pi/2)   cos(pi/2)   0; 
                0           0           1]

        if fl==1
            out = vertices[:,2]-vertices[:,1]
            return (1/norm(out))*(R*out)
        elseif fl==2
            out = vertices[:,1]-vertices[:,3]
            return (1/norm(out))*(R*out)
        elseif fl==3
            out = vertices[:,4]-vertices[:,3]
            return (1/norm(out))*(R*out)
        elseif fl==4
            out = vertices[:,4]-vertices[:,2]
            return (1/norm(out))*(R*out)
        end

    end
end


###############################################################################################################
#                                               Search Functions
###############################################################################################################

# Gets the element index
function getElem(x, mesh::Mesh)

    # x     Evaluation Point
    # mesh  The mesh

    E = 0

    for elem=1:mesh.NC[]
        if mesh.element_types[elem]==1 # 2-node line
            if (x >= mesh.vertices[1,mesh.j_cv[elem,1]])&&(x <= mesh.vertices[1,mesh.j_cv[elem,2]])
                E = elem
                break
            end
        elseif mesh.element_types[elem]==2 # 3-node triangle

            
            # if isinside(x,mesh.vertices[:,mesh.j_cv[elem,:]],SVector(1,2,3,1))
            if isinside(x,view(mesh.vertices,:,mesh.j_cv[elem,:]),SVector(1,2,3,1))
                E = elem
                break
            end
                # lambda = Phys2Barycentric(x, mesh.vertices[:,mesh.j_cv[elem,:]])
                # if !any(lambda .< 0.0)
                #     E = elem
                #     # display(E)
                #     break
                # end
        
        elseif mesh.element_types[elem]==3 # 4-node Quadrangle
            
            if isinside(x,mesh.vertices[:,mesh.j_cv[elem,[1,2,4,3]]],SVector(1,2,3,4,1))
                E = elem
                break
            end
        end
        
    end

    if E == 0
        @warn "Evaluation Point not in the Domain of the mesh! Returns 0."
    end

    return E

end

function getElem2(x,mesh::Mesh)

    E = 0

    if typeof(QT) == SearchTree{1} # LineTree
    if mesh.ST.Childs !== nothing
        for child in mesh.ST.Childs
            if (child.Nodes[1]<=x)&&(child.Nodes[2]>=x)
                E =  getElem2(x,mesh,child)
            end
        end
    else
        for elem in mesh.ST.ElemList
            xL =  minimum(mesh.vertices[mesh.j_cv[elem,:]])
            xR =  maximum(mesh.vertices[mesh.j_cv[elem,:]])
            if (xL<=x)&&(xR>=x)
                return elem
            end
        end
    end
    elseif typeof(QT) == SearchTree{2} # QuadTree
        if mesh.ST.Childs !== nothing
            for child in mesh.ST.Childs
                if isinside(x,child.Nodes)
                    E =  getElem2(x,mesh,child)
                end
            end
        else
            for elem in mesh.ST.ElemList
                if isinside(x,view(mesh.vertices,:,mesh.j_cv[elem,:]))
                    return elem
                end
            end
        end
    end

    return E

end


function getElem2(x, mesh::Mesh, QT::SearchTree)
    if typeof(QT) == SearchTree{1} # LineTree
        if QT.Childs !== nothing
            for child in QT.Childs
                if (child.Nodes[1]<=x)&&(child.Nodes[2]>=x)
                    E =  getElem2(x,mesh,child)
                    if E !== nothing
                        return E
                    end
                end
            end
            # If no child contained the point, still check the elements of this node (in case of overlapping)
            for elem in QT.ElemList
                xL =  minimum(mesh.vertices[mesh.j_cv[elem,:]])
                xR =  maximum(mesh.vertices[mesh.j_cv[elem,:]])
                if (xL<=x)&&(xR>=x)
                    return elem
                end
            end
            return nothing
        else
            # Leaf node: directly check elements here
            for elem in QT.ElemList
                xL =  minimum(mesh.vertices[mesh.j_cv[elem,:]])
                xR =  maximum(mesh.vertices[mesh.j_cv[elem,:]])
                if (xL<=x)&&(xR>=x)
                    return elem
                end
            end
            return nothing
        end

    elseif typeof(QT) == SearchTree{2} # QuadTree
        if QT.Childs !== nothing
            for child in QT.Childs
                if isinside(x, child.Nodes)
                    E = getElem2(x, mesh, child)
                    if E !== nothing
                        return E
                    end
                end
            end
            # If no child contained the point, still check the elements of this node (in case of overlapping)
            for elem in QT.ElemList
                if mesh.element_types[elem] == 3
                    if isinside(x, view(mesh.vertices, :, mesh.j_cv[elem, [1, 2, 4, 3]]))
                        return elem
                    end
                elseif isinside(x, view(mesh.vertices, :, mesh.j_cv[elem, :]))
                    return elem
                end
            end
            return nothing
        else
            # Leaf node: directly check elements here
            for elem in QT.ElemList
                if mesh.element_types[elem] == 3
                    if isinside(x, view(mesh.vertices, :, mesh.j_cv[elem, [1, 2, 4, 3]]))
                        return elem
                    end
                elseif isinside(x, view(mesh.vertices, :, mesh.j_cv[elem, :]))
                    return elem
                end
            end
            return nothing
        end
    end
end


# Get te element List for a region specified by the vertices
function get_localElementList(vertices, QT::SearchTree)
    if QT.Childs !== nothing
        for child in QT.Childs
            if all(isinside(vi,child.Nodes) for vi in eachcol(vertices))
                ElemList = get_localElementList(vertices, child)
                if ElemList !== nothing
                    return ElemList# 
                end
            end
        end
        # If no child contained the point, still check the elements of this node (in case of overlapping)
        if all(isinside(vi,QT.Nodes) for vi in eachcol(vertices))
            if !isempty(QT.ElemList)
                return QT.ElemList
            end
        end
        return nothing
    else
        # Leaf node: directly check elements here
        if all(isinside(vi,QT.Nodes) for vi in eachcol(vertices))
            if !isempty(QT.ElemList)
                return QT.ElemList
            end
        end
        return nothing
    end
end



function isinside(x,vertices,inds::SVector)::Bool
    nvert = size(vertices,2)
    # vert_closed = hcat(vertices,vertices[:,1])
    # vert_closed = view(vertices,:,collect(Iterators.take(cycle(1:nvert),nvert+1)))
    vert_closed = view(vertices,:,inds)

    inside = false

    # p1 = vertices[1:2,1]
    p1 = view(vertices,1:2,1)
    cc = 0
    for i=1:nvert

        # p2 = vert_closed[1:2,i+1]
        p2 = view(vert_closed,1:2,i+1)
        if x[2] > minimum((p1[2],p2[2]))
            if x[2] <= maximum((p1[2],p2[2]))
                if x[1] < maximum((p1[1],p2[1]))
                    x_intersect = (x[2]-p1[2])*(p2[1]-p1[1])/(p2[2]-p1[2]) + p1[1]

                    if (p1[1]==p2[1])||(x[1]<=x_intersect)
                        cc +=1
                    end
                end
            end
        end
        p1 = p2
    end
    
    if !iseven(cc)
        inside = true
    end

    return inside
end

function isinside(x, vertices)::Bool
    nvert = size(vertices, 2)
    # display(vertices)
    vert_closed = view(vertices, 1:2, collect(Iterators.take(cycle(1:nvert), nvert + 1)))

    function on_segment(p, q, r)
        # Check if point q lies on line segment pr
        return min(p[1], r[1]) ≤ q[1] ≤ max(p[1], r[1]) &&
               min(p[2], r[2]) ≤ q[2] ≤ max(p[2], r[2]) &&
               abs((r[1]-p[1])*(q[2]-p[2]) - (q[1]-p[1])*(r[2]-p[2])) < 1e-10
    end

    # Check if point lies exactly on any edge
    for i in 1:nvert
        p1 = view(vertices, 1:2, i)
        p2 = view(vert_closed, 1:2, i+1)
        if on_segment(p1, x, p2)
            return true
        end
    end

    # Ray casting algorithm
    inside = false
    p1 = view(vertices, 1:2, 1)
    cc = 0
    for i in 1:nvert
        p2 = view(vert_closed, 1:2, i+1)
        if x[2] > min(p1[2], p2[2])
            if x[2] ≤ max(p1[2], p2[2])
                if x[1] ≤ max(p1[1], p2[1])
                    if p1[2] != p2[2]
                        x_intersect = (x[2]-p1[2])*(p2[1]-p1[1])/(p2[2]-p1[2]) + p1[1]
                        if p1[1] == p2[1] || x[1] ≤ x_intersect
                            cc += 1
                        end
                    end
                end
            end
        end
        p1 = p2
    end

    return isodd(cc)
end




###############################################################################################################
#                                               Mesh Construction helpers
###############################################################################################################

function makeNodeConnectivity(eT::Integer, NC::Base.RefValue{Int64}, element_node_ids, node_ids, NV::Base.RefValue{Int64})

    nVertex, nEdges, nFaces, nVertEdge, nVertFace, ~, ~, ~ = getElementType_Info(eT)

    j_cv = zeros(Int,NC[],nVertex)
    @threads for elem=1:NC[]
        # edge = zeros(UInt,2,3)
        # edge[:,1] = vcat(element_node_ids[1,elem], element_node_ids[2,elem])
        # edge[:,2] = vcat(element_node_ids[2,elem], element_node_ids[3,elem])
        # edge[:,3] = vcat(element_node_ids[3,elem], element_node_ids[1,elem])
    
        for i=1:nVertex
            ind = findfirst(element_node_ids[i,elem] == node_ids[j] for j=1:NV[])
            j_cv[elem,NodeOrder(eT,i)] = ind
        end
    end

    return j_cv
end

# Turns Gmsh node ordering into increasing index vertex enumeration
function NodeOrder(eT::Integer, i::Int64)
    if eT==1
        if i==1
            return 1
        elseif i==2
            return 2
        end
    elseif eT==2
        if i==1
            return 1
        elseif i==2
            return 2
        elseif i==3
            return 3
        end
    elseif eT==3
        if i==1
            return 1
        elseif i==2
            return 2
        elseif i==3
            return 4
        elseif i==4
            return 3
        end
    end    

end

# Returns the edges (given in node ids) of an element in the canonical ordering
function Nodes2Edges(ElemNode_id, eT::Integer)
    if eT==1    # 2-node Line
        # The nodes are the edges!
        edge = zeros(UInt,1,2)
        edge[1,1] = ElemNode_id[1]
        edge[1,2] = ElemNode_id[2]
    
        return edge
    elseif eT==2    # 3-node triangle
        edge = zeros(UInt,2,3)
        edge[:,1] = vcat(ElemNode_id[2], ElemNode_id[3])
        edge[:,2] = vcat(ElemNode_id[1], ElemNode_id[3])
        edge[:,3] = vcat(ElemNode_id[1], ElemNode_id[2])

        return edge
    elseif eT==3    # 4-node Quadrilateral
        edge = zeros(UInt,2,4)
        edge[:,1] = vcat(ElemNode_id[1], ElemNode_id[2])
        edge[:,2] = vcat(ElemNode_id[1], ElemNode_id[4])
        edge[:,3] = vcat(ElemNode_id[4], ElemNode_id[3])
        edge[:,4] = vcat(ElemNode_id[2], ElemNode_id[3])

        return edge
    end
end

function makeEdgeConnectivity(eT::Integer, NC::Base.RefValue{Int64}, element_node_ids, edge_node_ids, NE::Base.RefValue{Int64})

    nVertex, nEdges, nFaces, nVertEdge, nVertFace, ~, ~, ~ = getElementType_Info(eT)

    j_ce = zeros(Int,NC[],nEdges)
    @threads for elem=1:NC[]
        # edge = zeros(UInt,2,3)
        # edge[:,1] = vcat(element_node_ids[1,elem], element_node_ids[2,elem])
        # edge[:,2] = vcat(element_node_ids[2,elem], element_node_ids[3,elem])
        # edge[:,3] = vcat(element_node_ids[3,elem], element_node_ids[1,elem])
    
        edge =  Nodes2Edges(element_node_ids[:,elem], eT)
        for i=1:nEdges
            ind = nothing
            for p_edge in permutations(edge[:,i])
                ind = findfirst(p_edge == edge_node_ids[:,j] for j=1:NE[])
                if ind !== nothing
                    break
                end
            end
            j_ce[elem,i] = ind
        end
    end

    return j_ce
end

function makeEdges(NE::Base.RefValue{Int64}, edge_node_ids, vertices, eT::Integer)
    edges = Array{Edge}(undef,NE[])

    if eT == 1 # 2-node Line
        # TODO: This is Bad!!
        @threads for e=1:NE[]

            edge = vertices[:,edge_node_ids[1,e]]
            tangent = edge/norm(edge)
            he = 0.0
    
            edges[e] = Edge(edge, tangent, he)
        end
    elseif eT == 2 || eT == 3 # 3-node Triangle / 4-node Quadrilateral
        @threads for e=1:NE[]
            edge = hcat(vertices[:,edge_node_ids[1,e]],vertices[:,edge_node_ids[2,e]])
        
            tangent = edge[:,2]-edge[:,1]
            he = norm(tangent)
            tangent = tangent/he
        
            edges[e] = Edge(edge, tangent, he)
        end
    end

    return edges
end


# Returns the edges (given in node ids) of an element in the canonical ordering
# This has to translate Gmsh's node ordering to the increasing vertex index ordering!!!
function Nodes2Faces(ElemNode_id, eT::Integer)
    if eT==1    # 2-node Line
        # The nodes are the Faces!
        face = zeros(UInt,1,2)
        face[1,1] = ElemNode_id[1]
        face[1,2] = ElemNode_id[2]

        return face
    elseif eT==2    # 3-node triangle
        face = zeros(UInt,2,3)
        face[:,1] = vcat(ElemNode_id[2], ElemNode_id[3])
        face[:,2] = vcat(ElemNode_id[1], ElemNode_id[3])
        face[:,3] = vcat(ElemNode_id[1], ElemNode_id[2])

        return face
    elseif eT==3    # 4-node Quadrilateral
        face = zeros(UInt,2,4)
        face[:,1] = vcat(ElemNode_id[1], ElemNode_id[2])
        face[:,2] = vcat(ElemNode_id[1], ElemNode_id[4])
        face[:,3] = vcat(ElemNode_id[4], ElemNode_id[3])
        face[:,4] = vcat(ElemNode_id[2], ElemNode_id[3])

        return face
    end
end

function makeFaceConnectivity(eT::Integer, NC::Base.RefValue{Int64}, element_node_ids, face_node_ids, NF::Base.RefValue{Int64})

    nVertex, nEdges, nFaces, nVertEdge, nVertFace, ~, ~, ~ = getElementType_Info(eT)

    j_cf = zeros(Int,NC[],nFaces)
    @threads for elem=1:NC[]
        # edge = zeros(UInt,2,3)
        # edge[:,1] = vcat(element_node_ids[1,elem], element_node_ids[2,elem])
        # edge[:,2] = vcat(element_node_ids[2,elem], element_node_ids[3,elem])
        # edge[:,3] = vcat(element_node_ids[3,elem], element_node_ids[1,elem])
    
        face =  Nodes2Faces(element_node_ids[:,elem], eT)
        for i=1:nFaces
            ind = nothing
            for p_face in permutations(face[:,i])
                # display(p_edge)
                ind = findfirst(p_face == face_node_ids[:,j] for j=1:NF[])
                if ind !== nothing
                    break
                end
            end
            j_cf[elem,i] = ind
        end
    end

    return j_cf
end

function makeFaces(NF::Base.RefValue{Int64}, face_node_ids, vertices, eT::Integer)
    faces = Array{Face}(undef,NF[])

    if eT == 1 # 2-node Line
        # TODO: This is Bad!!
        @threads for f=1:NF[]

            face = vertices[:,face_node_ids[1,f]]
            normal = [1.0; 0.0; 0.0]#face/norm(face)
            hf = 0.0
    
            faces[f] = Face(face, normal, hf)
        end
    elseif eT == 2 || eT == 3 # 3-node Triangle / 4-node Quadrilateral
        @threads for f=1:NF[]
            # Rotation matrix by pi/2
            R = [cos(pi/2)  -sin(pi/2)   0;
                 sin(pi/2)   cos(pi/2)   0; 
                 0           0           1]
    
            face = hcat(vertices[:,face_node_ids[1,f]],vertices[:,face_node_ids[2,f]])
            normal = R*(face[:,2]-face[:,1])
            hf = norm(normal)
            normal = normal/hf
    
            faces[f] = Face(face, normal, hf)
        end  
    end

    return faces
end


function makeEntityConnectivity_bd(eT::Integer, N_ent::Int64, EntTags, EntTags_bd, N::Base.RefValue{Int64})

    ent_c = zeros(Int64,N_ent)
    for k=1:N_ent
        ent_c[k] = findfirst(EntTags_bd[k] == EntTags[i] for i in 1:N[])      
    end

    return ent_c
end



###############################################################################################################
#                                               SearchTreeRelated Things
###############################################################################################################



# function InitQuadtree(vertices::Matrix{Float64}, NC::Int)

#     xmin = minimum(vertices)
#     xmax = maximum(vertices)

#     ymin = minimum(vertices)
#     ymax = maximum(vertices)

#     Nodes = [xmin xmax xmax xmin; ymin ymin ymax ymax]

#     ElemList = collect(1:NC)

#     return SearchTree(Nodes, nothing, ElemList, NC)
# end

# function InitLinetree(vertices::Matrix{Float64}, NC::Int)

#     xmin = minimum(vertices)
#     xmax = maximum(vertices)

#     Nodes = [xmin xmax]

#     ElemList = collect(1:NC)

#     return SearchTree(Nodes, nothing, ElemList, NC)
# end

# function spawnChild(QT::SearchTree, vertices::Matrix{Float64}, j_cv::Matrix{Int}, ChildInd::Int64)
#     # Child Indexing
#     #   -----------------
#     #  |       |        |
#     #  |   1   |   2    |
#     #  |_______|________|
#     #  |       |        |
#     #  |   3   |    4   |
#     #  |_______|________|

#     xmin = QT.Nodes[1,1]
#     xmax = QT.Nodes[1,2]
#     ymin = QT.Nodes[2,1]
#     ymax = QT.Nodes[2,3]

#     if ChildInd==1
#         xmax = (xmin+xmax)/2
#         ymin = (ymin+ymax)/2
#     elseif ChildInd==2
#         xmin = (xmin+xmax)/2
#         ymin = (ymin+ymax)/2
#     elseif ChildInd==3
#         xmax = (xmin+xmax)/2
#         ymax = (ymin+ymax)/2
#     elseif ChildInd==4
#         xmin = (xmin+xmax)/2
#         ymax = (ymin+ymax)/2
#     end
#     Nodes = [xmin xmax xmax xmin; ymin ymin ymax ymax]
#     # Lines = [[Nodes[:,1] Nodes[:,2]]; [Nodes[:,2] Nodes[:,3]]; [Nodes[:,3] Nodes[:,4]]; [Nodes[:,4] Nodes[:,1]]]

#     ElemList = []
#     for i in QT.ElemList
#         verts = view(vertices,1:2,j_cv[i,:])

#         if doesIntersect(verts,Nodes)#any( isinside(x, Nodes) for x in eachcol(verts))  
#         # if !isnothing(regions)
#             push!(ElemList,i)
#         end
#     end

#     return SearchTree(Nodes,nothing,ElemList, length(ElemList))

# end

# function BuildLineTree(QT::SearchTree, vertices::Matrix{Float64}, j_cv::Matrix{Int}; lv=1, maxLvls=4, ElemPerCell=50)

#     QT.Childs = Vector{SearchTree}(undef,2)
    
#      for i in eachindex(QT.Childs) 
#         QT.Childs[i] =  spawnChild(QT,vertices, j_cv,i)

#         if (lv < maxLvls)&&(QT.Childs[i].numElem > ElemPerCell)
#             BuildQuadtree(QT.Childs[i], vertices, j_cv, lv=lv+1, maxLvls=maxLvls, ElemPerCell=ElemPerCell)
#         end
#     end
# end

# function BuildQuadtree(QT::SearchTree, vertices::Matrix{Float64}, j_cv::Matrix{Int}; lv=1, maxLvls=4, ElemPerCell=50)

#     QT.Childs = Vector{SearchTree}(undef,4)
    
#      for i in eachindex(QT.Childs) 
#         QT.Childs[i] =  spawnChild(QT,vertices, j_cv,i)

#         if (lv < maxLvls)&&(QT.Childs[i].numElem > ElemPerCell)
#             BuildQuadtree(QT.Childs[i], vertices, j_cv, lv=lv+1, maxLvls=maxLvls, ElemPerCell=ElemPerCell)
#         end
#     end
# end

# # Checks if two polygons intersect (or contain another)
# function doesIntersect(polygon1::Any,polygon2::Any)::Bool

#     # check if polygon1 is inside polygon2
#     for v in eachcol(polygon1)
#         if isinside(v,polygon2)
#             return true
#         end
#     end

#     # check if polygon2 is inside polygon1
#     for v in eachcol(polygon2)
#         if isinside(v,polygon1)
#             return true
#         end
#     end

#     edges1 = hcat(polygon1,polygon1[:,1])
#     edges2 = hcat(polygon2,polygon2[:,1])

#     # check if edges of polygon intersect quad edges
#     for q=1:size(polygon2,2)
#         for p=1:size(polygon1,2)
#             if LinesIntersect(edges2[:,q:q+1],edges1[:,p:p+1])
#                 return true
#             end
#         end
#     end

#     return false
# end  

# function LinesIntersect(L1, L2)::Bool
#     # Li given as Li = [P1_i P2_i] = [x1 x2;
#     #                                 y1 y2]
#     # Parmetrizing Line 1
#     # L1 : r*P2_1 + (1-r)*P1_1, r ∈ [0,1]

#     # Parmetrizing Line 2
#     # L2 : s*P2_2 + (1-s)*P1_2, s ∈ [0,1]

#     # Intersection if (*):
#     #   r*P2_1 + (1-r)*P1_1 = s*P2_2 + (1-s)*P1_2
#     # 
#     #   <=> (P2_1-P1_1)*r + (P1_2-P2_2)*s = (P1_2-P1_1)  && r,s ∈ [0,1]
#     #   (If r,s ∉ [0,1], the Lines intersect outside the segment)

#     A = [(L1[:,2]-L1[:,1]) (L2[:,1]-L2[:,2])]
    
#     # Checks if lines are parallel
#     if det(A) == 0
#         return false
#     else
#         rs = A\(L2[:,1]-L1[:,1])

#         # Checks if Line Intersection is on segment (described in (*))
#         if all((rs .>= 0)) && all((rs .<= 1))
#             return true
#         end
#     end

#     # Lines don't intersect in segment
#     return false
    
# end

end # module