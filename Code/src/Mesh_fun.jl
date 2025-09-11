
# The Mesh Structure
mutable struct Mesh
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
function getElem(x::Float64, mesh::Mesh; type="standard")

    # x     Evaluation Point
    # mesh  The Mesh

    E = 0

    if type == "periodic"
        if x < mesh.xL
            x = mod(x,mesh.xR-mesh.xL)
        elseif  x > mesh.xR
            x = mod(x,mesh.xR)+mesh.xL
        end
    end

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
function RefineGlobal(mesh::Mesh; k::Int=2)

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
