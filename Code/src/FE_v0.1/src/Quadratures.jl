
module Quadratures

using QuadGK, LinearAlgebra

include("Polynomials.jl")
using .Polynomials

export Quadrature, InitQuadrature, Dunavant, InitQuadrature_Dict

# ToDo: Add dimensionally of Quadrature! This means need something like: pts_bary::Vector{Vector{2}}
# Use Static Arrays!
struct Quadrature
    npts::Int64
    w
    pts
    eT::Integer
end

function InitQuadrature_Dict(eTs, degs)
    # precomputes all Quadratures quad[[eT,deg]] for eT in eTs and deg in degs 

    quad =  Dict{Array{Integer},Quadrature}()
    # quad[[eT,deg]] is the quadrature of element type et, and degree deg

    for eT in eTs
        for deg in degs
            quad[[eT, deg]] = InitQuadrature(deg, eT)
        end
    end

    return quad
end

# type specifies the reference cell, n is the order
function InitQuadrature(deg::Int, eT::Integer)::Quadrature

    if eT==0 # Point
        return Quadrature(1,[1.0],[1.0],eT)

    elseif eT==1    # 2-node line
        pts, w = gauss(deg, -1, 1)
        npts = length(w)
    
        return Quadrature(npts,w,pts,eT)
    elseif eT==2    # 3-node Triangle
        # Using the quadrature according to 
        #   "Dunavant: High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle"

        w, pts =  Dunavant(deg)
        npts = length(w)

        return Quadrature(npts,w,pts,eT)
    elseif eT==3    # 4-Node quadrilateral
        
        pts_pre, w_pre = gauss(deg+1, 0, 1)
        npts = length(w_pre)

        pts = Array{Array{Float64}}(undef,npts^2)
        w = zeros(Float64,npts^2)

        for i=1:npts
            for j=1:npts
                ind = (i-1)*npts+j
                w[ind] = w_pre[j]*w_pre[i]
                pts[ind] = [pts_pre[i]; pts_pre[j]]
            end
        end

        return Quadrature(npts^2,w,pts,eT)
    
    elseif eT==4    # 5-node Pentagon This is inconsistent with Gmsh!!!!
        # Decomposes Pentagon into 5 triangles.
        # Define c1,c2,s1,s2 so that the polygon circumscribes the unit circle
        # 1. Quadrature on reference Simplex
        # 2. Transform points and weights to each triangle
        # 
        #                  (s2,c2+1)
        #                  /   |   \
        #               /      |      \ 
        #            /  (t4)   |  (t3)   \
        #         /            |            \
        # (s2-s1,c1+c2)-----(s2,c2)----(s1+s2,c1+c2)
        #         \           / \           / 
        #          \  (t5)  /     \  (t2)  /           
        #           \     /   (t1)  \     /
        #            \  /             \  /
        #          (0,0)-------------(2*s2,0)

        # Quadrature on reference Simplex
        w_pre, pts_pre =  Dunavant(deg)
        npts = length(w_pre)

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


        # A = [vertices[1,2]-vertices[1,1]  vertices[1,3]-vertices[1,1];
        #      vertices[2,2]-vertices[2,1]  vertices[2,3]-vertices[2,1]]
        dJ = t -> det([t[1,2]-t[1,1]  t[1,3]-t[1,1];
                       t[2,2]-t[2,1]  t[2,3]-t[2,1]])

        
        pts = Array{Array{Float64}}(undef,npts*5)
        w = zeros(Float64,npts*5)

        vol = sum(abs(dJ(Triangles[i])) for i=1:5)
        for i=1:5
            for j=1:npts
                ind = (i-1)*npts+j
                # w[ind] = w_pre[j]
                w[ind] = w_pre[j]*abs(dJ(Triangles[i]))/vol
                pts[ind] = Local2Global(pts_pre[j],Triangles[i],2)
            end
        end


        return Quadrature(npts*5,w,pts,eT)

    elseif eT==5    # 6-node Hexagon This is inconsistent with Gmsh!!!!
        # Decomposes Hexagon into 6 triangles.
        # Define c so that the edge length is one
        # 1. Quadrature on reference Simplex
        # 2. Transform points and weights to each triangle
        # 
        #          (0,2c)------------(1,2c)
        #            /  \             / \
        #           /     \   (t4)  /     \ 
        #          /  (t5)  \     /   (t3)  \
        #         /           \ /             \
        #    (-0.5,c)-------(0.5,c)--------(1.5,c)
        #         \           / \            / 
        #          \  (t6)  /     \  (t2)   /           
        #           \     /   (t1)  \      /
        #            \  /             \   /
        #          (0,0)--------------(1,0)

        # Quadrature on reference Simplex
        w_pre, pts_pre =  Dunavant(deg)
        npts = length(w_pre)

        c = sin(pi/3)

        # triangles
        t1 = [0.0 1.0 0.5;
              0.0 0.0 c]
        t2 = [1.0 1.5 0.5;
              0.0 c   c]
        t3 = [1.5 1.0 0.5;
              c   2c  c]
        t4 = [1.0 0.0 0.5;
              2c  2c  c]
        t5 = [0.0 -0.5 0.5;
              2c  c    c]
        t6 = [-0.5 0.0 0.5;
              c    0.0 c]
        Triangles = [[t1];[t2];[t3];[t4];[t5];[t6]]

        pts = Array{Array{Float64}}(undef,npts*6)
        w = zeros(Float64,npts*6)

        dJ = t -> det([t[1,2]-t[1,1]  t[1,3]-t[1,1];
                       t[2,2]-t[2,1]  t[2,3]-t[2,1]])

        vol = sum(abs(dJ(Triangles[i])) for i=1:6)
        for i=1:6
            for j=1:npts
                ind = (i-1)*npts+j
                # w[ind] = w_pre[j]
                w[ind] = w_pre[j]*abs(dJ(Triangles[i]))/vol
                pts[ind] = Local2Global(pts_pre[j],Triangles[i],2)
            end
        end

    

        return Quadrature(npts*6,w,pts,eT)

    elseif eT==6    # 7-node Heptagon This is inconsistent with Gmsh!!!!
        # Decomposes Heptagon into 7 triangles.
        # Points defined s.t. the radius of the circumscribing circle is 1
        # 1. Quadrature on reference Simplex
        # 2. Transform points and weights to each triangle
        # 
        #
        #                 __---(,)---__
        #            (,)´       |      `(,)
        #           /   \  (t5) | (t4) /   \ 
        #          /      \     |    /      \
        #         /  (t6)   \   |  /  (t3)   \
        #        /    ____----(bc)----____    \
        #      (,)----        /  \        ----(,) 
        #         \   (t7)  /      \  (t2)   /           
        #          \      /   (t1)   \      /
        #           \   /              \   /
        #           (0,0)---------------(,)

        # Quadrature on reference Simplex
        w_pre, pts_pre =  Dunavant(deg)
        npts = length(w_pre)


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

        bc = (1/7)*sum(hepta,dims=2)


        pts = Array{Array{Float64}}(undef,npts*7)
        w = zeros(Float64,npts*7)

        Triangles = []
        bc = (1/7)*sum(hepta,dims=2)
        for i=1:7
            push!(Triangles,hcat(hepta_edges[:,i:i+1],bc))
        end

        dJ = t -> det([t[1,2]-t[1,1]  t[1,3]-t[1,1];
                       t[2,2]-t[2,1]  t[2,3]-t[2,1]])

        vol = sum(abs(dJ(Triangles[i])) for i=1:7)
        for i=1:7
            for j=1:npts
                ind = (i-1)*npts+j
                # w[ind] = w_pre[j]
                w[ind] = w_pre[j]*abs(dJ(Triangles[i]))/vol
                pts[ind] = Local2Global(pts_pre[j],Triangles[i],2)
            end
        end

    

        return Quadrature(npts*7,w,pts,eT)

    end

end

# Computes the k-th Lagrange Basis function at point x, based on the interpolation points Pts
function Lagrange(x::Float64,Pts::Array,k::Int)
    npts = length(Pts)
    w = prod(Pts[k] - Pts[i]  for i in Iterators.filter(j->j!=k,1:npts))
    return prod( x - Pts[i] for i in Iterators.filter(j->j!=k,1:npts))/w
end


# function Warp(Np::Int)

#     # Definig the basic Warping function
#     pts_uni = [-1+ (2*i/Np) for i=0:Np]
#     pts_LGL, ~ = gauss(Np, -1, 1)
#     Lag1D == Array{Function}(undef,Np)
#     for k=1:deg+1
#         Lag1D[k] = x->Lagrange(x,pts_LGL,k)
#     end
#     w = r -> (1/(1-r^2))*sum((pts_LGL[i]-pts_uni[i])*Lag1D[i](r) for i=1:Np)

#     # Defining the barycentric coordinates 
    


# end



# A workflow for reading the txt
# f = open("Dunavant.txt","r")
# s = readline(f)
# a=split(s)
# parse(Int,a[1])
# parse(Float64,a[2])

function Dunavant(order::Int)

    f = open("src/Dunavant.txt","r")
    filedata = readlines(f)
    close(f)

    ndata = length(filedata) # number of lines in the data file

    # Index Matrix for the relevant lines
    # The information for a quadrature of order m starts at line StartStopInd[1,m], 
    # and stops including StartStopInd[2,m]
    StartStopInd = zeros(Int,2,12)

    LineData = Array{Array{String}}(undef,ndata-1)
    ind = 1
    for i=2:ndata
        LineData[i-1] = split(filedata[i])

        # get the start and stop index
        if (parse(Int,LineData[i-1][1])==ind)&&(StartStopInd[1,ind]==0)
            if ind==1
                StartStopInd[1,ind] = i-1
                ind = ind+1
            elseif ind==12
                StartStopInd[1,ind] = i-1
                StartStopInd[2,ind-1] = i-2
                StartStopInd[2,ind] = ndata
            else
                StartStopInd[1,ind] = i-1
                StartStopInd[2,ind-1] = i-2
                ind = ind+1
            end
        end
    end

    # Build the actual quadrature
    w = zeros(Float64,1)
    pts_bary = zeros(Float64,3,1)
    pts_loc = Array{Vector{Float64}}(undef,1)

    for i=StartStopInd[1,order]:StartStopInd[2,order]

        w_tmp = parse(Float64,LineData[i][2])

        a = parse(Float64,LineData[i][3])
        b = parse(Float64,LineData[i][4])
        c = parse(Float64,LineData[i][5])

        pts_tmp = [a a b b c c;
                   b c a c a b;
                   c b c a b a]

        for k=1:6
            if !any(pts_bary[:,j] == pts_tmp[:,k] for j=1:size(pts_bary,2))
                if pts_bary == zeros(Float64,3,1)
                    pts_bary[:,1] = pts_tmp[:,k]
                    pts_loc = [Barycentric2Local(pts_bary[:,1])]
                    w=w_tmp
                else
                    pts_bary =  hcat(pts_bary,pts_tmp[:,k])
                    push!(pts_loc,Barycentric2Local(pts_tmp[:,k]))
                    w = hcat(w,w_tmp)
                end
            end
        end

    end


    return w, pts_loc #bary

end

end #module

