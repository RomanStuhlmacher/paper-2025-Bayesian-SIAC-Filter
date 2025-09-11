module Polynomials

using SpecialFunctions, LinearAlgebra  # needed for Gamma-faunction

export Barycentric2Local, Phys2Barycentric, Global2Local, Local2Global, Face2Local, Jacobi, DJacobi, Legendre2D, DLegendre2D, getBasis, getGradBasis

###############################################################################################################
#                          Stuff
###############################################################################################################

function indexPairs(deg::Int)
    xy_ind = zeros(Int,2)
    for ix=1:deg+1
        for iy=1:deg+1
            if (ix+iy)<=(deg+2)
                if xy_ind == zeros(Int,2)
                    xy_ind = [ix;iy]
                else
                    xy_ind =  hcat(xy_ind,[ix;iy])
                end
            end
        end
    end
    return xy_ind
end


###############################################################################################################
#                                       Local Coordinate Mappings
###############################################################################################################

# Global2LocalMapping
# ToDo: How to are the mesh quantities stored in gmsh?
function Global2Local(x, vertices, eT::Integer)
    if eT==1 # 2-node-line
        return (2/(vertices[1,2]-vertices[1,1]))*(x[1]-vertices[1,1])-1
    elseif eT==2    #3-node Triangle
        ref_vertices = [0.0  1.0  0.0;
                        0.0  0.0  1.0]

        # Get the barycentric coordinates, from x, and physical vertices
        lambda = Phys2Barycentric(x, vertices; type="Simplex2")         
        
        # local coordinates
        return ref_vertices*lambda
    elseif eT==3 # 4-node Quadrangle
        maxiter = 20
        tol = 10e-14

        # Unpack vertices
        v1, v2, v3, v4 = eachcol(vertices)

        # Initial guess
        chi = 0.5
        eta = 0.5

        for iter = 1:maxiter
            # Bilinear interpolation
            x_hat = (1 - chi)*(1 - eta)*v1[1:2] + chi*(1 - eta)*v2[1:2] + (1 - chi)*eta*v3[1:2] + chi*eta*v4[1:2]

            # Residual
            r = x_hat[1:2] - x[1:2]

            # Compute derivatives
            dx_dchi = -(1 - eta)*v1[1:2] + (1 - eta)*v2[1:2] - eta*v3[1:2] + eta*v4[1:2]
            dx_deta = -(1 - chi)*v1[1:2] - chi*v2[1:2] + (1 - chi)*v3[1:2] + chi*v4[1:2]

            J = hcat(dx_dchi, dx_deta)  # 2x2 Jacobian

            # Solve for update
            Δ = J \ r

            # Update chi and eta
            chi -= Δ[1]
            eta -= Δ[2]

            if norm(Δ) < tol
                return [chi, eta]
            end
        end

        @warn "Newton iteration failed to converge"
        return [NaN, NaN]

    elseif eT==4

        barycenter = (1/5)*sum(vertices,dims=2)

        v_edge = hcat(vertices,vertices[:,1])
        for i=1:5
            if isinside(x,hcat(barycenter,v_edge[:,i,i+1]))
                return Global2Local(x, hcat(barycenter,v_edge[:,i,i+1]), 2)
            end
        end
   end

end

function Local2Global(z, vertices, eT::Integer)
    if eT==1 # 2-node-line
        return ((vertices[1,2]-vertices[1,1])/2)*(z[1]+1)+vertices[1,1]
    elseif eT==2    #3-node Triangle
        ref_vertices = [0.0  1.0  0.0;
                        0.0  0.0  1.0]

        # Get the barycentric coordinates, from x, and physical vertices
        lambda = Phys2Barycentric(z, ref_vertices; type="Simplex2")         
        
        # local coordinates
        return vertices[1:2,:]*lambda
    elseif eT==3    # 4-node Qudrangle
        ref_vertices = [0.0  1.0  0.0 1.0;
                        0.0  0.0  1.0 1.0]
        return (1-z[1])*(1-z[2])*vertices[:,1] + z[1]*(1-z[2])*vertices[:,2] + (1-z[1])*z[2]*vertices[:,3] + z[1]*z[2]*vertices[:,4]
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
        for i=1:5
            if isinside(z,Triangles[i])
                zt = Global2Local(z,Triangles[i],2)
                return Local2Global(zt,hcat(v_edge[:,i:i+1],barycenter), 2)
            end
        end
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
        for i=1:6
            if isinside(z,Triangles[i])
                zt = Global2Local(z,Triangles[i],2)
                return Local2Global(zt,hcat(v_edge[:,i:i+1],barycenter), 2)
            end
        end
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

        ref_barycenter =(1/7)*sum(hepta,dims=2)
        Triangles = []
        for i=1:7
            push!(Triangles,hcat(hepta_edges[:,i:i+1],ref_barycenter))
        end

        # The transformed Barycenter, given the vertices
        barycenter = (1/7)*sum(vertices,dims=2)
        v_edge = hcat(vertices,vertices[:,1])

        for i=1:7
            if isinside(z,Triangles[i])
                zt = Global2Local(z,Triangles[i],2)
                return Local2Global(zt,hcat(v_edge[:,i:i+1],barycenter), 2)
            end
        end
    end

end

# obtains face local coordinates, and returns the respective element local coordinates
# fl    local face index
# y     local face coordinate (i.e. y ∈ [0,1] for line faces)
function Face2Local(y, fl::Integer, eT::Integer)
    if eT==1 # 2-node line
        # reference line:
        #  (v1=e1=f1)-----(v2=e2=f2)

        # -> y not important, point face
        if fl==1
            return -1.0
        elseif fl==2
            return 1.0
        end
    elseif eT==2  # 3-node Triangle 
        #   expects y∈[0,1]
        # reference triangle:
        #       (v3)
        #        ^ ^
        #        |  \
        # (e2=f2)|    \ (e1=f1)
        #        |      \
        #       (v1)---->(v2)
        #          (e3=f3)

        if fl==1
            return [1.0; 0.0].*(1.0 .- y) +  [0.0; 1.0].*y
        elseif fl==2
            return [0.0; y]
        elseif fl==3
            return [y; 0.0]
        end

    elseif eT==3 # 4-node Quadrilateral 
        #   expects y∈[0,1]
        # reference quad:
        #      (v3)(e3=f3)(v4)
        #        ^ ------>^
        #        |        |
        # (e2=f2)|        |(e4=f4)
        #        |        |
        #       (v1)---->(v2)
        #          (e1=f1)

        if fl==1
            return [y; 0.0]
        elseif fl==2
            return [0.0; y]
        elseif fl==3
            return [y; 1.0]
        elseif fl==4
            return [1.0; y]
        end
    end

end

function isinside(x,vertices)::Bool
    vert_closed = hcat(vertices,vertices[:,1])
    nvert = size(vertices,2)

    inside = false

    p1 = vertices[1:2,1]
    cc = 0
    for i=1:nvert

        p2 = vert_closed[1:2,i+1]
        if x[2] > minimum([p1[2],p2[2]])
            if x[2] <= maximum([p1[2],p2[2]])
                if x[1] < maximum([p1[1],p2[1]])
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


###############################################################################################################
#                                       Mappings for Triangles
###############################################################################################################

function Barycentric2Local(lambda; type="Simplex2")
    # reference vertices (have vertices: [-1; -1], [-1;1], [1;-1])
        #
        #   v3=(-1,1)
        #       |   \
        #       |     \
        #       |       \
        # v1=(-1,-1)----v2=(1,-1)
    ref_vertices = [0.0  1.0  0.0;
                    0.0  0.0  1.0]
    # local coordinates
    return ref_vertices*lambda
end

function Phys2Barycentric(x, vertices; type="Simplex2")
    # if type == "Simplex2"
        # Get the barycentric coordinates, from x, and physical vertices
        # A = [vertices[1,1] vertices[1,2] vertices[1,3]; 
        #      vertices[2,1] vertices[2,2] vertices[2,3]; 
        #      1              1               1   ]
        # A = [vertices[1:2,:]; ones(1,3)]
        # display(A)
        if length(x)==2
            x = vcat(x,1.0)
        end
        # lambda = A\x            
    # end

    # return lambda
    return [vertices[1:2,:]; ones(1,3)]\x
end



###############################################################################################################
#                                       Jacobi polynomials in 1d
###############################################################################################################

# Jacobi polynomial on (-1,1)
function Jacobi(x::Float64,k::Int64,a::Number,b::Number)
    n = k-1 # to start internal indexing at 0

    if n<0
        return 0.0
    elseif n==0
        return one(x)
    elseif n == 1
        return (a-b+(a+b+2)*x)/2
    end

    p0 = one(x)
    p1 = (a-b+(a+b+2)*x)/2

    for i=2:n
        m = i-1
        a1 = 2*(m+1)*(m+a+b+1)*(2*m+a+b)
        a2 = (2*m+a+b+1)*(a^2-b^2)
        a3 = (2*m+a+b)*(2*m+a+b+1)*(2*m+a+b+2)
        a4 = 2*(m+a)*(m+b)*(2*m+a+b+2)

        p2 = ((a2+a3*x)*p1- a4*p0)/a1
        p0 = p1
        p1 = p2
    end

    return p1
end

function DJacobi(x::Float64,k::Int64,a::Number,b::Number, D::Int64)
   n = k-1

    return (gamma(a+b+n+1+D)/(2^D * gamma(a+b+n+1)))*Jacobi(x,k-D,a+D,b+D)
end



###############################################################################################################
#                                       2D Basis functions
###############################################################################################################

# TODO: Implement for other elements
function Legendre2D(x, vertices, kx::Int64, ky::Int64)

    z = Global2Local(x, vertices, 2)#type="Simplex2")

    a = 2*((z[1])/(1-z[2]))-1
    b = 2*z[2]-1
    return sqrt(2) * Jacobi(a,kx,0.0,0.0) * Jacobi(b,ky,2*kx-1,0.0) * (1 - z[2])^(kx-1)

end

# TODO: Works only for (Dx,Dy) ∈ {(1,0),(0,1)}, Implement higher order derivatives!
# TODO: Implement for other element types
function DLegendre2D(x, vertices, kx::Int64, ky::Int64, Dx::Int64, Dy::Int64)

    z = Global2Local(x, vertices, 2)

    a = 2*((z[1])/(1-z[2]))-1
    b = 2*z[2]-1

    if (Dx==1)&(Dy==0)  # Partial Derivative w.r.t. x
        return (2*sqrt(2))/(1-z[2]) * DJacobi(a,kx,0.0,0.0,Dx) * Jacobi(b,ky,2*kx-1,0.0) * (1 - z[2])^(kx-1)

    elseif (Dx==0)&&(Dy==1) # Partial Derivative w.r.t. y
        S1 = 2*(z[1]/(1-z[2])^2)*DJacobi(a,kx,0.0,0.0,Dy) * Jacobi(b,ky,2*kx-1,0.0) * (1 - z[2])^(kx-1)
        S2 = 2*Jacobi(a,kx,0.0,0.0) * DJacobi(b,ky,2*kx-1,0.0,Dy) * (1 - z[2])^(kx-1)
        S3 = 0.0

        if kx>1
            S3 = Jacobi(a,kx,0.0,0.0) * Jacobi(b,ky,2*kx-1,0.0) * (-1) * (kx-1)*(1 - z[2])^(kx-2)
        end

        return sqrt(2)*(S1+S2+S3)
    end

end


###############################################################################################################
#                                       Basis generation
###############################################################################################################

function getBasis(deg::Int64, eT::Integer)

    if eT==1  # 2-node line
        Basis = Array{Function}(undef,deg+1)
        for k=1:deg+1
            Basis[k] = x->Jacobi(x,k,0.0,0.0)
        end
        return Basis

    elseif eT==2   # 3-node triangle 
        ref_vertices = [0.0 1.0  0.0;
                        0.0 0.0  1.0]

        Np = sum(i for i=1:deg+1) # This sums all the terms in Pascals Triangle
        Basis = Array{Function}(undef,Np)
        # D1Basis = Array{Function}(undef,Np)

        ind = indexPairs(deg)
        for i=1:Np
            Basis[i] = x->Legendre2D(x, ref_vertices, ind[1,i], ind[2,i])
            # D1Basis[k] = x->DLegendre(x,k)
        end
        return Basis

    elseif eT==3  # 4-node quadrilateral
        ref_vertices = [0.0 1.0  0.0 1.0;
                        0.0 0.0  1.0 1.0]

        Np = sum(i for i=1:deg+1) # This sums all the terms in Pascals Triangle
        Basis = Array{Function}(undef,Np)
        # D1Basis = Array{Function}(undef,Np)

        ind = indexPairs(deg)
        for i=1:Np
            Basis[i] = x->Legendre2D(x, ref_vertices, ind[1,i], ind[2,i])
            # D1Basis[k] = x->DLegendre(x,k)
        end
        return Basis

    end
end

# Gradient of Basis
function getGradBasis(deg::Int64, eT::Integer)
    if eT==1  # 2-node line
        Grad_Basis = Array{Function}(undef,deg+1)
        for k=1:deg+1
            Grad_Basis[k] = x->DJacobi(x,k,0.0,0.0,1)
        end
        return Grad_Basis

    elseif eT==2  # 3-node triangle
        ref_vertices = [0.0  1.0  0.0;
                        0.0  0.0  1.0]

        Np = sum(i for i=1:deg+1) # This sums all the terms in Pascals Triangle
        Grad_Basis = Array{Function}(undef,Np)
        # D1Basis = Array{Function}(undef,Np)

        ind = indexPairs(deg)
        for i=1:Np
            Grad_Basis[i] = x->[DLegendre2D(x, ref_vertices, ind[1,i], ind[2,i],1,0); 
                                DLegendre2D(x, ref_vertices, ind[1,i], ind[2,i],0,1)]
        end
        return Grad_Basis

    elseif eT==3  # 4-node quadrilateral
        ref_vertices = [0.0  1.0  0.0 1.0;
                        0.0  0.0  1.0 1.0]

        Np = sum(i for i=1:deg+1) # This sums all the terms in Pascals Triangle
        Grad_Basis = Array{Function}(undef,Np)
        # D1Basis = Array{Function}(undef,Np)

        ind = indexPairs(deg)
        for i=1:Np
            Grad_Basis[i] = x->[DLegendre2D(x, ref_vertices, ind[1,i], ind[2,i],1,0); 
                                DLegendre2D(x, ref_vertices, ind[1,i], ind[2,i],0,1)]
        end
        return Grad_Basis

    end

end


end # module