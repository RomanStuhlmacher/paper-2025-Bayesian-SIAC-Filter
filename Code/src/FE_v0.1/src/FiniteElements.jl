# TODO: * Restructure Projections and FiniteElements: Projections only contains local Projections, FiniteElements everything for the FE struct
#       * Meshes: merge 1d and 2d mesh definitions
#       * Meshes: Enable more Elements types
#       * Meshes: Enable 3d mesh

module FiniteElements

import Gmsh, QuadGK
using LinearAlgebra, SpecialFunctions, Combinatorics, Base.Threads
using ProgressMeter

export check, readGmsh, L2, evalFE, getElem, getElem2, FE, Element, Quadrature, Barycentric2Local, getDoF, setDoF!, getElemDoF_ind, get_nDoF, Face2Local, getElementType_Info, Local2Global, getOutwardNormal, getJacobian, Interface2Local, getNormal
export Dunavant, InitQuadrature, Mesh, InitQuadrature_Dict, Global2Local, getOrthonormalBasis, getVandermodeDerivative, getDeterminant, SearchTree, get_localElementList, getReferenceElem, build_cube_mesh, Interpolate

###############################################################################################################
#                                           Including modules
###############################################################################################################

include("Polynomials.jl")
using .Polynomials

include("Meshes.jl")
using .Meshes

include("Quadratures.jl")
using .Quadratures 

###############################################################################################################
#                                               FE related
###############################################################################################################

# The Element Structure
mutable struct Element # Approximation Quantities
    eT::Integer                # DomainType (1: 2-node line, 2: 3-node triangle)
    deg::Int                    # degree
    nDoF::Int                  # Number of DoF
    DoF::Vector{Float64}       # The Degrees of Freedom

    Basis::Vector{Function}     # The Basis Functions
    GradBasis::Vector{Function} # Gradients of the Basis Functions (i.e. GradBasis contains all first derivatives)
end

mutable struct FE
    N::Int                  # Number of Elements
    eTs::Array{Integer}     # Array of existing (geometric) Element Types
    fTs::Array{Integer}     # Array of existing (geometric) Face Types
    degs::Array{Integer}    # Array of existing degrees

    quad::Dict{Integer,Quadrature}      # Quadrature formulas for the elements
    fquad::Dict{Integer, Quadrature}    # Quadrature formulas for the faces

    Bases::Dict{Array{Integer},Array{Function}}         # Bases[[eT,deg]] returns all occuring Bases in the mesh
    GradBases::Dict{Array{Integer},Array{Function}}     # GradBases[[eT,deg]] returns all occuring Gradient Bases in the mesh

    Bases_Eval::Dict{Array{Integer},Matrix{Float64}}        # Bases_Eval[[eT,deg]] returns the evaluated Bases at the corresponding quadrature points
    # The rows iterate the Basis functions, the colums the quadrature points
    GradBases_Eval::Dict{Array{Integer},Matrix{Union{Vector{Float64},Float64}}}    # GradBases_Eval[[eT,deg]] returns the evaluated Gradient Bases at the corresponding quadrature points
    # The rows iterate the Basis functions, the colums the quadrature points, the slices the derivative directions

#     Bases_FaceEval::Dict{Array{Integer},Matrix{Float64}}        # Bases_Eval[[eT,deg,fT]] returns the evaluated Bases at the corresponding face quadrature points
#     # The rows iterate the Basis functions, the colums the face quadrature points
#     GradBases_FaceEval::Dict{Array{Integer},Matrix{Vector{Float64}}}    # GradBases_Eval[[eT,deg,fT]] returns the evaluated Gradient Bases at the corresponding quadrature points
#    # The rows iterate the Basis functions, the colums the face quadrature points

    Elem::Array{Element}    # The Elements
    mesh::Mesh
end


# L2 projection of the function fun (as Function handle) in the intervall xL to xR, given a Quadrature and an Expansion Basis
function L2_Coeff(vertices, eT::Integer, deg::Int64, fun::Function, Basis::Array{Function}, quad::Quadrature)
    # display(quad.w)


    ~, JmT, detJ = getJacobian(vertices,eT)
    # ~, ~, ~, ~, ~, ~, Jacobian, ~ = getElementType_Info(eT)
    Np = get_nDoF(deg, eT; space="Pd")

    M = zeros(Float64,Np,Np)
    b = zeros(Float64,Np)

    basis_eval = Bases_Eval[[eT,deg]]  # size (Np, npts)
    detJ_vals = [detJ(ξ) for ξ in quad.pts]
    fun_eval = [fun(Local2Global(quad.pts[k], vertices, eT)) for k=1:quad.npts]

    for i = 1:Np
        @inbounds @simd for k = 1:quad.npts
            b[i] += quad.w[k] * fun_eval[k] * B_eval[i,k] * detJ_vals[k]
        end
        @inbounds @simd for j = 1:Np
            for k = 1:quad.npts
                M[i,j] += quad.w[k] * basis_eval[i,k] * basis_eval[j,k] * detJ_vals[k]
            end
        end
    end

    # for i=1:Np
    #     # fun_eval = z -> fun(Local2Global(z, vertices,eT)) .* Basis[i](z) .* detJ(z)#Jacobian(z)

    #     # b[i] = sum(quad.w[k]*fun_eval(quad.pts[k]) for k in 1:quad.npts)
    #     for k in 1:quad.npts
    #         b[i] += quad.w[k] * fun(Local2Global(quad.pts[k], vertices,eT)) * Basis[i](quad.pts[k]) * detJ(quad.pts[k])
    #     end

    #     for j=1:Np
    #         # fun_test = z ->  Basis[i](z) .* Basis[j](z) .* detJ(z) #Jacobian(z)
    #         for k in 1:quad.npts
    #             M[i,j] += quad.w[k]*Basis[i](quad.pts[k])*Basis[j](quad.pts[k])*detJ(quad.pts[k])
    #         end
    #         # M[i,j] = sum(quad.w[k]*fun_test(quad.pts[k]) for k in 1:quad.npts)
    #     end
    # end
    
    DoF = M\b

    return DoF
end

function L2_Coeff(vertices, eT::Integer, deg::Int64, fun::Function, quad::Quadrature, 
                    basis_eval::Matrix{Float64}, detJ_vals::Vector{Float64}, fun_eval::Vector{Float64}, 
                    M::Matrix{Float64}, b::Vector{Float64})

    _, _, detJ = getJacobian(vertices, eT)
    Np = get_nDoF(deg, eT; space="Pd")
    fill!(b, 0.0)
    fill!(M, 0.0)

    @inbounds for k = 1:quad.npts
        detJ_vals[k] = detJ(quad.pts[k])
        fun_eval[k] = fun(Local2Global(quad.pts[k], vertices, eT))
    end

    for i = 1:Np
        @inbounds @simd for k = 1:quad.npts
            b[i] += quad.w[k] * fun_eval[k] *basis_eval[i,k] * detJ_vals[k]
        end
        for j = 1:Np
            @inbounds @simd for k = 1:quad.npts
                M[i,j] += quad.w[k] * basis_eval[i,k] * basis_eval[j,k] * detJ_vals[k]
            end
        end
    end

    return M[1:Np, 1:Np] \ b[1:Np]
end

function Interpolate(fun::Function, mesh::Mesh, Pts::Array{Float64}, deg::Int, d::Int)

    # Computes the k-th Lagrange Basis function at point x, based on the interpolation points Pts
    function Lagrange(x::Float64,Pts::Array,k::Int)::Float64
        npts = length(Pts)
        w = prod(Pts[k] - Pts[i]  for i in Iterators.filter(j->j!=k,1:npts))
        return prod( x - Pts[i] for i in Iterators.filter(j->j!=k,1:npts))/w
    end

    function DLagrange(x::Float64, Pts::Array, k::Int)
        npts = length(Pts)
        # denominator (normalization factor)
        w = prod(Pts[k] - Pts[i] for i in 1:npts if i != k)

        # If x coincides with one of the interpolation points
        for m in 1:npts
            if isapprox(x, Pts[m]; atol=1e-12, rtol=1e-12)
                if m == k
                    # derivative of L_k at its own node
                    return sum(1/(Pts[k] - Pts[j]) for j in 1:npts if j != k)
                else
                    # derivative of L_k at another node
                    num = prod(Pts[m] - Pts[j] for j in 1:npts if j != k && j != m; init=1.0)
                    return num / w
                end
            end
        end

        # General case (x not equal to any node)
        lk = prod(x - Pts[i] for i in 1:npts if i != k) / w
        d  = sum(1/(x - Pts[i]) for i in 1:npts if i != k)
        return lk * d
    end

    eTs = unique(mesh.element_types)
    Bases = Dict{Vector{Integer},Array{Function}}()
    GradBases = Dict{Vector{Integer},Array{Function}}()

    Bases_Eval = Dict{Vector{Integer},Matrix{Float64}}()
    GradBases_Eval = Dict{Vector{Integer}, Matrix{Union{Vector{Float64},Float64}}}()
    quad =  Dict{Integer,Quadrature}()

    fTs = Array{Integer}(undef,0)
    # Bases_FaceEval = Dict{Vector{Integer},Matrix{Float64}}()
    # GradBases_FaceEval = Dict{Vector{Integer},Matrix{Vector{Float64}}}()
    fquad =  Dict{Integer,Quadrature}()
    for eT in eTs
        quad[eT] = InitQuadrature(2*deg, eT)
        Bases[[eT,deg]], ~ = getOrthonormalBasis(d, deg, quad[eT])
        GradBases[[eT,deg]] = getGradBasis(deg,eT)


        Bases[[eT,deg]] = [ x-> Lagrange(x,Pts,k) for k=1:deg+1]
        GradBases[[eT,deg]] =  [ x-> [DLagrange(x,Pts,k)] for k=1:deg+1]


        Bases_Eval[[eT,deg]] = [ Bases[[eT,deg]][i](quad[eT].pts[j]) for i in eachindex(Bases[[eT,deg]]), j in eachindex(quad[eT].pts)]
        GradBases_Eval[[eT,deg]] = [GradBases[[eT,deg]][i](quad[eT].pts[j]) for i in eachindex(GradBases[[eT,deg]]), j in eachindex(quad[eT].pts)]

        ~, ~, ~, ~, ~, fTs_loc, ~, ~ = getElementType_Info(eT)

        for fT in fTs_loc
            if !haskey(fquad,fT)
                fquad[fT] = InitQuadrature(2*deg, fT)

                # Bases_FaceEval[[eT,deg,fT]] = [ Bases[[eT,deg]][i](fquad[fT].pts[j]) for i in eachindex(Bases[[eT,deg]]), j in eachindex(fquad[fT].pts)]
                # GradBases_FaceEval[[eT,deg,fT]] = [GradBases[[eT,deg]][i](fquad[fT].pts[j]) for i in eachindex(GradBases[[eT,deg]]), j in eachindex(fquad[fT].pts)]
            end
        end

    end

    Elem = Array{Element}(undef,mesh.NC[])


    for j=1:mesh.NC[]
        eT = mesh.element_types[j]

        # Vandermonde = [ Bases[[eT,deg]][c](quad[eT].pts[r]) for c=1:deg+1, r=1:deg+1]
        DoF =[fun(Local2Global(Pts[k],[mesh.vertices[mesh.j_cv[j,1]] mesh.vertices[mesh.j_cv[j,2]]], 1))  for k=1:deg+1]
        # DoF = inv(Vandermonde)*DoF_pre

        Elem[j] = Element(eT,deg,length(DoF),
                          DoF,Bases[[eT,deg]],GradBases[[eT,deg]])

    end

    uh = FE(mesh.NC[],eTs,fTs,[deg],
            quad,fquad,
            Bases,GradBases,
            Bases_Eval,
            GradBases_Eval,
            Elem,mesh)

    return uh

end

    
# Generate a PPA of a function,using the Legendre Base
function L2(fun::Function, mesh::Mesh, deg::Int64, d::Int64)


    eTs = unique(mesh.element_types)
    Bases = Dict{Vector{Integer},Array{Function}}()
    GradBases = Dict{Vector{Integer},Array{Function}}()

    Bases_Eval = Dict{Vector{Integer},Matrix{Float64}}()
    GradBases_Eval = Dict{Vector{Integer},Matrix{Vector{Float64}}}()
    quad =  Dict{Integer,Quadrature}()

    fTs = Array{Integer}(undef,0)
    # Bases_FaceEval = Dict{Vector{Integer},Matrix{Float64}}()
    # GradBases_FaceEval = Dict{Vector{Integer},Matrix{Vector{Float64}}}()
    fquad =  Dict{Integer,Quadrature}()
    for eT in eTs
        quad[eT] = InitQuadrature(2*deg, eT)
        # Bases[[eT,deg]], ~ = getOrthonormalBasis(d, deg, quad[eT])
        # GradBases[[eT,deg]] = getGradBasis(deg,eT)


        Bases[[eT,deg]], Vandermonde, inds, monomial_scaling = getOrthonormalBasis(d, deg, quad[eT])#; space="Pd")
        GradBases[[eT,deg]] = Vandermonde2GradBasis(d,Vandermonde, inds, monomial_scaling)


        Bases_Eval[[eT,deg]] = [ Bases[[eT,deg]][i](quad[eT].pts[j]) for i in eachindex(Bases[[eT,deg]]), j in eachindex(quad[eT].pts)]
        GradBases_Eval[[eT,deg]] = [GradBases[[eT,deg]][i](quad[eT].pts[j]) for i in eachindex(GradBases[[eT,deg]]), j in eachindex(quad[eT].pts)]

        ~, ~, ~, ~, ~, fTs_loc, ~, ~ = getElementType_Info(eT)

        for fT in fTs_loc
            if !haskey(fquad,fT)
                fquad[fT] = InitQuadrature(2*deg, fT)

                # Bases_FaceEval[[eT,deg,fT]] = [ Bases[[eT,deg]][i](fquad[fT].pts[j]) for i in eachindex(Bases[[eT,deg]]), j in eachindex(fquad[fT].pts)]
                # GradBases_FaceEval[[eT,deg,fT]] = [GradBases[[eT,deg]][i](fquad[fT].pts[j]) for i in eachindex(GradBases[[eT,deg]]), j in eachindex(fquad[fT].pts)]
            end
        end

    end

    Elem = Array{Element}(undef,mesh.NC[])


    # Prepare Threading
    nthreads = Threads.nthreads()
    Np_max = maximum([get_nDoF(deg, et; space="Pd") for et in eTs])
    max_npts = maximum([q.npts for q in values(quad)])

    Mbufs = [zeros(Np_max, Np_max) for _ in 1:nthreads]
    bbufs = [zeros(Np_max) for _ in 1:nthreads]
    detJ_bufs = [zeros(max_npts) for _ in 1:nthreads]
    fun_bufs = [zeros(max_npts) for _ in 1:nthreads]
    

    projection_time = Progress(mesh.NC[]; desc="Projecting Data...")
    @threads for j=1:mesh.NC[]
    # for j=1:mesh.NC[]
        # # local_vert = mesh.vertices[:,mesh.j_cv[j,:]]
        # local_vert = view(mesh.vertices,:,mesh.j_cv[j,:])


        tid = Threads.threadid()
        local_vert = view(mesh.vertices,:,mesh.j_cv[j,:])
        eT = mesh.element_types[j]

        B_eval = Bases_Eval[[eT, deg]]
        DoF = L2_Coeff(local_vert, eT, deg, fun, quad[eT],
                        B_eval, detJ_bufs[tid], 
                        fun_bufs[tid], Mbufs[tid], bbufs[tid])


        # eT = mesh.element_types[j]
        # DoF =  L2_Coeff(local_vert,eT, deg,fun,Bases[[eT,deg]],quad[eT])
        Elem[j] = Element(eT,deg,length(DoF),
                          DoF,Bases[[eT,deg]],GradBases[[eT,deg]])

        next!(projection_time)
    end
    finish!(projection_time)

    uh = FE(mesh.NC[],eTs,fTs,[deg],
            quad,fquad,
            Bases,GradBases,
            Bases_Eval,
            GradBases_Eval,
            Elem,mesh)

    return uh

end


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

# Generate a PPA of a function,using the Legendre Base
# function L2(fun::Function, mesh::Mesh, deg::Int64)

#     eTs = unique(mesh.element_types)
#     Bases = Dict{Vector{Integer},Array{Function}}()
#     GradBases = Dict{Vector{Integer},Array{Function}}()
#     quad =  Dict{Integer,Quadrature}()

#     fTs = Array{Integer}(undef,0)
#     fquad =  Dict{Integer,Quadrature}()
#     for eT in eTs
#         Bases[[eT,deg]] = getBasis(deg, eT)
#         GradBases[[eT,deg]] = getGradBasis(deg,eT)
#         quad[eT] = InitQuadrature(2*deg, eT)

#         ~, ~, ~, ~, ~, fTs_loc, ~, ~ = getElementType_Info(eT)

#         for fT in fTs_loc
#             if !haskey(fquad,fT)
#                 fquad[fT] = InitQuadrature(2*deg, fT)
#             end
#         end

#     end

#     Elem = Array{Element}(undef,mesh.NC[])
#     for j=1:mesh.NC[]
#         local_vert = mesh.vertices[:,mesh.j_cv[j,:]]

#         eT = mesh.element_types[j]
#         DoF =  L2_Coeff(local_vert,eT, deg,fun,Bases[[eT,deg]],quad[eT])
#         Elem[j] = Element(eT,deg,length(DoF),
#                           DoF,Bases[[eT,deg]],GradBases[[eT,deg]])

#     end

#     uh = FE(mesh.NC[],eTs,fTs,[deg],
#             quad,fquad,
#             Bases,GradBases,
#             Elem,mesh)

#     return uh

# end

# Get the DoF of an FE Handle
function getDoF(uh::FE)

    N = Int(0)
    for i=1:uh.N
        N += uh.Elem[i].nDoF
    end
    # Assuming same deg polynomials on every element
    globalDoFs = zeros(Float64,N)
    ind = Int(0)
    for e = 1:uh.N
        for i = 1:uh.Elem[e].nDoF
            globalDoFs[ind+i] = uh.Elem[e].DoF[i]
        end
        ind += uh.Elem[e].nDoF
    end

    return globalDoFs
end


# Get the DoF of an FE Handle
function setDoF!(uh::FE, globalDoFs::Array{Float64})

    ind = Int(0)
    for e = 1:uh.N
        for i = 1:uh.Elem[e].nDoF
            uh.Elem[e].DoF[i] = globalDoFs[ind+i] 
        end
        ind += uh.Elem[e].nDoF
    end

end

# returns an arry of indices of the DoFs of an element in the globalDoFs
function getElemDoF_ind(uh::FE, elem_ind::Int64)

    cc = Int(0)

    for i=1:elem_ind-1
        cc += uh.Elem[i].nDoF
    end

    return collect(cc+1:cc+uh.Elem[elem_ind].nDoF)

end

function get_nDoF(deg::Integer, eT::Integer; space="Pd")

    # Pd -> the Taylor Polynomial Space 
    # Qd -> the Tensor product space

    if eT==1    # 2-npde line, space Pd
        return deg+1
    elseif (eT==2 || eT==3)&&(space=="Pd")   # 3-node triangle, space Pd
        return sum(i for i=1:deg+1)
    end
end

# Evaluate a FE Handle
function evalFE(x,uh::FE)::Float64
    # j = getElem2(x,uh.mesh)
    j = getElem2(x,uh.mesh, uh.mesh.ST)
    # display(j)
    # display(x)
    if isnothing(j)
        display(x)
        return 0.0
    end
    eT=uh.mesh.element_types[j]
    
    # z = Global2Local(x, uh.mesh.vertices[:,uh.mesh.j_cv[j,:]],eT)
    z = Global2Local(x, view(uh.mesh.vertices,:,uh.mesh.j_cv[j,:]),eT)
    
    uh_ev = Float64(0.0)
    for k=1:uh.Elem[j].nDoF
        uh_ev += uh.Elem[j].DoF[k] .* uh.Elem[j].Basis[k](z)
    end


    return uh_ev #sum( uh.Elem[j].DoF[k] .* uh.Elem[j].Basis[k](z) for k=1:uh.Elem[j].nDoF)
end


function Interface2Local(y, inds, uh::FE; opts="first")

    # Element indices
    e1 = inds[1][1]
    e2 = inds[2][1] 
    
    # local Face indices
    f1 = inds[1][2]
    f2 = inds[2][2]

    if opts=="first"
        z1 = Face2Local(y,f1,uh.Elem[e1].eT)    # local in e1
        x  = Local2Global(z1, uh.mesh.vertices[:,uh.mesh.j_cv[e1,:]], uh.Elem[e1].eT) # global coordinate
        z2 = Global2Local(x, uh.mesh.vertices[:,uh.mesh.j_cv[e2,:]], uh.Elem[e2].eT)    # local in e2
        return z1, z2
    elseif opts=="last"
        z2 = Face2Local(y,f2,uh.Elem[e2].eT)    # local in e2
        x  = Local2Global(z2, uh.mesh.vertices[:,uh.mesh.j_cv[e2,:]], uh.Elem[e2].eT) # global coordinate
        z1 = Global2Local(x, uh.mesh.vertices[:,uh.mesh.j_cv[e1,:]], uh.Elem[e1].eT)    # local in e2
        return z1, z2
    end

end


# Constructs an Orthonormal Basis on a simple polygon, given a corresponding quadrature
# Based on Gram-Schmidt, according to: 
#       Ern, DiPietro:  Mathematical aspects of DG, Appendix A
function getOrthonormalBasis(d::Int64, deg::Integer, quad::Quadrature; space="Pd")
    # starts with the monomials as Basis

    Np = 1
    if d==1
        Np = deg+1
    elseif d==2
        Np = sum(i for i=1:deg+1)
    end


    monomials = Array{Function}(undef,Np)
    monomial_scaling = Vector{Float64}(undef,Np)
    orthBasis = Array{Function}(undef,Np)
    inds = []

    A = diagm(ones(Np))

    # Initialize Monomials
    if d==1
        inds = reshape(collect(1:Np),1,:)
        for i=1:Np
            f = x -> (x[1]^(inds[i]-1))
            monomial_scaling[i] = sqrt(sum(quad.w[t]*(f(quad.pts[t]))^2  for t in 1:quad.npts))
            monomials[i] = x -> f(x)/monomial_scaling[i]
        end
    elseif d==2
        inds = indexPairs(deg)
        for i=1:Np
            f = x -> (x[1]^(inds[1,i]-1))*(x[2]^(inds[2,i]-1))
            monomial_scaling[i] = sqrt(sum(quad.w[t]*(f(quad.pts[t]))^2  for t in 1:quad.npts))
            monomials[i] = x -> f(x)/monomial_scaling[i]
        end
    end

    for i=1:Np

        scale = sum(quad.w[t]*(monomials[i](quad.pts[t]))^2  for t in 1:quad.npts)
        
        for j=1:i-1
            # *(orthBasis[j](quad.pts[t]))
            sp_ij = sum(quad.w[t]*(monomials[i](quad.pts[t])*(orthBasis[j](quad.pts[t])) )  for t in 1:quad.npts) 
            scale -= sp_ij^2 
            # Vandermonde_pre[i,j] = sp_ij
            A[i,j] = sp_ij
        end

        A[i,i] = 1/sqrt(scale)

        if i==1
            orthBasis[i] = x -> A[i,i]*monomials[i](x) 
        else
            A[i,1:i-1] = A[i,1:i-1]*A[i,i]
            orthBasis[i] = x -> A[i,i]*monomials[i](x) - sum( A[i,k] * orthBasis[k](x) for k=1:i-1)
        end
    end

    # println("=== Orthonormality check ===")
    for i = 1:Np, j = 1:Np
        inner_ij = sum(quad.w[t] * orthBasis[i](quad.pts[t]) * orthBasis[j](quad.pts[t]) for t = 1:quad.npts)
        if i == j
            if abs(inner_ij-1) > 1e-13
                println("⟨ϕ_$i, ϕ_$j⟩ = $inner_ij  (should be ≈ 1)")
            end
        else
            if abs(inner_ij) > 1e-13
                println("⟨ϕ_$i, ϕ_$j⟩ = $inner_ij  (should be ≈ 0)")
            end
            # println("⟨ϕ_$i, ϕ_$j⟩ = $inner_ij  (should be ≈ 0)")
        end
    end

    Vandermonde = A
    for i=1:Np
        Vandermonde[i,:] = A[i,:]/A[i,i]
    end

    return orthBasis, inv(Vandermonde), inds, monomial_scaling 

end


# Only for 2D so far!!!
function Vandermonde2GradBasis(d::Int,Vandermonde, inds, monomial_scaling)
    Np = size(inds, 2)

    Grad_Basis = Array{Function}(undef,Np)
    dmonomials = Array{Function}(undef, Np)

    if d==1
        # Differentiate Monomials
        for i in 1:Np
            a = inds[1, i] - 1  # because exponents are stored as +1
            dmonomials[i] = v-> (a==0 ? 0.0 : a*(v[1]^(a-1))/monomial_scaling[i])
        end

        # Now apply linear combinations via Vandermonde
        for i in 1:Np
            Grad_Basis[i] = v-> [ sum(Vandermonde[i, j] * dmonomials[j](v[1]) for j in 1:Np)]
        end

        return Grad_Basis
    elseif d==2
        # Differentiate Monomials
        for i in 1:Np
            a = inds[1, i] - 1  # because exponents are stored as +1
            b = inds[2, i] - 1

            dmonomials[i] = v-> [(a==0 ? 0.0 : a*(v[1]^(a-1))*(v[2]^b)/monomial_scaling[i]);
                                (b==0 ? 0.0 : b*(v[1]^(a))*(v[2]^(b-1))/monomial_scaling[i])]
        end

        # Now apply linear combinations via Vandermonde
        for i in 1:Np
            Grad_Basis[i] = v -> [ sum(Vandermonde[i, j] * dmonomials[j](v)[1] for j in 1:Np);
                                   sum(Vandermonde[i, j] * dmonomials[j](v)[2] for j in 1:Np)]
        end

        return Grad_Basis
    end
end



end # module FiniteElements
