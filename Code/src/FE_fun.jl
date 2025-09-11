###############################################################################################################
# Finite Element Implementation
###############################################################################################################

# The Element Structure
mutable struct Element
    # Mesh related Quantities
    xL::Float64     # Left Boundary
    xR::Float64     # Right Boundary
    dx::Float64     # Size

    # Approximation Quantities
    nDoF::Int                  # Number of DoF
    DoF::Vector{Float64}       # The Degrees of Freedom
    Basis::Vector{Function}     # The Basis Functions
    D1Basis::Vector{Function}   # First derivative of the Basis Functions
end

# The Finite Element (FE) Structure
mutable struct FE
    N::Int                  # Number of Elements
    Elem::Array{Element}   # The Elements
    mesh::Mesh
end

# A hnadle for Quadrature rules
struct Quadrature
    npts::Int64
    pts::Vector{Float64}
    w::Vector{Float64}
end


# L2 projection of the function fun (as Function handle) in the intervall xL to xR, given a Quadrature and an Expansion Basis
function L2_Coeff(xL::Float64 ,xR::Float64, deg::Int64, fun::Function, Basis::Array{Function}, quad::Quadrature)
    DoF = zeros(deg+1)
    for p = 1:deg+1
        fun_eval(x) = fun(Ref2Elem(x,xL,xR)) .* Basis[p](x)
        DoF[p] =((2*p-1)/2)* sum(quad.w[i]*fun_eval(quad.pts[i]) for i in 1:quad.npts)
    end
    return DoF
end

# # L2 projection of the function fun (as Vector, at the Quadrature Points) in the intervall xL to xR, given a Quadrature and an Expansion Basis
# function L2_Coeff(xL::Float64 ,xR::Float64, deg::Int64, fun::Vector{Float64}, Basis::Array{Function}, quad::Quadrature)
#     DoF = zeros(deg+1)
#     for p = 1:deg+1
#         DoF[p] = sum(quad.w[i]*fun[i]*Basis[p](quad.pts[i]) for i in 1:quad.npts)
#     end
#     return DoF
# end

# Generate a PPA of a function,using the Legendre Base
function L2(fun::Function, mesh::Mesh, deg::Int64)

    Basis = Array{Function}(undef,deg+1)
    D1Basis = Array{Function}(undef,deg+1)
    for k=1:deg+1
        Basis[k] = x->Legendre(x,k)
        D1Basis[k] = x->DLegendre(x,k)
    end

    # Setting up a Quadrature
    npts = 2*deg
    pts, w = gauss(npts, -1, 1)
    quad = Quadrature(npts,pts,w)

    Elem = Array{Element}(undef,mesh.N)
    for j=1:mesh.N
        DoF =  L2_Coeff(mesh.grid[j],mesh.grid[j+1],deg,fun,Basis,quad)
        Elem[j] = Element(mesh.grid[j],mesh.grid[j+1],mesh.dx[j],deg+1,DoF,Basis,D1Basis)
    end

    Approx = FE(mesh.N,Elem,mesh)

    return Approx

end

function Interpolate(fun::Function, mesh::Mesh, LocalNodes::Array{Float64})

    deg = length(LocalNodes)-1

    Basis = Array{Function}(undef,deg+1)
    D1Basis = Array{Function}(undef,deg+1)
    for k=1:deg+1
        Basis[k] = x->Lagrange(x,LocalNodes,k)
        D1Basis[k] = x->DLagrange(x,LocalNodes,k)
    end


    Elem = Array{Element}(undef,mesh.N)
    for j=1:mesh.N
        DoF =  [fun(Ref2Elem(LocalNodes[k],mesh.grid[j],mesh.grid[j+1])) for k=1:deg+1]
        Elem[j] = Element(mesh.grid[j],mesh.grid[j+1],mesh.dx[j],deg+1,DoF,Basis,D1Basis)
    end

    Approx = FE(mesh.N,Elem,mesh)

    return Approx

end

# Evaluate a FE Handle
function evalFE(x::Float64,Approx::FE; order="full", type="value", bd="standard")
    j = getElem(x,Approx.mesh)
    if bd=="periodic"
        j = getElem(x,Approx.mesh,type="periodic")
    end
    z = Elem2Ref(x,Approx.mesh.grid[j],Approx.mesh.grid[j+1])

    if (order == "full")&&(type == "value")
        return sum( Approx.Elem[j].DoF[k].*Approx.Elem[j].Basis[k](z) for k=1:Approx.Elem[j].nDoF)
    elseif (order == "average")&&(type == "value")
        return Approx.Elem[j].DoF[1].*Approx.Elem[j].Basis[1](z)
    elseif (order == "linear")&&(type == "value")
        return sum( Approx.Elem[j].DoF[k].*Approx.Elem[j].Basis[k](z) for k=1:2)
    elseif (order == "full")&&(type == "derivative")
        return sum( Approx.Elem[j].DoF[k].*Approx.Elem[j].D1Basis[k](z) for k=1:Approx.Elem[j].nDoF)
    elseif (order == "average")&&(type == "derivative")
        return Approx.Elem[j].DoF[1].*Approx.Elem[j].D1Basis[1](z)
    end

end

# Evaluate a FE Handle
function evalFE_av(x::Float64,Approx::FE)
    j = getElem(x,Approx.mesh)
    z = Elem2Ref(x,Approx.mesh.grid[j],Approx.mesh.grid[j+1])
    return sum( Approx.Elem[j].DoF[k].*Approx.Elem[j].Basis[k](z) for k=1:1)
end

# Get the DoF of an FE Handle
function getDoF(uh::FE)
    # Assuming same deg polynomials on every element
    globalDoFs = zeros(Float64,uh.N*uh.Elem[1].nDoF)
    for e = 1:uh.N
        for i = 1:uh.Elem[e].nDoF
            globalDoFs[uh.Elem[e].nDoF*(e-1)+i] = uh.Elem[e].DoF[i]
        end
    end

    return globalDoFs
end


# Get the DoF of an FE Handle
function getDoF_Elem(uh::FE, e::Int)
    # Assuming same deg polynomials on every element
    localDoFs = zeros(Float64,uh.Elem[e].nDoF)
    for i = 1:uh.Elem[e].nDoF
        localDoFs[i] = uh.Elem[e].DoF[i]
    end

    return localDoFs
end

# Set the DoF of an FE Handle
function setDoF!(uh::FE, globalDoFs::Array)

    # Assuming same deg polynomials on every element
    for e = 1:uh.N
        for i = 1:uh.Elem[e].nDoF
            uh.Elem[e].DoF[i] = globalDoFs[uh.Elem[e].nDoF*(e-1)+i]
        end
    end

    return globalDoFs
end

function setDoF_Elem!(uh::FE, localDoF::Array, e::Int)

    for i = 1:uh.Elem[e].nDoF
        uh.Elem[e].DoF[i] = 0.0
    end

    m = length(localDoF)
    for i = 1:m
        uh.Elem[e].DoF[i] = localDoF[i]
    end
end

###############################################################################################################
# Reference mappings (& Basis Isomorphisms... Later)
###############################################################################################################

# Mapping from Given Element to the Reference Element
function Elem2Ref(x::Float64, xL::Float64, xR::Float64)
    return (2/(xR-xL))*(x-xL)-1
end

# Mapping from Reference Element to a Given one
function Ref2Elem(z::Float64, xL::Float64, xR::Float64)
    return ((xR-xL)/2)*(z+1)+xL
end


###############################################################################################################
# Basis Functions
###############################################################################################################

# Computes the k-th Lagrange Basis function at point x, based on the interpolation points Pts
function Lagrange(x::Float64,Pts::Array,k::Int)::Float64
    npts = length(Pts)
    w = prod(Pts[k] - Pts[i]  for i in Iterators.filter(j->j!=k,1:npts))
    return prod( x - Pts[i] for i in Iterators.filter(j->j!=k,1:npts))/w
end

function DLagrange(x::Float64,Pts::Array{Float64},k::Int64)::Float64
    npts = length(Pts)
    w = prod(Pts[k] - Pts[i]  for i in Iterators.filter(j->j!=k,1:npts))
    lk = prod( x - Pts[i] for i in Iterators.filter(j->j!=k,1:npts))/w
    d = sum(1/(x-Pts[i]) for i in Iterators.filter(j->(j!=k),1:npts))

    return lk*d
end

# Computes the (non-normalized) k-th Legendre Polynomial at point X
function Legendre(x::Float64, k::Int; type="orthogonal")
    n = k-1

    if type == "orthogonal"
        if n==0
         return one(x)
        elseif n==1
          return(x)
        end

        p0 = one(x)
        p1 = x

        for i = 2:n
            p2 = ( (2i-1)*x*p1 - (i-1)*p0 ) / i
            p0 = p1
            p1 = p2
        end

        return p1
    end

    return sqrt(k-0.5)*Legendre(x,k;type="orthogonal")
end

# Evaluates the 1st derivative of the k-th Legendre Polynomial at x
function DLegendre(x::Float64, k::Int; type="orthonormal")
    n = k-1

    if n == 0
        return 0.0
    elseif n == 1
        return  one(x) #sqrt(1/(k-1/2))
    elseif n == 2
        return 3*x
    elseif n == 3
        return (15/2)*x*x - 3/2
    end

    return (n+1)*Legendre(x,n) + x .* DLegendre(x,n)
    
end



###############################################################################################################
# Flux related functions
###############################################################################################################

# Averaging
function av(fp::Float64,fm::Float64)
    return .5*(fp+fm)
end

# Jump
function jump(fp::Float64,fm::Float64)
    return fp-fm
end

# Lax-Friedrichs flux
function LF(up::Float64,um::Float64, flux::Function)

    # Local max evaluation?
    # h = uh.Elem[e].dx/10
    # xEval = collect(uh.Elem[e].xL:h:uh.Elem[e].xR)
    # u_vec = [evalFE(xj,uh) for xj in xEval] # uh evaluated
    # f_vec = [flux(uj) for uj in u_vec]      # f(uh) evaluated 

    # np = length(f_vec)
    # df_vec = [(f_vec[i+1]-f_vec[i])/h for i=1:np-1]  
    # alpha = maximum(df_vec) 

    # alpha = maximum([abs(flux(up)) abs(flux(um))]) # rusanov
    alpha = maximum([abs(up) abs(um)]) # burgers
    # alpha = 1 # advection
    f = av(flux(up),flux(um)) - (alpha/2)*jump(up,um)

    return f
end

# Upwind flux
function Up(up::Float64,um::Float64, flux::Function)

    fp = flux(up)
    fm = flux(um)

    if (fp-fm)/(up-um) >= 0
        return fm
    else
        return fp
    end
end



##############################################################################################################
# System assembly
##############################################################################################################


function discrete_system(du::Vector, u::Vector, t::Float64 ,M,K,F)

    # r = similar(u)
    # 2.1 compute r = (K - F)*u
    r = K*u        # r = K*u
    r -= F*u  # r .+= - F*u

    # 2.2 solve M * du = r  in place, overwriting r → du
    copy!(du, r)           # du = r
    ldiv!(M, du)      # now du = M^{-1}*r
    return nothing
end

# Mass Matrix Assembly
function assembleMassMatrix(uh::FE)
    # Assuming the same polynomial degree on each element
    order = uh.Elem[1].nDoF

    # Mass Matrix
    M = zeros(uh.N*order,uh.N*order)

    # Interior Points
    for e = 1:uh.N

        # Setting up a accurate enough Quadrature
        npts = 2*order
        pts, w = gauss(npts, -1, 1)
        # quad = Quadrature(npts,pts,w)

        # Ansatz functions (DG, columns)
        for j = 1:order
            # Test functions (VarForm, rows)
            for i = 1:order
                # Mass Matrix
                M[order*(e-1)+i,order*(e-1)+j] = (uh.Elem[e].dx/2)*sum(w[pp]*(uh.Elem[e].Basis[i](pts[pp])).*(uh.Elem[e].Basis[j](pts[pp])) for pp in 1:npts)
            end         
        end
    end
    return M
end

# Stiffness Assembly
function assembleStiffnessTerm(uh::FE, flux::Function, globalDoF::Array)
    
    # Assuming the same polynomial degree on each element
    order = uh.Elem[1].nDoF

    # Stiffness Matrix
    K =  Array{Float64}(undef,uh.N*order)

    # Setting up a accurate enough Quadrature
    npts = 2*order
    pts, w = gauss(npts, -1, 1)

    # Interior Points
    for e = 1:uh.N

        # Evaluating the flux
        flux_ev = zeros(npts)
        for pp=1:npts
            flux_ev[pp] = flux(sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](pts[pp]) for i=1:order))
        end

        # Test functions (VarForm, Rows)
        for j = 1:order
            # computing the stiffness term
            K[order*(e-1)+j] = sum( w[pp]*(uh.Elem[e].D1Basis[j](pts[pp]))*flux_ev[pp]  for pp in 1:npts)
        end   
    end
    return K
end

# Flux Assembly (periodic Boundary)
function assembleFlux(uh::FE, nflux::Function, flux::Function, globalDoF::Array; boundary="periodic")

    # Assuming the same polynomial degree on each element
    order = uh.Elem[1].nDoF

    # Numerical Flux 
    Flux = Array{Float64}(undef,uh.N*order)

    # Interior Points
    for e = 1:uh.N
        # Normal Vectors
        nR = 1.0
        nL = -1.0

        # Inner Elements
        if e > 1 && e < uh.N
            # Ansatz functions (DG)
            for j = 1:order
                 # Test functions
                vm = uh.Elem[e].Basis[j](1.0)       # v-
                vp = uh.Elem[e].Basis[j](-1.0)      # v+

                # Right element boundary
                upR = sum(globalDoF[order*(e)+i].*uh.Elem[e+1].Basis[i](-1.0) for i=1:order)      # u+ (in neighbor)
                umR = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](1.0) for i=1:order)     # u- (in element)
                fR = nflux(upR,umR, flux)     # right Flux

                # Left element boundary
                upL = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](-1.0) for i=1:order)  # u+ (in element)
                umL = sum(globalDoF[order*(e-2)+i].*uh.Elem[e-1].Basis[i](1.0) for i=1:order)   # u- (in neighbor)
                fL = nflux(upL,umL, flux)    # left Flux

                # Flux contribution
                Flux[order*(e-1)+j] =  (fR*vm*nR + fL*vp*nL)
                                        
            end
        # Leftmost element
        elseif e == 1

            if boundary == "periodic"
                # Ansatz functions (DG)
                for j = 1:order
                    # Test functions
                    vm = uh.Elem[e].Basis[j](1.0)       # v-
                    vp = uh.Elem[e].Basis[j](-1.0)      # v+

                    # Right element boundary
                    upR = sum(globalDoF[order*(e)+i].*uh.Elem[e+1].Basis[i](-1.0) for i=1:order)   # u+ (in neighbor)
                    umR = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](1.0) for i=1:order)  # u- (in element)
                    fR = nflux(upR,umR, flux)     # right Flux

                    # Left element boundary
                    upL = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](-1.0) for i=1:order)      # u+ (in element)
                    umL = sum(globalDoF[order*(uh.N-1)+i].*uh.Elem[uh.N].Basis[i](1.0) for i=1:order)    # u+ (in neighbor)
                    fL = nflux(upL,umL, flux)         # left Flux
                
                    # Flux contribution
                    Flux[order*(e-1)+j] =  (fR*vm*nR + fL*vp*nL)
                                        
                end

            elseif boundary == "extrapolate"
                # Ansatz functions (DG)
                for j = 1:order
                    # Test functions
                    vm = uh.Elem[e].Basis[j](1.0)       # v-
                    vp = uh.Elem[e].Basis[j](-1.0)      # v+

                    # Right element boundary
                    upR = sum(globalDoF[order*(e)+i].*uh.Elem[e+1].Basis[i](-1.0) for i=1:order)   # u+ (in neighbor)
                    umR = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](1.0) for i=1:order)  # u- (in element)
                    fR = nflux(upR,umR, flux)     # right Flux

                    # Left element boundary
                    upL = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](-1.0) for i=1:order)      # u+ (in element)
                    umL = upL   # extrapolated
                    fL = nflux(upL,umL, flux)         # left Flux
                
                    # Flux contribution
                    Flux[order*(e-1)+j] =  (fR*vm*nR + fL*vp*nL)
                                        
                end
            end
        # Rightmost element
        elseif e == uh.N

            if boundary == "periodic"
                # Ansatz functions (DG)
                for j = 1:order
                    # Test functions
                    vm = uh.Elem[e].Basis[j](1.0)       # v-
                    vp = uh.Elem[e].Basis[j](-1.0)      # v+

                    # Right element boundary
                    upR = sum(globalDoF[order*(0)+i].*uh.Elem[1].Basis[i](-1.0) for i=1:order)   # u+ (in neighbor)
                    umR = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](1.0) for i=1:order)   # u- (in element)
                    fR = nflux(upR,umR, flux)     # right Flux

                    # Left element boundary
                    upL = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](-1.0) for i=1:order) # u+ (in element)
                    umL = sum(globalDoF[order*(e-2)+i].*uh.Elem[e-1].Basis[i](1.0) for i=1:order)  # u+ (in neighbor)
                    fL = nflux(upL,umL, flux)    # left Flux

                    # Flux contribution
                    Flux[order*(e-1)+j] =  (fR*vm*nR + fL*vp*nL)                  
                end
            elseif boundary == "extrapolate"
                # Ansatz functions (DG)
                for j = 1:order
                    # Test functions
                    vm = uh.Elem[e].Basis[j](1.0)       # v-
                    vp = uh.Elem[e].Basis[j](-1.0)      # v+
    
                    # Right element boundary
                    umR = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](1.0) for i=1:order)   # u- (in element)
                    upR = umR   # extraploated
                    fR = nflux(upR,umR, flux)     # right Flux
    
                    # Left element boundary
                    upL = sum(globalDoF[order*(e-1)+i].*uh.Elem[e].Basis[i](-1.0) for i=1:order) # u+ (in element)
                    umL = sum(globalDoF[order*(e-2)+i].*uh.Elem[e-1].Basis[i](1.0) for i=1:order)  # u+ (in neighbor)
                    fL = nflux(upL,umL, flux)    # left Flux
    
                    # Flux contribution
                    Flux[order*(e-1)+j] =  (fR*vm*nR + fL*vp*nL)
                                            
                end

            end
        end
    end

    return Flux

end


function assembleFluxMatrix(uh::FE, a::Float64)
    order = uh.Elem[1].nDoF
    N     = uh.N
    ndofs = N * order

    # Preallocate index and value arrays for sparse assembly
    I = Int[]
    J = Int[]
    V = Float64[]

    Flux = zeros(Float64,ndofs,ndofs)

    # Interior Points
    for e = 1:uh.N

        # Test
        for i = 1:order
            # Ansatz
            for j in 1:order
                # Numerical Flux (Upwind):

                # Right Boundary
                Flux[order*(e-1)+i,order*(e-1)+j] += uh.Elem[e].Basis[i](1.0)*uh.Elem[e].Basis[j](1.0)

                # Periodic BC
                if e == 1
                    Flux[order*(e-1)+i,order*(N-1)+j] -= uh.Elem[e].Basis[i](-1.0)*uh.Elem[e].Basis[j](1.0)
                end

                # Left Boundary
                if e > 1
                    Flux[order*(e-1)+i,order*(e-2)+j] -= uh.Elem[e].Basis[i](-1.0)*uh.Elem[e].Basis[j](1.0)
                end
            
            end
        end
    end

    return sparse(Flux)
end

using SparseArrays

function assemble_global_matrices(uh::FE)
    order = uh.Elem[1].nDoF
    N = uh.N
    ndofs = N * order

    # Sparse matrix entries
    rows = Int[]
    cols = Int[]
    valsM = Float64[]
    valsK = Float64[]

    npts = 2*order
    pts, w = gauss(npts, -1, 1)

    # Loop over elements
    for e = 1:N
        dx = uh.Elem[e].dx
        # Local M, K
        M_loc = zeros(order, order)
        K_loc = zeros(order, order)

        for i = 1:order
            for j = 1:order
                M_loc[i, j] = (dx / 2) * sum(w[p] * uh.Elem[e].Basis[i](pts[p]) * uh.Elem[e].Basis[j](pts[p]) for p = 1:npts)
                K_loc[i, j] = sum(w[p] * uh.Elem[e].Basis[j](pts[p]) * uh.Elem[e].D1Basis[i](pts[p]) for p = 1:npts)
                # Note: no 1/dx here, you’ll apply it later
            end
        end

        # Insert into global matrices
        for i = 1:order
            gi = (e - 1) * order + i
            for j = 1:order
                gj = (e - 1) * order + j
                push!(rows, gi)
                push!(cols, gj)
                push!(valsM, M_loc[i, j])
                push!(valsK, K_loc[i, j])  
            end
        end
    end

    M = sparse(rows, cols, valsM, ndofs, ndofs)
    K = sparse(rows, cols, valsK, ndofs, ndofs)
    return M, K
end



###############################################################################################################
# Limiting related functions
###############################################################################################################

# This is not correct!!!
# Only truncates higher DoF, and keeps linear part
function limitSlopes!(uh::FE, globalDoF)

    # ToDo: Fix boundary condiditons
    for e=2:uh.N-1
        trouble = trouble_cell_indicator(uh, e)

        if trouble

            h = uh.Elem[e].dx
            vL = uh.Elem[e-1].DoF[1] 
            vC = uh.Elem[e].DoF[1] 
            vR = uh.Elem[e+1].DoF[1] 

            localDoF = getDoF_Elem(uh,e)
            slope = minmod([localDoF[2],(vC-vL)/(h/2),(vR-vC)/(h/2)])
            newDoF = [vC slope]

            setDoF_Elem!(uh,newDoF,e)
        end
    end


    return getDoF(uh)
end

# ToDo: Check correctness
function trouble_cell_indicator(uh::FE, e::Int)

    tol = uh.Elem[e].dx/2

    # Cell Averages
    v0C = uh.Elem[e].DoF[1]     # average of cell
    v0L = uh.Elem[e-1].DoF[1]   # average of left neighbor
    v0R = uh.Elem[e+1].DoF[1]   # average of right neighbor

    # Linear Slopes (in element)
    v1L = evalFE(uh.Elem[e].xL,uh; order="linear")  # value of linear approximation, left element boundary
    v1R = evalFE(uh.Elem[e].xR,uh; order="linear")  # value of linear approximation, right element boundary

    # Slopes to compare
    slopesL = [v0C-v1L, v0C-v0L, v0R-v0C]
    slopesR = [v1R-v0C, v0C-v0L, v0R-v0C]

    uL = v0C - minmod(slopesL)
    uR = v0C + minmod(slopesR)

    trouble = false

    if (abs(v1L-uL)>tol)||(abs(v1R-uR)>tol)
        trouble = true
    end

    return trouble
    
end

function minmod(u)
    m = length(u)
    s = (1/m)*sum(sign(u[i]) for i = 1:m)

    if abs(s) == 1
        return s*minimum(abs.(u))
    else
        return 0.0
    end
end

###############################################################################################################
# ODE Solver
###############################################################################################################

# Strong Stability preserving RK3
function SSPRK3(Approx::FE, fun::Function, u0::Array, t::Array; limiting=false)

    u = zeros(size(u0))
    u .= u0 

    u_all = u0

    for i in 1:length(t)-1
        h = t[i+1] - t[i]

        k1 = u + h*fun(u)
        k2 = (3/4)*u .+ (1/4)*k1 .+ (h/4)*fun(k1)
        u = (1/3)*u .+ (2/3)*k2  .+ h*(2/3)*fun(k2)

        if limiting
            setDoF!(Approx,u)
            u = limitSlopes!(Approx, u)
        end

        u_all = hcat(u_all,u)
    end

    
    return u, u_all
end

# Classical RK4
function RK4(fun::Function, u::Array, dt::Float64, t_end::Float64; t0=0.0, history=false)

    
    u_all = u
    t = t0

    while t <= t_end 

        if t+dt > t_end
            dt_hat = t_end-t

            k1 = dt_hat*fun(u,t)
            k2 = dt_hat*fun(u + k1/2,t+dt_hat/2)
            k3 = dt_hat*fun(u + k2/2,t+dt_hat/2)
            k4 = dt_hat*fun(u + k3,t+dt_hat)
            u = u + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += t_end
        else
            k1 = dt*fun(u,t)
            k2 = dt*fun(u + k1/2,t+dt/2)
            k3 = dt*fun(u + k2/2,t+dt/2)
            k4 = dt*fun(u + k3,t+dt)
            u = u + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

            t += dt
        end

        if history
            u_all = hcat(u_all,u)
        end

    end

    if history
        return u, u_all
    end

    return u
end

function RK4!(fun!::Function, u::Vector{T}, dt::Float64, t_end::Float64; t0=0.0, history=false) where T

    N = length(u)
    k1 = similar(u)
    k2 = similar(u)
    k3 = similar(u)
    k4 = similar(u)
    u_tmp = similar(u)

    t = t0
    if history
        us = Vector{T}[]
        push!(us, copy(u))
    end

    while t < t_end
        Δt = t + dt > t_end ? t_end - t : dt

        # stage 1
        fun!(k1, u,       t)
        @. k1 *= Δt

        # stage 2
        @. u_tmp = u + k1/2
        fun!(k2, u_tmp,   t + Δt/2)
        @. k2 *= Δt

        # stage 3
        @. u_tmp = u + k2/2
        fun!(k3, u_tmp,   t + Δt/2)
        @. k3 *= Δt

        # stage 4
        @. u_tmp = u + k3
        fun!(k4, u_tmp,   t + Δt)
        @. k4 *= Δt

        # combine
        @. u += (k1 + 2k2 + 2k3 + k4)/6

        t += Δt
        if history
            push!(us, copy(u))
        end
    end

    return history ? (u, hcat(us...)) : u
end


# Strong Stability preserving RK3
function SSPRK3(fun::Function, u::Array, dt::Float64, t_end::Float64; t0=0.0, history=false)

    # u = zeros(size(u0))
    # u .= u0 

    u_all = u
    t = t0

    while t <= t_end 

        if t+dt > t_end
            dt_hat = t_end-t
            stage = u + dt_hat*fun(u,t)
            stage = (3/4)*u .+ (1/4)*stage .+ (dt_hat/4)*fun(stage,t+dt_hat)
            u = (1/3)*u .+ (2/3)*stage  .+ dt_hat*(2/3)*fun(stage,t+dt_hat/2)
            t += t_end
        else
            stage = u + dt*fun(u,t)
            stage = (3/4)*u .+ (1/4)*stage .+ (dt/4)*fun(stage,t+dt)
            u = (1/3)*u .+ (2/3)*stage  .+ dt*(2/3)*fun(stage,t+dt/2)
            t += dt
        end

        if history
            u_all = hcat(u_all,u)
        end

    end
    
    if history
        return u, u_all
    end

    return u
end

