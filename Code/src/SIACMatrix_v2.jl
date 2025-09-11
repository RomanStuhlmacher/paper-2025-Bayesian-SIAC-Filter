# SIAC Matrix based on Lagrange Finite Elements on uniform grid

using LinearAlgebra # to do Linear Algebra
using QuadGK        # for Integrals
using BasicBSpline  # for the Kernel


###############################################################################################################
# Matrix construction
###############################################################################################################


function local_SIAC_Mat(r::Int, l::Int, pp::Int, LocalNodes::Array{Float64}, EvalNodes::Array{Float64})
    # r             Parameter of the SIAC Filter (#Splines-1, for classical SIAC r=2p if p degree of piecewise approximation)
    # l             Parameter of the SIAC Filter (order of BSpline, for classical SIAC l=p+1 if p degree of piecewise approximation)
    # pp            order (not degree!) of local approximation polynomial/number of local points
    # LocalNodes    Vector of Nodes for the Lagrange Polynomials/ Local evaluation Points, assumed to be in the Interval [-1,1]
    # EvallNodes    Vector of Nodes at which the Filtered solution is evaluated, assumed to be in the Interval [-1,1]

    # Setting up the Lagrange Basis
    Lag = Array{Function}(undef, pp)
    for k=1:pp
        Lag[k] = x -> Lagrange(x,LocalNodes,k)
    end

    nEval = length(EvalNodes)



    # knot vector for a single B-Spline
    knot = [-((l)/2)+i  for i=0:(l)] 

    # The Bspline Space defined by the knots -> contains just one B-Spline
    P0 = BSplineSpace{l-1}(KnotVector(knot))

    # Mesh breaks for the central B-Spline
    meshBreaks = collect(-(l-0.5):(l-0.5)) 
    KernelBreaks = collect(-(l-0.5):0.25:(l-0.5)) 

    # Number of  Elements in the support of one single B-Spline
    nEl = length(meshBreaks)-1

    # Number of parameters we have to consider for the filtering operation at each point
    # nEl -> Elements covered by one single B-Spline
    # r -> Counting additional Elements in support by adding B-Splines (1 element per spline)
    # pp -> DoF per Cell
    numParam = (nEl+r)*pp

    KMatLoc = zeros(nEval,numParam) # Vectors to multiply with the nodes to PostProcess
    KMatLoc_pre = zeros(Float64,nEval,nEl*pp)

    # Gauss Legendre Nodes for Integration
    npts = l+1
    pts, w = gauss(npts, -1, 1)


    for (i,xki) in enumerate(EvalNodes)   
    # i -> Enumerates the Basis functions in the filtered Element  

        xi = Ref2Elem(xki,-0.5,0.5)
        shifted_knots = knot.-xi

        for j=1:length(meshBreaks)-1
        # j -> Enumerates the mesh cells

            xL=meshBreaks[j]
            xR=meshBreaks[j+1]

            localBreaks = sort(unique(vcat(shifted_knots,KernelBreaks)))
            filter!(y->((y>=xL)&&(y<=xR)),localBreaks)


            for k=1:length(localBreaks)-1
            # k -> Enumerates the local Breaks

                # Transform to local Interval
                lb = localBreaks[k]
                ub = localBreaks[k+1]
                h = ub-lb
                pts_loc = (h/2)*(pts .+ 1) .+ lb

                for m=1:pp
                # m -> Enumerates the Basis functions of the mesh cells in the support

                    KMatLoc_pre[i,(j-1)*pp+m] += (h/2)*sum(w[ll]*bsplinebasis(P0,1,(xi-pts_loc[ll])).*Lag[m](Elem2Ref(pts_loc[ll],xL,xR)) for ll=1:npts )
                end
            end
        end

    end


    # BSpline centers
    xc = [r/2 - gamma for gamma in 0:r]

    # Kernel Coefficients
    c = analytic_kernel_coeff(r,l-1,xc)

    # display(KMatLoc_pre)
    for i=1:r+1
        # display(collect((i-1)*pp+1:(nEl+i-1)*pp))
        KMatLoc[:,(i-1)*pp+1:(nEl+i-1)*pp] += c[i]*KMatLoc_pre
    end
    
    return KMatLoc


end

# Creates a Matrix for the global SIAC Filtering operation
# Uses a mesh
function global_SIAC_Mat(p::Int64, numEl::Int64, LocalNodes::Array{Float64}, EvalNodes::Array{Float64}; boundaryTreatment="none")
    # p             degree of local approximation polynomial/number of local points -1
    # numEL         #Elements for the piecewise polynomial approximation
    # dx            Elementsize for local approximation
    # LocalNodes    Vector of Nodes for the Lagrange Polynomials/ Local evaluation Points
    # EvallNodes    Vector of Nodes at which the Filtered solution is evaluated, assumed to be in the Interval [-1,1]


    # Parameter of the SIAC Filter (#Splines-1, for classical SIAC m=2p if p degree of piecewise approximation)
    r = 2*p
    # Parameter of the SIAC Filter (order of BSpline, for classical SIAC n=p if p degree of piecewise approximation)
    l = p+1
    # order of local approximation polynomial/number of local points
    pp = p+1
    # Number of evaluation points per cell
    pp_ev = length(EvalNodes)

    KMatLoc = local_SIAC_Mat(r,l,pp,LocalNodes, EvalNodes)

    nPts = Int(pp*numEl)
    nPts_ev = Int(pp_ev*numEl)

    KMat = zeros(Float64,pp_ev*numEl,pp*numEl)

    nE_supp = Int(size(KMatLoc,2)/pp)
    n_bdPts = Int(ceil(((nE_supp-1)/2)*pp))


    KMat_pre = zeros(Float64,nPts_ev,nPts + 2*n_bdPts)

    cols = collect(1:size(KMatLoc,2))
    for elem=1:numEl
        KMat_pre[(elem-1)*pp_ev+1:(elem)*pp_ev,cols] = KMatLoc
        cols = cols.+pp
    end
    # Assuming Periodic Boundary
    if boundaryTreatment == "periodic"
        KMat = KMat_pre[:,n_bdPts+1:nPts+n_bdPts] + [zeros(nPts_ev,nPts-n_bdPts) KMat_pre[:,1:n_bdPts]] + [KMat_pre[:,(end-n_bdPts+1):end] zeros(nPts_ev,nPts-n_bdPts) ]

    # Assuming non-periodic Boundary, sets boundary nodes in KMat to identity
    else
        KMat = diagm(ones(nPts))
        KMat[:,n_bdPts+1:nPts-n_bdPts] = KMat_pre[:,2n_bdPts+1:nPts]
    end

    return sparse(KMat)

end

# Creates a Matrix for the global SIAC Filtering operation
# Uses a prescribed Set of evaluation Points and defines Mesh information accordingly
function global_SIAC_Mat(p::Int64, Pts::Array; boundaryTreatment="none" )
    # p             degree of local approximation polynomial/number of local points -1
    # Pts           Array of points on which the solution is evaluated.     

    # Infering Mesh related quantities from Evaluation Points
    # Local Nodes as Array of vectors of local evaluation points
    LocalNodes, numEl, dx = Pts2Nodes(p, Pts)

    return global_SIAC_Mat(p, numEl, LocalNodes, LocalNodes::Array{Float64}; boundaryTreatment=boundaryTreatment)

end




###############################################################################################################
# Handles for Lagrange Interpolation
###############################################################################################################

# Mapping to a Standard Element
function Elem2Ref(x::Float64, xL::Float64, xR::Float64)
    # Maps [xL,xR] to [-1,1]
    # xL    Left Boundary
    # xR    Right Boundary
    # x     Evaluation Point in [xL,xR]

    return (2/(xR-xL))*(x-xL)-1
end

# Mapping from a Standard Element 
function Ref2Elem(z::Float64, xL::Float64, xR::Float64)
    # Maps [-1,1] to [xL,xR]
    # xL    Left Boundary
    # xR    Right Boundary
    # x     Evaluation Point ibn [-1,1]

    return ((xR-xL)/2)*(z+1)+xL
end



# Computes the k-th Lagrange Basis function at point x, based on the interpolation points Pts
function Lagrange(x::Float64,Pts::Array,k::Int)
    # x     Evaluation Point
    # Pts   Nodes
    # k     Number of the Lagrange Polynomial
    npts = length(Pts)
    w = prod(Pts[k] - Pts[i]  for i in Iterators.filter(j->j!=k,1:npts))
    return prod( x .- Pts[i] for i in Iterators.filter(j->j!=k,1:npts))/w
end

# Computes Local Nodes from Evaluation points
# Interprets Evaluation Points as Nodes of a Piecewise (discontinuous) Polynomial Approximation
# Defines SIAC based on a uniform mesh struture
# Assumes uniformly spaced Evaluation points!
function Pts2Nodes(p::Int, Pts::Array{Float64})
    # p     prescribed polynomial degree of approximation
    # Pts   Global evaluation Points

    # Defining mesh based quantities
    numEL = ceil(Int,length(Pts)/(p+1))

    # Returns warning if number of Points and required elements don't match
    extraPts = 0
    if abs(numEL-length(Pts)/(p+1)) > 0
        @warn "Number of required Elements=$numEL for p=$p \n At least $(numEL*(p+1)) Points required, $(length(Pts)) Points given!"
        extraPts = numEL*(p+1)-length(Pts)
    end

    # Create the mesh
    mesh = zeros(numEL+1)
    LocalNodes = Array{Vector}(undef,numEL)
    sec = Pts[2]-Pts[1]     # The distance between two points

    for elem=1:numEL
        # Setting the Element Boundaries
        mesh[elem] = Pts[(elem-1)*(p+1)+1] - sec/2
    end
    mesh[end] = Pts[end] + sec/2
    dx = mesh[2]-mesh[1]


    # Creating the Local Nodes for the SIAC Matrix representation
    for elem=1:numEL

        # Setting Local Nodes
        loc = zeros(p+1)

        # Assuming number of points matches number of elements times polynomial order
        if extraPts == 0
            for k=1:p+1
                loc[k] = Elem2Ref(Pts[(elem-1)*(p+1)+k], mesh[elem], mesh[elem+1])
            end
            LocalNodes[elem] = loc
        
        # Assuming number of points not matches number of elements times polynomial order
        else
            # Normal procedure for the "complete" elements
            if elem < numEL
                for k=1:p+1
                    loc[k] = Elem2Ref(Pts[(elem-1)*(p+1)+k], mesh[elem], mesh[elem+1])
                end
                LocalNodes[elem] = loc
            # Takes local Nodes of previous element for "cut" element
            # Assumes uniformly spaced evaluation points!
            else
                LocalNodes[elem] =  LocalNodes[elem-1]
            end
        end
    end


    return LocalNodes, numEL, dx

end


function makeNodes(mesh::FiniteElements.Mesh, p::Int; type="uni")
    # xL        Left Boundary of the Domain
    # xR        Right Boundary of the Domain
    # numEL     #Elements for the piecwise polynomial Approximation
    # p         degree of piecewise polynomial Approximation
    # type      Type of local nodes
    #   "legendre"  Gauss-Legendre Nodes
    #   "uniform"   Uniformly distributed points


    # Setting up the approximation nodes 
    LocalNodes = zeros(Float64,p+1)

    if type == "GLeg"
        LocalNodes, LocalWeights = gauss(p+1,-1,1)    # Gauss-Legendre Nodes
        LocalNodes = reshape(LocalNodes,1,:)
        LocalWeights = reshape(LocalWeights,1,:)

    elseif type == "GLob"
        if p==1
            LocalNodes = [-1.0 1.0]
            LocalWeights = [1 1]
        elseif p==2
            LocalNodes = [-1.0 0.0 1.0]
            LocalWeights = [1/3 4/3 1/3]
        elseif p==3
            LocalNodes = [-1.0 -1/sqrt(5) 1/sqrt(5) 1.0]
            LocalWeights = [1/6 5/6 5/6 1/6]
        elseif p ==4
            LocalNodes = [-1.0 -sqrt(3/7) 0.0  sqrt(3/7) 1.0]
            LocalWeights = [1/10 49/90 32/45 49/90 1/10]
        elseif p==5
            LocalNodes = [-1.0 -sqrt(1/3+(2/21)*sqrt(7)) -sqrt(1/3-(2/21)*sqrt(7)) -sqrt(1/3-(2/21)*sqrt(7)) sqrt(1/3+(2/21)*sqrt(7)) 1.0]
            LocalWeights = [1/15 (14-sqrt(7))/30 (14+sqrt(7))/30 (14+sqrt(7))/30 (14-sqrt(7))/30 1/15]
        end

    elseif type == "uni"
        LocalNodes = collect(LinRange(-1,1,p+1))       # Uniformly Spaced Points
        if p==1
            LocalWeights = [1 1]
        elseif p==2
            LocalWeights = [1/3 4/3 1/3]
        elseif p==3
            LocalWeights = [3/8 9/8 9/8 3/8]
        elseif p ==4
            LocalWeights = [14/45 64/45 24/45 64/45 14/45]
        end
    end

    # GlobalNodes = [Ref2Elem(LocalNodes[k],mesh.grid[j],mesh.grid[j+1]) for k in 1:p+1, j in 1:mesh.N]
    # GlobalNodes = reshape(GlobalNodes,((p+1)*mesh.N,1))

    GlobalNodes = zeros(Float64,(p+1)*mesh.NC[])
    GlobalWeights = zeros(Float64,(p+1)*mesh.NC[])
    for e = 1:mesh.NC[]
        for i = 1:(p+1)
            GlobalNodes[(p+1)*(e-1)+i] = Local2Global(LocalNodes[i],[mesh.vertices[mesh.j_cv[e,1]] mesh.vertices[mesh.j_cv[e,2]]], 1)
            GlobalWeights[(p+1)*(e-1)+i] = LocalWeights[i]
        end
    end

    return GlobalNodes, GlobalWeights, LocalNodes, LocalWeights
end


###############################################################################################################
# Symmetric SIAC Kernel Functions
###############################################################################################################

# SIAC Kernel Coefficients 
function analytic_kernel_coeff(r::Int, l::Int, xShift::Vector)
    # r         Number of BSplines in the Kernel -1
    # n         Order of the Bsplines
    # xShift    BSpline Centers

    # Kernel Matrix
    A  = zeros(r+1,r+1)

    # Computation of coefficients (based on binomial representation of the shifted polynomials)
    for p = 0:r
        s1 = [ sum( [(-1.0)^(j) * binomial(l,j) * ((j-.5*(l-1) )^(l+1+i) - (j-.5*(l+1))^(l+1+i)) for j = 0:l] ) for i = 0:p]
        for γ = 0:r
            s2    = [binomial(p,i)*(-1)^(l)*(xShift[γ+1])^(p-i)*factorial(i)/factorial(l+1+i) for i = 0:p]
            A[p+1,γ+1] = sum(s1 .* s2)
        end
    end

    return reverse(factorize(A)\vcat(1.0,zeros(r)))
end

# Symmetric SIAC Kernel 
# The classical SIAC Kernel uses m=2p, n=p for p the degree of the piecewise polynomial approximation
function Symmetric_SIAC_Kernel(r::Int, l::Int)
    # r     Number of BSplines in the Kernel 
    # l     Order of the Bsplines

    # Creates knot Vector for the Kernel
    knot = [-((r+l)/2)+i  for i=0:(r+l)] 
    knotVec = KnotVector(knot)
    P0 = BSplineSpace{l-1}(knotVec)
    # knotVec = Knots(knot)           # old version  of BasicBSpline for Romans Laptop
    # P0 = BSplineSpace(n,knotVec)    # old version  of BasicBSpline for Romans Laptop

    # BSpline centers
    xᵧ = [r/2 - γ for γ in 0:r]

    # Kernel Coefficients
    c = analytic_kernel_coeff(r,l-1,xᵧ)
 
    return K(x) = sum([c[i].*bsplinebasis(P0,i,x) for i =1:r+1]) 
    # return K(x) = sum([c[i].*bsplinebasis(i,P0,x) for i =1:m+1])  # old version  of BasicBSpline for Romans Laptop
end



###############################################################################################################
# Continuous Postprocessing routine (only for visualization!)
###############################################################################################################

# Post Processing Routine
function PostProcess(x::Float64, H::Float64, r::Int, l::Int, Approx::FE, mesh::FiniteElements.Mesh, quad::Quadrature)

    u = x->evalFE(x,Approx)

    # Compute the KernelBreaks
    s = (r+l)/2
    BreakPoints = [ x.-(i)*H   for i=-s:s ]
    xL = minimum(BreakPoints)
    xR = maximum(BreakPoints)


    # Compute the BreakPoints
    for j=1:mesh.NV[]
        if (mesh.vertices[j]>=xL)&&(mesh.vertices[j]<=xR)
            BreakPoints = vcat(BreakPoints, mesh.vertices[j])
        end
    end
    sort!(unique!(BreakPoints))

    # Compute the SIAC Kernel
    K = Symmetric_SIAC_Kernel(r,l)

    # Do the convolution
    uPost = 0.0;
    numIntervalls = length(BreakPoints)-1
    for i=1:numIntervalls

        # Transform to local Interval
        lb = BreakPoints[i]
        ub = BreakPoints[i+1]
        h = ub-lb
        pts_loc = (h/2)*(quad.pts .+ 1) .+ lb

        uPost += (h/2)*sum(quad.w[k]*(K((pts_loc[k]-x)/H).*u(pts_loc[k]))  for k=1:quad.npts )
    end

    return uPost/(H)
end
