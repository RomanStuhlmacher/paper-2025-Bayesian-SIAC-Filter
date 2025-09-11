
###############################################################################################################
#                                     Bilinear Forms              
#   u Basis, v Testfunction, K Element, Ω Domain
###############################################################################################################

# Improve this function!!!
function assemble_BilinearForm(uh::FE, d::Int, type="Base_Base"; field::Union{Vector{Float64},Nothing}=nothing)
    # Possible Types
    # Base_Base,        Mass-Matrix
    # Grad_Base,        Advection-Matrix
    # Grad_Grad,        Stiffness-Matrix (ToDo!!)

    # Mass Matrix
    if type=="Base_Base"
        return  assembleMassMatrix(uh)

    # Advection Matrix of upwind scheme
    elseif type=="Grad_Base"

        # display("This happens")
        N_total = sum(e.nDoF for e in uh.Elem)
        max_Dof = maximum(e.nDoF for e in uh.Elem)

        # Check for the field
        if isnothing(field)
            field = ones(Float64,d)
        end

        # Estimate total entries (upper bound)
        total_entries = sum(e.nDoF^2 for e in uh.Elem)
        
        # Preallocate buffers
        rows = Vector{Int}(undef, total_entries)
        cols = Vector{Int}(undef, total_entries)
        vals = Vector{Float64}(undef, total_entries)

        # Setup multithreading for Cell-Contribution
        inds = zeros(Int, max_Dof)  # reused per thread
        entry_index = zeros(Int, nthreads())  # track insert position per thread

        # Thread-local storage for rows/cols/vals chunks
        local_rows = [Vector{Int}() for _ in 1:nthreads()]
        local_cols = [Vector{Int}() for _ in 1:nthreads()]
        local_vals = [Vector{Float64}() for _ in 1:nthreads()]

        
        #####################################
        # Cell Contribution
        #####################################
        assemly_time = Progress(uh.N; desc="Assembling Advection Matrix (Cell Contribution)..")
        Threads.@threads for e = 1:uh.N
            tid = threadid()
            elem = uh.Elem[e]
            num = elem.nDoF
            local_inds = similar(inds, num)
            local_inds .= getElemDoF_ind(uh, e)
            Kc_loc = zeros(Float64, num, num)

            computeLocalCellAdvection!(uh, e, field, Kc_loc)

            for i = 1:num
                ii = local_inds[i]
                for j = 1:num
                    jj = local_inds[j]
                    push!(local_rows[tid], ii)
                    push!(local_cols[tid], jj)
                    push!(local_vals[tid], Kc_loc[i,j])
                end
            end
            next!(assemly_time)
        end
        finish!(assemly_time)


        # Combine thread-local results
        rows = vcat(local_rows...)
        cols = vcat(local_cols...)
        vals = vcat(local_vals...)

        # Setup multithreading for Face-Contribution
        local_map = [Dict{Tuple{Int, Int}, Int}() for _ in 1:nthreads()]

        # Thread-local storage for rows/cols/vals chunks
        local_rows = [Vector{Int}() for _ in 1:nthreads()]
        local_cols = [Vector{Int}() for _ in 1:nthreads()]
        local_vals = [Vector{Float64}() for _ in 1:nthreads()]

        Kf_e1_e1 = [zeros(Float64,max_Dof,max_Dof) for _ in 1:nthreads()]
        Kf_e1_e2 = [zeros(Float64,max_Dof,max_Dof) for _ in 1:nthreads()]
        Kf_e2_e1 = [zeros(Float64,max_Dof,max_Dof) for _ in 1:nthreads()]
        Kf_e2_e2 = [zeros(Float64,max_Dof,max_Dof) for _ in 1:nthreads()]


        #####################################
        # Face Contribution (by Upwinding)
        #####################################
        assembly_time = Progress(length(uh.mesh.in_cf); desc="Assembling Advection Matrix (Face Contribution)..")
        Threads.@threads for f in uh.mesh.in_cf  # Loop over the interior faces
            tid = threadid()

            if !haskey(uh.mesh.face2elem, f)
                continue
            end

            computeLocalFaceAdvection!(uh, f, field, 
                                    Kf_e1_e1[tid], Kf_e1_e2[tid], Kf_e2_e1[tid], Kf_e2_e2[tid])

            (e1, f1), (e2, f2) = uh.mesh.face2elem[f]

            e1_DoF_inds = getElemDoF_ind(uh,e1)
            e2_DoF_inds = getElemDoF_ind(uh,e2)

            # Assign Contribution from Cell 1 
            for (i,j) in enumerate(e1_DoF_inds)
                for (ii,jj) in enumerate(e1_DoF_inds)
                    key = (j, jj)
                    if haskey(local_map[tid], key)
                        idx = local_map[tid][key]
                        local_vals[tid][idx] -= Kf_e1_e1[tid][i,ii]
                    else
                        push!(local_rows[tid], j)
                        push!(local_cols[tid], jj)
                        push!(local_vals[tid], -Kf_e1_e1[tid][i,ii])
                        local_map[tid][key] = length(local_vals[tid])
                    end
                end
                for (ii,jj) in enumerate(e2_DoF_inds)
                    key = (j, jj)
                    if haskey(local_map[tid], key)
                        idx = local_map[tid][key]
                        local_vals[tid][idx] -= Kf_e1_e2[tid][i,ii]
                    else
                        push!(local_rows[tid], j)
                        push!(local_cols[tid], jj)
                        push!(local_vals[tid], -Kf_e1_e2[tid][i,ii])
                        local_map[tid][key] = length(local_vals[tid])
                    end
                end
            end

            # Assign Contribution from Cell 2
            for (i,j) in enumerate(e2_DoF_inds)
                for (ii,jj) in enumerate(e1_DoF_inds)
                    key = (j, jj)
                    if haskey(local_map[tid], key)
                        idx = local_map[tid][key]
                        local_vals[tid][idx] -= Kf_e2_e1[tid][i,ii]
                    else
                        push!(local_rows[tid], j)
                        push!(local_cols[tid], jj)
                        push!(local_vals[tid], -Kf_e2_e1[tid][i,ii])
                        local_map[tid][key] = length(local_vals[tid])
                    end
                end
                for (ii,jj) in enumerate(e2_DoF_inds)
                    key = (j, jj)
                    if haskey(local_map[tid], key)
                        idx = local_map[tid][key]
                        local_vals[tid][idx] -= Kf_e2_e2[tid][i,ii]
                    else
                        push!(local_rows[tid], j)
                        push!(local_cols[tid], jj)
                        push!(local_vals[tid], -Kf_e2_e2[tid][i,ii])
                        local_map[tid][key] = length(local_vals[tid])
                    end
                end
            end

            next!(assembly_time)
        end
        finish!(assembly_time)


        # Combine the results from the threads
        global_map = Dict{Tuple{Int,Int}, Int}()
        for tid = 1:nthreads()
            for ind in eachindex(local_rows[tid])
                j = local_rows[tid][ind]
                jj = local_cols[tid][ind]
                val = local_vals[tid][ind]
                key = (j, jj)
                if haskey(global_map, key)
                    vals[global_map[key]] += val
                else
                    push!(rows, j)
                    push!(cols, jj)
                    push!(vals, val)
                    global_map[key] = length(vals)
                end
            end
        end

        return sparse(rows,cols,vals, N_total, N_total)


    end
    
end


###############################################################################################################
#                                       Mass Matrix                          
#                               I.e. Discretization of (u,v) on all K in Ω
#   u Basis, v Testfunction, K Element, Ω Domain
###############################################################################################################

using Base.Threads

function assembleMassMatrix(uh::FE)

    N_total = sum(e.nDoF for e in uh.Elem)
    max_Dof = maximum(e.nDoF for e in uh.Elem)

    # Estimate total entries (upper bound)
    total_entries = sum(e.nDoF^2 for e in uh.Elem)
    
    # Preallocate buffers
    rows = Vector{Int}(undef, total_entries)
    cols = Vector{Int}(undef, total_entries)
    vals = Vector{Float64}(undef, total_entries)

    inds = zeros(Int, max_Dof)  # reused per thread
    entry_index = zeros(Int, nthreads())  # track insert position per thread

    # Thread-local storage for rows/cols/vals chunks
    local_rows = [Vector{Int}() for _ in 1:nthreads()]
    local_cols = [Vector{Int}() for _ in 1:nthreads()]
    local_vals = [Vector{Float64}() for _ in 1:nthreads()]


    assemly_time = Progress(uh.N; desc="Assembling Mass Matrix..")
    Threads.@threads for e = 1:uh.N
        tid = threadid()
        elem = uh.Elem[e]
        num = elem.nDoF
        local_inds = similar(inds, num)
        local_inds .= getElemDoF_ind(uh, e)
        M_loc = zeros(Float64, num, num)

        computeLocalMassMatrix!(uh, e, M_loc)

        for i = 1:num
            ii = local_inds[i]
            for j = 1:num
                jj = local_inds[j]
                push!(local_rows[tid], ii)
                push!(local_cols[tid], jj)
                push!(local_vals[tid], M_loc[i,j])
            end
        end
        next!(assemly_time)
    end
    finish!(assemly_time)

    # Combine thread-local results
    rows = vcat(local_rows...)
    cols = vcat(local_cols...)
    vals = vcat(local_vals...)

    return sparse(rows, cols, vals, N_total, N_total)
end


function computeLocalCellAdvection!(uh::FE, e::Integer, field::Vector{Float64}, Kc_loc::Matrix{Float64})

    Np = uh.Elem[e].nDoF
    eT = uh.Elem[e].eT
    deg = uh.Elem[e].deg

    J, JmT, detJ = getJacobian(view(uh.mesh.vertices,:,uh.mesh.j_cv[e,:]), eT)

    # Zero out first
    @inbounds for i = 1:Np
        @inbounds for j = 1:Np
            Kc_loc[i,j] = 0.0
        end
    end

    quad = uh.quad[eT]
    basis_eval = uh.Bases_Eval[[eT,deg]]
    gradbasis_eval = uh.GradBases_Eval[[eT,deg]]

    @inbounds for k = 1:quad.npts
        w_k = quad.w[k]
        detJ_k = detJ(quad.pts[k])
        JmT_k = JmT(quad.pts[k])

        # Test functions (VarForm, Rows)
        for i = 1:Np
            # Basis functions (VarForm, Columns)
            for j = 1:Np
                Kc_loc[i,j] += w_k * dot( detJ_k * JmT_k * gradbasis_eval[i,k],  field) * basis_eval[j,k] 
            end
        end
    end

    return nothing
end

function computeLocalFaceAdvection!(uh::FE, f::Integer, field::Vector{Float64}, 
                                    Kf_e1_e1::Matrix{Float64}, Kf_e1_e2::Matrix{Float64}, Kf_e2_e1::Matrix{Float64}, Kf_e2_e2::Matrix{Float64})

    (e1, f1), (e2, f2) = uh.mesh.face2elem[f]
    e_inds = [CartesianIndex(e1,f1); CartesianIndex(e2,f2)]

    # Get DoF Indices
    Np1 = uh.Elem[e1].nDoF
    Np2 = uh.Elem[e2].nDoF

    # Zero out first
    @inbounds for i = 1:maximum((Np1,Np2))
        @inbounds for j = 1:maximum((Np1,Np2))
            Kf_e1_e1[i,j] = 0.0
            Kf_e1_e2[i,j] = 0.0
            Kf_e2_e1[i,j] = 0.0
            Kf_e2_e2[i,j] = 0.0
        end
    end

    # Transformations
    J1, JmT1, detJ_e1 = getJacobian(view(uh.mesh.vertices,:,uh.mesh.j_cv[e1,:]),uh.Elem[e1].eT)
    J2, JmT2, detJ_e2 = getJacobian(view(uh.mesh.vertices,:,uh.mesh.j_cv[e2,:]),uh.Elem[e2].eT)

    # Face information
    ~, ~, ~, ~, ~, fTs1, ~, ~ = getElementType_Info(uh.Elem[e1].eT)
    ~, ~, ~, ~, ~, fTs2, ~, ~ = getElementType_Info(uh.Elem[e2].eT)

    # Determine which Face-Quadrature to use
    fquad = uh.fquad[fTs1[f1]].npts >= uh.fquad[fTs2[f2]].npts ? uh.fquad[fTs1[f1]] : uh.fquad[fTs2[f2]]

    # Get the reference outward normals and the face lengths
    ref_vertices_1, ref_faces_1 = getReferenceElem(uh.Elem[e1].eT)
    ref_vertices_2, ref_faces_2 = getReferenceElem(uh.Elem[e2].eT)

    if uh.Elem[e1].eT==1
        ref_face_length_1 = 1.0
    else
        ref_face_length_1 = norm(ref_faces_1[f1][1]-ref_faces_1[f1][2])
    end

    if uh.Elem[e2].eT==1
        ref_face_length_2 = 1.0
    else
        ref_face_length_2 = norm(ref_faces_2[f2][1]-ref_faces_2[f2][2])
    end


    # Get reference Outward normals
    nl1 = getOutwardNormal(ref_vertices_1, f1, uh.Elem[e1].eT)
    nl2 = getOutwardNormal(ref_vertices_2, f2, uh.Elem[e2].eT)

    # Local Coordinates and (global) face normals
    # According to: Ern, Finite Elements I, Chapter 9, Lemma 9.11
    z=[]
    if uh.Elem[e1].eT==1
        z =  [Interface2Local(0.0,e_inds,uh) for k=1:fquad.npts]
        n1 = [ (1/norm(JmT1(z[k])*nl1))*JmT1(z[k])*nl1 for k=1:fquad.npts]
        n2 = [ (1/norm(JmT2(z[k])*nl2))*JmT2(z[k])*nl2 for k=1:fquad.npts]
    else
        z =  [Interface2Local(0.5*(fquad.pts[k]+1),e_inds,uh) for k=1:fquad.npts]
        n1 = [ (1/norm(JmT1(z[k][1])*nl1[1:2]))*JmT1(z[k][1])*nl1[1:2] for k=1:fquad.npts]
        n2 = [ (1/norm(JmT2(z[k][2])*nl2[1:2]))*JmT2(z[k][2])*nl2[1:2] for k=1:fquad.npts]
    end

    # Scalar product with field
    # According to: Di Pietro, Mathematical Aspects of Discontinuous Galerkin Methods, Chapter 2
    sp_p = []
    sp_m = []
    if uh.Elem[e1].eT==1
        sp = [dot(field,n1[k]) for k=1:fquad.npts]
        sp_p = [(abs(sp[k])+sp[k])/2 for k=1:fquad.npts]
        sp_m = [(abs(sp[k])-sp[k])/2 for k=1:fquad.npts]
    else
        sp = [dot(field,n1[k][1:2]) for k=1:fquad.npts]
        sp_p = [(abs(sp[k])+sp[k])/2 for k=1:fquad.npts]
        sp_m = [(abs(sp[k])-sp[k])/2 for k=1:fquad.npts]
    end

    # Bases evaluation
    base_e1_ev = [uh.Elem[e1].Basis[i](z[k][1]) for i=1:Np1, k=1:fquad.npts]
    base_e2_ev = [uh.Elem[e2].Basis[i](z[k][2]) for i=1:Np2, k=1:fquad.npts]
    w = fquad.w

    # Face differentials from Geometric mapping of Cell
    # According to: Ern, Finite Elements I, Chapter 9, Lemma 9.12
    # Sign according to: Di Pietro, Mathematical Aspects of Discontinuous Galerkin Methods, Chapter 2.2.3, Numerical Fluxes
    ds_e1=[]
    ds_e2=[]

    if uh.Elem[e1].eT==1
        ds_e1 = [dot(n1[k],n1[k])*detJ_e1(z[k])*norm(JmT1(z[k])*nl1) * ref_face_length_1  for k=1:fquad.npts] #[dot(n1,n1)*lF for k=1:fquad.npts] 
        ds_e2 = [dot(n2[k],n1[k])*detJ_e2(z[k])*norm(JmT2(z[k])*nl2)* ref_face_length_2  for k=1:fquad.npts]
    else
        ds_e1 = [dot(n1[k],n1[k])*detJ_e1(z[k][1])*norm(JmT1(z[k][1])*nl1[1:2]) * ref_face_length_1  for k=1:fquad.npts] #[dot(n1,n1)*lF for k=1:fquad.npts] 
        ds_e2 = [dot(n2[k],n1[k])*detJ_e2(z[k][2])*norm(JmT2(z[k][2])*nl2[1:2]) * ref_face_length_2  for k=1:fquad.npts] #[dot(n2,n1)*lF for k=1:fquad.npts] # ds = detJ_e2 * norm(JmT2)*nl2 * ref_face_length_2 * (n1,n1)
    end


    @inbounds @simd for k in 1:fquad.npts

        # computing elem 1
        # Test functions (VarForm, Rows)
        @inbounds for i=1:Np1
            # Basis functions (um) (VarForm, Cols)
            @inbounds for ii=1:Np1
                Kf_e1_e1[i,ii] +=  w[k] * (sp_p[k] * base_e1_ev[ii,k]) * base_e1_ev[i,k] * ds_e1[k]
            end

            # Basis functions (up) (VarForm, Cols)
            @inbounds for ii=1:Np2
                Kf_e1_e2[i,ii] += w[k] * (-sp_m[k] * base_e2_ev[ii,k]) * base_e1_ev[i,k] * ds_e1[k]
            end  
        end 
        
        # computing elem 2
        # Test functions (VarForm, Rows)
        @inbounds for i=1:Np2
            # Basis functions (um) (VarForm, Cols)
            @inbounds for ii=1:Np1
                Kf_e2_e1[i,ii] += w[k] * (sp_p[k] * base_e1_ev[ii,k]) * base_e2_ev[i,k] * ds_e2[k]
            end  

            # Basis functions (up) (VarForm, Cols)
            @inbounds for ii=1:Np2
                Kf_e2_e2[i,ii] += w[k] * (-sp_m[k] * base_e2_ev[ii,k]) * base_e2_ev[i,k] * ds_e2[k]
            end   
        end
    end


end


function computeLocalMassMatrix!(uh::FE, e::Integer, M_loc::Matrix{Float64})
    Np = uh.Elem[e].nDoF
    eT = uh.Elem[e].eT
    deg = uh.Elem[e].deg

    ~, ~, detJ = getJacobian(view(uh.mesh.vertices,:,uh.mesh.j_cv[e,:]), eT)

    # Zero out first
    @inbounds for i = 1:Np
        @inbounds for j = 1:Np
            M_loc[i,j] = 0.0
        end
    end

    quad = uh.quad[eT]
    basis_eval = uh.Bases_Eval[[eT,deg]]

    @inbounds for k = 1:quad.npts
        w_k = quad.w[k]
        detJ_k = detJ(quad.pts[k])
        for i = 1:Np
            φi = basis_eval[i,k]
            for j = 1:Np
                M_loc[i,j] += w_k * φi * basis_eval[j,k] * detJ_k
            end
        end
    end

    return nothing
end

function prescribeBoundaryData(uh::FE, bd::Integer, d::Integer, data::Function; data_vec::Union{Vector{Float64},Nothing}=nothing, field::Union{Vector{Float64},Nothing}=nothing)

    N_total = sum(e.nDoF for e in uh.Elem)
    max_Dof = maximum(e.nDoF for e in uh.Elem)

    local_rows = [Vector{Int}() for _ in 1:nthreads()]
    local_vals = [Vector{Float64}() for _ in 1:nthreads()]
    localBd_data = [zeros(Float64,max_Dof) for _ in 1:nthreads()]

    # Check for the field
    if isnothing(field)
        field = ones(Float64,d)
    end

    Threads.@threads for f in uh.mesh.bd_cf[bd]  # Loop over the boundary faces
        tid = threadid()
        if !haskey(uh.mesh.face2elem,f)
            continue
        end

        # Compute Boundary contribution
        faceIntegral(uh, f, data, field, localBd_data[tid])

        # Element indexing
        (e,~), ~ = uh.mesh.face2elem[f]
        num = uh.Elem[e].nDoF
        local_inds = getElemDoF_ind(uh, e)

        # write to local indices
        for i = 1:num
            push!(local_rows[tid], local_inds[i])
            push!(local_vals[tid], localBd_data[tid][i])
        end
    end

    # Check for the field
    if isnothing(data_vec)
        data_vec = zeros(Float64,N_total)
    else
        data_vec .= 0.0
    end
    
    for tid=1:nthreads()
        for j in eachindex(local_vals[tid])
            data_vec[local_rows[tid][j]] += local_vals[tid][j]
        end
    end

    return data_vec

end


function faceIntegral(uh::FE, f::Integer, fun::Function, field::Vector{Float64}, Bd_vec::Vector{Float64})
    # element index and local face index
    (e, fl), ~ = uh.mesh.face2elem[f]
    Np = uh.Elem[e].nDoF

    # Zero out first
    @inbounds for i = 1:Np
        Bd_vec[i] = 0.0
    end

    # Transformations
    J, JmT, detJ = getJacobian(view(uh.mesh.vertices,:,uh.mesh.j_cv[e,:]),uh.Elem[e].eT)

    # Face information
    ~, ~, ~, ~, ~, fTs, ~, ~ = getElementType_Info(uh.Elem[e].eT)

    
    # Get the reference outward normals and the face lengths
    ref_vertices, ref_faces = getReferenceElem(uh.Elem[e].eT)
    if uh.Elem[e].eT==1
        ref_face_length = 1.0
    else
        ref_face_length = norm(ref_faces[fl][1]-ref_faces[fl][2])
    end
    nl = getOutwardNormal(ref_vertices, fl, uh.Elem[e].eT)

    # Quadrature
    fquad = uh.fquad[fTs[fl]]
    w = fquad.w

    # Local Coordinates and Normal
    z=[]
    n = []
    if uh.Elem[e].eT==1
        z =  [Face2Local(0.0,fl,uh.Elem[e].eT) for k=1:fquad.npts]
        n = [ (1/norm(JmT(z[k])*nl))*JmT(z[k])*nl for k=1:fquad.npts]
    else
        z =  [Face2Local(0.5*(fquad.pts[k]+1),fl,uh.Elem[e].eT) for k=1:fquad.npts]
        n = [ (1/norm(JmT(z[k])*nl[1:2]))*JmT(z[k])*nl[1:2] for k=1:fquad.npts]
    end

    # Scalar product with field
    sp_m = []
    if uh.Elem[e].eT==1
        sp = [dot(field,n[k]) for k=1:fquad.npts]
        sp_m = [(abs(sp[k])-sp[k])/2 for k=1:fquad.npts]
    else
        sp = [dot(field,n[k][1:2]) for k=1:fquad.npts]
        sp_m = [(abs(sp[k])-sp[k])/2 for k=1:fquad.npts]
    end

    # Function and Bases Evaluations
    fun_ev = [fun(Local2Global(z[k], view(uh.mesh.vertices,:,uh.mesh.j_cv[e,:]),uh.Elem[e].eT)) for k=1:fquad.npts]
    base_ev = [uh.Elem[e].Basis[i](z[k]) for i=1:Np, k=1:fquad.npts]

    # Differential
    ds=[]
    if uh.Elem[e].eT==1
        ds = [ detJ(z[k])*norm(JmT(z[k])*nl) * ref_face_length for k=1:fquad.npts]
    else
        ds = [ detJ(z[k])*norm(JmT(z[k])*nl[1:2]) * ref_face_length for k=1:fquad.npts]
    end
    # ds = [lF for k=1:fquad.npts] # ds = detJ * norm(JmT)*normal * ref_face_length

    # if sp < 0
        # computing elem 1
        @inbounds for k in 1:fquad.npts
            # computing elem 1
            # Test functions (VarForm, Rows)
            for i=1:Np
                Bd_vec[i] += w[k] * sp_m[k] *  fun_ev[k]*base_ev[i,k] * ds[k]
            end
        end
    # end
end


function assembleBoundaryMatrix(uh::FE, bd::Integer, d::Integer; field::Union{Vector{Float64},Nothing}=nothing)

    N_total = sum(e.nDoF for e in uh.Elem)
    max_Dof = maximum(e.nDoF for e in uh.Elem)

    local_rows = [Vector{Int}() for _ in 1:nthreads()]
    local_cols = [Vector{Int}() for _ in 1:nthreads()]
    local_vals = [Vector{Float64}() for _ in 1:nthreads()]

    Kf_bd = [zeros(Float64,max_Dof,max_Dof) for _ in 1:nthreads()]

    # Check for the field
    if isnothing(field)
        field = ones(Float64,d)
    end


    Threads.@threads for f in uh.mesh.bd_cf[bd]  # Loop over the boundary faces
        tid = threadid()
        if !haskey(uh.mesh.face2elem,f)
            continue
        end

        # Compute Boundary contribution
        computeLocalFaceAdvection_Bd!(uh, f, field, Kf_bd[tid])

        # Element indexing
        (e,~), ~ = uh.mesh.face2elem[f]
        num = uh.Elem[e].nDoF
        local_inds = getElemDoF_ind(uh, e)

        # write to local indices
        for i = 1:num
            ii = local_inds[i]
            for j = 1:num
                jj = local_inds[j]
                push!(local_rows[tid], ii)
                push!(local_cols[tid], jj)
                push!(local_vals[tid], Kf_bd[tid][i,j])
            end
        end
    end

    # Combine thread-local results
    rows = vcat(local_rows...)
    cols = vcat(local_cols...)
    vals = vcat(local_vals...)

    return sparse(rows, cols, vals, N_total, N_total)
end



function computeLocalFaceAdvection_Bd!(uh::FE, f::Integer, field::Vector{Float64}, Kf_bd::Matrix{Float64})

        # element index and local face index
        (e, fl), ~ = uh.mesh.face2elem[f]
        Np = uh.Elem[e].nDoF

        # Zero out first
        @inbounds for i = 1:Np
            @inbounds for j = 1:Np
                Kf_bd[i,j] = 0.0
            end
        end
        # Transformations
    ~, JmT, detJ = getJacobian(view(uh.mesh.vertices,:,uh.mesh.j_cv[e,:]),uh.Elem[e].eT)

    # Face information
    ~, ~, ~, ~, ~, fTs, ~, ~ = getElementType_Info(uh.Elem[e].eT)

    # Get the reference outward normals and the face lengths
    ref_vertices, ref_faces = getReferenceElem(uh.Elem[e].eT)

    if uh.Elem[e].eT==1
        ref_face_length = 1.0
    else
        ref_face_length = norm(ref_faces[fl][1]-ref_faces[fl][2])
    end
    nl = getOutwardNormal(ref_vertices, fl, uh.Elem[e].eT)

    # Quadrature
    fquad = uh.fquad[fTs[fl]]
    w = fquad.w

    # Local Coordinates and Normal
    z=[]
    n = []
    if uh.Elem[e].eT==1
        z =  [Face2Local(0.0,fl,uh.Elem[e].eT) for k=1:fquad.npts]
        n = [ (1/norm(JmT(z[k])*nl))*JmT(z[k])*nl for k=1:fquad.npts]
    else
        z =  [Face2Local(0.5*(fquad.pts[k]+1),fl,uh.Elem[e].eT) for k=1:fquad.npts]
        n = [ (1/norm(JmT(z[k])*nl[1:2]))*JmT(z[k])*nl[1:2] for k=1:fquad.npts]
    end

    # Scalar product with field
    sp_p = []
    if uh.Elem[e].eT==1
        sp = [dot(field,n[k]) for k=1:fquad.npts]
        sp_p = [(abs(sp[k])+sp[k])/2 for k=1:fquad.npts]
    else
        sp = [dot(field,n[k][1:2]) for k=1:fquad.npts]
        sp_p = [(abs(sp[k])+sp[k])/2 for k=1:fquad.npts]
    end

    # Bases Evaluations
    base_ev = [uh.Elem[e].Basis[i](z[k]) for i=1:Np, k=1:fquad.npts]

    # Differential
    ds=[]
    if uh.Elem[e].eT==1
        ds = [ detJ(z[k])*norm(JmT(z[k])*nl) * ref_face_length for k=1:fquad.npts]
    else
        ds = [ detJ(z[k])*norm(JmT(z[k])*nl[1:2]) * ref_face_length for k=1:fquad.npts]
    end

   @inbounds for k in 1:fquad.npts
        # computing elem 1
        # Test functions (VarForm, Rows)
        for i=1:Np
            # Basis functions (up) (VarForm, Cols)
            for ii=1:Np
                Kf_bd[i,ii] += w[k] * sp_p[k] *  base_ev[i,k] * base_ev[ii,k] * ds[k]
            end
        end
    end
end





###############################################################################################################
# ODE Solver
###############################################################################################################

# Strong Stability preserving RK3
function SSPRK3(Approx::FE, fun::Function, u::Array, dt::Float64, t_end::Float64; t0=0.0, history=false)#; limiting=false)

    # u = zeros(size(u0))
    # u .= u0 

    # u_all = u

    expected_iter = ceil(Int, (t_end-t0)/dt + 1)
    t = t0


    p_time = Progress(expected_iter; desc="Running time evolution...")
    while t <= t_end 

        if t+dt > t_end
            dt_hat = t_end-t
            stage = u + dt*fun(u,t)
            stage = (3/4)*u .+ (1/4)*stage .+ (dt_hat/4)*fun(stage,t+dt_hat)
            u = (1/3)*u .+ (2/3)*stage  .+ dt_hat*(2/3)*fun(stage,t+dt_hat/2)
            t += t_end
        else
            stage = u + dt*fun(u,t)
            stage = (3/4)*u .+ (1/4)*stage .+ (dt/4)*fun(stage,t+dt)
            u = (1/3)*u .+ (2/3)*stage  .+ dt*(2/3)*fun(stage,t+dt/2)
            t += dt
        end

        # if limiting
        #     setDoF!(Approx,u)
        #     u = limitSlopes!(Approx, u)
        # end

        if history
            u_all = hcat(u_all,u)
        end

        next!(p_time)
    end
    finish!(p_time)

    
    if history
        return u, u_all
    end

    return u
end

# FOur Stage SSPRK3
function SSPRK3_4(Approx::FE, fun::Function, u::Array, dt::Float64, t_end::Float64; t0=0.0, history=false)#; limiting=false)

    # u = zeros(size(u0))
    # u .= u0 

    # u_all = u

    expected_iter = ceil(Int, (t_end-t0)/dt + 1)
    t = t0


    p_time = Progress(expected_iter; desc="Running time evolution...")
    while t <= t_end 

        if t+dt > t_end
            t += (t+dt)-t_end
        else
            t += dt
        end


        k1 = (1/2)*(2u + dt*fun(u,t))
        k2 = (1/2)*(2k1 + dt*fun(k1,t+dt/2))
        k3 = (1/6)*(4u + 2k2 + dt*fun(k2,t+dt))
        u = (1/2)*(2k3 + dt*fun(k3,t+dt/2))

        # if limiting
        #     setDoF!(Approx,u)
        #     u = limitSlopes!(Approx, u)
        # end

        if history
            u_all = hcat(u_all,u)
        end

        next!(p_time)
    end
    finish!(p_time)

    
    if history
        return u, u_all
    end

    return u
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


# Classical RK4
function RK4(fun::Function, u0::Array, t::Array)#; limiting=false)


    u = u0
    u_all = u0

    for i in 1:length(t)-1
        h = t[i+1] - t[i]

        k1 = h*fun(u)
        k2 = h*fun(u + k1/2)
        k3 = h*fun(u + k2/2)
        k4 = h*fun(u + k3)
        u = u + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

        # if limiting
        #     setDoF!(Approx,u)
        #     u = limitSlopes!(Approx, u)
        # end

        u_all = hcat(u_all,u)
    end
end
