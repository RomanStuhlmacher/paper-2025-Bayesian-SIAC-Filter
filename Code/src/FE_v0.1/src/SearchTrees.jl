
###############################################################################################################
#                                               SearchTreeRelated Things
###############################################################################################################

mutable struct SearchTree{dim}
    Nodes::SMatrix{dim,2}
    Childs::Union{Vector{SearchTree{dim}},Nothing}
    ElemList::Array{Int}
    numElem::Int
end



function InitSearchtree(dim::Int, vertices::Matrix{Float64}, NC::Int)

    if dim==1 # LineTree
    
        xmin = minimum(vertices)
        xmax = maximum(vertices)

        Nodes = SMatrix{dim,2}(xmin, xmax)

        ElemList = collect(1:NC)

        return SearchTree{1}(Nodes, nothing, ElemList, NC)

    elseif dim==2 # QuadTree

        xmin = minimum(vertices)
        xmax = maximum(vertices)

        ymin = minimum(vertices)
        ymax = maximum(vertices)

        Nodes = SMatrix{dim,2}(xmin, ymin, xmax, ymax)

        ElemList = collect(1:NC)

        return SearchTree{2}(Nodes, nothing, ElemList, NC)
    end
end

function spawnChild(QT::SearchTree, vertices::Matrix{Float64}, j_cv::Matrix{Int}, ChildInd::Int64)
    if typeof(QT) == SearchTree{1} # LineTree
        # Child Indexing
        #       1       2    
        #  |--------|--------|
    
        xmin = QT.Nodes[1]
        xmax = QT.Nodes[2]
    
        if ChildInd==1
            xmax = (xmin+xmax)/2
        elseif ChildInd==2
            xmin = (xmin+xmax)/2
        end
       Nodes = SMatrix{1,2}(xmin, xmax)
    
        ElemList = []
        for i in QT.ElemList
           verts = view(vertices,1,j_cv[i,:])
   
           if (Nodes[1]<=verts[1])&&(Nodes[2]>=verts[1]) || (Nodes[1]<=verts[2])&&(Nodes[2]>=verts[2])
               push!(ElemList,i)
           end
        end
    
        return SearchTree{1}(Nodes,nothing,ElemList, length(ElemList))

    elseif typeof(QT) == SearchTree{2} # QuadTree
        # Child Indexing
        #   -----------------
        #  |       |        |
        #  |   1   |   2    |
        #  |_______|________|
        #  |       |        |
        #  |   3   |    4   |
        #  |_______|________|
    
        xmin = QT.Nodes[1]
        xmax = QT.Nodes[2]
        ymin = QT.Nodes[3]
        ymax = QT.Nodes[4]
    
        if ChildInd==1
            xmax = (xmin+xmax)/2
            ymin = (ymin+ymax)/2
        elseif ChildInd==2
            xmin = (xmin+xmax)/2
            ymin = (ymin+ymax)/2
        elseif ChildInd==3
            xmax = (xmin+xmax)/2
            ymax = (ymin+ymax)/2
        elseif ChildInd==4
            xmin = (xmin+xmax)/2
            ymax = (ymin+ymax)/2
        end
       Nodes = SMatrix{2,2}(xmin, ymin, xmax, ymax)
        # Lines = [[Nodes[:,1] Nodes[:,2]]; [Nodes[:,2] Nodes[:,3]]; [Nodes[:,3] Nodes[:,4]]; [Nodes[:,4] Nodes[:,1]]]
    
        ElemList = []
        for i in QT.ElemList
           verts = view(vertices,1:2,j_cv[i,:])
   
           if doesIntersect(verts,Nodes)#any( isinside(x, Nodes) for x in eachcol(verts))  
           # if !isnothing(regions)
               push!(ElemList,i)
           end
        end
    
        return SearchTree{2}(Nodes,nothing,ElemList, length(ElemList))
    end
end


function BuildSearchtree(QT::SearchTree, vertices::Matrix{Float64}, j_cv::Matrix{Int}; lv=1, maxLvls=4, ElemPerCell=50)

    QT.Childs = if typeof(QT) == SearchTree{1}
        Vector{SearchTree{1}}(undef,2)
    elseif typeof(QT) == SearchTree{2}
        Vector{SearchTree{2}}(undef,4)
    elseif typeof(QT) == SearchTree{3}
        Vector{SearchTree{3}}(undef,8)
    end
    
     for i in eachindex(QT.Childs) 
        QT.Childs[i] =  spawnChild(QT,vertices, j_cv,i)

        if (lv < maxLvls)&&(QT.Childs[i].numElem > ElemPerCell)
            BuildSearchtree(QT.Childs[i], vertices, j_cv, lv=lv+1, maxLvls=maxLvls, ElemPerCell=ElemPerCell)
        end
    end
end

# Checks if two polygons intersect (or contain another)
function doesIntersect(polygon1::Any,polygon2::Any)::Bool

    # check if polygon1 is inside polygon2
    for v in eachcol(polygon1)
        if isinside(v,polygon2)
            return true
        end
    end

    # check if polygon2 is inside polygon1
    for v in eachcol(polygon2)
        if isinside(v,polygon1)
            return true
        end
    end

    edges1 = hcat(polygon1,polygon1[:,1])
    edges2 = hcat(polygon2,polygon2[:,1])

    # check if edges of polygon intersect quad edges
    for q=1:size(polygon2,2)
        for p=1:size(polygon1,2)
            if LinesIntersect(edges2[:,q:q+1],edges1[:,p:p+1])
                return true
            end
        end
    end

    return false
end  

function LinesIntersect(L1, L2)::Bool
    # Li given as Li = [P1_i P2_i] = [x1 x2;
    #                                 y1 y2]
    # Parmetrizing Line 1
    # L1 : r*P2_1 + (1-r)*P1_1, r ∈ [0,1]

    # Parmetrizing Line 2
    # L2 : s*P2_2 + (1-s)*P1_2, s ∈ [0,1]

    # Intersection if (*):
    #   r*P2_1 + (1-r)*P1_1 = s*P2_2 + (1-s)*P1_2
    # 
    #   <=> (P2_1-P1_1)*r + (P1_2-P2_2)*s = (P1_2-P1_1)  && r,s ∈ [0,1]
    #   (If r,s ∉ [0,1], the Lines intersect outside the segment)

    A = [(L1[:,2]-L1[:,1]) (L2[:,1]-L2[:,2])]
    
    # Checks if lines are parallel
    if det(A) == 0
        return false
    else
        rs = A\(L2[:,1]-L1[:,1])

        # Checks if Line Intersection is on segment (described in (*))
        if all((rs .>= 0)) && all((rs .<= 1))
            return true
        end
    end

    # Lines don't intersect in segment
    return false
    
end

