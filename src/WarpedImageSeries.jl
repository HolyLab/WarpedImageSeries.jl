module WarpedImageSeries

#using Images, BlockRegistration, Interpolations
using Images, RegisterDeformation, Interpolations, MappedArrays

import Base: getindex, size, show
export WarpedSeriesView, warpedseries

struct WarpedSeriesView{TO,N,TI} <: AbstractArray{TO,N}
    img::AbstractArray{TI,N}
    tfms #can be anything that warpedview() accepts
    views #output of warpedview(img, tfm[i]) for each i
    shared_idxs #intersection of all sets of warped indices
end

#default to Float64 output precision
WarpedSeriesView(img::AbstractArray{T,N}, tfms, views, shared_idxs) where {T,N} =
    WarpedSeriesView{Float64,T,N}(img, tfms, views, shared_idxs)

#This is conservative in that we only include in the view pixels that are
#common to all images post-warping.
#Note: this index intersection may cause problems for general non-affine deformations;
#only tested on rigid deformations so far
function warpedseries(img::AbstractArray{T,N}, tfms, precision=Float64) where {T,N}
    @assert length(tfms) == size(img, N) #assume series is along last dimension
    sh_idxs = axes(img)
    vs = []
    colons = (fill(Colon(), N-1)...,)
    for i=1:length(tfms)
        v = warpedview(mappedarray(precision, view(img, colons..., i)), tfms[i])
        sh_idxs = ([intersect(x, y) for (x,y) in zip(sh_idxs, axes(v))]...,)
        push!(vs, v)
    end
    return WarpedSeriesView{precision,N,T}(img, tfms, vs, sh_idxs)
end

show(io::IO, A::T) where {T<:WarpedSeriesView} = print(io, "WarpedSeriesView of size $(size(A)), eltype $(eltype(A))\n")
show(io::IO, ::MIME"text/plain", A::T) where {T<:WarpedSeriesView} = show(io, A)

size(A::WarpedSeriesView{T,N}) where {T,N} = (map(length, A.shared_idxs)...,size(A.img, N))
getindex(A::WarpedSeriesView, inds...) = inner_index(A, last(inds), A.shared_idxs, Base.front(inds)...)
getindex(A::WarpedSeriesView, inds::CartesianIndex) = getindex(A, inds.I...) #inner_index(A, last(inds), A.shared_idxs, Base.front(inds)...)

inner_index(A::WarpedSeriesView, series_inds::T, inds_safe, inds_user...) where {T<:Union{Int, CartesianIndex}} = A.views[series_inds][calc_inds(inds_safe, inds_user...)...]

#when multiple stacks are included
function inner_index(A::WarpedSeriesView{T,N}, series_inds, inds_safe, inds_user...) where {T,N}
    if isa(series_inds, Colon)
        series_inds = last(axes(A.img))
    end
    prealloc = zeros(T, Base.front(_idx_shape(A, (inds_user...,2)))..., length(series_inds))
    inner_index!(prealloc, A, series_inds, inds_safe, inds_user...)
end

#Note: this is about half as efficient as it could be
function inner_index!(prealloc::AbstractArray{TA,N}, A::WarpedSeriesView{TA,N}, series_inds::T, inds_safe, inds_user...) where {TA, N, T<:AbstractUnitRange{Int}}
    colons = (fill(Colon(), N-1)...,) #performance hit?
    for (i, idx) in enumerate(series_inds)
        prealloc[colons...,i] =  A.views[idx][calc_inds(inds_safe, inds_user...)...]
    end
    return prealloc
end

calc_inds(inds_safe, inds_user...) = (first(inds_safe)[first(inds_user)], calc_inds(Base.tail(inds_safe), Base.tail(inds_user)...)...)
calc_inds(inds_safe::Tuple{}, inds_user::Tuple{}...) = ()
calc_inds(inds_safe, inds_user::Tuple{}...) = error("Too few indices passed to getindex")
calc_inds(inds_safe::Tuple{}, inds_user...) = begin @show inds_user; error("Too many indices passed to getindex") end

#note: duplicated from CachedSeries.jl
_idxs(A, dim, idxs) = idxs
_idxs(A, dim, idxs::Colon) = axes(A,dim)
function _idx_shape(A::AbstractArray, idxs)
    lens = zeros(Int, ndims(A))
    for i = 1:ndims(A)
        lens[i] = length(_idxs(A, i, idxs[i]))
    end
    return (lens...,)
end

end # module
