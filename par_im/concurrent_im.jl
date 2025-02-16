using CUDA
using Pandas
using OptimalTransport


struct Params
    dt::Float64
    blocks::Int
    size::Int
    replicas::Int
    epoch::Vector{Float64}
    beta::Vector{Float64}
end

struct SpinGlassProblem
    N::Int
    J::CuArray{Float64}
    h::CuArray{Float64}
    JExt::CuArray{Float64}
    JInt::CuArray{Float64}
end


function do_step()
end
