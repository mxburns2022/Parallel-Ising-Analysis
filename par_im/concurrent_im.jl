# using CUDA
using Pandas
using OptimalTransport
using SparseArrays
using Debugger
using LinearAlgebra
using Plots
using DelimitedFiles

struct Params
    dt::Float64
    size::Int
    replicas::Int
    epoch::Vector{Float64}
    beta::Vector{Float64}
end

struct SpinGlassProblem
    N::Int
    J::Array{Float64}
    h::Array{Float64}
    JExt::Array{Float64}
    JInt::Array{Float64}
    par::Params
end

function make_block(J::Array{Float64}, JInt::Array{Float64}, JExt::Array{Float64}, par::Params)

    nblocks = ceil(size(J)[1] / par.size)
    n_extra = size(J)[1] % par.size
    size_per_block = Int(floor(size(J)[1] / nblocks))
    low = 1
    for i in 1:nblocks
        bump = size_per_block
        if i <= n_extra
            bump += 1
        end
        indexrows = low:low+bump
        indexvalues = low:low+bump,low:low+bump
        JInt[indexvalues...] = J[indexvalues...]
        JExt[indexrows,:] = J[indexrows,:] - JInt[indexrows,:]
        low += bump
    end
    
end


function read_gset(path::String, par::Params)
    
    edgelist, header  = readdlm(path, 
                                ' ', 
                                Int32, 
                                '\n',
                                true, 
                                0,
                                true)
    nodes, edges = map(
        x -> parse(Int, x), header[1:2])
    J = fill(nodes, nodes)
    for (u, v, w) in edgelist

    end

    println(header)
    infile = open(path)
end
J = rand(10, 10)
J = triu(convert(Matrix{Float64}, J .>= .8))
J[diagind(J)] .= 0
println(size(J), J)
temp = (J + transpose(J))
J = temp
JInt = zeros(size(J))
JExt = zeros(size(J))
par = Params(1e-3,
             3,
             3,
             [1e-5],
             [1.0]
             )
make_block(J, JInt, JExt, par)
