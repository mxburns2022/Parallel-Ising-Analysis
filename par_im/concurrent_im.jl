# using CUDA
using Pandas
using StochasticOptimalTransport
using Debugger
using LinearAlgebra
using DelimitedFiles
# using DifferentialEquations, ProgressLogging
using Statistics
using Clp
using PythonOT


mutable struct Params
    dt::Float64
    size::Int
    replicas::Int
    epoch::Vector{Float64}
    beta::Vector{Float64}
    tstop::Float64
    R::Float64
    C::Float64
    function Params(
           replicas::Int = 1, 
           size::Int = 500, 
           tstop::Float64 = 1e-6,
           epoch::Float64 = 1e-9,
           beta::Float64 = 1.0)
        return new(1e-12, size, replicas, [epoch], [beta], tstop, 310e3, 50e-15)
    end
end

struct SpinGlassProblem
    N::Int
    M::Int
    J::Array{Float64}
    h::Array{Float64}
    JExt::Array{Float64}
    JInt::Array{Float64}
    par::Params
end

function make_block(J::Array{Float64}, JInt::Array{Float64}, JExt::Array{Float64}, par::Params)
    nblocks = 1
    n_extra = 0
    size_per_block = size(J)[1]
    if par.size < size(J)[1]
        nblocks = ceil(size(J)[1] / par.size)
        n_extra = size(J)[1] % par.size
        size_per_block = Int(floor(size(J)[1] / nblocks))
    end
    
    low = 1
    for i in 1:nblocks
        bump = size_per_block - 1
        if i <= n_extra
            bump += 1
        end
        indexrows = low:low+bump
        indexvalues = low:low+bump,low:low+bump
        JInt[indexvalues...] = J[indexvalues...]
        JExt[indexrows,:] = J[indexrows,:] - JInt[indexrows,:]
        low += (bump + 1)
    end
end


function read_gset(path::String, par::Params)
    
    # Read header and edgelist
    edgelist, header  = readdlm(path, 
                                ' ', 
                                Float64, 
                                '\n',
                                header=true, 
                                skipblanks=true)
    # Account for linear biases in the file
    nlin = 0
    if size(header)[1] == 3 && header[3] != ""
        nodes, nlin, edges = map(x->parse(Int, x), header)
    else 
        nodes, edges = map(x->parse(Int, x), header[1:2])
    end
    offset = 0
    if contains(path, ".gset") || contains(path, ".ising")
        offset = 1
    end

    J = fill(0.0, nodes, nodes)
    h = fill(0.0, nodes)

    # Read the linear terms
    for edge in edgelist[1:nlin, :]
        ui = Int(edge[1]) + offset
        h[ui] = -edge[2]
    end
    # Read the quadratic terms
    for index in nlin+1:edges
        edge = edgelist[index,:]
        ui = Int(edge[1]) + offset
        vi = Int(edge[2]) + offset
        J[ui, vi] = -edge[3]
        J[vi, ui] = -edge[3]
    end
    JInt = fill(0.0, size(J))
    JExt = fill(0.0, size(J))
    
    make_block(J, JInt, JExt, par)
    prob = SpinGlassProblem(
        nodes, edges, J, h, JExt, JInt, par
    )
    return prob
end

function run_brim(prob::SpinGlassProblem, par::Params)
    t = 0.0
    x = sign.(randn(prob.N, par.replicas))
    x0 = copy(x)
    xideal = copy(x)
    function brim_gradient!(du, u, p, t)
        du[1:end,:] = (prob.JInt * u + prob.JExt * x0) / (par.R * par.C)
    end
    function brim_gradient_full!(du, u, p, t)
        du[1:end,:] = (prob.J * u) / (par.R * par.C)
    end
    function σ!(du, u, p, t)
        du[1:end,:] = sqrt(2 * par.dt/(p * par.R * par.C)) .* randn(size(u))
    end
    grad = fill(0.0, size(x))
    grad_ideal = fill(0.0, size(x))
    noise = fill(0.0, size(x))
    samples_ideal = []
    samples_concurrent = []
    grad_error = []
    grad_difference = []
    while (t < par.tstop) 
        copy!(x0, x)
        copy!(xideal, x)
        τ = 0
        β0 = 0.5
        β1 = par.beta[1]
        βStep = (β1 - β0) / (ceil(par.epoch[1] / par.dt))
        β = β0
        grad_error_local = fill(0.0, size(x))
        grad_difference_local = fill(0.0, size(x))
        while τ < par.epoch[1]
            # print(x)
            grad_error_local += prob.JExt * (x - x0) * par.dt / (par.R * par.C)
            grad_difference_local += prob.J * (x - xideal) * par.dt / (par.R * par.C)
            brim_gradient!(grad, x, 0, t+τ)
            σ!(noise, x, β, t+τ)
            x += grad .* par.dt + noise
            x[x .>= 1.] .= 1.
            x[x .<= -1.] .= -1.
            brim_gradient_full!(grad_ideal, xideal, 0, t+τ)
            xideal += grad_ideal .* par.dt + noise
            xideal[xideal .>= 1.] .= 1.
            xideal[xideal .<= -1.] .= -1.
            τ += par.dt
            β += βStep
        end
        t += par.epoch[1]
        grad_error
        push!(grad_error, mean(abs.(sum(grad_error_local, dims=1))))
        push!(grad_difference, mean(abs.(sum(grad_difference_local, dims=1))))
        push!(samples_ideal, xideal)
        push!(samples_concurrent, x)
    end
    return (samples_ideal, samples_concurrent, grad_error, grad_difference)
end

function bit_to_integer(arr)
    arr = reverse(arr)
    sum(((i, x),) -> Int(x) << ((i-1) * sizeof(x)), enumerate(arr.chunks))
end

function integer_to_bit(arr)
    arr = reverse(arr)
    sum(((i, x),) -> Int(x) << ((i-1) * sizeof(x)), enumerate(arr.chunks))
end


function compute_distance_matrix(N)
    distance = fill(0.0, 2^N, 2^N)
    for i in 0:2^N-2
        for j in i+1:2^N-1
            overlap = (i ⊻ j)
            hamming_distance = sum(overlap & 2^j > 0 for j in 0:N-1)
            distance[i+1, j+1] = hamming_distance
            distance[j+1, i+1] = hamming_distance
        end
    end
    return distance
end


function get_empirical_distribution(samples)
    samplecount = size(samples)[1]
    N, reps = size(samples[1])
    nsamples = samplecount * reps
    distribution = fill(0.0, 2^N)
    increment = 1.0 / nsamples
    for i in 1:samplecount
        for rep in 1:reps
            bitvec = (samples[i][:, rep] .> 0)
            distribution[bit_to_integer(bitvec)+1] += increment
        end
    end
    return distribution
end

function get_gibbs_distribution(prob::SpinGlassProblem)
    distribution = fill(0.0, 2^prob.N)
    for i in 1:2^prob.N
        bits = BitArray((i-1) & 2^j > 0 for j in 0:prob.N-1)
        spins = convert(Vector{Float64}, bits) .* 2 .- 1
        ene = -0.5 * (spins' * prob.J * spins)
        distribution[i] = exp(-prob.par.beta[1]*ene)
    end
    distribution /= sum(distribution)
    return distribution

end
localARGS = isdefined(Main, :newARGS) ? newARGS : ARGS
for blocks in [2, 3, 4, 6]
    for epoch in 10.0.^(range(-10,stop=-7,length=30))
    # epoch = 1e-7
        par = Params(3000, blocks, epoch, epoch, 10.0)
        # if !isdefined(Main, :ideal) || (isdefined(Main, :repeat) && repeat)
        prob = read_gset(localARGS[1], par)
        ideal, conc, err, diff = run_brim(prob, par)
        repeat = false;
        # end
        μ = get_empirical_distribution(conc)
        ν = get_empirical_distribution(ideal)
        η = get_gibbs_distribution(prob)
        C = compute_distance_matrix(prob.N)
        w1_νη = emd2(ν, η, C)
        w1_νμ = emd2(ν, μ, C)
        w1_μη = emd2(μ, η, C)
        println("$(blocks),$(epoch),$(w1_νη),$(w1_νμ),$(w1_μη),$(err[1]),$(diff[1])")
    end
end
# M = compute_distance_matrix(prob.N)
# dist = emd2(μ, ν, hamming_distance, max_iter=10_000_000, rtol=1e-6)