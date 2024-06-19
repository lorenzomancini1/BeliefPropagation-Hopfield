module BP

using Random, Statistics, LinearAlgebra

function hopfield_sample(N, α)
    
    #Generate an Hopfield sample, ie the patterns ξ and the coupling matrix J obtained with the Hebbian rule
    
    M = round(Int, N * α)
    ξ = rand([-1,1], N, M)
    J = ξ * ξ'
    J[diagind(J)] .= 0
    return ξ, J ./ N
end

function perturb(σ::AbstractVector, p)
    
    #Function that perturbs a configuration σ by flipping all its element with probability p
    
    N = length(σ)
    σ_new = copy(σ)
    for i in 1:N
        if rand() < p
            σ_new[i] *= -1
        end
    end
    return σ_new
end

function overlap(σ1::AbstractVector, σ2::AbstractVector)
    # Function that computes the overlap between two configurations
    return σ1 ⋅ σ2 / length(σ1)
end

function init_messages(σ, m)
    # Function that initializes the h messages (I should ask to the professor the initial condition
    N = length(σ)
    #println(N)
    h = zeros(N, N)
    for i in 1:N
        h[i, :] .= atanh(m * σ[i])
    end
    h[diagind(h)] .= 0
    
    return h   
end

function f(J, h, β)
    return atanh( tanh(β*J) * tanh( β*h) ) / β
end


function update_node_msg!(h, J, i, β)
    N = size(J, 1)
    h_t = h'
    hi = sum( f(J[i, k], h_t[i, k], β) for k in 1:N) # here maybe the implementation is suboptimal
    for j in 1:N
        h[i, j] = hi - f(J[j, i], h[j, i], β)
    end
end

function update_messages!(h, J, β)
    #h_copy = deepcopy(h)
    N = size(J, 1)
    Threads.@threads for i in 1:N
        update_node_msg!(h, J, i, β)
    end
    h[diagind(h)] .= 0
    #return h_copy
end

function H(i, J, h, β)
    N = size(J, 1)
    Hi = sum( f(J[i, k], h[k, i], β) for k in 1:N )
    return Hi
end

function marginal(σ, H, β)
    return exp(σ * H * β)
end

#function node_belief(i, ν)
#    # get the incoming messages to the variable node i
#    inc_msg = ν[:, i]
#    deleteat!(inc_msg, i)
#    
#    bel = 1
#    for a in eachindex(inc_msg)
#        bel *= inc_msg[a]
#    end
#    
#    bel /= sum(bel)
#    
#    return bel   
#end

function run_bp(σ, J, β, p; maxiter = 200)
    
    N = length(σ)
    m = 1 - 2*p
    h = init_messages(σ, m)
    h_copy = deepcopy(h)
    diff = 1 # keep track of the changes related to the messages matrix
    mags = zeros(N) # vector that will be filled with magnetizations
    iter = 0

    while diff > 0 && iter <= maxiter
        update_messages!(h, J, β)
        diff = norm(h_copy - h) / N^2 
        h_copy .= h
        iter += 1
    end
    
    
    #compute marginals
    # compute directly mags (tanh(\beta * Hi)) --->
    for i in eachindex(σ)
        mags[i] = tanh(β * H(i, J, h, β))
    end
    return mags
end

end

