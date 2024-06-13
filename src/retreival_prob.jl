include("./bp_hopfield.jl")
using Plots, DelimitedFiles, Statistics
using Profile, ProfileView

function retreive(N::Int64, α::Float64, nsamples::Int64, β::Int64, p::Float64; m0 = 0.8)
    
    # Function that runs the BP algorithm to retreive the initial pattern. It iterates over a number of samples
    # and each time it generates a random pattern, perturbs it and then runs the BP algorithm to retreive the initial pattern.
    # Finally it returns the probability of success and the average magnetization of the final patterns
    
    
    mi, mf = zeros(nsamples), zeros(nsamples)
    #probs, mags = zeros(nsamples), zeros(nsamples)
    #Threads.@threads for sample in 1:nsamples
    for sample in 1:nsamples
        # generate the Hopfield samples
        ξ, J = BP.hopfield_sample(N, α)
        # take a random sample and perturb it with spin-flip probability p
        k = rand(1:size(ξ, 2))
        σ = ξ[:, k]
        σ_pert = BP.perturb(σ, p)
        
        # compute initial magnetization
        mi[sample] = BP.overlap(σ, σ_pert)

        # run bp to retreive the initial pattern
        σ_rec = BP.run_bp(σ_pert, J, β, p)

        # compute final magnetization
        mf[sample] = BP.overlap(σ, σ_rec)
    end

    success = mf .>= m0
    prob = mean(success)
    mag  = mean(mf)
    prob_error = std(success) / sqrt(nsamples)
    mag_error = std(mf) / sqrt(nsamples)
    return prob, mag, prob_error, mag_error
end

function run_retreive(N::Int64, α::Float64, nsamples::Int64, β::Int64, pp::AbstractVector;
     m0 = 0.8, show = false, save = false)

    # Function that calls the retreive function for a range of p values and saves the results in a file
    np = length(pp)

    probs, probs_errors, mags, merrors = zeros(np), zeros(np), zeros(np), zeros(np)

    for i in 1:np
        p = pp[i]
        prob, mag, prob_error, mag_error = retreive(N, α, nsamples, β, p; m0 = m0)
        probs[i] = prob
        probs_errors[i] = prob_error
        mags[i] = mag
        merrors[i] = mag_error
    end

    show && plotf(N, α, pp, probs)
    save && savedata(N, α, hcat(pp, probs, probs_errors, mags, merrors))
end

function plotf(N::Int, α::Float64, pp::AbstractVector, probs::AbstractVector)
    fig = plot(pp, probs, size = (500,300), markershape =:circle, label = "N = $N, α = $α",
                    xlabel = "p", ylabel = "probs") 
    display(fig)
    return nothing
end

function savedata(N::Int, α::Float64, data::AbstractMatrix; dir = "./src/data")

    folder = replace(string(α),"." => "" )
    path = dir*"/alpha_"*folder

    if isdir(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            writedlm(io, data)
        end
    else
        mkpath(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            writedlm(io, data)
        end
    end
    return nothing
end

function main()
    
    NN = [100, 200, 500, 700, 1000, 1500]
    nsamples = [2000, 1500, 800, 500, 200]
    α = 0.1
    println("------------- α = $α ----------------")
    β = 10
    pp = range(0.15, stop=0.5, length=20)

    for i in eachindex(NN)
        println("--------------------")
        println("Running simulation for N = $(NN[i])")
        println("...")
        N = NN[i]
        ns = nsamples[i]
        run_retreive(N, α, ns, β, pp; save = false, show = true)
        println("Simulation for N = $(NN[i]) completed")
    end 
end


