using LinearAlgebra
using Einsum
using Distributions
using PyPlot
import PyPlot as plt

include("matrix_stuff.jl")
include("structs.jl")
include("circuit_blocks.jl")
include("generic_noise.jl")
include("wootters_decomposition.jl")
include("cost_functions.jl")
include("data_management.jl")




@doc """ apply_unitary!(myMPS,H,ind,left_to_right,chi_max)

    Application of a layer of unitary gates to a MPS
"""
function apply_unitary!(
        myMPS::Vector{Array{ComplexF64, 3}}, 
        H::Array{ComplexF64, 4},
        ind::Int64, 
        chi_max::Union{Int64, Nothing}; 
        right_canonical_form::Bool=true, 
        renormalize::Bool=false
        )

    # select the two affected tensors from the MPS
    A = myMPS[ind]               # dimensions (d,D_{ind-1},D_{ind})
    B = myMPS[ind+1]             # dimensions (d,D_{ind},D_{ind+1})
    # if size(A)[3] > 1
    #     @assert is_isometry(reshape(A, prod(size(A)[[1,2]]), size(A)[3])) || is_isometry(reshape(B, prod(size(B)[2]), prod(size(B)[[1,3]])))
    # end

    # Contract the two MPS tensors together
    @einsum AB[d1,d2,Dl,Dr] := A[d1,Dl,D] * B[d2,D,Dr]
    @assert norm(AB) <= 1 + sqrt(eps()) "norm(AB) = $(norm(AB)) > 1 + sqrt(eps())"

    # Contract the joint tensor with the unitary gate
    @einsum HAB[d1,Dl,d2,Dr] := AB[d1in,d2in,Dl,Dr] * H[d1in,d2in,d1,d2]    # dimensions (d,D_{ind},d,D_{ind+1})
    dl,Dl,dr,Dr = size(HAB)
    # @show norm(A), norm(B), norm(AB), norm(HAB)

    # Compute the MPS tensors through SVD
    # Reshape dim-4 tensor into matrix
    HAB_matrix = reshape(HAB,prod(size(HAB)[1:2]),prod(size(HAB)[3:4]))     # dimensions (dl*Dl, dr*Dr)
    @assert !any(isnan, HAB_matrix)
    F = SVD{ComplexF64, Float64, Matrix{ComplexF64}}
    try
        F = svd(HAB_matrix)
        # F=svd(mat,alg=LinearAlgebra.DivideAndConquer())
    catch e
        F = svd(HAB_matrix, alg=LinearAlgebra.QRIteration())
        @show cond(HAB_matrix) 
    end
    u, s, v = F
    v = conj.(v)
    # @show norm(u), norm(v)
    # @show size(u), size(v)

    # Choose right or left canonical form
    if right_canonical_form #right or left normal form
        v = v * diagm(0=>s)     # dimensions (dr*Dr,D=min(d*Dl,dr*Dlr))
        @assert is_isometry(u)
    else
        u = u * diagm(0=>s)     # dimensions (dl,Dl,D=min(d*Dl,dr*Dlr))
        @assert is_isometry(v)
    end

    if isnothing(chi_max)
        # Shape back into dim-3 tensors
        u = reshape(u, dl,Dl,size(u)[2])            # dimensions (dl,Dl,D)
        v = reshape(v, dr,Dr,size(v)[2])            # dimensions (dr,Dr,D)
        v = permutedims(v,[1,3,2])                  # dimensions (dr,D,Dr)
    else
        # Truncate and shape back into dim-3 tensors
        u = reshape(u[:,1:min(chi_max,size(u)[2])], dl,Dl,min(chi_max,size(u)[2]))  # dimensions (dl,Dl,chi_max)
        v = reshape(v[:,1:min(chi_max,size(v)[2])], dr,Dr,min(chi_max,size(v)[2]))  # dimensions (dr,Dr,chi_max)
        v = permutedims(v, [1,3,2])                                             # dimensions (dr,chi_max,Dr)
        if renormalize
            truncated_norm = norm(s[1:min(chi_max,length(s))])
            if right_canonical_form #right or left normal form
                v /= truncated_norm
            else
                u /= truncated_norm
            end
        end
    end
    # @show norm(u), norm(v)
    # @show size(u), size(v)

    # Update the tensors of the MPS
    myMPS[ind] = u
    myMPS[ind+1] = v
    # return norm(s[size(u)[3]+1:end])/norm(s)
    return s
end

@doc """ apply_noise!(myMPS,K,ind,do_wootters=false)

    Application of a layer of noise processes
"""
function apply_noise!(
        myMPS::Vector{Array{ComplexF64, 3}}, 
        K::Array{ComplexF64, 3},
        ind::Int64,
        do_wootters::Bool=false
        )

    # Target tensor from the MPS
    A = myMPS[ind]                                              # dimensions (d,Dl,Dr)
    # Computing the effective 2-qubit state
    # dim-3 tensor to matrix for SVD
    A_matrix = reshape(A, size(A)[1], prod(size(A)[2:3]))       # dimensions (d,Dl*Dr)
    F = SVD{ComplexF64, Float64, Matrix{ComplexF64}}
    try
        F = svd(A_matrix)
        # F=svd(mat,alg=LinearAlgebra.DivideAndConquer())
    catch e
        F = svd(A_matrix, alg=LinearAlgebra.QRIteration())
    end
    u, s, v = F
    # effective 2-qubit state as a matrix
    t = u * diagm(0=>s)                                         # dimensions (d, dred)
    # Apply the noise tensor
    @einsum tk[dout,dred,k] := K[dout,din,k] * t[din,dred]
    # Optimize using Wootters' method
    if do_wootters
        tkp = try 
            wootters(tk; do_checks=true)
        catch e 
            println("Wootters failed: ")
            println(e) 
            wootters(tk; do_checks=false)
        end
    else 
        tkp = tk
    end
    # Compute the probability distribution of the paths
    pi = [norm(tkp[:,:,k])^2 for k in 1:size(tkp)[3]]
    pi = pi./sum(pi)
    if any(isnan, pi)
        @show tkp, K
    end
    # @show pi, sum(pi)
    # Sample one path
    kch = argmax(rand(Multinomial(1,pi)))
    tf = tkp[:,:,kch]/sqrt(pi[kch])             # dimensions (d, dred)
    # Contract the sampled 2-qubit state back into the MPS
    KA = reshape(tf*v',size(A)...)              # dimensions (d,Dl,Dr)
    myMPS[ind] = KA
end

@doc """ trajectory_sampling(settings)

    Application of a full circuit of unitaries + noise to sample a single 
    trajectory of the Monte Carlo simulation
"""
function trajectory_sampling(sample::Int64, data::Dict{String, Matrix{Float64}}, settings::SimulationSettings)
    
    n = settings.qubits
    # state = deepcopy(settings.initial_MPS)
    state = generate_MPS(n, settings.initial_state)
    chi_max = settings.max_bond_dim
    n_layers = settings.depth
    K = settings.noise_tensor
    do_wootters = (settings.unraveling_type == "wootters")
    renormalize = get(settings.circuit_details, :renormalize, false)

    for l in 1:n_layers                  # each layer
        for oe in 0:1                   # odd and even sub-layers
            # layer of unitary gates
            for i in 1:n-1              # each qubit
                if i%2 == 1-oe          # odd layer  (oe==0) => odd qubits
                                        # even layer (oe==1) => even qubits
                    gate = get_gate(settings, l, i, sample)
                    svals = apply_unitary!(state, gate, i, chi_max; right_canonical_form=true)
                    # r[l] += norm(svals[chi_max+1:end])/norm(svals)          # accumulate error
                    if !isnothing(chi_max)
                        data["truncation"][sample, l+1] += norm(svals[chi_max+1:end])/norm(svals)       # accumulate error
                    end
                # shift canonical center
                elseif i%2 == oe        # second qubits of each unitary
                    # shift the canonical center by applying an identity gate
                    svals = apply_unitary!(state, identity_gate, i, chi_max; right_canonical_form=true, renormalize=renormalize) #/!\ set chi_max to nothing/infinity?
                    # error should be 0, since we applied an identity gate?
                    if !isnothing(chi_max)
                        er = norm(svals[chi_max+1:end])/norm(svals)
                        @assert er < sqrt(eps()) "truncation error after identity $(er) > sqrt(eps())"
                    end
                end
            end
            # layer of noise processes
            for i in n:-1:1
                apply_noise!(state, K, i, do_wootters)
                if i>1
                    svals = apply_unitary!(state, identity_gate, i-1, chi_max; right_canonical_form=false)
                    if !isnothing(chi_max)
                        er = norm(svals[chi_max+1:end])/norm(svals)
                        @assert er < sqrt(eps()) "truncation error after identity $(er) > sqrt(eps())"
                    end
                end
                if (i == div(n,2,RoundUp)+1) && (oe == 1)
                    for quantity in settings.monitoring_quantities
                        if quantity == "entropy" || occursin("Renyi-", quantity)
                        # if !occursin("truncation", quantity)
                            data[quantity][sample, l+1] = compute_quantity(quantity, svals)
                        elseif occursin("purity", quantity)
                            single_site_tensor = state[i-1]
                            single_site_matrix = reshape(single_site_tensor, size(single_site_tensor)[1], prod(size(single_site_tensor)[2:3]))
                            data[quantity][sample, l+1] = compute_quantity(quantity, svdvals(single_site_matrix))
                        end
                    end
                end
            end
        end
    end
    # return state, r #returns the sum of the truncation errors
end

