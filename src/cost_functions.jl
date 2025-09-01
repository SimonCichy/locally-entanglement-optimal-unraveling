""" alpha_entropy(A::Matrix, alpha::Float64)

    Compute the (Renyi-alpha) entropy entropy of (the singular values) of a matrix
    (or the bond of an MPS). Von Neumann entropy for alpha = 1, Renyi entropy otherwise.
"""
function alpha_entropy(A::Matrix, alpha::Float64)
    s = svdvals(A)
    if length(findall(i -> i<0, s)) > 0
        println("WARNING: some negative singular values:", s[findall(i -> i<0, s)])
        @assert max(abs.(findall(i -> i<0, s))) < sqrt(eps()) "can't compute entropy due to large negative singular values"
        s = s[findall(i -> i>0, s)]         # pick only the non-zero ones
    end
    return alpha_entropy(s.^2, alpha)
end

""" alpha_entropy(p::Vector{<:Float64}, alpha::Float64)

    Compute the (Renyi-alpha) entropy of the probability distribution p. It is the 
    Shanon entropy for alpha=1.
"""
function alpha_entropy(p::Vector{<:Float64}, alpha::Float64=1.)
    if any(isnan, p)
        throw(ArgumentError("NaN values in the probability distribution"))
    end
    if length(findall(i -> i<0, p)) > 0
        println("WARNING: some negative probabilities:", s[findall(i -> i<0, p)])
        @assert max(abs.(findall(i -> i<0, p))) < sqrt(eps()) "can't compute entropy due to large negative probabilities"
    end
    p = p[findall(i -> i>0, p)]             # pick only the non-zero ones
    N2 = sum(p)                             # probability of the path
    if N2 < sqrt(eps())                     # how bad of an approximation is that?
        return 0
    else
        if alpha == 1
            E = - sum(p .* log2.(p))
            entropy = E + N2 * log2(N2)         # = E for normalized distributions
        else
            N_alpha_squared = sum(p.^alpha)
            entropy = (N2/(1-alpha)) * (log2(N_alpha_squared)-log2(N2^(alpha)))
        end
        @assert !isnan(entropy)
        return entropy
    end
end

""" compute_quantity(quantity::String, singular_values::Vector{<:Float64})

    Compute the quantity of interest (entropy, Renyi, truncation) based on singular values.
"""
function compute_quantity(quantity::String, singular_values::Vector{<:Float64})
    prob_dist = singular_values.^2 / sum(singular_values.^2)
    if quantity == "entropy"
        if isnan(alpha_entropy(prob_dist, 1.))
            @show singular_values
            throw(ArgumentError("NaN entropy"))
        end
        return alpha_entropy(prob_dist, 1.)
    elseif occursin("Renyi-", quantity)
        alpha = parse(Float64, quantity[7:end])
        return bond_entropy(prob_dist, alpha)
    elseif occursin("truncation", quantity)
        # max_bond_dim = parse(Int64, cost_type[12:end])
        # return bond_trunctation_error(singular_values, max_bond_dim; with_root=false)
        throw(ArgumentError("truncation to be implemented as independent quantity to compute"))
    elseif occursin("purity", quantity)
        return sum(singular_values.^2)
    else
        throw(ArgumentError("Undefined monitored quantity"))
    end
end
