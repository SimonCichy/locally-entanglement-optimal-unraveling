
@doc """ canonical_kraus(noise_model::String, noise_rate::Float64)

    Creation of the canonical Kraus tensor for some known noise models 
    (depolarizing, dephasing, amplitude damping). Decomposition obtained from 
    the diagonailziation of the Choi state, resulting in an orthogonal Kraus 
    decomposition
"""
function canonical_kraus(noise_model::String, noise_rate::Float64)
    d_physical = 2
    if noise_model == "depolarizing" || noise_model == "DP"
        p = noise_rate
        d_Kraus = 4
        Kraus_array = zeros(ComplexF64, d_physical, d_physical, d_Kraus)
        Kraus_array[:,:,1] = sqrt(1-3p/4) * [1+0im 0; 0 1]
        Kraus_array[:,:,2] = sqrt(p/4) * [0 1+0im; 1 0]
        Kraus_array[:,:,3] = sqrt(p/4) * [0 -1im; 1im 0]
        Kraus_array[:,:,4] = sqrt(p/4) * [1+0im 0; 0 -1]
    elseif noise_model == "dephasing" || noise_model == "DF"
        p = noise_rate
        d_Kraus = 2
        Kraus_array = zeros(ComplexF64, d_physical, d_physical, d_Kraus)
        Kraus_array[:,:,1] = sqrt(1-p/2) * [1+0im 0; 0 1]
        Kraus_array[:,:,2] = sqrt(p/2) * [1+0im 0; 0 -1]
    elseif noise_model == "amplitude damping" || noise_model == "AD"
        gamma = noise_rate
        d_Kraus = 2
        Kraus_array = zeros(ComplexF64, d_physical, d_physical, d_Kraus)
        Kraus_array[:,:,1] = [1+0im 0; 0 sqrt(1-gamma)]
        Kraus_array[:,:,2] = [0 sqrt(gamma)+0im; 0 0]
    # elseif noise_model == "jump 1->0"
    #     return canonical_kraus("AD", 1-exp(-noise_rate))
    # elseif noise_model == "jump proj-0"
    #     return canonical_kraus("DF", 1-exp(-noise_rate/2))
    else
        throw(ArgumentError("Undefined noise model"))
    end
    return Kraus_array
end

@doc """ projective_kraus(noise_model::String, noise_rate::Float64)

    Creation of the projective Kraus tensor for some known noise models 
    (depolarizing, dephasing). Decomposition in terms of projectors (on the 
    computational basis)
"""
function projective_kraus(noise_model::String, noise_rate::Float64)
    d_physical = 2
    if noise_model == "depolarizing" || noise_model == "DP"
        p = noise_rate
        d_Kraus = 5
        Kraus_array = zeros(ComplexF64, d_physical, d_physical, d_Kraus)
        Kraus_array[:,:,1] = sqrt(1-p) * [1+0im 0; 0 1]
        Kraus_array[:,:,2] = sqrt(p/2) * [1+0im 0; 0 0]
        Kraus_array[:,:,3] = sqrt(p/2) * [0 1+0im; 0 0]
        Kraus_array[:,:,4] = sqrt(p/2) * [0 0; 1+0im 0]
        Kraus_array[:,:,5] = sqrt(p/2) * [0 0; 0 1+0im]
    elseif noise_model == "dephasing" || noise_model == "DF"
        p = noise_rate
        d_Kraus = 3
        Kraus_array = zeros(ComplexF64, d_physical, d_physical, d_Kraus)
        Kraus_array[:,:,1] = sqrt(1-p) * [1+0im 0; 0 1]
        Kraus_array[:,:,2] = sqrt(p) * [1+0im 0; 0 0]
        Kraus_array[:,:,3] = sqrt(p) * [0 0; 0 1+0im]
    else
        throw(ArgumentError("No projective decomposition defined for the specified noise model"))
    end
    return Kraus_array
end

@doc """ rotated_kraus(noise_model::String, noise_rate::Float64)

    Creation of the rotated Kraus tensor for some known noise models 
    (depolarizing, dephasing, amplitude damping). Decomposition obtained from 
    applying a Hadamard matrix on the typical/canonical decomposition.
"""
function rotated_kraus(noise_model::String, noise_rate::Float64)
    K = canonical_kraus(noise_model, noise_rate)
    d_physical, _, d_Kraus = size(K)
    K_rotated = zeros(ComplexF64, d_physical, d_physical, d_Kraus)
    if d_Kraus == 2
        H = (1/sqrt(2)) * [1 1; 1 -1]
    elseif d_Kraus == 4
        H = (1/2) * [1 1 1 1; 1 1 -1 -1; 1 -1 1 -1; 1 -1 -1 1]
    else
        throw(ArgumentError("No rotated unraveling defined for Kraus rank different from 2 or 4"))
    end
    @einsum K_rotated[din,dout,dkout] := K[din,dout,dk] * H[dkout,dk]
    return K_rotated
end

channel_rank(K::Array{ComplexF64, 3}) = size(K)[3]

function minimal_kraus_rank(noise_model::String)
    if noise_model == "depolarizing" || noise_model == "DP"
        return 4
    elseif noise_model == "dephasing" || noise_model == "DF"
        return 2
    elseif noise_model == "amplitude damping" || noise_model == "AD"
        return 2
    # elseif noise_model == "jump 1->0"
    #     return noise_minimal_kraus("AD")
    # elseif noise_model == "jump proj-0"
    #     return noise_minimal_kraus("DF")
    else
        throw(ArgumentError("Undefined noise model"))
    end
end

function get_noise_tensor(noise_model::String, unraveling_type::String, noise_rate::Float64)
    if unraveling_type == "typical" || unraveling_type == "canonical" || unraveling_type == "orthogonal"
        return canonical_kraus(noise_model, noise_rate)
    elseif unraveling_type == "projective"
        return projective_kraus(noise_model, noise_rate)
    elseif unraveling_type == "rotated"
        return rotated_kraus(noise_model, noise_rate)
    elseif unraveling_type == "wootters"
        # will be optimized later during the simulation
        return canonical_kraus(noise_model, noise_rate)
    elseif unraveling_type == "optimized"
        throw(ArgumentError("optimization of the unraveling through unitary gradient descent not implemented"))
    else
        throw(ArgumentError("Undefined unraveling type"))
    end 
end