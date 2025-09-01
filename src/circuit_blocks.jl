# kwargs is intended to be a dictionary containing some (the relevant ones for
# the given simulation) of the fileds below. Some are given default values if 
# not user-specified (e.g. time_step, entangling_angle, local_rotations), 
# others will give errors if missing 
# (e.g. n_samples for circuit="random Hamiltonian" with fixed=true)
# details = Dict(
#     :fixed => false,
#     :n_samples => 0,
#     :entangling_angle => 0.05,
#     :local_rotations => false, 
#     :Hamiltonian_unitary => zeros(4,4),
#     :Hamiltonian_matrix => I(4), 
#     :time_step => 0.05, 
#     :coupling_fields => [1, 1, 1],
#     :local_fields => [0, 0, 0]
# )

identity_gate = reshape(diagm(0=>(1.0+0im)*ones(4)),2,2,2,2)

@doc """ generate_algorithm_gates(circuit_type, n_qubits, depth, fixed, details)

    Function to generate the set of gates (or not) used in the simulation, as 
    Julia arrays. For fixed circuits with different gates, it eturns a vector of
    matrices (of for each gate). For fixed Hamiltonian circuits, it returns a 
    single matrix (the one gate used repeatedly). For randomized circuits, it 
    returns an empty vector (as the new gate has to be generated for each sample).

    Considered circuit types: 
    - "random circuit": brickwork circuit of haar random 2-qubit gates
    - "low entangling random circuit": same but with random rates with small rotation angle
    - "Hamiltonian": single fixed 2-qubit gate used for all the circuit. Either random or structured
    Additional keywords:
    - with_local_rotation: adding 1-qubit Haar random gates after the 2-qubit gates
    - fixed: fixing one circuit for all samples of the simulation
"""
function generate_algorithm_gates(circuit_type::String, n_qubits::Int64, depth::Int64, n_samples::Int64; kwargs...)
    if occursin("circuit", circuit_type)
        fixed = get(kwargs, :fixed, false)
        if !fixed
            return []
        else
            return generate_random_brickwork_circuit(n_qubits, depth; kwargs...)
        end
    elseif occursin("Hamiltonian", circuit_type)
        return create_Hamiltonian_gate(circuit_type, n_samples; kwargs...)
    elseif circuit_type == "only Bell"
        throw(ArgumentError("only Bell circuits not yet implemented"))
    else
        throw(ArgumentError("Undefined circuit type"))
    end
end

@doc """ generate_random_brickwork_circuit(n_qubits::Int64, depth::Int64; kwargs...)

    Generate the gates of the random brickwork circuit (Haar random, low-
    entangling, with or without local rotations, ... depending on the kwargs).
    Returns a list of lists of 4x4 unitary matrices (for each layer, for each 
    target qubit)
"""
function generate_random_brickwork_circuit(n_qubits::Int64, depth::Int64; kwargs...)
    full_circuit = Vector{Vector{Matrix{ComplexF64}}}()
    for l in 1:depth
        single_layer = Vector{Matrix{ComplexF64}}()
        # first_qubit = mod1(l, 2)
        # for qubit in first_qubit:2:(n_qubits-1)
        #     push!(single_layer, random_2qubit_gate(; kwargs...))
        # end
        for qubit in 1:(n_qubits-1)
            push!(single_layer, random_2qubit_gate(; kwargs...))
        end
        push!(full_circuit, single_layer)
    end
    return full_circuit
end

@doc """ create_Hamiltonian_gate(circuit_type::String; kwargs...)

    Create the (single) unitary matrix corresponding to the trotterized 
    evolution of a local Hamiltonian (resulting in a circuit where the gate
    is always the same)
"""
function create_Hamiltonian_gate(circuit_type::String, n_samples::Int64; kwargs...)
    if occursin("random", circuit_type)
        fixed = get(kwargs, :fixed, false)
        entangling_angle = get(kwargs, :entangling_angle, 0.05)
        local_rotations = get(kwargs, :local_rotations, false)
        if fixed
            return low_entangling_matrix(; entangling_angle=entangling_angle, with_local_rot=local_rotations)
        else
            return [low_entangling_matrix(; entangling_angle=entangling_angle, with_local_rot=local_rotations) for _ in 1:n_samples]
        end
    elseif occursin("user", circuit_type)
        if haskey(kwargs, :Hamiltonian_unitary)
            # @assert is_a_unitary(details["Hamiltonian unitary"])
            return get(kwargs, :Hamiltonian_unitary, zeros(4,4))
        elseif haskey(kwargs, :Hamiltonian_matrix)
            H = get(kwargs, :Hamiltonian_matrix, zeros(4,4))
            # @assert ishermitian(H)
            dt = get(kwargs, :time_step, 0.05)
            return exp(-1im * H * dt)
        else
            throw(ArgumentError("Need to specify Hamiltonian_matrix or Hamiltonian_unitary in the kwargs to use the user-defined Hamiltonian"))
        end
    elseif occursin("Heisenberg", circuit_type)
        coupling_generators = [kron(PauliX, PauliX), kron(PauliY, PauliY), kron(PauliZ, PauliZ)] 
        local_generators = [kron(I(2), PauliX) + kron(PauliX, I(2)), kron(I(2), PauliY) + kron(PauliY, I(2)), kron(I(2), PauliZ) + kron(PauliZ, I(2))]
        if !haskey(kwargs, :coupling_fields) || !haskey(kwargs, :local_fields)
            println("WARNING: some Hamiltonian parameters not defined by user. Using default values")
        end
        coupling_fields = get(kwargs, :coupling_fields, [1, 1, 1])
        local_fields = get(kwargs, :local_fields, [0, 0, 0])
        Hamiltonian_term = coupling_fields' * coupling_generators + (1/2) * local_fields' * local_generators
        # @assert size(Hamiltonian_term) == (4,4)                 # 2-site unitary evolution
        # @assert ishermitian(Hamiltonian_term) 
        time_step = get(kwargs, :time_step, .05)
        # @assert time_step >= 0
        return exp(-1im * Hamiltonian_term * time_step)
        # @assert is_a_unitary(U)
    else
        throw(ArgumentError("Undefined Hamiltonian. Specify a name (e.g. Heisenberg, user, random) and the necessary settings"))
    end
end

@doc """ get_gate(settings::SimulationSettings, layer::Int64=0, qubit::Int64=0, sample::Int64=0)

    Get or generate the right gate to be applied on the target qubit, on the specified layer, for the given sample, 
    depending on the type of simulated circuit
"""
function get_gate(settings::SimulationSettings, layer::Int64=0, qubit::Int64=0, sample::Int64=0)
    if occursin("random circuit", circuit_type)
        fixed = get(settings.circuit_details, :fixed, false)
        if fixed
            # gate = settings.gates[layer][div(qubit+1, 2)]
            gate = settings.gates[layer][qubit]
        else
            gate = random_2qubit_gate(;settings.circuit_details...)
        end
    elseif occursin("Hamiltonian", circuit_type)
        if !occursin("random", circuit_type)    # Heisenberg or user-defined Hamiltonian
            gate = settings.gates
        else
            fixed = get(settings.circuit_details, :fixed, false)
            if fixed                            # randomly generated Hamiltonian but re-used for all samples
                gate = settings.gates
            else                                # new random Hamiltonian for each sample
                if sample == 0
                    throw(ArgumentError("Need sample number to select the gate of non-fixed random Hamiltonian simulation"))
                else
                    gate = settings.gates[sample]
                end
            end
        end
    else
        throw(ArgumentError("Undefined circuit type"))
    end
    return reshape(gate, 2,2,2,2) 
    # TODO check the need for permuting dimensions
end

@doc """ generate_MPS(n_qubits::Int64, state::String="0")

    Generate an MPS state on n_qubits qubits in a specific state. 
    Implemented initial states are 
    the all zero state ("ground" or "0"), 
    the all one state ("excited" or "1"), 
    the all plus state ("plus" or "+"), 
    the all minus state ("minus" or "-"), 
    the all right state ("right" or "R"), 
    the all left state ("left" or "L"), 
    the alternating zero and one ("alternate computational" or "0101"), 
    a sequence of Bell pairs ("Bell"), 
    magic states on each qubit ("magic" or "T"), 
    local random states on each qubit ("local random").
"""
function generate_MPS(n_qubits::Int64, state::String="0")::Vector{Array{ComplexF64, 3}}
    if state=="ground" || state=="0"
        return [reshape([1.0+0im,0], 2,1,1) for _ in 1:n_qubits]
    elseif state=="excited" || state=="1"
        return [reshape([0,1.0+0im], 2,1,1) for _ in 1:n_qubits]
    elseif state=="plus" || state=="+"
        return [reshape([1.0+0im,1.0+0im]./sqrt(2), 2,1,1) for _ in 1:n_qubits]
    elseif state=="minus" || state=="-"
        return [reshape([1.0+0im,-1.0+0im]./sqrt(2), 2,1,1) for _ in 1:n_qubits]
    elseif state=="right" || state=="R"
        return [reshape([1.0+0im,1.0im]./sqrt(2), 2,1,1) for _ in 1:n_qubits]
    elseif state=="left" || state=="L"
        return [reshape([1.0+0im,-1.0im]./sqrt(2), 2,1,1) for _ in 1:n_qubits]
    elseif state=="alternate computational" || state=="0101"
        return [isodd(q) ? reshape([1.0+0im,0], 2,1,1) : reshape([0,1.0+0im], 2,1,1) for q in 1:n_qubits]
    elseif state=="Bell"
        A = reshape([1, 0, 0, 1], 2,1,2)/sqrt(2)
        B = reshape([1, 0, 0, 1], 2,2,1)
        MPS = [X for _ in 1:div(8,2) for X in (A, B)]
        if n_qubits%2 == 1
            append!(MPS, reshape([1.0+0im,1.0+0im]/sqrt(2), 2,1,1))
        end
        return MPS
    elseif state=="magic" || state=="T"
        beta = acos(1/sqrt(3))
        T = reshape([cos(beta/2)+0im, exp(1im*pi/4)*sin(beta/2)], 2,1,1)
        return [T for _ in 1:n_qubits]
    elseif state=="local random"
        return[reshape(Haar_random_qubit_state(1), 2,1,1) for _ in 1:n_qubits]
    elseif state=="Haar"
        # vec = Haar_random_qubit_state(n_qubits)
        throw(ArgumentError("Haar random initial state is still being implemented"))
    else
        throw(ArgumentError("Undefined initial state"))
    end
end

