
@doc """ SimulationSettings(
            circuit_type::String, qubits::Integer, depth::Integer, initial_state::String, 
            noise_model::String, noise_rate::Real, unraveling_type::String, 
            n_samples::Integer, 
            gates::Union{Vector, Matrix}, initial_MPS::Vector{<:Array{<:Number, 3}}, 
            circuit_details::Union{Nothing, Dict}, unraveling_details::Union{Nothing, Dict}
            )

    Data type containing the information about the settings of the simulation 
    (simulated circuit, noise process, unraveling choise, etc). 
    - circuit_type: type of circuit to be simulated (e.g. "random circuit", 
    "random Hamiltonian", "user Hamiltonian", "Heisenberg Hamiltonian")
    - qubits: number of qubits (width)
    - depth: number of layers
    - initial_state: type of state to initialize the simulation with
    - noise_model: type of single-qubit noise applied (e.g. "depolarizing")
    - noise_rate: rate of the noise process
    - unraveling_type: choice of unraveling strategy (e.g. "typical", "projective", "wootters")
    - n_samples: number of samples computed for the simulation
    - max_bond_dim: maximum bond dimension above which each bond is truncated
    - gates: sequence of gates in the circuit (automatically constructed)
    - initial_MPS: MPS representation of the initial state (automatically constructed)
    - circuit_details: further information on the simulated algorithm/circuit/Hamiltonian. 
        circuit_details = Dict(
            :fixed => true/false,
            :low_entangling => true/false,
            :entangling_angle => 0.05,                  # (some float)
            :local_rotations => true/false, 
            :Hamiltonian_unitary => zeros(4,4),         # (some 4x4 unitary matrix)
            :Hamiltonian_matrix => I(4),                # (some 4x4 Hermitian matrix)
            :time_step => 0.05,                         # (some small float)
            :coupling_fields => [1, 1, 1],
            :local_fields => [0, 0, 0], 
            :renormalize => true/false
        )
        Not all fields are required for each circuit type. 
        "random circuit" requires {fixed, low_entangling, entangling_angle (if low_entangling = true) and local_rotations}.
        "random Hamiltonian" requires {fixed, time_step, local_rotations}
        "user Hamiltonian" requires {Hamiltonian_unitary} or {Hamiltonian_matrix and time_step}
        "Heisenberg Hamiltonian" requires {coupling_fields, local_fields, time_step}
        renormalize is to be considered if max_bond_dim!=nothing
    - unraveling_details: further information for the unraveling (needed for 
    optimized unraveling, not implemented yet)
"""
struct SimulationSettings
    circuit_type::String
    qubits::Int64
    depth::Int64
    initial_state::String
    noise_model::String
    noise_rate::Float64
    unraveling_type::String
    n_samples::Int64
    max_bond_dim::Union{Nothing, Int64}
    gates::Union{Vector, Matrix}            # make a clear type out of that?
    initial_MPS::Vector{Array{ComplexF64, 3}}
    noise_tensor::Array{ComplexF64, 3}
    monitoring_quantities::Vector{String}
    circuit_details::Dict
    unraveling_details::Dict
end
SimulationSettings(
    circuit_type::String, n_qubits::Int64, depth::Int64, initial_state::String, 
    noise_model::String, noise_rate::Float64, unraveling_type::String, 
    n_samples::Int64, max_bond_dim::Union{Nothing, Int64}=nothing,
    monitoring_quantities::Vector{String}=["entropy"],
    circuit_details::Dict=Dict(), unraveling_details::Dict=Dict()
) = SimulationSettings(
    circuit_type, n_qubits, depth, initial_state, 
    noise_model, noise_rate, unraveling_type, 
    n_samples, max_bond_dim,
    generate_algorithm_gates(circuit_type, n_qubits, depth, n_samples; circuit_details...),
    generate_MPS(n_qubits, initial_state), 
    get_noise_tensor(noise_model, unraveling_type, noise_rate),
    monitoring_quantities,
    circuit_details, unraveling_details
)

