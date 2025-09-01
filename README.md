# Locally entanglement-optimal unraveling

Code base for the simulations of the publication [Cichy et al. (2025)](https://arxiv.org/abs/2508.05745),
*Classical simulation of noisy quantum circuits via locally entanglement-optimal unravelings*.


## Recap of the paper (simulation method)

Unraveling of quantum open system evolutions into locally entanglement-optimal trajectories:
1. Noise processes (quantum channels) can be interpreted as stochastic mixtures of (non-unitary) pure state evolutions
2. One can sample these pure states (*trajectories*) to estimate properties of interest (expectation values, samples from the computational basis): this is the *Monte Carlo wave function* approach (or jump method)
3. Channels have non-unique representations, and one can leverage the unitary degree of freedom do generate favorable unravelings into trajectories (ones with reduced simulation cost)
4. If representing each pure state with MPS, one can use the unitary freedom to minimize entanglement
5. Minimizing entanglement (computing entanglement of formation) is a hard problem, but we can solve it exactly for single-qubit noise channels
6. We use this "optimal" unraveling to simulated open quantum systems (noisy circuits and Lindbladian evolutions)


## How to use the code


### Example

```julia
# Including necessary sources and dependencies
include("src/unraveling_simulation.jl")
using Printf
using ProgressBars

begin
    # Settings of the simulation 
    do_save = true 
    do_plot = !do_save 
    circuit_type = "Heisenberg Hamiltonian"  
    n_qubits = 16 
    depth = 600
    initial_state = "0"
    chi_max = 20
    n_samples = 20
    noise_model = "AD" 
    noise_rates = [0.005, 0.01]
    unravelings = ["typical", "rotated", "wootters"]
    monitoring_quantities=["truncation", "entropy"]
    global circuit_details = Dict(
        :time_step => 0.05, 
        :coupling_fields => [0, 1, 0],
        :local_fields => [0.35, 0.35, 0.5],
        :renormalize => true
    )
    date = 20250628
    global counter = get_smallest_counter(date, 1)

    do_plot && (r = [])

    for unraveling in unravelings               # Repeating the whole simulation for all unravlings of the list
        @time for noise_rate in noise_rates     # Repeating the simulation for all noise rates
            println("Sampling of "*circuit_type*" with "*noise_model*" at rate $(noise_rate) under "*unraveling*" unraveling")
            simulation_settings = SimulationSettings(
                circuit_type, n_qubits, depth, initial_state, 
                noise_model, noise_rate, unraveling, 
                n_samples, chi_max, 
                monitoring_quantities, circuit_details
            )

            filename = do_save ? "path/to/file/location/$(date)_unraveling_$(counter).jld2" : nothing
            @show filename
            create_empty_file(filename, simulation_settings)            # Empty file to save the data

            simulation_outcomes = Dict{String, Matrix{Float64}}()       # Empty dictionnary for temporary saving of the data
            for quantity in monitoring_quantities
                simulation_outcomes[quantity] = zeros(Float64, n_samples, depth+1) # Empty arrays of the right dimension
            end

            iter = ProgressBar(1:n_samples, unit=" samples")
            timing = @timed for sample in iter  # Repeating for each sample
                set_description(iter, noise_model*", "*string(@sprintf("gamma = %.1e, ", noise_rate))*unraveling)

                # Computation of one trajectory
                trajectory_sampling(sample, simulation_outcomes, simulation_settings)
            end

            do_save && save_data(filename, simulation_outcomes)
            do_save && save_data(filename, timing, "runtime")
            do_plot && push!(r, vec(mean(simulation_outcomes["truncation"], dims=1)))
            
            global counter += 1
        end

        if do_plot
            plt.figure()
            [plt.plot(r, label=noise_rates[i]) for (i,r) in enumerate(r)]
            plt.xlabel("rounds")
            plt.ylabel("total error")
            plt.legend()
        end
    end
end
```


**Explanation of the code**

The main part of the code happens in the line 
```julia
trajectory_sampling(sample, simulation_outcomes, simulation_settings)
```
which calls the main routing to generate a sample tranjectory and corresponding monitored quantities. 
The rest of the file is there to define the settings of the simulation or to save/plot the results

After including the necessary sources and dependencies, the settings of the simulation are defined
- `do_save = true` : Whether to store the simulation results or only plot them
- `do_plot = !do_save` : If not saving, generate plots at the end
- `circuit_type = "Heisenberg Hamiltonian"` : Type of circuit to simulated. To be chosed from `"random circuit"`, `"user Hamiltonian"`, `"Heisenberg Hamiltonian"`
- `n_qubits = 16` : Number of qubits
- `depth = 600` : Depth of the circuit (number of layers)
- `initial_state = "0"` : Initial state of the *n* qubits (see `generate_MPS` in `circuit_blocks` for possible choices)
- `chi_max = 20` : Maximum bond dimension for the simulation (may be an integer or `nothing` for exact simulation)
- `n_samples = 20` : Number of samples generated and then averaged over to extract the final quantities
- `noise_model = "AD"` : Noise model (currently supported: depolarizing `"DP"`, dephasing `"DF"` and amplitude damping `"AD"`)
- `noise_rates = [0.005, 0.01]` : List of noise rates to simulate (floats between 0 and 1)
- `unravelings = ["typical", "rotated", "wootters"]` : List of unravelings to be simulated to be compated
- `monitoring_quantities = ["truncation", "entropy"]` : List of quantities to save/plot (see also `compute_quantity` in `cost_functions` for a list of supported quantities)
- `global circuit_details = Dict(...)` : Dictionnary of details of the construction of the circuit (see `SimulationSettings` in `structs` for the details of how to set it up for different kinds of circuits)
- `date = 20250628` : Setting the date for the automatic naming of the files where data is stored
- `global counter = get_smallest_counter(date, 1)` : Automatically determining the number of the file to save the data


**Plotting from saved files**

When saving the simulation results in files for later processing, they can be represented using the functions in `src/data_management`. 
For instance, a set of 24 files containing the results of simulations for 8 noise rates for 3 different unravelings can be plotted with 
```julia
plot_comparison_vs_noise(["path/to/file/location/yymmdd_unraveling_$k.jld2" for k in 1:24]; n_lines=8)
```
Alternatively, data from different the simulations of 3 unravelings for 4 different bond dimensions can be plotted with
```julia
plot_comparison_vs_bonddim(["path/to/file/location/yymmdd_unraveling_$k.jld2" for k in 1:12]; n_lines=4)
```
Both methods result in a set of plots where the different monitored quantities are plotted with respect to depth and with respect to noise/bond dimension.


### Explanation: how samples are computed

One simulation (one run) is built together from the simulation of several samples (trajectories):
- each sample is one randomly sampled trajectory of the noisy circuit,
- circuits have a local brickwork structure on *n* qubits with *L* layers,
- layers are composed of non-overlapping 2-body nearest-neighbour unitaries (drawn from different possible ensembles) composed with single-qubit noise channels,
- the noise processes are unraveled on each qubit greedily (in a sweeping fashion) by computing the Kraus decomposition giving the entanglement of formation and sampling one Kraus operator to apply.


## Dependencies

Code written in [Julia](https://julialang.org/) 
Tensor operations based on [Einsum](https://github.com/ahwillia/Einsum.jl)
Saving/storing based on [JLD2](https://github.com/JuliaIO/JLD2.jl)

Self-written code base:
- `unraveling_simulation` : Main subroutines of the simulation (application of unitary gates, application of noise processes, and the combination thereof into the full simulation of one trajectory)
- `structs` : Definition of the new data type `SimulationSettings` containing all the information of the simulation settings 
- `circuit_blocks` : Functions to define the circuit to simulate (gates to be applied and initial state)
- `generic_noise` : Functions to define the noise tensors
- `wootters_decomposition` : Set of functions necessary to compute the optimal decomposition (following Appendix F of the paper)
- `cost_functions` : Functions to compute some quantities of interest
- `matrix_stuff` : Some useful matrix definitions
- `data_management` : Functions to manage simulation data (saving, extracting from saved files and plotting)
