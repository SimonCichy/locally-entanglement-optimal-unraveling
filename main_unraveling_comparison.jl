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
    do_plot && (r = [])
    repetitions = 10

    # Random.seed!(unraveling.seed)
    date = 20250628


    for run in 1:repetitions

        global circuit_details = Dict(
            # :fixed => false,
            # :low_entangling => true,
            # :entangling_angle => 0.05,
            # :local_rotations => false, 
            # :Hamiltonian_unitary => ComplexF64[-0.49730319762396313 - 0.3486759325979161im -0.6095261365943906 + 0.4575601027874342im 0.07213746798364404 - 0.11396436707701024im -0.12036552862517742 - 0.13248287841774609im; 0.06925139336930236 + 0.7598650153038152im -0.5645560032452138 - 0.13635164691075388im -0.15257536262976557 + 0.03306147458532581im -0.00892504110740796 - 0.23673232114801881im; 0.06771768642272098 + 0.017573669031166073im 0.21051731885776387 - 0.10548599043185101im 0.324985438530976 - 0.4874479987285134im -0.4529532110187102 - 0.6255181430559172im; -0.047483105126327634 - 0.20438899804930163im 0.15994994468066107 + 0.027795111264176477im -0.7145668267665053 + 0.32188378295516495im -0.031940684532568174 - 0.5606949368572363im],
            # :Hamiltonian_matrix => I(4), 
            :time_step => 0.05, 
            # :coupling_fields => [1, 1, 1],
            :coupling_fields => 2 .* rand(3) .- 1,
            # :local_fields => [1, 0, 0],
            :local_fields => [rand(), 0, 0],
            :renormalize => true
        )
    
        global counter = get_smallest_counter(date, 1)

        for unraveling in unravelings
        @time for noise_rate in noise_rates                         # @time ProfileView.@profview t = @timed
            println("Sampling of "*circuit_type*" with "*noise_model*" at rate $(noise_rate) under "*unraveling*" unraveling")
            simulation_settings = SimulationSettings(
                circuit_type, n_qubits, depth, initial_state, 
                noise_model, noise_rate, unraveling, 
                n_samples, chi_max, 
                monitoring_quantities, circuit_details
            )

            filename = do_save ? "../../data/5_clean-up_version/$(date)_unraveling_$(counter).jld2" : nothing
            @show filename
            create_empty_file(filename, simulation_settings)

            simulation_outcomes = Dict{String, Matrix{Float64}}()
            for quantity in monitoring_quantities
                simulation_outcomes[quantity] = zeros(Float64, n_samples, depth+1)
            end

            iter = ProgressBar(1:n_samples, unit=" samples")
            timing = @timed for sample in iter
                set_description(iter, noise_model*", "*string(@sprintf("gamma = %.1e, ", noise_rate))*unraveling)
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
            # [plt.plot(r, label=unravelings[i]) for (i,r) in enumerate(r)]
            # plt.title("the two norm of the final state for 1000 qubits and different amp_noise")
            plt.xlabel("rounds")
            plt.ylabel("total error")
            plt.legend()
        end

        end
    end
end