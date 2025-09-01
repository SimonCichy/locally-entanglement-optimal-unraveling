
using JLD2

@doc """create_empty_file(filename::Union{String, Nothing}, settings::SimulationSettings)
    
    Generate an empty jld2 file containing the meta-data of a single 
    unraveling simulation simulation, to be complemented by the simulation data
    (monitored quantities at each layer for each sample). 
    See SimulationSettings for all the settings.
"""
function create_empty_file(
    filename::Union{String, Nothing}, 
    settings::SimulationSettings 
    )

    # (@isdefined filename) || (filename = nothing)
    if !isnothing(filename)
        if isfile(filename)
            throw(ArgumentError("Existing file under this filename"))
        end
        save(filename, "settings", settings)
    end
end

function get_smallest_counter(date, counter=1)
    while isfile("../../data/5_clean-up_version/$(date)_unraveling_$(counter).jld2")
        counter += 1
    end
    return counter
end

@doc """save_data(filename::Union{String, Nothing}, data)

    Save the data for the simulation of one unraveling in the JLD2 file. The 
    data comes as a dictionnary of stored monitored quantities (e.g. 
    entanglement entropy, truncation) saved in matrices of dimension 
    n_samples x depth
"""
function save_data(filename::Union{String, Nothing}, data, key::String="data")
    if !isnothing(filename)
        f = jldopen(filename, "r+")
        write(f, key, data)
        close(f)
    end
end

@doc """show_simulation_settings(file)

    Show all the settings of the simulation run. See SimulationSettings for 
    a list.
"""
function show_simulation_settings(file::String)
    f = jldopen(file, "r")
    settings =  f["settings"]
    for parameter in fieldnames(SimulationSettings) 
        in(parameter, (:gates, :initial_MPS, :noise_tensor)) && continue
        if in(parameter, (:circuit_details, :unraveling_details))
            println(parameter, " "^(21-length(string(parameter))), " : ")
            for (subparameter, value) in getproperty(settings, parameter) 
                println("  ",subparameter, " "^(19-length(string(subparameter))), " : ", value) 
            end 
        else
            println(parameter, " "^(21-length(string(parameter))), " : ", getproperty(settings, parameter)) 
        end
    end
    if haskey(f, "runtime")
        println("runtime",  " "^14, " : ")
        for key in keys(f["runtime"])
            println("  ",key, " "^(19-length(string(key))), " : ", f["runtime"][key])
        end
    end
    f = nothing
    GC.gc(true)
end

@doc """get_data(filename::String)

    Retrieve the stored information of the simulation in the file filename. 
    Both the settings of the simulation and all saved quantities.
"""
function get_data(filename::String)
    jldopen(filename, "r") do f
        settings = f["settings"]
        data = f["data"]
        return settings, data
    end
end

@doc """plot_comparison_vs_noise(filenames::Vector{String})

    Plotting the comparison of the monitored quantities between the different 
    runs of unraveling simulations. Each file contains simulation data for one 
    circuit setting, noise model and unraveling setting.
    Each file contains the simulation settings (as a SimulationSettings object)
    and the simulation data (a collection of monitored quantities stored in a 
    matrix of dimension n_samples x depth)
"""
function plot_comparison_vs_noise(
    filenames::Vector{String}; 
    noise_rates::Union{Nothing, Vector{Float64}}=nothing, 
    plot_final::Bool=true,
    # target_depth::Union{Int64, Nothing}=nothing,
    n_lines::Union{Int64, Nothing}=nothing,            # for the color scale
    cumulative::Bool=false
    )

    show_simulation_settings(filenames[1])

    monitored_quantities = Vector{String}()
    counters = Dict{String, Int64}()              # counting lines for typical, projective, rotated and wootters
    # final_values = Dict{String, Vector}()
    if !isnothing(n_lines)
        plot_colors = lineplot_colors(n_lines)
    end

    for (c, file) in enumerate(filenames)
        @show file
        settings, data = get_data(file)
        if c == 1
            monitored_quantities = settings.monitoring_quantities
            for q in monitored_quantities
                plt.figure(q*"-vs-depth")
                plt.xlabel("circuit depth")
                plt.ylabel(q)
                if plot_final
                    plt.figure(q*"-vs-noise")
                    if q == "truncation"
                        plt.loglog()
                    else
                        plt.semilogx()
                    end
                    plt.xlabel("noise_rate")
                    plt.ylabel(q)
                end
            end
        end
        # for q in monitored_quantities
        #     if !haskey(final_values, q*" "*settings.unraveling_type)
        #         final_values[q*" "*settings.unraveling_type] = [mean(data[q][:,end])]
        #     else
        #         push!(final_values[q*" "*settings.unraveling_type], mean(data[q][:,end]))
        #     end
        # end
        !isnothing(noise_rates) && !in(settings.noise_rate, noise_rates) && continue
        if !haskey(counters, settings.unraveling_type)
            counters[settings.unraveling_type] = 1
        else
            counters[settings.unraveling_type] += 1
        end
        for q in monitored_quantities
            !haskey(data, q) && continue
            if (q == "truncation") 
                if cumulative
                    growth_curve = cumsum(vec(mean(4 .* sqrt(2) .* sqrt.(data[q]), dims=1)))
                else
                    # growth_curve = vec(mean(data[q], dims=1))
                    growth_curve = vec(mean(sqrt(2) .* sqrt.(data[q]), dims=1))
                end
            else
                growth_curve = vec(mean(data[q], dims=1))
            end
            plt.figure(q*"-vs-depth")
            # for gamma in noise_rates
            legend = settings.noise_model*" $(settings.noise_rate) "*settings.unraveling_type
            colour = isnothing(n_lines) ? nothing : plot_colors[settings.unraveling_type][counters[settings.unraveling_type],:]
            plt.plot(0:settings.depth, growth_curve, label=legend, color=colour)

            plt.figure(q*"-vs-noise")
            plt.scatter(settings.noise_rate, growth_curve[end], color=colour)
        end
        # plot_colors = lineplot_colors(1)
        # for q in monitored_quantities
        #     plt.figure(q*"-vs-noise")
        #     for line in keys(final_values)
        #         !occursin(q, line) && continue
        #         unraveling = line[length(q)+1:end]
        #         legend = unraveling
        #         colour = plot_colors[unraveling]
        #         plt.plot(0:noise_rates, final_values[line], label=legend, c=colour) #noise rates?
        #     end
        # end
    end
    for fig in plt.get_fignums()
        plt.figure(fig)
        plt.legend()
    end
end

@doc """plot_comparison_vs_bonddim(filenames::Vector{String})

    Plotting the comparison of the monitored quantities between the different 
    runs of unraveling simulations. Each file contains simulation data for one 
    circuit setting, noise model and unraveling setting.
    Each file contains the simulation settings (as a SimulationSettings object)
    and the simulation data (a collection of monitored quantities stored in a 
    matrix of dimension n_samples x depth)
"""
function plot_comparison_vs_bonddim(
    filenames::Vector{String}; 
    bond_dims::Union{Nothing, Vector{Float64}}=nothing, 
    plot_final::Bool=true,
    # target_depth::Union{Int64, Nothing}=nothing,
    n_lines::Union{Int64, Nothing}=nothing,            # for the color scale
    cumulative::Bool=false
    )

    show_simulation_settings(filenames[1])
    
    monitored_quantities = Vector{String}()
    counters = Dict{String, Int64}()              # counting lines for typical, projective, rotated and wootters
    # final_values = Dict{String, Vector}()
    if !isnothing(n_lines)
        plot_colors = lineplot_colors(n_lines)
    end

    for (c, file) in enumerate(filenames)
        @show file
        settings, data = get_data(file)
        if c == 1
            monitored_quantities = settings.monitoring_quantities
            for q in monitored_quantities
                plt.figure(q*" vs depth")
                if (q == "truncation") && cumulative
                #     plt.semilogy()
                    plt.ylim([0, 3])
                end
                plt.xlabel("circuit depth")
                plt.ylabel(q)
                if plot_final
                    plt.figure(q*" vs bond dim")
                    if q == "truncation"
                        plt.loglog()
                    else
                        plt.semilogx()
                    end
                    plt.xlabel("bond dim")
                    plt.ylabel(q)
                end
            end
        end
        !isnothing(bond_dims) && !in(settings.max_bond_dim, bond_dims) && continue
        if !haskey(counters, settings.unraveling_type)
            counters[settings.unraveling_type] = 1
        else
            counters[settings.unraveling_type] += 1
        end
        for q in monitored_quantities
            if (q == "truncation") 
                if cumulative
                    growth_curve = cumsum(vec(mean(4 .* sqrt(2) .* sqrt.(data[q]), dims=1)))
                else
                    # growth_curve = vec(mean(data[q], dims=1))
                    growth_curve = vec(mean(2 .* data[q], dims=1))
                    # growth_curve = vec(mean(sqrt(2) .* sqrt.(data[q]), dims=1))
                end
            else
                growth_curve = vec(mean(data[q], dims=1))
            end
            plt.figure(q*" vs depth")
            # for gamma in noise_rates
            legend = settings.noise_model*", gamma=$(settings.noise_rate), chi=$(settings.max_bond_dim), "*settings.unraveling_type
            colour = isnothing(n_lines) ? nothing : plot_colors[settings.unraveling_type][counters[settings.unraveling_type],:]
            plt.plot(0:settings.depth, growth_curve, label=legend, color=colour)

            plt.figure(q*" vs bond dim")
            plt.scatter(settings.max_bond_dim, growth_curve[end], color=colour)
        end
    end
    for fig in plt.get_fignums()
        plt.figure(fig)
        plt.legend()
    end
end

function lineplot_colors(n_lines::Int64)
    unitary_cmap = get_cmap(:Blues)
    projective_cmap = get_cmap(:Purples)
    rotated_cmap = get_cmap(:Greens)
    wootters_cmap = get_cmap(:Reds)                 # :Greys
    if n_lines == 1
        color_dict = Dict(
            "unitary" => unitary_cmap(1), 
            "typical" => unitary_cmap(1),
            "projective" => projective_cmap(1), 
            "rotated" => rotated_cmap(1), 
            "wootters" => wootters_cmap(1), 
        )
    else
        colorrange = range(0.2, 1, n_lines)
        color_dict = Dict(
            "unitary" => unitary_cmap(colorrange), 
            "typical" => unitary_cmap(colorrange),
            "projective" => projective_cmap(colorrange), 
            "rotated" => rotated_cmap(colorrange), 
            "wootters" => wootters_cmap(colorrange), 
        )
    end
    return color_dict
end
