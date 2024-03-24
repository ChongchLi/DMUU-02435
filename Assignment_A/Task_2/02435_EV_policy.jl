# import useful packages
using Random
using JuMP
using Gurobi
using Printf
using Clustering
using Distances

# include files
include("V2_price_process.jl")
include("V2_02435_multistage_problem_data.jl")

function Make_EV_here_and_now_decision(number_of_simulation_periods, number_of_warehouses, tau, current_stock, current_prices, lookahead_days, nb_initial_scenarios)
    # Define the number of look-ahead days
    lookahead_days = check_lookahead(lookahead_days, number_of_simulation_periods, tau)

    # Generate your scenarios: Price(w,t,s)
    price_scenarios = generate_scenarios(number_of_warehouses, W, current_prices, initial_scenarios, lookahead_days)

    # Mean on those scenarios to have only one path of prices
    cost_coffee = reshape(mean(prices_trajectory_scenarios, dims=3), size(prices_trajectory_scenarios, 1), size(prices_trajectory_scenarios, 2))

    #Declare model with Gurobi solver
    model_EB = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))

    #Declare the variables to optimize
    # Quantities of coffee ordered, W rows and T columns
    @variable(model_EB, quantities_ordered[1:number_of_warehouses, 1:actual_look_ahead_days]>=0)
    # Quantities send from w to q, W rows W columns and T layers
    @variable(model_EB, quantities_send[1:number_of_warehouses, 1:number_of_warehouses, 1:actual_look_ahead_days]>=0)
    # Quantities recieved by w from q, W rows W columns and T layers
    @variable(model_EB, quantities_recieved[1:number_of_warehouses, 1:number_of_warehouses, 1:actual_look_ahead_days]>=0)
    # Quantities in the warehouse stockage, W rows and T columns
    @variable(model_EB, quantities_stocked[1:number_of_warehouses, 1:actual_look_ahead_days]>=0)
    # Quantities mising to complete the demand, W rows and T columns
    @variable(model_EB, quantities_missed[1:number_of_warehouses, 1:actual_look_ahead_days]>=0)
    
    #Objective function
    @objective(model_EB, Min, sum(quantities_ordered[w,t]*cost_coffee[w,t] for w in 1:number_of_warehouses, t in 1:actual_look_ahead_days) 
    + sum(quantities_send[w,q,t]*cost_tr[w,q] for w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:actual_look_ahead_days)
    + sum(quantities_missed[w,t]*cost_miss[w] for w in 1:number_of_warehouses, t in 1:actual_look_ahead_days))

    #Constraints of the problem
    # Constraint on stockage capacities limited to the maximum capacities
    @constraint(model_EB, Stockage_limit[w in 1:number_of_warehouses, t in 1:actual_look_ahead_days], quantities_stocked[w,t] <= warehouse_capacities[w])
    # Constraint on transport capacities limited to the maximum capacities
    @constraint(model_EB, Transport_limit[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:actual_look_ahead_days], quantities_send[w,q,t] <= transport_capacities[w,q])
    # Constraint on quantity send equal quantity recieved
    @constraint(model_EB, Send_recieved[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:actual_look_ahead_days], quantities_send[w,q,t] == quantities_recieved[q,w,t])
    # Constraint on a warehouse can only send to others warehouse
    # Useless cause the self-transport capacity is equal to 0
    @constraint(model_EB, Self_transport[w in 1:number_of_warehouses, t in 1:actual_look_ahead_days], quantities_send[w,w,t] == 0)
    # Constraint on quantity send limited to previous stock
    @constraint(model_EB, Transport_stock[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 2:actual_look_ahead_days], sum(quantities_send[w,q,t] for q in 1:number_of_warehouses) <= quantities_stocked[w,t-1])
    @constraint(model_EB, Transport_stock_start[w in 1:number_of_warehouses, q in 1:number_of_warehouses], sum(quantities_send[w,q,1] for q in 1:number_of_warehouses) <= current_stock[w])
    # Constraint on quantity stock at time t with input and output
    @constraint(model_EB, Stockage[w in 1:number_of_warehouses, t in 2:actual_look_ahead_days], quantities_stocked[w,t] == quantities_stocked[w,t-1]+quantities_ordered[w,t]
    +sum(quantities_recieved[w,q,t] - quantities_send[w,q,t] for q in 1:number_of_warehouses)- demand_trajectory[w,t] + quantities_missed[w,t])
    @constraint(model_EB, Stockage_start[w in 1:number_of_warehouses], quantities_stocked[w,1] == current_stock[w]+quantities_ordered[w,1]
    +sum(quantities_recieved[w,q,1] - quantities_send[w,q,1] for q in 1:number_of_warehouses)- demand_trajectory[w,1] + quantities_missed[w,1])

    optimize!(model_EB)

    #Check if optimal solution was found
    if termination_status(model_EB) == MOI.OPTIMAL
        println("Optimal solution found")

        # Return interesting values
        return value.(quantities_ordered[:,1]),value.(quantities_send[:,:,1]),value.(quantities_recieved[:,:,1]),value.(quantities_stocked[:,1]),value.(quantities_missed[:,1])
    else
        return error("No solution.")
    end
end

function check_lookahead(look_ahead_days, number_of_simulation_periods, tau)
    if look_ahead_days > number_of_simulation_periods - tau
        lookahead_days = number_of_simulation_periods-tau+1
    else 
        lookahead_days = look_ahead_days+1
    end    
    return lookahead_days
end


function generate_scenarios(number_of_warehouses, W, current_prices, initial_scenarios, lookahead_days)

    Scen = collect(1:initial_scenarios)
    scenarios = zeros(number_of_warehouses, lookahead_days, initial_scenarios)
    for s in Scen
        for w in W
            scenarios[w,1,s] = current_prices[w]
            for t in 2:lookahead_days
                scenarios[w,t,s] = sample_next(scenarios[w,t-1,s])
            end
        end
    end
    return scenarios
end

function discretize_scenarios(price_scenarios, granularity)
    # Assuming price_scenarios is a 3-dimensional array with dimensions (number_of_warehouses, lookahead_days, number_of_scenarios)
    
    # Copy price_scenarios to keep the original data intact
    discretized_scenarios = copy(price_scenarios)
    
    # Loop over each entry in the 3D array using proper indexing
    for i in 1:size(price_scenarios, 1)
        for j in 1:size(price_scenarios, 2)
            for k in 1:size(price_scenarios, 3)
                # Round each price to the nearest value based on the granularity
                discretized_scenarios[i, j, k] = round(price_scenarios[i, j, k] / granularity) * granularity
            end
        end
    end
    
    return discretized_scenarios
end

function reduce_scenarios(price_scenarios, number_of_warehouses, num_of_reduced_scenarios, lookahead_days, granularity, reduce_type)
    if reduce_type == "kmeans" 
        reduced_scenarios, probabilities = cluster_kmeans(price_scenarios, num_of_reduced_scenarios, granularity)
    elseif reduce_type == "kmedoids"
        reduced_scenarios, probabilities = cluster_kmedoids(price_scenarios, number_of_warehouses, num_of_reduced_scenarios, lookahead_days)
    else 
        reduced_scenarios, probabilities = fast_forward(price_scenarios, number_of_warehouses, num_of_reduced_scenarios, lookahead_days)
    end
end

function create_non_anticipativity_sets(lookahead_days, reduced_prices, num_of_reduced_scenarios)
    non_anticipativity_sets = Dict()
    Scen = collect(1:num_of_reduced_scenarios)
    Day = collect(1:lookahead_days)

    for s in Scen 
        for sp in s+1:num_of_reduced_scenarios
            sim = 1
            for d in Day 
                if sim == 1
                    if reduced_prices[:,d,s] == reduced_prices[:,d,sp]
                        key = (s,d)
                        if haskey(non_anticipativity_sets, key)
                            push!(non_anticipativity_sets[key], sp)
                        else 
                            non_anticipativity_sets[key] = [sp]
                        end
                    else 
                        sim = 0
                    end
                end
            end
        end
    end
    return non_anticipativity_sets
end