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
number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data()
function make_multistage_here_and_now_decision(number_of_sim_periods, tau, current_stock, current_prices, lookahead_days, initial_scenarios, granularity)
    # Step 1: Define the number of look-ahead days
    lookahead_days = check_lookahead(lookahead_days, number_of_sim_periods, tau)

    # Step 2：Define the initial number of scenarios

    # Step 3: Generate your scenarios: Price(w,t,s)
    price_scenarios = generate_scenarios(number_of_warehouses, W, current_prices, initial_scenarios, lookahead_days)

    # Step 4: Discretize Scenarios
    discretized_scenarios = discretize_scenarios(price_scenarios, granularity)

    # Step 5: Reduce scenarios
    reduced_scenarios, probabilities = reduce_scenarios(discretized_scenarios)

    # Step 6: Reassign probabilities
    reassigned_probabilities = reassign_probabilities(reduced_scenarios, probabilities)

    # Step 7: Create and populate the “non-anticipativity” sets
    non_anticipativity_sets = create_non_anticipativity_sets(reassigned_probabilities)

    # Step 8: Solve the program
    decisions = solve_stochastic_program(non_anticipativity_sets, current_stock, current_prices)

    return decisions
end

function check_lookahead(look_ahead_days, number_of_sim_periods, tau)
    if look_ahead_days > number_of_sim_periods - tau
        lookahead_days = number_of_sim_periods-tau+1
        return lookahead_days
    else 
        lookahead_days = look_ahead_days+1
        return lookahead_days
    end    

function generate_scenarios(number_of_warehouses, W, current_prices, initial_scenarios, lookahead_days)

    Scen = collect(1:initial_scenarios)
    scenarios = zeros(number_of_warehouses, actual_look_ahead_days, initial_scenarios)
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
    # Create an empty array to hold the discretized scenarios
    discretized_scenarios = copy(price_scenarios)
    
    # Loop over each scenario using eachindex for safer indexing
    for i in eachindex(price_scenarios)
        for j in eachindex(price_scenarios[i])
            discretized_scenarios[i][j] = round(price_scenarios[i][j] / granularity) * granularity
        end
    end
    
    return discretized_scenarios
end


function reduce_scenarios(price_scenarios, reduce_type)
    if reduce_type == "kmeans" 
        reduced_scenarios, probabilities = cluster_kmeans(price_scenarios, nb_initial_scenarios, nb_reduced_scenarios, granularity)
    elseif reduce_type == "kmedoids"
        reduced_scenarios, probabilities = cluster_kmedoids(number_of_warehouses, price_scenarios, nb_initial_scenarios, nb_reduced_scenarios, actual_look_ahead_days)
    else 
        reduced_scenarios, probabilities = fast_forward(number_of_warehouses, price_scenarios, nb_initial_scenarios, nb_reduced_scenarios, actual_look_ahead_days)
    end
end

function reassign_probabilities(reduced_scenarios, probabilities)
    # Reassign probabilities to the reduced set of scenarios
end

function create_non_anticipativity_sets(probabilities)
    # Create sets to ensure non-anticipativity across scenarios
end

function solve_stochastic_program(non_anticipativity_sets, current_stock, current_prices)
    # Solve the optimization problem using a stochastic programming solver
end


##################################################################################################################
###################           reduction function               ###################################################
##################################################################################################################

#Performs fast forward selection for the given parameters
#D = Symmetric distance matrix
#p = vector of probabilities
#n = target number of scenarios
#Returns Array with 2 element, [1] = list of probabilities, [2] = list of selected scenario indices
function FastForwardSelection(D, p, n)
    init_d = D
    not_selected_scenarios = collect(range(1,length(D[:,1]);step=1))
    selected_scenarios = []
    while length(selected_scenarios) < n
        selected = select_scenario(D, p, not_selected_scenarios)
        deleteat!(not_selected_scenarios, findfirst(isequal(selected), not_selected_scenarios))
        push!(selected_scenarios, selected)
        D = UpdateDistanceMatrix(D, selected, not_selected_scenarios)
    end
    result_prob = RedistributeProbabilities(D, p, selected_scenarios, not_selected_scenarios)
    return [result_prob, selected_scenarios]
end

#Redistributes probabilities at the end of the fast forward selection
#D = original distance matrix
#p = probabilities
#selected_scenarios = indices of selected scenarios
#not_selected_scenarios = indices of non selected scenarios
function RedistributeProbabilities(D, p, selected_scenarios, not_selected_scenarios)
    probabilities = p
    for s in not_selected_scenarios
        min_idx = -1
        min_dist = Inf
        for i in selected_scenarios
            if D[s,i] < min_dist
                min_idx = i
                min_dist = D[s,i]
            end
        end
        probabilities[min_idx] = probabilities[min_idx] + p[s]
        probabilities[s] = 0.0
    end
    new_probabilities = [probabilities[i] for i in selected_scenarios]
    return new_probabilities
end

#Updates the distance matrix in the fast forward selection
#D = current distance matrix
#selected = index of scenario selected in this iteration
#scenarios = index list of not selected scenarios
function UpdateDistanceMatrix(D, selected, not_selected_scenarios)
    for s in not_selected_scenarios
        if s!=selected
            for s2 in not_selected_scenarios
                if s2!=selected
                    D[s,s2] = min(D[s,s2], D[s,selected])
                end
            end
        end
    end
    return D
end

#Selects the scenario idx with minimum Kantorovic distance
#D = Distance matrix
#p = probabilities
#scenarios = not selected scenarios
function select_scenario(D, p, not_selected_scenarios)
    min_dist = Inf
    min_idx = -1
    for s in not_selected_scenarios
        dist = sum(p[s2]*D[s2,s] for s2 in not_selected_scenarios if s!=s2)
        if dist < min_dist
            min_dist = dist
            min_idx = s
        end
    end
    return min_idx
end


function fast_forward(price_scenarios, number_of_warehouses, num_of_reduced_scenarios, lookahead_days)

    if lookahead_days != 1
        # reshapped
        reshape_price_scenarios = reshape(price_scenarios, :, size(price_scenarios, 3))
        #Calculate distance matrix (euclidean distance)
        Distance_matrix = zeros(Float64, reshape_price_scenarios, reshape_price_scenarios)
        for i in 1:reshape_price_scenarios
            for j in 1:reshape_price_scenarios
                distance = sqrt(sum((prices[l, i] - prices[l, j])^2 for l in eachindex(1:size(prices, 1))))
                Distance_matrix[i, j] = distance
            end
        end

        #Initialize equiprobable probabilities
        probabilities = repeat([1.0/reshape_price_scenarios], 1, reshape_price_scenarios)[1,:]
        #Include fast forward selection and apply it

        result = FastForwardSelection(Distance_matrix, probabilities, num_of_reduced_scenarios)

        #Resulting probabilities
        new_probabilities = result[1]
        #Selected scenario indices
        scenario_indices = result[2]

        reduced_prices = zeros(Float64, number_of_warehouses * lookahead_days, num_of_reduced_scenarios)
        for i = 1:num_of_reduced_scenarios
            reduced_prices[:,i] = reshape_price_scenarios[:,scenario_indices[i]]
        end
        reduced_prices = reshape(reduced_prices, size(price_scenarios)[1], size(price_scenarios)[2], :)
    else
        reduced_prices = price_scenarios[:,:,1:num_of_reduced_scenarios]
        new_probabilities = zeros(num_of_reduced_scenarios) 
        for i in 1:num_of_reduced_scenarios
            new_probabilities[i] = 1/num_of_reduced_scenarios
        end
    end

    return reduced_prices, new_probabilities

end

function cluster_kmeans(price_scenarios, num_of_reduced_scenarios, granularity)

    reshape_price_scenarios = reshape(price_scenarios, :, size(price_scenarios, 3))
    # perform K-means clustering
    results = kmeans(reshape_price_scenarios, num_of_reduced_scenarios; maxiter=2000)
    # Get the cluster centers
    reduced_prices = results.centers

    # Assignments of data points to clusters
    assigned = assignments(results) # get the assignments of points to clusters
    new_probabilities = zeros(Float64, num_of_reduced_scenarios)
    for a in assigned
        new_probabilities[a] = new_probabilities[a] + (1/length(assigned))
    end

    reduced_prices = reshape(reduced_prices, size(price_scenarios)[1], size(price_scenarios)[2], :)
    reduced_prices = discretize_scenarios(reduced_prices, granularity)
   

    return reduced_prices, new_probabilities

end

function cluster_kmedoids(price_scenarios, number_of_warehouses, num_of_reduced_scenarios, lookahead_days)

    if lookahead_days != 1
        reshape_price_scenarios = reshape(price_scenarios, :, size(price_scenarios, 3))
        #Calculate distance matrix (euclidean distance)
        Distance_matrix = zeros(Float64, num_sampled_scenarios, num_sampled_scenarios)
        for i in 1:num_sampled_scenarios
            for j in 1:num_sampled_scenarios
                distance = sqrt(sum((prices[l, i] - prices[l, j])^2 for l in eachindex(1:size(prices, 1))))
                Distance_matrix[i, j] = distance
            end
        end
        
        # Apply k-medoids algorithm
        result = kmedoids(Distance_matrix, num_of_reduced_scenarios;display=:iter)
        # resulting medoids(indices of data points)
        M = result.medoids

        reduced_prices = zeros(number_of_warehouses * lookahead_days, num_of_reduced_scenarios)

        for i in 1:num_of_reduced_scenarios
            reduced_prices[:,i] = reshape_price_scenarios[:,M[i]]
        end

        #Assignments of data points to clusters
        assigned = assignments(result) # get the assignments of points to clusters
        new_probabilities = zeros(Float64, num_of_reduced_scenarios)
        for a in assigned
            new_probabilities[a] = new_probabilities[a] + (1/length(assigned))
        end
        reduced_prices = reshape(reduced_prices, size(price_scenarios)[1], size(price_scenarios)[2], :)
    else
        reduced_prices = price_scenarios[:,:,1:num_of_reduced_scenarios]
        new_probabilities = zeros(num_of_reduced_scenarios) 
        for i in 1:num_of_reduced_scenarios
            new_probabilities[i] = 1/num_of_reduced_scenarios
        end

    return reduced_prices, new_probabilities
end




