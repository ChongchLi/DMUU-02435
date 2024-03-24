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

function make_multistage_here_and_now_decision(number_of_simulation_periods, num_of_reduced_scenarios, tau, current_stock, current_prices, lookahead_days, initial_scenarios, granularity, reduce_type)
    # Step 1: Define the number of look-ahead days
    lookahead_days = check_lookahead(lookahead_days, number_of_simulation_periods, tau)

    # Step 2：Define the initial number of scenarios(finished)

    # Step 3: Generate your scenarios: Price(w,t,s)
    price_scenarios = generate_scenarios(number_of_warehouses, W, current_prices, initial_scenarios, lookahead_days)

    # Step 4: Discretize Scenarios
    discretized_scenarios = discretize_scenarios(price_scenarios, granularity)

    # Step 5 & 6: Reduce scenarios and reassign probabilities
    reduced_prices, probabilities = reduce_scenarios(discretized_scenarios, number_of_warehouses, num_of_reduced_scenarios, lookahead_days, granularity, reduce_type)

    # Step 7: Create and populate the “non-anticipativity” sets
    non_anticipativity_sets = create_non_anticipativity_sets(lookahead_days, reduced_prices, num_of_reduced_scenarios)

    # Step 8: Solve the program
    model_MP = Model(Gurobi.Optimizer)
    
    # DEclare the Variables
    # amount of the coffee oredered
    @variable(model_MP, x_order[1:number_of_warehouses, 1:lookahead_days, 1:num_of_reduced_scenarios]>=0)
    # storage level of w at t
    @variable(model_MP, z_storage[1:number_of_warehouses, 1:lookahead_days, 1:num_of_reduced_scenarios]>=0)
    # the missing amount
    @variable(model_MP, m_missing[1:number_of_warehouses, 1:lookahead_days, 1:num_of_reduced_scenarios]>=0)
    # At stage t, the amount of coffee is sent from warehouse w to the neighboring warehouse q
    @variable(model_MP, y_send[1:number_of_warehouses, 1:number_of_warehouses, 1:lookahead_days, 1:num_of_reduced_scenarios]>=0)
    # At stage t, the amount of coffee is received by the neighboring warehouse q
    @variable(model_MP, y_received[1:number_of_warehouses, 1:number_of_warehouses, 1:lookahead_days, 1:num_of_reduced_scenarios]>=0)
    

    # objective function
    @objective(model_MP, Min, sum(probabilities[s]*(sum(reduced_prices[w,t,s] * x_order[w,t,s] for w in 1:number_of_warehouses, t in 1:lookahead_days)
    + sum(cost_tr[w,q] * y_send[w,q,t,s] for w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:lookahead_days)
    + sum(cost_miss[w] * m_missing[w,t,s] for w in 1:number_of_warehouses, t in 1:lookahead_days)) for s in 1:num_of_reduced_scenarios))

    # constraints
    # storage capacity
    @constraint(model_MP, storage_capacity[w in 1:number_of_warehouses, t in 1:lookahead_days, s in 1:num_of_reduced_scenarios], z_storage[w,t,s] <= warehouse_capacities[w])
    # transport capacity
    @constraint(model_MP, transport_capacity[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:lookahead_days, s in 1:num_of_reduced_scenarios], y_send[w,q,t,s] <= transport_capacities[w,q])
    # quantity send equal quantity recieved
    @constraint(model_MP, SendReceiveBalance[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:lookahead_days, s in 1:num_of_reduced_scenarios], y_send[w,q,t,s] == y_received[q,w,t,s])
    # inventory balance
    @constraint(model_MP, inventory_balance_start[w in 1:number_of_warehouses, s in 1:num_of_reduced_scenarios], demand_trajectory[w,1] == current_stock[w] - z_storage[w,1,s] + x_order[w,1,s] + sum(y_received[w,q,1,s] - y_send[w,q,1,s] for q in 1:number_of_warehouses) + m_missing[w,1,s])
    @constraint(model_MP, inventory_balance[w in 1:number_of_warehouses, t in 2:lookahead_days, s in 1:num_of_reduced_scenarios], demand_trajectory[w,t] == z_storage[w,t-1,s] - z_storage[w,t,s] + x_order[w,t,s] + sum(y_received[w,q,t,s] - y_send[w,q,t,s] for q in 1:number_of_warehouses) + m_missing[w,t,s])
    # Constraint on amount send limited to previous
    @constraint(model_MP, send_limitied_start[w in 1:number_of_warehouses, q in 1:number_of_warehouses, s in 1:num_of_reduced_scenarios], sum(y_send[w,q,1,s] for q in 1:number_of_warehouses) <= current_stock[w])
    @constraint(model_MP, send_limitied[w in 1:number_of_warehouses, q in 1:number_of_warehouses,t in 2:lookahead_days, s in 1:num_of_reduced_scenarios], sum(y_send[w,q,t,s] for q in 1:number_of_warehouses) <= z_storage[w,t-1,s])
    # a warehouse can only send to other warehouses
    @constraint(model_MP, self_send[w in 1:number_of_warehouses, t in 1:lookahead_days, s in 1:num_of_reduced_scenarios], y_send[w,w,t,s] == 0)
    # Constraints of non-anticipativity
    Keys_SetsList = collect(keys(non_anticipativity_sets))
    for key in Keys_SetsList
        set = key[1]
        time = key[2]
        Others_sets = non_anticipativity_sets[key]
        for set_p in Others_sets 
            @constraint(model_MP, [w in 1:number_of_warehouses], x_order[w,time,set] == x_order[w,time,set_p])
            @constraint(model_MP, [w in 1:number_of_warehouses, q in 1:number_of_warehouses], y_send[w,q,time,set] == y_send[w,q,time,set_p])
            @constraint(model_MP, [w in 1:number_of_warehouses, q in 1:number_of_warehouses], y_received[w,q,time,set] == y_received[w,q,time,set_p])
            @constraint(model_MP, [w in 1:number_of_warehouses], z_storage[w,time,set] == z_storage[w,time,set_p])
            @constraint(model_MP, [w in 1:number_of_warehouses], m_missing[w,time,set] == m_missing[w,time,set_p])
        end
    end


    optimize!(model_MP)

    # Check if the model was solved successfully
    if termination_status(model_MP) == MOI.OPTIMAL
        # Extract decisions
        x_order_MP = value.(x_order[:,1,1])
        y_send_MP = value.(y_send[:,:,1,1])
        y_received_MP = value.(y_received[:,:,1,1])
        z_storage_MP = value.(z_storage[:,1,1])
        m_missing_MP = value.(m_missing[:,1,1])

        # System's total cost
        total_cost = objective_value(model_MP)

        # Return the decisions and cost
        return x_order_MP, y_send_MP, y_received_MP, z_storage_MP, m_missing_MP
    else
        error("The model did not solve to optimality.")
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

    num_scenarios = size(price_scenarios, 3)
    
    if lookahead_days != 1
        # Reshape price_scenarios into a 2D matrix where rows correspond to scenarios
        reshape_price_scenarios = reshape(price_scenarios, :, num_scenarios)
        # Initialize the Distance_matrix to store the pairwise distances between scenarios
        Distance_matrix = zeros(Float64, size(reshape_price_scenarios, 2), size(reshape_price_scenarios, 2))
        for i in 1:num_scenarios
            for j in 1:num_scenarios
                distance = sqrt(sum((reshape_price_scenarios[:, i] - reshape_price_scenarios[:, j]).^2))
                Distance_matrix[i, j] = distance
            end
        end

        #Initialize equiprobable probabilities
        probabilities = fill(1.0 / size(reshape_price_scenarios, 2), size(reshape_price_scenarios, 2))
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
    
    num_scenarios = size(price_scenarios, 3)
    
    if lookahead_days != 1
        reshape_price_scenarios = reshape(price_scenarios, :, num_scenarios)
        # Calculate distance matrix (euclidean distance)
        Distance_matrix = zeros(Float64, num_scenarios, num_scenarios)
        for i in 1:num_scenarios
            for j in 1:num_scenarios
                distance = sqrt(sum((reshape_price_scenarios[:, i] - reshape_price_scenarios[:, j]).^2))
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
    end

    return reduced_prices, new_probabilities
end



#####################################################
#### TEST FUNCTION ############
#####################################################
# number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data()

# include("V2_simulation_experiments.jl")
# # Creating the random experiments on which the policy will be evaluated
# number_of_experiments, Expers, Price_experiments = simulation_experiments_creation(number_of_warehouses, W, number_of_simulation_periods)
# num_of_reduced_scenarios = 20
# tau = 1
# current_stock = initial_stock
# current_prices = Price_experiments[1,:,1]
# lookahead_days = 3
# initial_scenarios = 100
# granularity = 0.5
# x_order_MP, y_send_MP, y_received_MP, z_storage_MP, m_missing_MP = make_multistage_here_and_now_decision(number_of_simulation_periods, num_of_reduced_scenarios, tau, current_stock, current_prices, lookahead_days, initial_scenarios, granularity,"fast_forward")


#####################################################
#### VALIDATE ############
#####################################################
using BenchmarkTools

# Define your input parameters
number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data()

include("V2_simulation_experiments.jl")
# Creating the random experiments on which the policy will be evaluated
number_of_experiments, Expers, Price_experiments = simulation_experiments_creation(number_of_warehouses, W, number_of_simulation_periods)
num_of_reduced_scenarios = 20
tau = 1
current_stock = initial_stock
current_prices = Price_experiments[1,:,1]
lookahead_days = 3
initial_scenarios = 100
granularity = 0.5
# Run the function and measure execution time
execution_time = @elapsed begin
    x_order_MP, y_send_MP, y_received_MP, z_storage_MP, m_missing_MP = make_multistage_here_and_now_decision(
        number_of_simulation_periods,
        num_of_reduced_scenarios,
        tau,
        current_stock,
        current_prices,
        lookahead_days,
        initial_scenarios,
        granularity,
        "kmeans" # or "kmeans" / "kmedoids"
    )
end

println("Execution time: $execution_time seconds")

# Check if the execution time is within the allowed limit
if execution_time > 3
    println("Error: The execution time exceeds the 3 seconds limit.")
else
    println("Success: The execution time is within the allowed limit.")
end

# Check the number of scenarios generated
if initial_scenarios > 1000
    println("Error: The number of initial scenarios exceeds 1000.")
else
    println("Success: The number of initial scenarios is within the limit.")
end

# Calculate the total number of decision variables in the model
total_vars = number_of_warehouses * lookahead_days * num_of_reduced_scenarios
# Multiply by the number of decision variable arrays you have
total_vars *= 5 # For x_order, z_storage, m_missing, y_send, y_received

println("Total number of decision variables: $total_vars")

if total_vars > 6500
    println("Error: The number of decision variables exceeds 6500.")
else
    println("Success: The number of decision variables is within the limit.")
end
