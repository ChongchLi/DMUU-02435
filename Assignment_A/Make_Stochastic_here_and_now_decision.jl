using JuMP
using Gurobi
using Printf
using Random
using Clustering

include("V2_Assignment_A_codes/V2_price_process.jl")
include("02435_two_stage_problem_data.jl")
include("reduction_function.jl")

function Make_Stochastic_here_and_now_decision(prices, num_of_scenarios)
    number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data(2)

    num_sampled_scenarios = 1000

    next_prices = Array{Float64}(undef, number_of_warehouses, num_sampled_scenarios)
    for w in 1:number_of_warehouses
        for n in 1:num_sampled_scenarios
            next_prices[w,n] = sample_next(prices[w])
        end
    end

    ## select scenarios reduction function 
    reduced_prices, probs = cluster_kmeans(next_prices, num_of_scenarios)
    next_prices = reduced_prices
    num_sampled_scenarios = num_of_scenarios


    demand_coffee = demand_trajectory  # coffee demand

    model_MS = Model(Gurobi.Optimizer) # declare model with Gurobi

    # DEclare the Variables
    # amount of the coffee oredered
    @variable(model_MS, x_order_1[1:number_of_warehouses]>=0)
    @variable(model_MS, x_order_2[1:number_of_warehouses, 1:num_of_scenarios]>=0)
    # storage level of w at t
    @variable(model_MS, z_storage_1[1:number_of_warehouses]>=0)
    @variable(model_MS, z_storage_2[1:number_of_warehouses, 1:num_of_scenarios]>=0)
    # the missing amount
    @variable(model_MS, m_missing_1[1:number_of_warehouses]>=0)
    @variable(model_MS, m_missing_2[1:number_of_warehouses, 1:num_of_scenarios]>=0)
    # At stage t, the amount of coffee is sent from warehouse w to the neighboring warehouse q
    @variable(model_MS, y_send_1[1:number_of_warehouses, 1:number_of_warehouses]>=0)
    @variable(model_MS, y_send_2[1:number_of_warehouses, 1:number_of_warehouses, 1:num_of_scenarios]>=0)
    # At stage t, the amount of coffee is received by the neighboring warehouse q
    @variable(model_MS, y_received_1[1:number_of_warehouses, 1:number_of_warehouses]>=0)
    @variable(model_MS, y_received_2[1:number_of_warehouses, 1:number_of_warehouses, 1:num_of_scenarios]>=0)
    

    # objective function
    @objective(model_MS, Min, sum(x_order_1[w]*prices[w] for w in 1:number_of_warehouses) + sum(y_send_1[w,q]*cost_tr[w,q] for w in 1:number_of_warehouses, q in 1:number_of_warehouses)
    + sum(m_missing_1[w]*cost_miss[w] for w in 1:number_of_warehouses) + sum(probs[n]*(sum(x_order_2[w,n]*next_prices[w,n] for w in 1:number_of_warehouses)
    + sum(y_send_2[w,q,n]*cost_tr[w,q] for w in 1:number_of_warehouses, q in 1:number_of_warehouses)
    + sum(m_missing_2[w,n]*cost_miss[w] for w in 1:number_of_warehouses)) for n in 1:num_of_scenarios))

    # constraints
    # storage capacity
    @constraint(model_MS, storage_capacity_1[w in 1:number_of_warehouses], z_storage_1[w] <= warehouse_capacities[w])
    @constraint(model_MS, storage_capacity_2[w in 1:number_of_warehouses, n in 1:num_of_scenarios], z_storage_2[w,n] <= warehouse_capacities[w])
    # transport capacity
    @constraint(model_MS, transport_capacity_1[w in 1:number_of_warehouses, q in 1:number_of_warehouses], y_send_1[w,q] <= transport_capacities[w,q])
    @constraint(model_MS, transport_capacity_2[w in 1:number_of_warehouses, q in 1:number_of_warehouses, n in 1:num_of_scenarios], y_send_2[w,q,n] <= transport_capacities[w,q])
    
    # quantity send equal quantity recieved
    @constraint(model_MS, Send_recieved_1[w in 1:number_of_warehouses, q in 1:number_of_warehouses], y_send_1[w,q] == y_received_1[q,w])
    @constraint(model_MS, Send_recieved_2[w in 1:number_of_warehouses, q in 1:number_of_warehouses, n in 1:num_of_scenarios], y_send_2[w,q,n] == y_received_2[q,w,n])
    
    # inventory balance
    @constraint(model_MS, inventory_balance_1[w in 1:number_of_warehouses], demand_coffee[w,1] == initial_stock[w] - z_storage_1[w] + x_order_1[w] + sum(y_received_1[w,q] - y_send_1[w,q] for q in 1:number_of_warehouses) + m_missing_1[w])
    @constraint(model_MS, inventory_balance_2[w in 1:number_of_warehouses, n in 1:num_of_scenarios], demand_coffee[w,2] == z_storage_1[w] - z_storage_2[w,n] + x_order_2[w,n] + sum(y_received_2[w,q,n] - y_send_2[w,q,n] for q in 1:number_of_warehouses) + m_missing_2[w,n])
    # Constraint on amount send limited to previous
    @constraint(model_MS, send_limited_1[w in 1:number_of_warehouses, q in 1:number_of_warehouses], sum(y_send_1[w,q] for q in 1:number_of_warehouses) <= initial_stock[w])
    @constraint(model_MS, send_limited_2[w in 1:number_of_warehouses, q in 1:number_of_warehouses, n in 1:num_of_scenarios], sum(y_send_2[w,q,n] for q in 1:number_of_warehouses) <= z_storage_1[w])
    # a warehouse can only send to other warehouses
    @constraint(model_MS, self_send_1[w in 1:number_of_warehouses], y_send_1[w,w] == 0)
    @constraint(model_MS, self_send_2[w in 1:number_of_warehouses, n in 1:num_of_scenarios], y_send_2[w,w,n] == 0)



    optimize!(model_MS)

    # Check if the model was solved successfully
    if termination_status(model_MS) == MOI.OPTIMAL
        # Extract decisions
        x_order_opt = value.(x_order_1)
        z_storage_opt = value.(z_storage_1)
        m_missing_opt = value.(m_missing_1)
        y_send_opt = value.(y_send_1)
        y_received_opt = value.(y_received_1)

        # System's total cost
        
        total_cost = sum(value.(x_order_1[w])*prices[w] for w in 1:number_of_warehouses) + sum(value.(y_send_1[w,q])*cost_tr[w,q] for w in 1:number_of_warehouses, q in 1:number_of_warehouses)
        + sum(value.(m_missing_1[w])*cost_miss[w] for w in 1:number_of_warehouses)

        # Return the decisions and cost
        return x_order_opt, z_storage_opt, m_missing_opt, y_send_opt, y_received_opt, total_cost
    else
        error("The model did not solve to optimality.")
    end





end

prices=round.(10 * rand(3), digits=2)    
x_order_opt, z_storage_opt, m_missing_opt, y_send_opt, y_received_opt, total_cost=Make_Stochastic_here_and_now_decision(prices,50)


