#inlcude useful files
include("V2_Assignment_A_codes/V2_02435_two_stage_problem_data.jl")
include("V2_Assignment_A_codes/V2_price_process.jl")

# import packages
using Gurobi
using JuMP
function Calculate_OiH_solution(day1_price, day2_price)
    number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data()
    price_coffee = [day1_price, day2_price] # coffee prices
    demand_coffee = demand_trajectory  # coffee demand

    model_OH = Model(Gurobi.Optimizer) # declare model with Gurobi

    # DEclare the Variables
    # amount of the coffee oredered
    @variable(model_OH, x_order[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    # storage level of w at t
    @variable(model_OH, z_storage[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    # the missing amount
    @variable(model_OH, m_missing[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    # At stage t, the amount of coffee is sent from warehouse w to the neighboring warehouse q
    @variable(model_OH, y_send[1:number_of_warehouses, 1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    # At stage t, the amount of coffee is received by the neighboring warehouse q
    @variable(model_OH, y_received[1:number_of_warehouses, 1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    

    # objective function
    @objective(model_OH, Min, sum(price_coffee[t] * x_order[w, t] for w in 1:number_of_warehouses, t in 1:number_of_simulation_periods)
    + sum(cost_tr[w, q] * y_send[w, q, t] for w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods)
    + sum(cost_miss[w] * m_missing[w, t] for w in 1:number_of_warehouses, t in 1:number_of_simulation_periods))

    # constraints
    # storage capacity
    @constraint(model_OH, storage_capacity[w in 1:number_of_warehouses, t in 1:number_of_simulation_periods], z_storage[w,t] <= warehouse_capacities[w])
    # transport capacity
    @constraint(model_OH, transport_capacity[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w,q,t] <= transport_capacities[w,q])
    # quantity send equal quantity recieved
    @constraint(model_OH, Send_recieved[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w,q,t] == y_received[q,w,t])
    # inventory balance
    @constraint(model_OH, inventory_balance[w in 1:number_of_warehouses, t in 2:number_of_simulation_periods], demand_coffee[w,t] == z_storage[w, t-1] - z_storage[w, t] + x_order[w,t] + sum(y_received[w,q,t] - y_send[w,q,t] for q in 1:number_of_warehouses) + m_missing[w,t])
    # initial balance
    @constraint(model_OH, inventory_balance_start[w in 1:number_of_warehouses], demand_coffee[w,1] == initial_stock[w] - z_storage[w, 1] + x_order[w,1] + sum(y_received[w,q,1] - y_send[w,q,1] for q in 1:number_of_warehouses) + m_missing[w,1])
    # Constraint on amount send limited to previous
    @constraint(model_OH, send_limitied[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 2:number_of_simulation_periods], sum(y_send[w,q,t] for q in 1:number_of_warehouses) <= z_storage[w, t-1])
    @constraint(model_OH, send_limitied_start[w in 1:number_of_warehouses, q in 1:number_of_warehouses], sum(y_send[w,q,1] for q in 1:number_of_warehouses) <= initial_stock[w])
    # a warehouse can only send to other warehouses
    @constraint(model_OH, self_send[w in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w,w,t] == 0)

    optimize!(model_OH)

    # Check if the model was solved successfully
    if termination_status(model_OH) == MOI.OPTIMAL
        # Extract decisions
        x_order_opt = value.(x_order)
        z_storage_opt = value.(z_storage)
        m_missing_opt = value.(m_missing)
        y_send_opt = value.(y_send)
        y_received_opt = value.(y_received)

        # System's total cost
        total_cost = objective_value(model_OH)

        # Return the decisions and cost
        return x_order_opt, z_storage_opt, m_missing_opt, y_send_opt, y_received_opt, total_cost
    else
        error("The model did not solve to optimality.")
    end


end

day1_price = 5
day2_price = 6

# Call your function with the sample data
results = Calculate_OiH_solution(day1_price, day2_price)

# Print the results to inspect them
println("Results: ", results)


    





