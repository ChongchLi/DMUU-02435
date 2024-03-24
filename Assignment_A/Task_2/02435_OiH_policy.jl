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

function OiH_policy(price_coffee)

    #Declare model with Gurobi solver
    model_OiH = Model(Gurobi.Optimizer)

    # DEclare the Variables
    # amount of the coffee oredered
    @variable(model_OiH, x_order[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    # storage level of w at t
    @variable(model_OiH, z_storage[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    # the missing amount
    @variable(model_OiH, m_missing[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    # At stage t, the amount of coffee is sent from warehouse w to the neighboring warehouse q
    @variable(model_OiH, y_send[1:number_of_warehouses, 1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    # At stage t, the amount of coffee is received by the neighboring warehouse q
    @variable(model_OiH, y_received[1:number_of_warehouses, 1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
    
    # objective function
    @objective(model_OiH, Min, sum(price_coffee[w,t] * x_order[w,t] for w in 1:number_of_warehouses, t in 1:number_of_simulation_periods)
    + sum(cost_tr[w,q] * y_send[w,q,t] for w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods)
    + sum(cost_miss[w] * m_missing[w,t] for w in 1:number_of_warehouses, t in 1:number_of_simulation_periods))

    # constraints
    # storage capacity
    @constraint(model_OiH, storage_capacity[w in 1:number_of_warehouses, t in 1:number_of_simulation_periods], z_storage[w,t] <= warehouse_capacities[w])
    # transport capacity
    @constraint(model_OiH, transport_capacity[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w,q,t] <= transport_capacities[w,q])
    # quantity send equal quantity recieved
    @constraint(model_OiH, SendReceiveBalance[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w,q,t] == y_received[q,w,t])
    # inventory balance
    @constraint(model_OiH, inventory_balance_start[w in 1:number_of_warehouses], demand_trajectory[w,1] == current_stock[w] - z_storage[w,1] + x_order[w,1] + sum(y_received[w,q,1] - y_send[w,q,1] for q in 1:number_of_warehouses) + m_missing[w,1])
    @constraint(model_OiH, inventory_balance[w in 1:number_of_warehouses, t in 2:number_of_simulation_periods], demand_trajectory[w,t] == z_storage[w,t-1] - z_storage[w,t] + x_order[w,t] + sum(y_received[w,q,t] - y_send[w,q,t] for q in 1:number_of_warehouses) + m_missing[w,t])
    # Constraint on amount send limited to previous
    @constraint(model_OiH, send_limitied_start[w in 1:number_of_warehouses, q in 1:number_of_warehouses], sum(y_send[w,q,1] for q in 1:number_of_warehouses) <= current_stock[w])
    @constraint(model_OiH, send_limitied[w in 1:number_of_warehouses, q in 1:number_of_warehouses,t in 2:number_of_simulation_periods], sum(y_send[w,q,t] for q in 1:number_of_warehouses) <= z_storage[w,t-1])
    # a warehouse can only send to other warehouses
    @constraint(model_OiH, self_send[w in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w,w,t] == 0)
  
    optimize!(model_OiH)

    # Check if the model was solved successfully
    if termination_status(model_OiH) == MOI.OPTIMAL
        # Extract decisions
        x_order_OiH = value.(x_order[:,1])
        y_send_OiH = value.(y_send[:,:,1])
        y_received_OiH = value.(y_received[:,:,1])
        z_storage_OiH = value.(z_storage[:,1])
        m_missing_OiH = value.(m_missing[:,1])

        # System's total cost
        total_cost = objective_value(model_OiH)

        # Return the decisions and cost
        return x_order_OiH, y_send_OiH, y_received_OiH, z_storage_OiH, m_missing_OiH
    else
        error("The model did not solve to optimality.")
    end
end