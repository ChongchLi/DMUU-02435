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

function Make_EV_here_and_now_decision(number_of_simulation_periods, tau, current_stock, current_prices, lookahead_days, initial_scenarios)
    # Define the number of look-ahead days
    lookahead_days = check_lookahead(lookahead_days, number_of_simulation_periods, tau)

    # Generate your scenarios: Price(w,t,s)
    price_scenarios = generate_scenarios(number_of_warehouses, W, current_prices, initial_scenarios, lookahead_days)

    # Mean on those scenarios to have only one path of prices
    price_coffee = reshape(mean(price_scenarios, dims=3), size(price_scenarios, 1), size(price_scenarios, 2))

    #Declare model with Gurobi solver
    model_EV = Model(Gurobi.Optimizer)

    # DEclare the Variables
    # amount of the coffee oredered
    @variable(model_EV, x_order[1:number_of_warehouses, 1:lookahead_days]>=0)
    # storage level of w at t
    @variable(model_EV, z_storage[1:number_of_warehouses, 1:lookahead_days]>=0)
    # the missing amount
    @variable(model_EV, m_missing[1:number_of_warehouses, 1:lookahead_days]>=0)
    # At stage t, the amount of coffee is sent from warehouse w to the neighboring warehouse q
    @variable(model_EV, y_send[1:number_of_warehouses, 1:number_of_warehouses, 1:lookahead_days]>=0)
    # At stage t, the amount of coffee is received by the neighboring warehouse q
    @variable(model_EV, y_received[1:number_of_warehouses, 1:number_of_warehouses, 1:lookahead_days]>=0)
    
    # objective function
    @objective(model_EV, Min, sum(price_coffee[w,t] * x_order[w,t] for w in 1:number_of_warehouses, t in 1:lookahead_days)
    + sum(cost_tr[w,q] * y_send[w,q,t] for w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:lookahead_days)
    + sum(cost_miss[w] * m_missing[w,t] for w in 1:number_of_warehouses, t in 1:lookahead_days))

    # constraints
    # storage capacity
    @constraint(model_EV, storage_capacity[w in 1:number_of_warehouses, t in 1:lookahead_days], z_storage[w,t] <= warehouse_capacities[w])
    # transport capacity
    @constraint(model_EV, transport_capacity[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:lookahead_days], y_send[w,q,t] <= transport_capacities[w,q])
    # quantity send equal quantity recieved
    @constraint(model_EV, SendReceiveBalance[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:lookahead_days], y_send[w,q,t] == y_received[q,w,t])
    # inventory balance
    @constraint(model_EV, inventory_balance_start[w in 1:number_of_warehouses], demand_trajectory[w,1] == current_stock[w] - z_storage[w,1] + x_order[w,1] + sum(y_received[w,q,1] - y_send[w,q,1] for q in 1:number_of_warehouses) + m_missing[w,1])
    @constraint(model_EV, inventory_balance[w in 1:number_of_warehouses, t in 2:lookahead_days], demand_trajectory[w,t] == z_storage[w,t-1] - z_storage[w,t] + x_order[w,t] + sum(y_received[w,q,t] - y_send[w,q,t] for q in 1:number_of_warehouses) + m_missing[w,t])
    # Constraint on amount send limited to previous
    @constraint(model_EV, send_limitied_start[w in 1:number_of_warehouses, q in 1:number_of_warehouses], sum(y_send[w,q,1] for q in 1:number_of_warehouses) <= current_stock[w])
    @constraint(model_EV, send_limitied[w in 1:number_of_warehouses, q in 1:number_of_warehouses,t in 2:lookahead_days], sum(y_send[w,q,t] for q in 1:number_of_warehouses) <= z_storage[w,t-1])
    # a warehouse can only send to other warehouses
    @constraint(model_EV, self_send[w in 1:number_of_warehouses, t in 1:lookahead_days], y_send[w,w,t] == 0)
  
    optimize!(model_EV)

    # Check if the model was solved successfully
    if termination_status(model_EV) == MOI.OPTIMAL
        # Extract decisions
        x_order_EV = value.(x_order[:,1])
        y_send_EV = value.(y_send[:,:,1])
        y_received_EV = value.(y_received[:,:,1])
        z_storage_EV = value.(z_storage[:,1])
        m_missing_EV = value.(m_missing[:,1])

        # System's total cost
        total_cost = objective_value(model_EV)

        # Return the decisions and cost
        return x_order_EV, y_send_EV, y_received_EV, z_storage_EV, m_missing_EV
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