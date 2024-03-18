using Random
using JuMP
using Gurobi
using Printf

include("V2_Assignment_A_codes/V2_02435_two_stage_problem_data.jl")
include("V2_Assignment_A_codes/V2_price_process.jl")

# Prices are known for today
prices = round.(10 * rand(3), digits=2)

function Make_EV_here_and_now_decision(prices)
    # Import the data
    number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data()
    
    # Prices are unknown for the next day
    prices2 = Array{Float64}(undef, number_of_warehouses)
    for w in 1:number_of_warehouses
        price_samples = 0.0
        for i in 1:1000
            price_samples += sample_next(prices[w])
        end
        # Calculate the expected second-stage prices
        prices2[w] = round.(price_samples / 1000, digits=2)
    end
    
    CoffeePrice = [prices, prices2]

    model = Model(Gurobi.Optimizer)

    #Variables
    @variable(model, x[1:number_of_warehouses, 1:number_of_simulation_periods] >= 0)  # Quantities of coffee ordered, W rows and T columns
    @variable(model, z[1:number_of_warehouses, 1:number_of_simulation_periods] >= 0)  # Quantities in the warehouse stockage, W rows and T columns
    @variable(model, m[1:number_of_warehouses, 1:number_of_simulation_periods] >= 0)  # Quantities missing to complete the demand, W rows and T columns
    @variable(model, y_send[1:number_of_warehouses, 1:number_of_warehouses, 1:number_of_simulation_periods] >= 0)  # Quantities send from w to q, W rows W columns and T layers
    @variable(model, y_receive[1:number_of_warehouses, 1:number_of_warehouses, 1:number_of_simulation_periods] >= 0)  # Quantities recieved by w from q, W rows W columns and T layers

    #Objective function
    @objective(model, Min, sum(x[w, t] * CoffeePrice[t][w] for w in 1:number_of_warehouses, t in 1:number_of_simulation_periods) +
                             sum(y_send[w, q, t] * cost_tr[w, q] for w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods) +
                             sum(m[w, t] * cost_miss[w] for w in 1:number_of_warehouses, t in 1:number_of_simulation_periods))

    #Constraints
    @constraint(model, StorageCap[w in 1:number_of_warehouses, t in 1:number_of_simulation_periods], z[w, t] <= warehouse_capacities[w])
    @constraint(model, TransportationCap[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w, q, t] <= transport_capacities[w, q])
    @constraint(model, SendReceiveBalance[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w, q, t] == y_receive[q, w, t])
    @constraint(model, SelfTransport[w in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w, w, t] == 0)
    @constraint(model, TransportationStock[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 2:number_of_simulation_periods], sum(y_send[w, q, t] for q in 1:number_of_warehouses) <= z[w, t-1])
    @constraint(model, TransportationStockStart[w in 1:number_of_warehouses, q in 1:number_of_warehouses], sum(y_send[w, q, 1] for q in 1:number_of_warehouses) <= initial_stock[w])
    @constraint(model, InventoryBalance[w in 1:number_of_warehouses, t in 2:number_of_simulation_periods], z[w, t] == z[w, t-1] + x[w, t] +
                                 sum(y_receive[w, q, t] - y_send[w, q, t] for q in 1:number_of_warehouses) - demand_trajectory[w, t] + m[w, t])
    @constraint(model, InventoryBalanceStart[w in 1:number_of_warehouses], z[w, 1] == initial_stock[w] + x[w, 1] +
                                 sum(y_receive[w, q, 1] - y_send[w, q, 1] for q in 1:number_of_warehouses) - demand_trajectory[w, 1] + m[w, 1])

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        
        println("Optimal solution found")
        println("Minimal Cost: $(round(objective_value(model), digits=4))")
        println("-----------------")
        for t in 1:number_of_simulation_periods
            println("Time Step: $t")
            println("----") 
            for w in 1:number_of_warehouses 
                println("Warehouse $w : Demand $(demand_trajectory[w, t]) / Ordered $(round(value(x[w, t]), digits=2)) / Price $(CoffeePrice[t][w])")
                if t != 1 
                    println("Previous Stock $(round(value(z[w, t-1]), digits=2)) / Sent $(sum(round(value(y_send[w, q, t]), digits=2) for q in 1:number_of_warehouses)) / Recieved $(sum(round(value(y_receive[w, q, t]), digits=2) for q in 1:number_of_warehouses))")
                else 
                    println("Previous Stock 2.00 / Sent $(sum(round(value(y_send[w, q, t]), digits=2) for q in 1:number_of_warehouses)) / Recieved $(sum(round(value(y_receive[w, q, t]), digits=2) for q in 1:number_of_warehouses))")
                end
                println("Missed $(round(value(m[w, t]), digits=2)) / Stock $(round(value(z[w, t]), digits=2))")
                println("----")  
            end
            println("-----------------") 
        end

    #======================================================#    
        # Extract decisions
        x_order_EV = value.(x[:,1])
        z_storage_EV = value.(z[:,1])
        m_missing_EV = value.(m[:,1])
        y_send_EV = value.(y_send[:,:,1])
        y_receive_EV = value.(y_receive[:,:,1])

        # System's total cost
        total_cost_1 = objective_value(model)

        # Return the decisions and cost
        return x_order_EV, z_storage_EV, m_missing_EV, y_send_EV, y_receive_EV, total_cost_1
    #========================================================#    


    else
        println("No solution.")
    end
end

Make_EV_here_and_now_decision(prices)
