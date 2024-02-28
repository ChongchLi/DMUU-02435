using Random
using JuMP
using Gurobi
using Printf

include("/Users/xiaoqian/Desktop/Decision making under uncertainty/Assignmenrt/V2_Assignment_A_codes/V2_02435_two_stage_problem_data.jl")
include("/Users/xiaoqian/Desktop/Decision making under uncertainty/Assignmenrt/V2_Assignment_A_codes/V2_price_process.jl")

# Prices are known for today
prices = round.(10 * rand(3), digits=2)

function Make_EV_here_and_now_decision(prices)
    # Import the data
    load_the_data() = number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory
    
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
    return prices2
end

prices2 = Make_EV_here_and_now_decision(prices)
CoffeePrice = [prices, prices2]


model_1 = Model(Gurobi.Optimizer)

 #Variables
 #At stage t, warehouse w can order an amount x_(w,t)≥0 of coffee from external suppliers
 @variable(model_1, x[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
 #At stage t, storage level of w
 @variable(model_1, z[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
 #At stage t, the missing amount of w
 @variable(model_1, m[1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
 #At stage t, the amount of coffee is sent from warehouse w to the neighboring warehouse q
 @variable(model_1, y_send[1:number_of_warehouses, 1:number_of_warehouses, 1:number_of_simulation_periods]>=0)
 #At stage t, the amount of coffee is received by the neighboring warehouse q
 @variable(model_1, y_receive[1:number_of_warehouses, 1:number_of_warehouses, 1:number_of_simulation_periods]>=0)

 #Objective model 
        @objective(model_1, Min, sum(x[w,t]*CoffeePrice[t][w] for w in 1:number_of_warehouses, t in 1:number_of_simulation_periods) 
        + sum(y_send[w,q,t]*cost_tr[w,q] for w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods)
        + sum(m[w,t]*cost_miss[w] for w in 1:number_of_warehouses, t in 1:number_of_simulation_periods))
    
 #Constraints 
 #Storage capacity should be under warehouse capacity
 @constraint(model_1, StorageCap[w in 1:number_of_warehouses, t in 1:number_of_simulation_periods], z[w,t] <= warehouse_capacities[w])
 #The amount of warehouse sent should not beyond the transpotation capicity
 @constraint(model_1, TransportationCap[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w,q,t]<=transport_capacities[w,q])
 #Inventory balance 
 @constraint(model_1, Storage[w in 1:number_of_warehouses, t in 2:number_of_simulation_periods], demand_trajectory[w,t]== z[w,t-1]-z[w,t]+x[w,t]
    +sum(y_receive[w,q,t] - y_send[w,q,t] for q in 1:number_of_warehouses) + m[w,t])
 @constraint(model_1, Storage_start[w in 1:number_of_warehouses], demand_trajectory[w,1] == initial_stock[w]-z[w,1]+x[w,1]
    +sum(y_receive[w,q,1] - y_send[w,q,1] for q in 1:number_of_warehouses) + m[w,1])
 #warehouse w must already have this amount previously w,q,t stored (i.e., the coffee ordered by w at t, cannot be sent to q at the same day
 @constraint(model_1, Transportation[w in 1:number_of_warehouses, q in 1:number_of_warehouses, t in 2:number_of_simulation_periods], sum(y_send[w,q,t] for q in 1:number_of_warehouses) <= z[w,t-1])
 @constraint(model_1, Transportion_start[w in 1:number_of_warehouses, q in 1:number_of_warehouses], sum(y_send[w,q,1] for q in 1:number_of_warehouses) <= initial_stock[w])
 #warehouse can not send coffee to itself 
 @constraint(model_1, Limitation[w in 1:number_of_warehouses, t in 1:number_of_simulation_periods], y_send[w,w,t] == 0)
 
 optimize!(model_1)

 if termination_status(model_1) == MOI.OPTIMAL
    println("Optimal solution")
    
    println("Minimal Cost : $(round.(objective_value(model_1), digits=4))")
    println("-----------------")
    for t in 1:number_of_simulation_periods
        println("Time Step: $t")
        println("----") 
        for w in 1:number_of_warehouses 
            println("Warehouse $w : Demand $(demand_trajectory[w,t]) / Ordered $(round.(value.x[w,t], digits=2)) / Price $(CoffeePrice[t][w])")
            if t != 1 
                println("Previous Stock $(round.(value.(z)[w,t-1], digits=2)) / Sent $(sum(round.(value.y_send[w,q,t], digits=2) for q in 1:number_of_warehouses)) / Recieved $(sum(round.(value.y_recieved[w,q,t], digits=2) for q in 1:number_of_warehouses))")
            else 
                println("Previous Stock 2.00 / Sent $(sum(round.(value.y_send[w,q,t], digits=2) for q in 1:number_of_warehouses)) / Recieved $(sum(round.(value.y_recieved[w,q,t], digits=2) for q in 1:number_of_warehouses))")
            end
            println("Missed $(round.(value.m[w,t], digits=2)) / Stock $(round.(value.z[w,t], digits=2))")
            println("----")  
        end
        println("-----------------") 
    end

end

   