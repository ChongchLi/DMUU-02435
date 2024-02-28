using JuMP
using Gurobi
using Random
using Printf
include("V2_Assignment_A_codes /V2_price_process.jl")
include("V2_Assignment_A_codes /V2_02435_two_stage_problem_data.jl")

## Parameters, load the data 
number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, T, demand_trajectory = load_the_data()


Coffee = Model(Gurobi.Optimizer)

#variables
@variable(Coffee, 0 <= x[w in W, t in T])
@variable(Coffee, 0 <= z[w in W, t in T])
@variable(Coffee, 0 <= m[w in W, t in T])
@variable(Coffee, 0 <= ys[w in W, q in W, t in T])
@variable(Coffee, 0 <= yr[w in W, q in W, t in T])

#objective
@objective(Coffee, Min, sum((sample_next(0)*x[w,t])
                              +sum(cost_tr[w,q]*ys[w,q,t] for q in W, t in T)
                              +cost_miss[w]*m[w,t] for w in W, t in T))
#constraint
# @constraint(Coffee, storage_capicity[w in W, t in T], z[w,t] <= warehouse_capacities)
# @constraint(Coffee, transportation_capicity[w in W, q in W, t in T], ys[w,q,t] <= transport_capacities[w,q])
# @constraint(Coffee, inventory_balance[w in W, q in W, t in T], demand_trajectory[w,t]=z[w,t-1]+x[w,t]+sum[yr]-sum[ys]+m[w,t])
# @constraint(Coffee, inventory_balance2[w in W, q in W, t in T], ys[w,q,t]<=yr[w,q,t-1]+x[w,q,t-1])

# demand fulfillment and inventory balance
for w in W, t in T
    if t == 1
        @constraint(Coffee, x[w,t] + initial_stock[w] + sum(yr[w,q,t] for q in W) - sum(ys[w,q,t] for q in W) + m[w,t] == demand_trajectory[w,t])
    else
        @constraint(Coffee, x[w,t] + z[w,t-1] + sum(yr[w,q,t] for q in W) - sum(ys[w,q,t] for q in W) + m[w,t] == demand_trajectory[w,t])
    end
end

# Warehouse capacity
for w in W, t in T
    @constraint(Coffee, z[w,t] <= warehouse_capacities[w])
end

# Transportation capacity
for w in W, q in W, t in T
    @constraint(Coffee, ys[w,q,t] <= transport_capacities[w,q])
end



optimize!(Coffee)

if termination_status(Coffee) == MOI.OPTIMAL
    println("Optimal solution found")
    
    # println("order for external suppliers: $([value.(x[m,t]) for m in M, t in T])")
    println("order for external suppliers: $([value.(x[w,t]) for w in W, t in T])")
    println()
   

    

@printf "\nObjective value: %0.3f\n\n" objective_value(Coffee)

else
    error("No solution.")
end

