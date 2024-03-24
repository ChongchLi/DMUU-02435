using BenchmarkTools
using JLD2
using FileIO
using Plots
using JuMP
using Gurobi
using Statistics

# Assuming all required modules are in the same directory and correctly include all necessary dependencies
include("V2_02435_multistage_problem_data.jl")
include("V2_simulation_experiments.jl")
include("V2_price_process.jl")
include("V2_feasibility_check.jl")
include("02435_multistage_policy.jl")
include("02435_EV_policy.jl")

number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, transport_capacities, initial_stock, number_of_simulation_periods, sim_T, demand_trajectory = load_the_data()
number_of_experiments, Expers, Price_experiments = simulation_experiments_creation(number_of_warehouses, W, number_of_simulation_periods)

function evaluate_policy(policy, reduce_type, lookahead_days, num_of_reduced_scenarios, initial_scenarios, granularity)
    # Initialization of the decision variables and policy cost
    x = Dict()
    send = Dict()
    receive = Dict()
    z = Dict()
    m = Dict()
    policy_cost = 99999999*ones(number_of_experiments, number_of_simulation_periods)
    policy_cost_at_experiment = 99999999*ones(number_of_experiments)

    total_time = @elapsed begin
        # for each experiment
        for e in Expers
            println("EXPERIMENT N°$e LAUNCHED")
            # and for each timeslot of the horizon
            for tau in sim_T
                println("    CURRENTLY OPTIMIZING DAY $tau")
                # Set each warehouse's stock level 
                if tau == 1
                    current_stock = initial_stock
                else
                    current_stock = z[(e,tau-1)]
                end
                # Observe current demands and prices
                current_demands = demand_trajectory[:,tau]
                current_prices = Price_experiments[e,:,tau]

                # Call policy to make a decision for here and now
                elapsed_time = @elapsed begin
                    if policy == "MP"
                        x[(e,tau)], send[(e,tau)], receive[(e,tau)], z[(e,tau)], m[(e,tau)] = make_multistage_here_and_now_decision(number_of_simulation_periods, num_of_reduced_scenarios, tau, current_stock, current_prices, lookahead_days, initial_scenarios, granularity, reduce_type)
                    elseif policy == "EV"
                        x[(e,tau)], send[(e,tau)], receive[(e,tau)], z[(e,tau)], m[(e,tau)] = Make_EV_here_and_now_decision(number_of_simulation_periods, tau, current_stock, current_prices, lookahead_days, initial_scenarios)
                    end
                end

                # Print the elapsed time
                println("Elapsed time: $elapsed_time seconds")
                
                #Check whether the policy's here and now decisions are feasible/meaningful
                successful = check_feasibility(x[(e,tau)], send[(e,tau)], receive[(e,tau)], z[(e,tau)], m[(e,tau)], current_stock, current_demands,  warehouse_capacities, transport_capacities)
                # If not, then the policy's decisions are discarded for this timeslot, and the dummy policy is used instead
                if successful == 0
                    println("DECISION DOES NOT MEET THE CONSTRAINTS FOR THIS TIMESLOT. THE DUMMY POLICY WILL BE USED INSTEAD")
                    println(e, number_of_simulation_periods, tau, current_stock, current_demands, x[(e,tau)], send[(e,tau)], receive[(e,tau)], z[(e,tau)], m[(e,tau)])
                    global ep = e
                    global taup = tau
                    global current_stockp = current_stock
                    global current_demandsp = current_demands
                    global xp = x[(e,tau)]
                    global sp = send[(e,tau)]
                    global rp = receive[(e,tau)] 
                    global zp = z[(e,tau)]
                    global mp = m[(e,tau)]
                    x[(e,tau)], send[(e,tau)], receive[(e,tau)], z[(e,tau)], m[(e,tau)] = make_dummy_decision(number_of_simulation_periods, tau, current_stock, current_demands, current_prices)
                end

                policy_cost[e,tau] = sum(current_prices[w]*x[(e,tau)][w] + cost_miss[w]*m[(e,tau)][w] + sum(cost_tr[w,q]*receive[(e,tau)][w,q] for q in W) for w in W)  
            end
            policy_cost_at_experiment[e] = sum(policy_cost[e,tau] for tau in sim_T)

        end

        FINAL_POLICY_COST = sum(policy_cost_at_experiment[e] for e in Expers) / number_of_experiments
    end
    # Print the total time and the final policy cost
    println("Total time: $total_time seconds")
    println("THE FINAL POLICY EXPECTED COST IS ", FINAL_POLICY_COST)
    return(FINAL_POLICY_COST, total_time)     
end

# Define your evaluation parameters
lookahead_days = 3
num_of_reduced_scenarios = 20
initial_scenarios = 100
granularity = 0.5

# Initialize a dictionary to hold the results
results = Dict()

# Evaluate each policy method
for method in ["kmeans", "kmedoids", "fastforward", "EV"]
    println("Evaluating policy: $method")
    policy_cost, total_time = evaluate_policy(
        method == "EV" ? "EV" : "MP",
        method,
        lookahead_days,
        num_of_reduced_scenarios,
        initial_scenarios,
        granularity
    )
    results[method] = (policy_cost, total_time)
end

# Save the results to a file
FileIO.save("evaluation_results.jld2", "results", results)

# Convert keys of the results dictionary to a list for plotting
methods = collect(keys(results))
costs = [results[method][1] for method in methods]
times = [results[method][2] for method in methods]

# Plotting the results with axis labels
cost_plot = bar(methods, costs, title = "Policy Cost Comparison", legend = false, xlabel = "Policy Method", ylabel = "Cost")
time_plot = bar(methods, times, title = "Policy Time Comparison", legend = false, xlabel = "Policy Method", ylabel = "Time (seconds)")

# Save the plots
savefig(cost_plot, "policy_costs_comparison.png")
savefig(time_plot, "policy_times_comparison.png")

# Display the plots
display(cost_plot)
display(time_plot)


# Print results
println("Policy comparison results:")
for (method, (cost, time)) in results
    println("$method: Cost = $cost, Time = $time seconds")
end
