include("02435_two_stage_problem_data.jl")
include("V2_Assignment_A_codes/V2_price_process.jl")
include("Make_Stochastic_here_and_now_decision.jl")
include("Calculate_OiH_solution.jl")
include("1.b.jl")

using JLD
using Plots

## Generate 100 values (experiments) for the initial prices randomly (uniformly from [0, 10])

number_of_experiments = 100

function Evaluation(number_of_experiments)

    # INITIALIZATION
    # price
    day1_price = zeros(number_of_experiments, 3)
    day2_price = zeros(number_of_experiments, 3)
    # order amount
    day1_x_order = zeros(number_of_experiments, 3, 5) 
    day2_x_order = zeros(number_of_experiments, 3, 5)
    # order sent
    day1_y_send = zeros(number_of_experiments, 3, 3, 5)
    day2_y_send = zeros(number_of_experiments, 3, 3, 5)
    # order received
    day1_y_receive = zeros(number_of_experiments, 3, 3, 5)
    day2_y_receive = zeros(number_of_experiments, 3, 3, 5)
    # storage level
    day1_z_storage = zeros(number_of_experiments, 3, 5)
    day2_z_storage = zeros(number_of_experiments, 3, 5)
    # order missed
    day1_m_missing = zeros(number_of_experiments, 3, 5)
    day2_m_missing = zeros(number_of_experiments, 3, 5)
    # total cost
    total_cost = zeros(number_of_experiments, 5)

    # 100 experiments
    for i in 1:number_of_experiments

        ## Day 1
        prices_experiments = round(10 * rand(3), digits=2)
        day1_price[i,:] = prices_experiments

        # For each experiment, call each program to make a here-and-now decision.
        x_order_EV, z_storage_EV, m_missing_EV, y_send_EV, y_received_EV, total_cost_EV = Make_EV_here_and_now_decision(prices_experiments)
        day1_x_order[i,:,1] = x_order_EV
        day1_z_storage[i,:,1] = z_storage_EV
        day1_m_missing[i,:,1] = m_missing_EV
        day1_y_send[i,:,:,1] = y_send_EV
        day1_y_receive[i,:,:,1] = y_received_EV
        total_cost[i,1] = total_cost_EV
        ## for N = 5
        x_order_ST5, z_storage_ST5, m_missing_ST5, y_send_ST5, y_received_ST5, total_cost_ST5 = Make_Stochastic_here_and_now_decision(prices_experiments, 5)
        day1_x_order[i,:,2] = x_order_ST5
        day1_z_storage[i,:,2] = z_storage_ST5
        day1_m_missing[i,:,2] = m_missing_ST5
        day1_y_send[i,:,:,2] = y_send_ST5
        day1_y_receive[i,:,:,2] = y_received_ST5
        total_cost[i,2] = total_cost_ST5    
        ## for N = 20
        x_order_ST20, z_storage_ST20, m_missing_ST20, y_send_ST20, y_received_ST20, total_cost_ST20 = Make_Stochastic_here_and_now_decision(prices_experiments, 20)
        day1_x_order[i,:,3] = x_order_ST20
        day1_z_storage[i,:,3] = z_storage_ST20
        day1_m_missing[i,:,3] = m_missing_ST20
        day1_y_send[i,:,:,3] = y_send_ST20
        day1_y_receive[i,:,:,3] = y_received_ST20
        total_cost[i,3] = total_cost_ST20
        ## for N = 50
        x_order_ST50, z_storage_ST50, m_missing_ST50, y_send_ST50, y_received_ST50, total_cost_ST50 = Make_Stochastic_here_and_now_decision(prices_experiments, 50)
        day1_x_order[i,:,4] = x_order_ST50
        day1_z_storage[i,:,4] = z_storage_ST50
        day1_m_missing[i,:,4] = m_missing_ST50
        day1_y_send[i,:,:,4] = y_send_ST50
        day1_y_receive[i,:,:,4] = y_received_ST50
        total_cost[i,4] = total_cost_ST50


        ## Day2
        new_prices_experiments = round.(map(Float64,map(sample_next, prices_exp)),digits=2)
        Prices_day2[i,:] = new_prices_experiments

        










