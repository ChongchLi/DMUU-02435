using JuMP
using Gurobi
using Printf
using Clustering
using Distances

include("V2_Assignment_A_codes/V2_02435_two_stage_problem_data.jl")
include("V2_Assignment_A_codes/V2_price_process.jl")
include("fast-forward-selection.jl")



function fast_forward(prices, num_of_scenarios)
    number_of_warehouses = 3
    num_sampled_scenarios = 1000
    #Calculate distance matrix (euclidean distance)
    Distance_matrix = zeros(Float64, num_sampled_scenarios, num_sampled_scenarios)
    for i in 1:num_sampled_scenarios
        for j in 1:num_sampled_scenarios
            distance = sqrt(sum((prices[l, i] - prices[l, j])^2 for l in eachindex(1:size(prices, 1))))
            Distance_matrix[i, j] = distance
        end
    end

    #Initialize equiprobable probabilities
    probabilities = repeat([1.0/num_sampled_scenarios], 1, num_sampled_scenarios)[1,:]
    #Include fast forward selection and apply it

    result = FastForwardSelection(Distance_matrix, probabilities, num_of_scenarios)

    #Resulting probabilities
    new_probabilities = result[1]
    #Selected scenario indices
    scenario_indices = result[2]

    reduced_prices = zeros(Float64, number_of_warehouses, num_of_scenarios)
    for i = 1:num_of_scenarios
        reduced_prices[:,i] = prices[:,scenario_indices[i]]
    end

    return reduced_prices, new_probabilities

end

function cluster_kmeans(prices, num_of_scenarios)

    # perform K-means clustering
    results = kmeans(prices, num_of_scenarios; display=:iter)
    # Get the cluster centers
    reduced_prices = results.centers

    # Assignments of data points to clusters
    assigned = assignments(results) # get the assignments of points to clusters
    new_probabilities = zeros(Float64, num_of_scenarios)
    for a in assigned
        new_probabilities[a] = new_probabilities[a] + (1/length(assigned))
    end

    return reduced_prices, new_probabilities

end

function cluster_kmedoids(prices, num_of_scenarios)

    number_of_warehouses = 3
    num_sampled_scenarios = 1000
    #Calculate distance matrix (euclidean distance)
    Distance_matrix = zeros(Float64, num_sampled_scenarios, num_sampled_scenarios)
    for i in 1:num_sampled_scenarios
        for j in 1:num_sampled_scenarios
            distance = sqrt(sum((prices[l, i] - prices[l, j])^2 for l in eachindex(1:size(prices, 1))))
            Distance_matrix[i, j] = distance
        end
    end
    
    # Apply k-medoids algorithm
    result = kmedoids(Distance_matrix, num_of_scenarios;display=:iter)
    # resulting medoids(indices of data points)
    M = result.medoids

    reduced_prices = zeros(number_of_warehouses, num_of_scenarios)

    for i in 1:num_of_scenarios
        reduced_prices[:,i] = prices[:,M[i]]
    end

    #Assignments of data points to clusters
    assigned = assignments(result) # get the assignments of points to clusters
    new_probabilities = zeros(Float64, num_of_scenarios)
    for a in assigned
        new_probabilities[a] = new_probabilities[a] + (1/length(assigned))
    end
    return reduced_prices, new_probabilities
end

    

