using JuMP
using Gurobi
using Printf
using Clustering
using Distances

#Performs fast forward selection for the given parameters
#D = Symmetric distance matrix
#p = vector of probabilities
#n = target number of scenarios
#Returns Array with 2 element, [1] = list of probabilities, [2] = list of selected scenario indices
function FastForwardSelection(D, p, n)
    init_d = D
    not_selected_scenarios = collect(range(1,length(D[:,1]);step=1))
    selected_scenarios = []
    while length(selected_scenarios) < n
        selected = select_scenario(D, p, not_selected_scenarios)
        deleteat!(not_selected_scenarios, findfirst(isequal(selected), not_selected_scenarios))
        push!(selected_scenarios, selected)
        D = UpdateDistanceMatrix(D, selected, not_selected_scenarios)
    end
    result_prob = RedistributeProbabilities(D, p, selected_scenarios, not_selected_scenarios)
    return [result_prob, selected_scenarios]
end

#Redistributes probabilities at the end of the fast forward selection
#D = original distance matrix
#p = probabilities
#selected_scenarios = indices of selected scenarios
#not_selected_scenarios = indices of non selected scenarios
function RedistributeProbabilities(D, p, selected_scenarios, not_selected_scenarios)
    probabilities = p
    for s in not_selected_scenarios
        min_idx = -1
        min_dist = Inf
        for i in selected_scenarios
            if D[s,i] < min_dist
                min_idx = i
                min_dist = D[s,i]
            end
        end
        probabilities[min_idx] = probabilities[min_idx] + p[s]
        probabilities[s] = 0.0
    end
    new_probabilities = [probabilities[i] for i in selected_scenarios]
    return new_probabilities
end

#Updates the distance matrix in the fast forward selection
#D = current distance matrix
#selected = index of scenario selected in this iteration
#scenarios = index list of not selected scenarios
function UpdateDistanceMatrix(D, selected, not_selected_scenarios)
    for s in not_selected_scenarios
        if s!=selected
            for s2 in not_selected_scenarios
                if s2!=selected
                    D[s,s2] = min(D[s,s2], D[s,selected])
                end
            end
        end
    end
    return D
end

#Selects the scenario idx with minimum Kantorovic distance
#D = Distance matrix
#p = probabilities
#scenarios = not selected scenarios
function select_scenario(D, p, not_selected_scenarios)
    min_dist = Inf
    min_idx = -1
    for s in not_selected_scenarios
        dist = sum(p[s2]*D[s2,s] for s2 in not_selected_scenarios if s!=s2)
        if dist < min_dist
            min_dist = dist
            min_idx = s
        end
    end
    return min_idx
end


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



