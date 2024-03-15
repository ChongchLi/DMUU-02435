using Random
using Distributions

function sample_next(previous_point)
    sample = previous_point + rand(Gamma(1.0, 2.0)) * rand(Normal((5 - previous_point) * 0.3, 1))

    if sample < 0
        sample = 0
    end

    if sample > 10
        sample = 10
    end

    rand_num = rand()

    if rand_num < 0.9
        price_sample = sample
    else
        price_sample = 10
    end

    return price_sample
end

 function simulate_second_day_price(pw_1)
           total_price = 0.0
           for _ in 1:1000
               price_sample = sample_next(pw_1)
               total_price += price_sample
           end
           expected_price = total_price / 1000
           return expected_price
       end

pw_1 = 7.0  # we assume that the price of frist day is 7
simulate_second_day_price(pw_1)

