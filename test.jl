using JuMP
using Xpress

function test()
    # 1. Create the model and specify the solver
    model = Model(Xpress.Optimizer)

    # 2. Define variables
    # JuMP handles indexing and naming very efficiently
    @variable(model, x[i=1:2] >= 0)

    # 3. Add constraints
    # You can write math naturally!
    @constraint(model, x[1] + x[2] == 1)

    # 4. Set the objective
    # Note: x[1]^2 is natively supported
    @objective(model, Min, x[1]^2 + 2 * x[2]^2)

    # 5. Optimize
    optimize!(model)

    # 6. Print results
    println("x1 = $(value(x[1])), x2 = $(value(x[2]))")
end

test()