struct ManifoldNewton{IL, L} <: SecondOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    retract!::Function
end

"""
# Newton
## Constructor
```julia
Newton(; alphaguess = LineSearches.InitialStatic(),
linesearch = LineSearches.HagerZhang())
```

## Description
The `Newton` method implements Newton's method for optimizing a function. We use
a special factorization from the package `PositiveFactorizations.jl` to ensure
that each search direction is a direction of descent. See Wright and Nocedal and
Wright (ch. 6, 1999) for a discussion of Newton's method in practice.

## References
 - Nocedal, J. and S. J. Wright (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function ManifoldNewton(retract!; alphaguess = LineSearches.InitialStatic(), # Good default for Newton
                        linesearch = LineSearches.HagerZhang())    # Good default for Newton

    ManifoldNewton(_alphaguess(alphaguess), linesearch, retract!)
end

Base.summary(::ManifoldNewton) = "Manifold Newton's Method"

# mutable struct NewtonState{Tx, T, F<:Cholesky} <: AbstractOptimizerState
#     x::Tx
#     x_previous::Tx
#     f_x_previous::T
#     F::F
#     s::Tx
#     @add_linesearch_fields()
# end

function initial_state(method::ManifoldNewton, options, d, initial_x)
    T = eltype(initial_x)
    n = length(initial_x)
    # Maintain current gradient in gr
    s = similar(initial_x)

    x0 = copy(initial_x)
    method.retract!(x0)
    value_gradient!!(d, x0)
    hessian!!(d, x0)

    NewtonState(copy(x0), # Maintain current state in state.x
                copy(x0), # Maintain previous state in state.x_previous
                T(NaN), # Store previous f in state.f_x_previous
                Cholesky(similar(d.H, T, 0, 0), :U, BLAS.BlasInt(0)),
                similar(initial_x), # Maintain current search direction in state.s
                @initial_linesearch()...)
end

function update_state!(d, state::NewtonState, method::ManifoldNewton)
    # Search direction is always the negative gradient divided by
    # a matrix encoding the absolute values of the curvatures
    # represented by H. It deviates from the usual "add a scaled
    # identity matrix" version of the modified Newton method. More
    # information can be found in the discussion at issue #153.
    T = eltype(state.x)


    state.F = cholesky!(Positive, NLSolversBase.hessian(d))
    if typeof(gradient(d)) <: Array
        # is this actually StridedArray?
        ldiv!(state.s, state.F, -gradient(d))
    else
        error("Wrong")
    end
    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Update current position # x = x + alpha * s
    @. state.x = state.x + state.alpha * state.s

    method.retract!(state.x)
    lssuccess == false # break on linesearch error
end

function trace!(tr, d, state, iteration, method::ManifoldNewton, options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["h(x)"] = copy(NLSolversBase.hessian(d))
        dt["Current step size"] = state.alpha
    end
    g_norm = norm(gradient(d), Inf)
    update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
