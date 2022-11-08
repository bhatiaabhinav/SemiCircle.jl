module SemiCircle

using Random
using LinearAlgebra
using MDPs
import MDPs: state_space, action_space, discount_factor, horizon, action_meaning, state, action, reward, reset!, step!, in_absorbing_state, visualize
using Luxor
using Luxor.Colors
using UnPack

export SemiCircleEnv

mutable struct SemiCircleEnv{T<:AbstractFloat} <: AbstractMDP{Vector{T}, Vector{T}}
    θg::Float64
    Rc::Float64
    Rg::Float64

    horizon::Int
    𝕊::VectorSpace{T}
    𝔸::VectorSpace{T}

    state::Vector{T}  # sinθ, cosθ, x, y
    action::Vector{T} # ω, 𝑣
    reward::Float64

    function SemiCircleEnv{T}(θg; Rc::Real=0.5, Rg::Real=0.2, horizon::Integer=50) where T<:AbstractFloat
        sspace = VectorSpace{T}(T[-1, -1, -Rc, 0], T[1, 1, Rc, Rc])
        aspace = VectorSpace{T}(T[-2π, -1], T[2π, 1])
        new{T}(θg, Rc, Rg, horizon, sspace, aspace, zeros(T, 4), zeros(T, 2), 0.0)
    end
end

@inline state_space(sc::SemiCircleEnv) = sc.𝕊
@inline action_space(sc::SemiCircleEnv) = sc.𝔸
@inline horizon(sc::SemiCircleEnv) = sc.horizon
@inline discount_factor(sc::SemiCircleEnv) = 0.99

function reset!(sc::SemiCircleEnv; rng::AbstractRNG=Random.GLOBAL_RNG)
    θ, x, y = rand(rng) * π, 0, 0
    sc.state .= (sin(θ), cos(θ), x, y)
    sc.reward = 0
    sc.action .= 0
    return nothing
end

function step!(sc::SemiCircleEnv{T}, a::Vector{T}; rng=Random.GLOBAL_RNG) where T
    @assert a ∈ action_space(sc)
    sc.action .= a
    if in_absorbing_state(sc)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        sc.reward = 0.0
    else
        dt = 0.01
        ω, ν = a
        for arepeat in 1:10
            sinθ, cosθ, x, y = sc.state
            θ = atan(sinθ, cosθ) # -π..π
            θ = θ + ω * dt
            x = clamp(x + ν * cos(θ) * dt, -sc.Rc, sc.Rc)
            y = clamp(y + ν * sin(θ) * dt, 0, sc.Rc)

            sc.state .= (sin(θ), cos(θ), x, y)
            reached_goal = distance_to_goal(sc) <= sc.Rg
            sc.reward = Float64(reached_goal) # + sc.θg
            reached_goal && break
        end
    end
    nothing
end

function distance_to_goal(sc::SemiCircleEnv)::Float64
    sinθ, cosθ, x, y = sc.state
    xg, yg = sc.Rc .* (cos(sc.θg), sin(sc.θg))
    d = norm((xg - x, yg - y))
    return d
end

function in_absorbing_state(sc::SemiCircleEnv)::Bool
    return distance_to_goal(sc) <= sc.Rg
end

function visualize(sc::SemiCircleEnv, s::Vector{Float32})::Matrix{ARGB32}
    @unpack Rc, Rg, θg = sc
    W, H = Int.(ceil.((200(Rc+Rg), 100(Rc+Rg))))
    Drawing(W, H + 6, :image)
    origin(Point(W ÷ 2, H))

    background("white")

    setcolor("red")
    setline(3Rc)
    setdash("dot")
    arc(Point(0, 0), 100Rc, π, 2π, :stroke)
    setline(1Rc)
    arc(Point(0, 0), 100(Rc+Rg), π, 2π, :stroke)
    arc(Point(0, 0), 100(Rc-Rg), π, 2π, :stroke)
    xg, yg = Rc .* (cos(θg), sin(θg))
    circle(Point(100xg, -100yg), 100Rg, :fill)

    setcolor("blue")
    sinθ, cosθ, x, y = s
    θ = atan(sinθ, cosθ) # -π..π
    circle(Point(100x, -100y), 5Rc, :fill)
    setdash("solid")
    line(Point(100x, -100y), Point(100(x + 0.1 * cosθ), -100(y + 0.1 * sinθ)), :stroke)

    return image_as_matrix()
end

end # module SemiCircle
