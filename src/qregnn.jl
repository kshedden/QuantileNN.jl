using JuMP, Tulip, Random, Statistics, NearestNeighbors, StatsBase
using LightGraphs, MathOptInterface

# Implement the nonparametric quantile regression approach described here:
# https://arxiv.org/abs/2012.01758

# Representation of a fitted model
mutable struct Qreg

    # The outcome variable
    y::Vector{Float64}

    # The covariates used to define the nearest neighbors
    x::Matrix{Float64}

    # Indices of the nearest neighbors
    nn::Matrix{Int}

    # A tree for finding neighbors
    kt::KDTree

    # The optimization model
    model::JuMP.Model

    # The fitted values from the most recent call to fit
    fit::Vector{Float64}

    # The probability point for the most recent fit
    p::Float64

    # Retain these references from the optimization model
    rpos::Array{JuMP.VariableRef,1}
    rneg::Array{JuMP.VariableRef,1}
    dcap::Array{JuMP.VariableRef,2}
    rfit::Array{JuMP.VariableRef,1}
end

# Returns the degrees of freedom of the fitted model, which is the
# number of connected components of the graph defined by all edges
# of the covariate graph that are fused in the regression fit.
function degf(qr::Qreg; e = 1e-2)::Int

    nn = qr.nn
    fv = qr.fit
    g = SimpleGraph(size(nn, 1))

    for i = 1:size(nn, 1)
        for j = 1:size(nn, 2)
            if abs(fv[i] - fv[nn[i, j]]) < e
                add_edge!(g, i, nn[i, j])
            end
        end
    end

    return length(connected_components(g))
end

# Returns the BIC for the given fitted model.
function bic(qr::Qreg)::Tuple{Float64,Int}
    d = degf(qr)
    p = qr.p
    resid = qr.y - qr.fit
    pos = sum(x -> clamp(x, 0, Inf), resid)
    neg = -sum(x -> clamp(x, -Inf, 0), resid)
    check = p * pos + (1 - p) * neg
    sig = (1 - abs(1 - 2 * p)) / 2
    n = length(qr.y)
    return tuple(2 * check / sig + d * log(n), d)
end

function fittedvalues(qr::Qreg)
	return qr.fit
end

# Predict the quantile at the point z using k nearest neighbors.
function predict(qr::Qreg, z::AbstractVector{Float64}; k = 5)::Float64
    ii, _ = knn(qr.kt, z, k)
    return mean(qr.fit[ii])
end

# Predict the quantile at the point z using local linear regression.
function predict_smooth(qr::Qreg, z::AbstractVector{Float64}, bw::AbstractVector{Float64})::Float64
	@assert minimum(bw) > 0
    f = qr.fit
    x = qr.x
    n, r = size(x)
    xtx = zeros(r + 1, r + 1)
    xty = zeros(r + 1)
    xr = ones(r + 1)
    for i = 1:n
        xr[2:end] = x[i, :] - z
        e2 = sum(abs2, xr[2:end] ./ bw)
        w = exp(-e2 / 2)
        xtx .= xtx + w * xr * xr'
        xty .= xty + w * f[i] * xr
    end

    b = xtx \ xty
    return b[1]
end

# Construct a structure that can be used to perform quantile regression
# of y on x.  k is the number of nearest neighbors used for regularization.
function qregnn(y::Vector{Float64}, x::Matrix{Float64}; p::Float64=0.5, k::Int = 5, 
                lam::Float64=0.1, dofit::Bool=true)::Qreg

    n = length(y)

    # Build the nearest neighbor tree, exclude each point from its own
    # neighborhood.
    kt = KDTree(x')
    nx, _ = knn(kt, x', k + 1, true)
    nn = hcat(nx...)'
    nn = nn[:, 2:end]

    model = Model(Tulip.Optimizer)

    # The estimated quantile for each row of the design matrix.
    @variable(model, rfit[1:n])

    # The residuals y - rfit are decomposed into their positive
    # and negative parts.
    rpos = @variable(model, rpos[1:n])
    rneg = @variable(model, rneg[1:n])

    # The distance between the fitted value of each point
    # and its nearest neighbor is bounded by dcap.
    dcap = @variable(model, dcap[1:n, 1:k])

    @constraint(model, rpos - rneg .== y - rfit)
    @constraint(model, rpos .>= 0)
    @constraint(model, rneg .>= 0)
    @constraint(model, dcap .>= 0)
    for j = 1:k
        @constraint(model, rfit - rfit[nn[:, j]] .<= dcap[:, j])
        @constraint(model, rfit[nn[:, j]] - rfit .<= dcap[:, j])
    end

    qr = Qreg(y, x, nn, kt, model, Vector{Float64}(), -1, rpos, rneg, dcap, rfit)

	if dofit
		fit!(qr, p; lam=lam)
	end

	return qr
end


# Estimate the p'th quantiles for the population represented by the data
# in qr. lam is a penalty parameter controlling the smoothness of the
# fit.
function StatsBase.fit!(qr::Qreg, p::Float64; lam::Float64=0.1)

    @objective(qr.model, Min, sum(p * qr.rpos + (1 - p) * qr.rneg) + lam * sum(qr.dcap))

    optimize!(qr.model)
    if termination_status(qr.model) != MathOptInterface.OPTIMAL
        error("Qreg fit did not converge")
    end
    qr.fit = value.(qr.rfit)
end


# Search for a tuning parameter based on BIC.  Starting from
# lambda=0.1, increase the tuning parameter sequentially
# by a factor of 'fac'.  The iterations stop when the current
# BIC is greater than the previous BIC, or when the degrees of
# freedom is less than or equal to 'dof_min', or when the value of
# lambda is greater than 'lam_max'. The path is returned as an array
# of triples containing the tuning parameter value, the BIC
# value, and the degrees of freedom.
function bic_search(qr::Qreg, p::Float64; fac = 1.2, lam_max = 1e6, dof_min = 2)

    pa = []

    lam = 0.1
    while lam < lam_max
        _ = fit!(qr, p; lam=lam)
        b, d = bic(qr)
        push!(pa, [lam, b, d])
        if (d <= dof_min) || (length(pa) > 1 && b > pa[end-1][2])
            break
        end
        lam = lam * fac
    end

    la = minimum([x[2] for x in pa])

    return tuple(la, pa)
end
