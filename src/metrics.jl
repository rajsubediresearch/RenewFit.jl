# Performance metrics. The WIS alpha set matches computeWIS.m in the MATLAB
# toolboxes, so scores are directly comparable across SubFit and RenewFit.

const WIS_ALPHAS = vcat([0.02, 0.05], collect(0.1:0.1:0.9))

mae(yhat, y) = sum(abs, yhat .- y) / length(y)
mse(yhat, y) = sum(abs2, yhat .- y) / length(y)

function coverage(curves::AbstractMatrix, y::AbstractVector; level::Float64=0.95)
    a = (1 - level) / 2
    c = 0
    for i in eachindex(y)
        row = @view curves[i, :]
        (y[i] >= quantile(row, a) && y[i] <= quantile(row, 1 - a)) && (c += 1)
    end
    return 100 * c / length(y)
end

function wis(curves::AbstractMatrix, y::AbstractVector)
    K = length(WIS_ALPHAS)
    total = 0.0
    for i in eachindex(y)
        row = @view curves[i, :]
        s = 0.5 * abs(y[i] - quantile(row, 0.5))
        for a in WIS_ALPHAS
            l = quantile(row, a / 2); u = quantile(row, 1 - a / 2)
            IS = (u - l) + (2 / a) * (l - y[i]) * (y[i] < l) +
                 (2 / a) * (y[i] - u) * (y[i] > u)
            s += (a / 2) * IS
        end
        total += s / (K + 0.5)
    end
    return total / length(y)
end

performance(curves::AbstractMatrix, point::AbstractVector, y::AbstractVector) =
    (MAE = mae(point, y), MSE = mse(point, y),
     Coverage95 = coverage(curves, y), WIS = wis(curves, y))

"""
    performance_by_horizon(curves, point, y)

Metrics at each forecast horizon separately, so degradation with lead time is
visible rather than averaged away.

Note on coverage: at a SINGLE origin the `covered` column is binary — the
observation either falls inside the 95% interval or it does not. It only
becomes a coverage rate when averaged across many origins, which is what
`examples/horizon_degradation.jl` does. Reading a single-origin coverage
column as a percentage is a mistake.
"""
function performance_by_horizon(curves::AbstractMatrix, point::AbstractVector,
                                y::AbstractVector)
    H = min(size(curves, 1), length(y), length(point))
    rows = NamedTuple[]
    for h in 1:H
        row = curves[h:h, :]
        lo = quantile(@view(curves[h, :]), 0.025)
        hi = quantile(@view(curves[h, :]), 0.975)
        push!(rows, (horizon = h, truth = float(y[h]), point = float(point[h]),
                     AE = abs(point[h] - y[h]), SE = (point[h] - y[h])^2,
                     lo95 = lo, hi95 = hi, width95 = hi - lo,
                     covered = (y[h] >= lo && y[h] <= hi) ? 1 : 0,
                     WIS = wis(row, [float(y[h])])))
    end
    return rows
end
