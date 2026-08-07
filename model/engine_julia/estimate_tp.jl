#!/usr/bin/env julia
# Estimate a yields-only AFNS-restricted term-structure model and decompose the
# term premium, using TermStructureModels.jl (MIT).
#
# Usage:
#   julia --project=<this dir> estimate_tp.jl PANEL_CSV OUT_CSV \
#         [iteration=6000] [burnin=1000] [maturities_years=1,2,3,5,7,10] [upper_p=24]
#
# PANEL_CSV : columns = `date` + one column per maturity (header = maturity in
#             YEARS), yields in PERCENT per annum. (See us.engine.write_panel_csv.)
# OUT_CSV   : canonical long frame `date,maturity,term_premium,expected_rate,
#             fitted_yield` in DECIMAL — the schema validation/compare_afns reads.
#
# Convention (TermStructureModels docs): data_scale=1200 for monthly percent p.a.
# Term-premium / EH outputs come back in percent p.a.; we divide by 100 to emit
# decimals. fitted = EH + TP by the model's decomposition identity.
using TermStructureModels, CSV, DataFrames, Statistics, Dates

function main()
    panel_csv = ARGS[1]
    out_csv   = ARGS[2]
    iteration = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6000
    burnin    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000
    mats_yr   = length(ARGS) >= 5 ? parse.(Float64, split(ARGS[5], ",")) : [1.0,2.0,3.0,5.0,7.0,10.0]
    upper_p   = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 24

    df = CSV.read(panel_csv, DataFrame)
    dates = df.date
    ycols = names(df)[2:end]                       # maturity-in-years headers
    yields = Matrix{Float64}(df[:, 2:end])         # T×N, PERCENT
    T, N = size(yields)
    tau_all_months = [round(Int, parse(Float64, string(c)) * 12) for c in ycols]
    tp_tau_months  = [round(Int, m * 12) for m in mats_yr]
    macros = zeros(T, 0)                            # yields-only
    rho    = zeros(0)
    medium_tau = collect(24:3:48)

    println("panel: T=$T N=$N  $(dates[1])..$(dates[end])  tau(months)=$tau_all_months")
    println("TP maturities (months)=$tp_tau_months  iter=$iteration burnin=$burnin upper_p=$upper_p")
    flush(stdout)

    tuned, _ = tuning_hyperparameter(yields, macros, tau_all_months, rho; medium_tau, upper_p=upper_p)
    println("tuned p=$(tuned.p)"); flush(stdout)

    saved, accept = posterior_sampler(yields, macros, tau_all_months, rho, iteration, tuned; medium_tau)
    saved = saved[burnin+1:end]
    saved, pr_stat = erase_nonstationary_param(saved)
    println("draws kept=$(length(saved))  Pr_stationary=$(round(pr_stat,digits=3))  accept=$(round.(accept,digits=2))")
    flush(stdout)

    saved_TP = term_premium(tp_tau_months, tau_all_months, saved, yields, macros)[1]

    # Posterior-mean TP and EH; orient rows = time.
    orient(M) = size(M,1) == length(tp_tau_months) && size(M,2) != length(tp_tau_months) ? permutedims(M) : M
    TPsum = zeros(size(orient(saved_TP[1].TP)))
    EHsum = zeros(size(orient(saved_TP[1].EH)))
    for s in saved_TP
        TPsum .+= orient(s.TP)
        EHsum .+= orient(s.EH)
    end
    TP = TPsum ./ length(saved_TP)                  # (nrow × n_tau), percent
    EH = EHsum ./ length(saved_TP)
    nrow = size(TP, 1)
    tp_dates = dates[end-nrow+1:end]                # TP defined over t=p+1:T

    # Emit canonical long frame in DECIMAL.
    rows = DataFrame(date=Date[], maturity=Float64[], term_premium=Float64[],
                     expected_rate=Float64[], fitted_yield=Float64[])
    for (j, mat) in enumerate(mats_yr)
        tp = TP[:, j] ./ 100
        eh = EH[:, j] ./ 100
        for i in 1:nrow
            push!(rows, (tp_dates[i], mat, tp[i], eh[i], tp[i] + eh[i]))
        end
    end
    CSV.write(out_csv, rows)
    println("WROTE $out_csv  ($(nrow) dates × $(length(mats_yr)) maturities)")
end

main()
