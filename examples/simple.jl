using QuantileNN, UnicodePlots, Distributions

function example1()

    n = 1000
    x = randn(n, 2)
    ey = x[:, 1] .^ 2 
    y = ey + randn(n)

    qr = qregnn(y, x; dofit=false)

	for p in [0.25, 0.5, 0.75]
		fit!(qr, p)
		for mode in [1, 2]

			# True quantiles
			yq = ey .+ quantile(Normal(), p)
			
			fv = if mode == 1
				# Predict using local averaging
				[predict(qr, r; k=10) for r in eachrow(x)]
			else
				# Predict using local linear fitting
				[predict_smooth(qr, r, [0.1, 10]) for r in eachrow(x)]
			end
			plt = lineplot([-3, 3], [-3, 3], xlim=(-3, 3), ylim=(-3, 3))
			scatterplot!(plt, yq, fv)
			println("Quantiles at probability p=", p)
			println(plt)
		end
	end
end

function example2()
    nrep = 20
    n = 1000
    p = 0.5
    for j = 1:nrep
        for k in [1, 2]

            x = randn(n, k)
            y = x[:, 1] .^ 2 + randn(n)

            qr = qregnn(y, x)

            la, pa = bic_search(qr, p, lam_max = 1e6)
            x = [z[2] for z in pa]
            x = x .- minimum(x)
            plt = lineplot(x[3:end])
            println(plt)
        end
    end
end
