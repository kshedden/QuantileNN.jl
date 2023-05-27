module QuantileNN

	import StatsAPI: fit, predict, RegressionModel, fitted

	export QNN, fit, fit!, predict, predict_smooth, fitted
	export bic_search

	include("qregnn.jl")

end # module
