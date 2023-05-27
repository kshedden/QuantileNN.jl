using Test
using QuantileNN
using StableRNGs
using Statistics

@testset "test1" begin

    rng = StableRNG(123)
    n = 2000
    X = randn(rng, n, 2)
    y = X[:, 1] + randn(rng, n)

    qr1 = fit(QNN, X, y; p=0.25)
    qr2 = fit(QNN, X, y; p=0.5)
    qr3 = fit(QNN, X, y; p=0.75)

    yq = zeros(n, 3)
    yq[:, 1] = fitted(qr1)
    yq[:, 2] = fitted(qr2)
    yq[:, 3] = fitted(qr3)

    ax = [-0.67, 0, 0.67] # True intercepts
    for j = 1:3
        c = cov(yq[:, j], X[:, 1])
        b = c / var(X[:, 1])
        a = mean(yq[:, j]) - b * mean(X[:, 1])
        @test abs(b - 1) < 0.1 # True slope is 1
        @test abs(a - ax[j]) < 0.15
    end

    bw = Float64[2, 2]
    @test abs(predict_smooth(qr1, [0.0, 0.0], bw) - ax[1]) < 0.15
    @test abs(predict_smooth(qr2, [0.0, 0.0], bw) - ax[2]) < 0.15
    @test abs(predict_smooth(qr3, [0.0, 0.0], bw) - ax[3]) < 0.15
end
