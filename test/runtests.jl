using Test, QuantileNN, Random, Statistics

@testset "test1" begin

    Random.seed!(342)
    n = 2000
    x = randn(n, 2)
    y = x[:, 1] + randn(n)

    qr1 = qregnn(y, x; p=0.25)
    qr2 = qregnn(y, x; p=0.5)
    qr3 = qregnn(y, x; p=0.75)

    yq = zeros(n, 3)
    yq[:, 1] = fittedvalues(qr1)
    yq[:, 2] = fittedvalues(qr2)
    yq[:, 3] = fittedvalues(qr3)

    ax = [-0.67, 0, 0.67] # True intercepts
    for j = 1:3
        c = cov(yq[:, j], x[:, 1])
        b = c / var(x[:, 1])
        a = mean(yq[:, j]) - b * mean(x[:, 1])
        @test abs(b - 1) < 0.1 # True slope is 1
        @test abs(a - ax[j]) < 0.1
    end

    bw = Float64[2, 2]
    @test abs(predict_smooth(qr1, [0.0, 0.0], bw) - ax[1]) < 0.1
    @test abs(predict_smooth(qr2, [0.0, 0.0], bw) - ax[2]) < 0.1
    @test abs(predict_smooth(qr3, [0.0, 0.0], bw) - ax[3]) < 0.1
end
