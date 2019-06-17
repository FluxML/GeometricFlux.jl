
@testset "Test MessagePassing layer" begin
    x = randn(Float32, 10, 10, 3, 2)
    MessagePassing(msg_func, upd_func, aggr=+)
end
