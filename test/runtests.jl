using Test

@testset "EntropicLearning.jl" begin
    include("utilities.jl")
    include("common.jl")
    include("eSPA.jl")
end

include("Aqua.jl")
