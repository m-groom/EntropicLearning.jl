using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        EntropicLearning;
        ambiguities=false, # TODO: switch to true
        unbound_args=true,
        undefined_exports=true,
        project_extras=true,
        deps_compat=false, # TODO: Switch to true after fixing compat
    )
end
