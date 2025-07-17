# Reformat - supervised
function MMI.reformat(eos::EOSWrapper, X, y)
    T_instances = MMI.nrows(X)
    args = MMI.reformat(eos.model, X, y)
    Tf = EntropicLearning.get_promoted_eltype(X)

    return (args, T_instances, Tf)
end

# Reformat - unsupervised and predict/transform
function MMI.reformat(eos::EOSWrapper, X)
    T_instances = MMI.nrows(X)
    args = MMI.reformat(eos.model, X)
    Tf = EntropicLearning.get_promoted_eltype(X)
    # Note: only the first argument is used in the case of predict/transform
    return (args, T_instances, Tf)
end

# Select rows
function MMI.selectrows(eos::EOSWrapper, I, args, T_instances::Int, Tf::Type)
    return (MMI.selectrows(eos.model, I, args...), T_instances, Tf)
end
