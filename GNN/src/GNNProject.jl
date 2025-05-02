module GNNProject
    using Reexport

    include("types.jl")
    include("data_loader.jl")
    include("loss.jl")
    include("model_evaluation.jl")
    include("training.jl")
    include("plotting.jl")

    @reexport using .Types
    @reexport using .DataLoader
    @reexport using .Loss
    @reexport using .ModelEvaluation
    @reexport using .ModelTraining
    @reexport using .Plotting

end