using Documenter
using LinearAlgebra
using SpecializingFactorizations

makedocs(;
    sitename = "SpecializingFactorizations.jl",
    modules = [SpecializingFactorizations],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/SpecializingFactorizations/stable/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/SciML/SpecializingFactorizations.jl",
    devbranch = "main",
)
