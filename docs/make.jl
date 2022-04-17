using RadialBasisFiniteDifferences
using Documenter

DocMeta.setdocmeta!(RadialBasisFiniteDifferences, :DocTestSetup, :(using RadialBasisFiniteDifferences); recursive=true)

makedocs(;
    modules=[RadialBasisFiniteDifferences],
    authors="Jesus Arias",
    repo="https://github.com/jarias9/RadialBasisFiniteDifferences.jl/blob/{commit}{path}#{line}",
    sitename="RadialBasisFiniteDifferences.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jarias9.github.io/RadialBasisFiniteDifferences.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jarias9/RadialBasisFiniteDifferences.jl",
    devbranch="main",
)
