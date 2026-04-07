using Documenter, ClosedFormExpectations

DocMeta.setdocmeta!(ClosedFormExpectations, :DocTestSetup, :(using ClosedFormExpectations, Distributions, ExponentialFamily, BayesBase); recursive = true)

makedocs(
    modules  = [ClosedFormExpectations],
    clean    = true,
    sitename = "ClosedFormExpectations.jl",
    pages    = [
        "Introduction" => "index.md",
        "Library" => [
            "API Reference"        => "lib/api.md",
            "Expression Types"     => "lib/expressions.md",
            "Supported Pairs"      => "lib/supported-pairs.md",
            "Custom Distributions" => "lib/distributions.md",
        ],
        "Extensions" => [
            "Enzyme" => "extensions/enzyme.md",
        ],
        "Extra" => [
            "Contributing" => "extra/contributing.md",
        ]
    ],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    )
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(repo = "github.com/biaslab/ClosedFormExpectations.jl.git", devbranch = "main", forcepush = true)
end
