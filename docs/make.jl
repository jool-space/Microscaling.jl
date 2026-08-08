using Microscaling
using Microfloats, BitPacking
using Documenter

DocMeta.setdocmeta!(Microscaling, :DocTestSetup, :(using Microscaling); recursive=true)
DocMeta.setdocmeta!(Microfloats, :DocTestSetup, :(using Microfloats); recursive=true)
DocMeta.setdocmeta!(BitPacking, :DocTestSetup, :(using BitPacking); recursive=true)

makedocs(;
    # Microfloats and BitPacking are listed so `@docs` can pull in the
    # docstrings of names Microscaling re-publics. Only a curated subset of
    # their docstrings is included, so the missing-docs check is disabled,
    # and cross-reference errors are downgraded (the included upstream
    # docstrings link to names outside this manual).
    modules=[Microscaling, Microfloats, BitPacking],
    checkdocs=:none,
    warnonly=[:cross_references],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Microscaling.jl",
    format=Documenter.HTML(;
        canonical="https://docs.jool.space/Microscaling.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Microscaling.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Microscaling.jl",
)
