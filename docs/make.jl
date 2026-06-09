using Microscaling
using Documenter

DocMeta.setdocmeta!(Microscaling, :DocTestSetup, :(using Microscaling); recursive=true)

makedocs(;
    modules=[Microscaling],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Microscaling.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/Microscaling.jl",
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
