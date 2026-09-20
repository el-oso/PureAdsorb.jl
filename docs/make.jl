using Documenter, DocumenterVitepress, PureAdsorb

makedocs(;
    sitename = "PureAdsorb.jl", authors = "Jorge Vieyra", modules = [PureAdsorb],
    format = DocumenterVitepress.MarkdownVitepress(; repo = "github.com/el-oso/PureAdsorb.jl", devbranch = "master", devurl = "dev"),
    repo = "github.com/el-oso/PureAdsorb.jl",
    source = "src", build = "build",
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Design" => "design.md",
        "Validation" => "validation.md",
        "Benchmarks" => "benchmarks.md",
        "API" => "api.md",
    ]
)

DocumenterVitepress.deploydocs(; repo = "github.com/el-oso/PureAdsorb.jl", devbranch = "master", push_preview = true)
