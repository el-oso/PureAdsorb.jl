using Documenter, DocumenterVitepress, PureAdsorb

makedocs(;
    sitename = "PureAdsorb.jl", authors = "Jorge Vieyra", modules = [PureAdsorb], warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(; repo = "github.com/el-oso/PureAdsorb.jl", devbranch = "master", devurl = "dev"),
    source = "src", build = "build",
    pages = ["Home" => "index.md", "API" => "api.md"]
)

DocumenterVitepress.deploydocs(; repo = "github.com/el-oso/PureAdsorb.jl", devbranch = "master", push_preview = true)
