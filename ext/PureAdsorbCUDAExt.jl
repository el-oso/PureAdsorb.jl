module PureAdsorbCUDAExt

using CUDA, PureAdsorb

PureAdsorb.backend_loaded(::CUDA.CUDABackend) = true

end
