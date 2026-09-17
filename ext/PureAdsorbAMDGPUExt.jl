module PureAdsorbAMDGPUExt

using AMDGPU, PureAdsorb

PureAdsorb.backend_loaded(::AMDGPU.ROCBackend) = true

end
