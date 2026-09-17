using TestItemRunner
@run_package_tests filter = ti -> !(:gpu in ti.tags) && !(:slow in ti.tags)
