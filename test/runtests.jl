using CpuId, Aqua, ReTestItems, ClosedFormExpectations

nthreads = max(cputhreads(), 1)
ncores = max(cpucores(), 1)

Aqua.test_all(
    ClosedFormExpectations,
    ambiguities = false,
    piracies = false,
)

runtests(ClosedFormExpectations,
    nworkers = ncores,
    nworker_threads = Int(nthreads / ncores),
    memory_threshold = 1.0
)