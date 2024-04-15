using CpuId, ReTestItems, ClosedFormExpectations

nthreads = max(cputhreads(), 1)
ncores = max(cpucores(), 1)

runtests(ClosedFormExpectations,
    nworkers = ncores,
    nworker_threads = Int(nthreads / ncores),
    memory_threshold = 1.0
)
