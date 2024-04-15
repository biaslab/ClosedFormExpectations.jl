using CpuId, ReTestItems, LogExpectations

nthreads = max(cputhreads(), 1)
ncores = max(cpucores(), 1)

runtests(LogExpectations,
    nworkers = ncores,
    nworker_threads = Int(nthreads / ncores),
    memory_threshold = 1.0
)
