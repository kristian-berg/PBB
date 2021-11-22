import multiprocessing

class MonteCarloSimulation:
    def __init__(self, num_processes, initializer, initargs):
        self.pool = multiprocessing.Pool(processes=num_processes, initializer = initializer, initargs = initargs)

    def start(self, num_cycles, cycle, do_result):
        # Start a Pool with 8 processes
        jobs = range(0, num_cycles)
        for res in self.pool.imap_unordered(cycle, jobs):
            do_result(res)
        # Safely terminate the pool
        self.pool.close()
        self.pool.join()
    
