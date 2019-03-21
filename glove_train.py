from libraries import *
import queue
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from glove_cython_D import *

class Glove(object):
    def __init__(self, cooccurrences, alpha=0.75, x_max=100.0, d=50, seed=1234):
        """
        Glove model for obtaining dense embeddings from a
        co-occurence (sparse) matrix.
        """
        self.alpha           = alpha
        self.xMax           = x_max
        self.d               = d
        self.cooccurrences     = cooccurrences
        self.seed            = seed
        np.random.seed(seed)
        self.W               = np.random.uniform(-0.5/d, 0.5/d, (len(cooccurrences), d)).astype(np.float64)
        self.contextW        = np.random.uniform(-0.5/d, 0.5/d, (len(cooccurrences), d)).astype(np.float64)
        self.b               = np.random.uniform(-0.5/d, 0.5/d, (len(cooccurrences), 1)).astype(np.float64)
        self.contextB        = np.random.uniform(-0.5/d, 0.5/d, (len(cooccurrences), 1)).astype(np.float64)
        self.gradsqW         = np.ones_like(self.W, dtype=np.float64)
        self.gradsqContextW  = np.ones_like(self.contextW, dtype=np.float64)
        self.gradsqb         = np.ones_like(self.b, dtype=np.float64)
        self.gradsqContextB  = np.ones_like(self.contextB, dtype=np.float64)

    def train(self, stepSize=0.05, threadsCount = 9, batchSize=50, verbose=False):

        jobs = queue.Queue()
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)
        total_error = [0.0]
        total_done = [0]

        total_els = len(self.cooccurrences)
            # Batch co-occurrence pieces
        numExamples = 0
        def put_jobs(numExamples):
            batchLength = 0
            batch = []
            index = 0
            for cooccurrence in self.cooccurrences:
                batch.append(cooccurrence)
                # print(batchLength)
                batchLength += 1
                if batchLength >= batchSize:
                    print("putting jobs ", index)
                    jobs.put(
                        (
                            np.array([item[0] for item in batch], dtype=np.int32),
                            np.array([item[1] for item in batch], dtype=np.int32),
                            np.array([item[2] for item in batch], dtype=np.float64)
                        )
                    )
                    numExamples += len(batch)
                    batch = []
                    batchLength = 0
                    index += 1
            if len(batch) > 0:
                jobs.put(
                    (
                        np.array([item[0] for item in batch], dtype=np.int32),
                        np.array([item[1] for item in batch], dtype=np.int32),
                        np.array([item[2] for item in batch], dtype=np.float64)
                    )
                )
                numExamples += len(batch)
                batch = []
                batchLength = 0
            print("Finished putting jobs")
        # thread function:
        def thread_train():

            error = np.zeros(1, dtype=np.float64)
            print("error ", error)
            while True:
                job = jobs.get()
                print(error)
                if job is None:  # data finished, exit
                    break
                print("Train")
                train_glove(self, job, stepSize, error)
                jobs.task_done()
                with lock:
                    total_error[0] += error[0]
                    total_done[0] += len(job[0])
                    if verbose:
                        if total_done[0] % 1000 == 0:
                            print("Completed %.3f%%\r" % (100.0 * total_done[0] / total_els))
                error[0] = 0.0


        # Create workers
        threads = []
        print("Before threads assignment")
        put_jobs(numExamples)
        for i in range(threadsCount):
            thread = threading.Thread(target=thread_train)
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()
            threads.append(thread)
        print(len(self.cooccurrences))

        jobs.join()

        for _ in range(threadsCount):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in threads:
            thread.join()

        return total_error[0] / numExamples