from libraries import *
import queue
import numpy as np
# import pyximport
# pyximport.install(setup_args={'include_dirs': np.get_include()})
# from glove_cython_D import *

class Glove(object):
    def __init__(self, cooccurrences,vocabSize, alpha=0.75, x_max=100.0, d=50):
        """
        Glove model for obtaining dense embeddings from a
        co-occurence (sparse) matrix.
        """
        doubleVocab = 2*vocabSize
        self.alpha           = alpha
        self.xMax           = x_max
        self.d               = d
        self.cooccurrences     = cooccurrences
        seed            = 1234
        np.random.seed(seed)
        self.W               = np.random.uniform(-0.5/d, 0.5/d, (doubleVocab, d)).astype(np.float64)
        self.contextW        = np.random.uniform(-0.5/d, 0.5/d, (doubleVocab, d)).astype(np.float64)
        self.bias               = np.random.uniform(-0.5/d, 0.5/d, (doubleVocab, 1)).astype(np.float64)
        self.contextB        = np.random.uniform(-0.5/d, 0.5/d, (doubleVocab, 1)).astype(np.float64)
        self.gradsqW         = np.ones_like(self.W, dtype=np.float64)
        self.gradsqContextW  = np.ones_like(self.contextW, dtype=np.float64)
        self.gradsqb         = np.ones_like(self.bias, dtype=np.float64)
        self.gradsqContextB  = np.ones_like(self.contextB, dtype=np.float64)

    def train_glove_thread(self, i, j, Xij, batchSize, stepSize, error):

        l1 = 0;
        l2 = 0
        alpha = self.alpha
        batchIndex = 0
        diff = 0;
        fdiff = 0

        for batchIndex in range(batchSize):
            if len(i) <= batchIndex or len(j) <= batchIndex:
                break
            # Calculate cost, save diff for gradients
            l1 = i[batchIndex]
            l2 = j[batchIndex]

            diff = np.dot(self.W[l1].T,self.contextW[l2])  # dot product of word and context word vector
            diff += self.bias[i[batchIndex]] + self.contextB[j[batchIndex]] - np.log(Xij[batchIndex])
            fdiff = diff if (Xij[batchIndex] > self.xMax) else pow(Xij[batchIndex] / self.xMax, alpha) * diff
            error += 0.5 * fdiff * diff  # weighted squared error

            # # Adaptive gradient updates
            fdiff *= stepSize  # for ease in calculating gradient
            for b in range(self.d):
                # learning rate times gradient for word vectors
                temp1 = fdiff * self.contextW[b + l2]
                temp2 = fdiff * self.W[b + l1]
                # adaptive updates
                self.W[b + l1] -= (temp1 / np.sqrt(self.gradsqW[b + l1]))
                self.contextW[b + l2] -= (temp2 / np.sqrt(self.gradsqContextW[b + l2]))
                self.gradsqW[b + l1] += temp1 * temp1
                self.gradsqContextW[b + l2] += temp2 * temp2
            # updates for bias terms
            self.bias[i[batchIndex]] -= fdiff / np.sqrt(self.gradsqb[i[batchIndex]]);
            self.contextB[j[batchIndex]] -= fdiff / np.sqrt(self.gradsqContextB[j[batchIndex]]);

            fdiff *= fdiff;
            self.gradsqb[i[batchIndex]] += fdiff
            self.gradsqContextB[j[batchIndex]] += fdiff
        return error

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
                batch = []
                batchLength = 0
            print("Finished putting jobs")
        # thread function:
        def thread_train():

            error = 0.0
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                print("Train")
                error = self.train_glove_thread(job[0], job[1], job[2], batchSize, stepSize, error)
                jobs.task_done()
                with lock:
                    total_error[0] += error
                    print("Total error is ", total_error[0])
                    total_done[0] += len(job[0])
                    if verbose:
                        if total_done[0] % 1000 == 0:
                            print("Completed %.3f%%\r" % (100.0 * total_done[0] / total_els))
                error = 0.0


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

        return total_error[0] / len(self.cooccurrences)

