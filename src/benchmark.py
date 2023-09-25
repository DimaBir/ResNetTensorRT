import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch

class Benchmark:
    def __init__(self, model, device="cuda", input_shape=(32, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=100):
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.dtype = dtype
        self.nwarmup = nwarmup
        self.nruns = nruns
        cudnn.benchmark = True

    def run(self):
        input_data = torch.randn(self.input_shape).to(self.device)
        print("Warm up ...")
        with torch.no_grad():
            for _ in range(self.nwarmup):
                features = self.model(input_data)
        torch.cuda.synchronize()
        print("Start timing ...")
        timings = []
        with torch.no_grad():
            for i in range(1, self.nruns + 1):
                start_time = time.time()
                features = self.model(input_data)
                torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)
                if i % 10 == 0:
                    print('Iteration %d/%d, ave batch time %.2f ms' % (i, self.nruns, np.mean(timings) * 1000))
        print("Input shape:", input_data.size())
        print("Output features size:", features.size())
        print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))
