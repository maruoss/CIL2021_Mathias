# %%
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b

print("Hello World TEST123")
print("a + b:", a+b)


# from __future__ import print_function
import torch

x = torch.rand(5, 3)
print(x)

if torch.cuda.is_available():
   print ("Cuda is available")
   device_id = torch.cuda.current_device()
   gpu_properties = torch.cuda.get_device_properties(device_id)
   print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
          device_id,
          gpu_properties.name,
          gpu_properties.major,
          gpu_properties.minor,
          gpu_properties.total_memory / 1e9))
else:    
   print ("Cuda is not available")
