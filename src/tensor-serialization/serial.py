import time
import torch
import struct

import torch
import pickle


def get_tensors_size(tensors: torch.Tensor):
    # Calculate memory usage in bytes
    memory_usage_bytes = tensors.element_size() * tensors.nelement()

    # Convert to megabytes (MB)
    memory_usage_mb = memory_usage_bytes / (1024 ** 2)

    print(f'Tensor shape: {tensors.shape}')
    print(f'Memory usage: {memory_usage_mb:.2f} MB')


# Create a tensor
tensor = torch.randn(10000, 10000)
get_tensors_size(tensor)


# Apply pickle dumps to the tensor
start = time.time()
pickled_tensor = pickle.dumps(tensor)
end = time.time()
pickle_duration = round(end-start, 2)
pickle_bytes = len(pickled_tensor)
print(f'Pickled tensor: {pickle_bytes/ (1024 ** 2)} MB')
print(f'Creating pickle took: {pickle_duration} s')


start = time.time()
packed_tensor = struct.pack(
    f'{tensor.nelement()}f', *tensor.flatten().tolist())
end = time.time()
packing_duration = round(end-start, 2)
print(f'Packed tensor: {len(packed_tensor)/ (1024 ** 2)} MB')
print(f'Packing took: {packing_duration} s')

if packing_duration < pickle_duration:
    print(
        f'Packing is {round(packing_duration/pickle_duration,2)} times faster')
else:
    print(
        f'Pickle is {round(packing_duration/pickle_duration,2)} times faster')
