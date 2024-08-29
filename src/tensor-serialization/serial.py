import time
import torch
import struct


import torch
import pickle

# Create a tensor
tensor = torch.randn(10000, 10000)
t_bytes = tensor.element_size() * tensor.nelement()

print(f'Tensor shape: {tensor.shape}\n')
print(f'Tensor size: {t_bytes} bytes')


# Apply pickle dumps to the tensor
start = time.time()
pickled_tensor = pickle.dumps(tensor)
end = time.time()
pickle_duration = round(end-start, 2)
pickle_bytes = len(pickled_tensor)
print(f'Pickled tensor: {pickle_bytes} bytes')
print(f'Creating pickle took: {pickle_duration} s')


start = time.time()
packed_tensor = struct.pack(
    f'{tensor.nelement()}f', *tensor.flatten().tolist())
end = time.time()
packing_duration = round(end-start, 2)
print(f'Packed tensor: {len(packed_tensor)} bytes')
print(f'Packing took: {packing_duration} s')

if packing_duration < pickle_duration:
    print(
        f'Packing is {round(packing_duration/pickle_duration,2)} times faster')
else:
    print(
        f'Pickle is {round(packing_duration/pickle_duration,2)} times faster')
