import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(torch.cuda.is_available())
    print(dir(torch.device))
    print('MPS device found')
    print (x)
else:
    print ("MPS device not found.")
