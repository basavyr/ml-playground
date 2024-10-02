import torch

import timm
from adnet.configuration_adnet import AdnetConfig
from adnet.modeling_adnet import Adnet_HF


AdnetConfig.register_for_auto_class()
Adnet_HF.register_for_auto_class("AutoModel")

model = Adnet_HF(AdnetConfig())

model.save_pretrained("./adnet-model")


x = model(torch.tensor([[4, 5]], dtype=torch.float))
print(x)
