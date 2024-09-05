from transformers import AutoConfig, AutoModel


models = ["1bitLLM/bitnet_b1_58-large",
          "1bitLLM/bitnet_b1_58-xl", "1bitLLM/bitnet_b1_58-3B"]


for model_name in models:
    print(f'--------- {model_name} ---------')
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_config(config)
    print(model)
