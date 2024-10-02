from transformers import PretrainedConfig


class AdnetConfig(PretrainedConfig):
    model_type = "adnet"

    def __init__(self, input_size=2, hidden_size1=512, hidden_size2=1024, output_size=1, architectures=["adnet"], pipeline_tag="adnet", license="mit", repo_url="https://huggingface.co/basavyr/adnet", tags=["release"], **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.architectures = architectures
        self.pipeline_tag = pipeline_tag
        self.license = license
        self.model_tags = tags
        self.repo_url = repo_url
