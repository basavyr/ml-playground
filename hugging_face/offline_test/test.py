# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100")
model = AutoModelForSeq2SeqLM.from_pretrained("alirezamsh/small100")
