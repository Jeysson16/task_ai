# convert_gguf.py
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.convert_llama_weights_to_hf import (
    convert_llama_weights_to_hf,
)

GGUF_PATH = r"C:\Users\Jeysson\.lmstudio\models\lmstudio-community\DeepSeek-R1-Distill-Llama-8B-GGUF\model.gguf"
HF_OUTPUT = r"C:\Users\Jeysson\.lmstudio\models\DeepSeek-8b-hf"

# 1. Cargar la configuración base (necesitarás un config.json plantillla para Llama)
config = AutoConfig.from_pretrained("decapoda-research/llama-8b-hf")
# 2. Convertir
convert_llama_weights_to_hf(
    gguf_file=GGUF_PATH,
    pytorch_dump_folder_path=HF_OUTPUT,
    config=config,
)
# 3. Copiar tokenizer
tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-8b-hf", use_fast=True)
tokenizer.save_pretrained(HF_OUTPUT)
