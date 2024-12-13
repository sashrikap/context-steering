"""Model paths for Hugging Face and local models.

Create soft link for local models/ to point to the actual model directory.

```
ln -s models ACTUAL_MODEL_DIR
```

Replace with your own model or local model path
"""

import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_LLAMA_7B_TEXT_DIR = f'{root_path}/models/hf-llama-2-7b/'
HF_LLAMA_13B_TEXT_DIR = f'{root_path}/models/hf-llama-2-13b/'
HF_LLAMA_7B_CHAT_DIR = f'{root_path}/models/hf-llama-2-7b-chat/'
HF_LLAMA_13B_CHAT_DIR = f'{root_path}/models/hf-llama-2-13b-chat/'
HF_ALPACA_DIR = f'{root_path}/models/alpaca/'



LLAMA_7B_TEXT_DIR = f'{root_path}/models/llama-2-7b/'
LLAMA_13B_TEXT_DIR = f'{root_path}/models/llama-2-13b/'
LLAMA_7B_CHAT_DIR = f'{root_path}/models/llama-2-7b-chat/'
LLAMA_13B_CHAT_DIR = f'{root_path}/models/llama-2-13b-chat/'
LLAMA_TOKENIZER_PATH = f'{root_path}/models/llama-2-tokenizer/tokenizer.model'


HF_LLAMA_3_8B_TEXT_DIR = f'{root_path}/models/hf-llama-3-8b/'
HF_LLAMA_3_8B_CHAT_DIR = f'{root_path}/models/hf-llama-3-8b-chat/'
HF_LLAMA_3_TOKENIZER_PATH = f'{root_path}/models/hf-llama-3-8b/tokenizer.json'
