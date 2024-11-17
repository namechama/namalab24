from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
import ollama
import torch

llm_path = "/workspace/input/llm/gemma-2-2b-jpn-it"
# system_prompt = self.system_prompts[direction]
messages = [
    {"role": "system", "content": "あなたは有名な詩人です"},
    {"role": "user", "content": "マシーンラーニングについての詩を書いてください。"}
]
response = ollama.chat(model='schroneko/gemma-2-2b-jpn-it', messages=messages)
assistant_message = response['message']['content']
print(assistant_message)

# ローカルに保存したモデルとトークナイザーを読み込む
# tokenizer = AutoTokenizer.from_pretrained(llm_path)

# # 下記はCPUを使用する場合。
# # GPUの場合は、device_map="auto",　torch_dtype=torch.bfloat16　とします。
# model = AutoModelForCausalLM.from_pretrained(
#     llm_path,
#     device_map="cuda",
#     torch_dtype=torch.bfloat16
# )

# # テキスト生成の例
# # messages = [
# #     {"role": "user", "content": "マシーンラーニングについての詩を書いてください。"}
# # ]
# messages=[
#           {
#            "role": "user",
#            "content": "マシーンラーニングについての詩を書いてください。"
#           }
#           ]

# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
# outputs = model.generate(inputs, max_new_tokens=3000,temperature=0.7)
# generated_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

# print(generated_text.strip())
