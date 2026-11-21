import ollama

reply = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(reply["message"]["content"])