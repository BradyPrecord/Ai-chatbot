from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pretrained DialoGPT model (small/medium/large)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

print("🤖 Local AI Chatbot ready! Type 'exit' to quit.\n")

# Keep chat history for context
chat_history_ids = None

for step in range(100):
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Bye! 👋")
        break

    # Encode user input + history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode and print response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Bot:", response)
