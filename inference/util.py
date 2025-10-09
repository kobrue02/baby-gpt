import torch


def complete(model, prompt, enc, device, max_tokens=100, temperature=0.8, top_k=200):
    """ Generate text completions using the provided model and tokenizer."""
    # Encode prompt
    prompt_ids = enc.encode_ordinary(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # Decode and display output
    return enc.decode(output_ids[0].tolist())