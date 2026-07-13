from openai import OpenAI

client = OpenAI(api_key="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")

def llm(prompt, temperature=0.3, max_tokens=4000):
    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    
    return response.output[0].content[0].text.strip()