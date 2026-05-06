from langchain_core.prompts import PromptTemplate
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

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

full_code = read_file("test_code.txt")
prompt_template_text = read_file("test_prompt.txt")

def draft_model(full_code, prompt_template_text):
    prompt_gen_template = PromptTemplate(
        input_variables=["full_code"],
        template=prompt_template_text
    )

    prompt = prompt_gen_template.format(
        full_code=full_code
    )

    return llm(prompt)

def generate_bn(full_code, prompt_template_text):
    draft = draft_model(full_code, prompt_template_text)
    return draft

re_optimized_code = generate_bn(full_code, prompt_template_text)
print(re_optimized_code)
