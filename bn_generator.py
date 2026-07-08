from langchain_core.prompts import PromptTemplate
from openai import OpenAI
import json

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

proposed_bn_filename="last_proposed_bn.jsonl"
full_context = read_file("context_agent.txt")
success_report = read_file("merged_flawed_success_train.json")
failure_report = read_file("merged_flawed_failure_train.json")
prompt_template_text = read_file("gen_prompt.txt")
flawed_bn=read_file("flawed_BN_0.json")

# -------------------------------
# 1️⃣ Draft Stage (generation only)
# -------------------------------

def draft_model(full_context, flawed_bn, success_report, failure_report, prompt_template_text, temperature):
    prompt_gen_template = PromptTemplate(
        input_variables=[
            "full_context",
            "flawed_bn",
            "success_report",
            "failure_report"
        ],
        template=prompt_template_text
    )

    prompt = prompt_gen_template.format(
        full_context=full_context,
        flawed_bn=flawed_bn,
        success_report=success_report,
        failure_report=failure_report
    )

    return llm(prompt, temperature=temperature)

def generate_bn(full_context, flawed_bn, success_report, failure_report, prompt_template_text, temperature=0.3):
    draft = draft_model(full_context, flawed_bn, success_report, failure_report, prompt_template_text, temperature)
    return draft

def store_bn_proposal(bn_new, bn_number, filename=proposed_bn_filename):
    record = {
        "bn_number": bn_number,
        "bn": bn_new
    }

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

### -------------------------------
if __name__ == "__main__":
    bn_proposal = generate_bn(full_context, flawed_bn, success_report, failure_report, prompt_template_text, temperature=0.3)
    print(bn_proposal)
    bn_json = json.loads(bn_proposal)
    store_bn_proposal(bn_json, bn_number=1)
