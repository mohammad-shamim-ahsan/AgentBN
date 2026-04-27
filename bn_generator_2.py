from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate

# -------------------------------
# 1️⃣ Setup local LLM
# -------------------------------
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
model_name = "Qwen/Qwen2.5-72B-Instruct"
#model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=7000,
    temperature=0.3,
    #do_sample=True,
    #repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------------
# 2️⃣ Read Single Context File
# -------------------------------
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

full_context = read_file("Context for Gen-Agent.txt")

# -------------------------------
# 3️⃣ Prompt Template (from file)
# -------------------------------
prompt_template_text = read_file("gen_prompt_A.txt")

prompt_gen_template = PromptTemplate(
    input_variables=["full_context"],
    template=prompt_template_text
)

# -------------------------------
# 4️⃣ Sub-task A
# -------------------------------
def subtask_A(full_context):
    prompt = prompt_gen_template.format(
        full_context=full_context
    )
    return llm.invoke(prompt)

# -------------------------------
# 5️⃣ Run
# -------------------------------
bn_proposal = subtask_A(full_context)

print("=== BN Proposal ===")
print(bn_proposal)
