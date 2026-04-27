from langgraph_reflection import create_reflection_graph
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict
from openevals.llm import create_llm_as_judge

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# -------------------------
# Initialize local Mistral model
# -------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",       # automatically load on GPU if available
    torch_dtype=torch.float16,
)

# Wrap the model in a callable to mimic LangChain chat model
class LocalChatModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def invoke(self, messages):
        """Concatenate messages and generate response from the model."""
        # Flatten conversation into a single string
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"
        prompt += "assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return [{"role": "assistant", "content": response}]


# -------------------------
# Define the main assistant model
# -------------------------
def call_model(state):
    """Process the user query with a local Mistral model."""
    local_model = LocalChatModel(model, tokenizer)
    return {"messages": local_model.invoke(state["messages"])}


# -------------------------
# Assistant graph
# -------------------------
assistant_graph = (
    StateGraph(MessagesState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile()
)


# -------------------------
# Judge tool
# -------------------------
class Finish(TypedDict):
    """Tool for the judge to indicate the response is acceptable."""
    finish: bool


# -------------------------
# Critique prompt
# -------------------------
critique_prompt = """You are an expert judge evaluating AI responses. Your task is to critique the AI assistant's latest response in the conversation below.

Evaluate the response based on these criteria:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the user's query?
3. Clarity - Is the explanation clear and well-structured?
4. Helpfulness - Does it provide actionable and useful information?
5. Safety - Does it avoid harmful or inappropriate content?

If the response meets ALL criteria satisfactorily, set pass to True.

If you find ANY issues with the response, do NOT set pass to True. Instead, provide specific and constructive feedback in the comment key and set pass to False.

Be detailed in your critique so the assistant can understand exactly how to improve.

<response>
{outputs}
</response>"""


# -------------------------
# Judge function
# -------------------------
def judge_response(state, config):
    """Evaluate the assistant's response using a local or remote judge model."""
    # Here, you can switch to a local judge by creating a LocalChatModel as above if desired
    evaluator = create_llm_as_judge(
        prompt=critique_prompt,
        model="openai:o3-mini",  # Can be replaced with a local model if desired
        feedback_key="pass",
    )
    eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)

    if eval_result["score"]:
        print("✅ Response approved by judge")
        return
    else:
        print("⚠️ Judge requested improvements")
        return {"messages": [{"role": "user", "content": eval_result["comment"]}]}


# -------------------------
# Judge graph
# -------------------------
judge_graph = (
    StateGraph(MessagesState)
    .add_node(judge_response)
    .add_edge(START, "judge_response")
    .add_edge("judge_response", END)
    .compile()
)


# -------------------------
# Reflection app
# -------------------------
reflection_app = create_reflection_graph(assistant_graph, judge_graph)
reflection_app = reflection_app.compile()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    example_query = [
        {
            "role": "user",
            "content": "Explain how nuclear fusion works and why it's important for clean energy",
        }
    ]

    print("Running example with reflection using local Mistral...")
    result = reflection_app.invoke({"messages": example_query})
    print(result)
    
