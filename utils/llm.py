# import os
# import json
# from openai import OpenAI
# from config.settings import TOKEN_USAGE_FILE

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# total_input_tokens = 0
# total_output_tokens = 0
# total_api_calls = 0

# def llm(prompt, temperature=0.3, max_tokens=8000):
#     global total_input_tokens, total_output_tokens, total_api_calls

#     response = client.responses.create(
#         model="gpt-5.4",
#         input=prompt,
#         temperature=temperature,
#         max_output_tokens=max_tokens,
#     )

#     usage = response.usage

#     total_input_tokens += usage.input_tokens
#     total_output_tokens += usage.output_tokens
#     total_api_calls += 1

#     save_token_usage()

#     return response.output_text.strip()

# def save_token_usage(filename=TOKEN_USAGE_FILE):

#     data = {
#         "model": "gpt-5.4",
#         "api_calls": total_api_calls,
#         "input_tokens": total_input_tokens,
#         "output_tokens": total_output_tokens,
#         "total_tokens": total_input_tokens + total_output_tokens,
#     }

#     with open(filename, "w") as f:
#         json.dump(data, f, indent=2)



#----------------------------------------------------------------
import os
import json
from openai import OpenAI
from config.settings import TOKEN_USAGE_FILE

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

total_input_tokens = 0
total_output_tokens = 0
total_api_calls = 0

def llm(prompt, temperature=0.3, max_tokens=8000):
    global total_input_tokens, total_output_tokens, total_api_calls

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt,
        reasoning={"effort": "medium"},
        max_output_tokens=max_tokens,
    )

    usage = response.usage

    total_input_tokens += usage.input_tokens
    total_output_tokens += usage.output_tokens
    total_api_calls += 1

    save_token_usage()

    return response.output_text.strip()

def save_token_usage(filename=TOKEN_USAGE_FILE):

    data = {
        "model": "gpt-5.5",
        "api_calls": total_api_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
