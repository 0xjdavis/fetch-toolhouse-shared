import streamlit as st
import openai
from toolhouse import Toolhouse
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
import asyncio

# Set the OpenAI API key correctly
openai.api_key = st.secrets["OPENAI_KEY"]
th = Toolhouse(access_token=st.secrets["TOOLHOUSEAI_KEY"], provider="openai")
AGENT_MAILBOX_KEY = "911d1aac-059c-4cae-ab16-cfc75362953f"

class ToolHouseAIRequest(Model):
    query: str

agent = Agent(
    name="toolhouseai-test-agent",
    seed="toolhouseai-seed",
    mailbox=f"{AGENT_MAILBOX_KEY}@https://agentverse.ai"
)

fund_agent_if_low(agent.wallet.address())

toolhouseai_proto = Protocol(name="ToolhouseAI-Protocol", version="0.1.0")

async def get_answer(query):
    # Define the OpenAI model we want to use
    MODEL = 'gpt-4o-mini'

    messages = [{
        "role": "user",
        "content": query
    }]

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        # Passes Code Execution as a tool
        tools=th.get_tools()
    )

    # Get the generated code
    generated_code = response.choices[0].message.content

    # Runs the Code Execution tool, gets the result,
    # and appends it to the context
    messages += th.run_tools(response)

    final_response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=th.get_tools()
    )

    execution_result = final_response.choices[0].message.content

    if "```python" in execution_result and "```" in execution_result:
        start = execution_result.find("```python") + len("```python")
        end = execution_result.find("```", start)
        execution_result_code = execution_result[start:end].strip()
    else:
        execution_result_code = execution_result

    return generated_code, execution_result_code

@agent.on_event("startup")
async def introduce(ctx: Context):
    ctx.logger.info(ctx.agent.address)

@toolhouseai_proto.on_message(ToolHouseAIRequest)
async def handle_request(ctx: Context, sender: str, msg: ToolHouseAIRequest):
    ctx.logger.info(f"Received query : {msg.query}")
    try:
        generated_code, execution_result = await get_answer(msg.query)
        ctx.logger.info(execution_result)
        return execution_result
    except Exception as err:
        ctx.logger.error(err)
        return str(err)

agent.include(toolhouseai_proto, publish_manifest=True)

# Streamlit app
st.title("ToolhouseAI Agent Interface")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        with st.spinner("Processing your query..."):
            try:
                generated_code, execution_result = asyncio.run(get_answer(query))
                
                st.subheader("Generated Code:")
                st.code(generated_code, language="python")
                
                st.subheader("Execution Result:")
                st.code(execution_result, language="python")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query.")

# Run the agent in the background
if __name__ == "__main__":
    import threading
    agent_thread = threading.Thread(target=agent.run)
    agent_thread.start()
