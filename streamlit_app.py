import streamlit as st
import openai
from toolhouse import Toolhouse
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
import asyncio
import threading
import uuid

# API KEYS
openai.api_key = st.secrets["OPENAI_KEY"]
TOOLHOUSE_KEY = st.secrets["TOOLHOUSE_KEY"]
AGENT_MAILBOX_KEY = st.secrets["TH_AGENT_MAILBOX_KEY"]

# Generate a unique user ID for Toolhouse
USER_ID = str(uuid.uuid4())

th = Toolhouse(
    access_token=TOOLHOUSE_KEY,
    provider="openai",
    user_id=USER_ID  # Add the user ID here
)

class ToolHouseAIRequest(Model):
    query: str

toolhouseai_proto = Protocol(name="ToolhouseAI-Protocol", version="0.1.0")

# AGENT
def initialize_agent():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    agent = Agent(
        name="toolhouseai-test-agent",
        seed="toolhouseai-seed",
        mailbox=f"{AGENT_MAILBOX_KEY}@https://agentverse.ai"
    )

    fund_agent_if_low(agent.wallet.address())

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
    
    return agent, loop

# OPEN AI QUERY
async def get_answer(query):
    MODEL = 'gpt-4o-mini'

    messages = [{
        "role": "user",
        "content": query
    }]

    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=th.get_tools()
        )

        generated_code = response.choices[0].message.content

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
    except openai.OpenAIError as e:
        st.error(f"OpenAI API error: {str(e)}")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        raise

# Initialize agent and event loop
agent, loop = initialize_agent()

# Run the agent in a separate thread
def run_agent():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(agent.run())

agent_thread = threading.Thread(target=run_agent, daemon=True)
agent_thread.start()

# Streamlit app
st.title("Toolhouse & Fetch Agent with Code Interpreter")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        with st.spinner("Processing your query..."):
            try:
                generated_code, execution_result = asyncio.run(get_answer(query))
                
                st.subheader("Response:")
                st.code(generated_code, language="python")
                
                st.subheader("Code:")
                st.code(execution_result, language="python")
            except Exception as e:
                st.error(f"An error occurred: {type(e).__name__}: {str(e)}")
                st.error("Please check your API keys and try again.")
    else:
        st.warning("Please enter a query.")

# Display the generated user ID
st.sidebar.write(f"User ID: {USER_ID}")
