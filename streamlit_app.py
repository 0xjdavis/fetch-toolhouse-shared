import streamlit as st
import groq
from toolhouse import Toolhouse
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
import asyncio
import threading

# API KEYS
groq_client = groq.Groq(api_key=st.secrets["GROQ_KEY"]) # Groq
AGENT_MAILBOX_KEY = st.secrets["TH_AGENT_MAILBOX_KEY"] # Fetch
th = Toolhouse(access_token=st.secrets["TOOLHOUSE_KEY"]) # Toolhouse
th.set_metadata("id", "user_id")



# Setting page layout
st.set_page_config(
    page_title="Generate and run code using a Fetch Agent with Toolhouse Code Interpreter",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar for API Key and User Info
st.sidebar.header("About App")
st.sidebar.markdown('This is an app utilizes a Fetch.ai Agent to call Toolhouse code interpreter tool to generate and run code, created by <a href="https://ai.jdavis.xyz" target="_blank">0xjdavis</a>.', unsafe_allow_html=True)

# Select Groq Model
MODEL = st.sidebar.selectbox(
    "Select a model:",
    ("llama3-8b-8192", "llama3-groq-70b-8192-tool-use-preview", "mixtral-8x7b-32768", "gemma-7b-it"),
)

# Calendly
st.sidebar.markdown("""
    <hr />
    <center>
    <div style="border-radius:8px;padding:8px;background:#fff";width:100%;">
    <img src="https://avatars.githubusercontent.com/u/98430977" alt="Oxjdavis" height="100" width="100" border="0" style="border-radius:50%"/>
    <br />
    <span style="height:12px;width:12px;background-color:#77e0b5;border-radius:50%;display:inline-block;"></span> <b style="color:#000000">I'm available for new projects!</b><br />
    <a href="https://calendly.com/0xjdavis" target="_blank"><button style="background:#126ff3;color:#fff;border: 1px #126ff3 solid;border-radius:8px;padding:8px 16px;margin:10px 0">Schedule a call</button></a><br />
    </div>
    </center>
    <br />
""", unsafe_allow_html=True)

# Copyright
st.sidebar.caption("©️ Copyright 2024 J. Davis")



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

# GROQ QUERY
async def get_answer(query):

    # Define the Groq model we want to use
    # USING SELECT FROM ABOVE!!!!!!
    #MODEL = 'mixtral-8x7b-32768'  # No result, but got code. 
    #MODEL = 'llama2-70b-4096'  # Don't use this.
    #MODEL = 'llama3-groq-70b-8192-tool-use-preview'
    # MODEL = 'llama3-8b-8192'
    # MODEL = 'gemma-7b-it' 
    
    messages = [{
        "role": "user",
        "content": query
    }]

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=th.get_tools()
    )

    # Get the generated code
    generated_code = response.choices[0].message.content

    # Runs the Code Execution tool, gets the result,
    # and appends it to the context
    messages += th.run_tools(response)

    final_response = groq_client.chat.completions.create(
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

# Initialize agent and event loop
agent, loop = initialize_agent()

# Run the agent in a separate thread
def run_agent():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(agent.run())

agent_thread = threading.Thread(target=run_agent, daemon=True)
agent_thread.start()

# APP
st.title("Generate and Run Code"
st.caption("Fetch Agent with Toolhouse Code Interpreter")

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
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query.")
