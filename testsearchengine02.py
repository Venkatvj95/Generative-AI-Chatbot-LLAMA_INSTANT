import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import  GoogleSearchAPIWrapper
from langchain_community.tools import  GoogleSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# 🔐 API keys
groq_api_key = "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"
google_api_key = "AIzaSyBquYJm-4MzJiMyzyrcoWQaR0g-q62vwR0"
google_cse_id = "755f1b7126e8c49a6"

# 🔍 Google Search Tool
google_wrapper = GoogleSearchAPIWrapper(
    google_api_key=google_api_key,
    google_cse_id=google_cse_id
)
search_tool = GoogleSearchRun(api_wrapper=google_wrapper)

# ⚙️ Initialize model and agent once
llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", streaming=True)
tools = [search_tool]
search_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

# 🧠 Chat UI
st.title("AI Chatbot with Google Search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am an AI chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 💬 Handle user input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
