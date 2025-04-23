from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
from langchain_groq import ChatGroq
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()


def load_agent() -> AgentExecutor:
    llm = ChatGroq(
        model='deepseek-r1-distill-llama-70b',
        streaming=True,
        temperature=0,

    )
    # DuckDuckGoSearchRun, arxiv search, wikipedia
    tools = load_tools(
        tool_names=['wikipedia', 'ddg-search'],  #'arxiv', wolframe-alpha
        llm=llm
    )
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )


chain = load_agent()
st_callback = StreamlitCallbackHandler(st.container())

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.invoke(prompt, callbacks=[st_callback])
        result = response.get('output')
        st.write(str(result))
