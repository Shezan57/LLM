from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
from langchain_groq import ChatGroq
import streamlit as st
from typing import Optional, Literal
from langchain.chains.base import Chain
from dotenv import load_dotenv
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.callbacks import StreamlitCallbackHandler

load_dotenv()

ReasoningStrategies = Literal['zero-shot-react', 'plan-and-solve']


def load_agent(
        tool_names: list[str],
        strategy: ReasoningStrategies = 'zero-shot-react'
) -> Chain:
    llm = ChatGroq(
        model='deepseek-r1-distill-llama-70b',
        streaming=True,
        temperature=0,
    )
    tools = load_tools(
        tool_names=tool_names,
        llm=llm
    )
    if strategy == 'plan-and-solve':
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        return PlanAndExecute(planner=planner, executor=executor, verbose=True)
    else:
        return initialize_agent(
            tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True, verbose=True)


strategy = st.radio(
    "Reasoning Strategy",
    ("plan-and-solve", "zero-shot-react")
)

tool_names = st.multiselect(
    "Which tools do you want to use?",
    [
        "google-search", "ddg-search", "wolfram-alpha", "wikipedia", "arxiv",
        "python_repl", "pal-math", "llm_math"
    ],
    ["ddg-search", "wikipedia", "wolfram-alpha"]
)
agent_chain = load_agent(tool_names=tool_names, strategy=strategy)

st_callback = StreamlitCallbackHandler(st.container())

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent_chain.invoke({"input": prompt}, callbacks=[st_callback])
            if isinstance(response, dict):
                result = response.get("output", (str(response)))
            else:
                result = str(response)
            st.write(result)
        except Exception as e:
            st.error(f"Error executing agent: {e}")
            st.info("Try using a different strategy of tool combination")
