import streamlit as st
import pandas as pd
import tempfile
import re
from mitosheet.streamlit.v1 import spreadsheet

from langchain.agents import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.utilities import PythonREPL
from langchain_community.chat_models import ChatOllama

# Streamlit setup
st.set_page_config(page_title="🧠 CSV Chat Agent", layout="wide")
st.title("🧠 CSV Chat Agent with Code Execution & LLM Explanation")
st.markdown("""
Upload your CSV file, explore it visually with **Mito**,  
ask questions in natural language, and get:
- 🧾 Python code from a CSV agent
- 🐍 Code execution via Python REPL
- 🧠 Human-readable explanation from LLM

**Powered by LangChain Experimental Agents + Python REPL**
""")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV", type=["csv"])

if uploaded_file is not None:
    # Read and display data
    df = pd.read_csv(uploaded_file)
    spreadsheet(df)

    # Initialize LLM
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOllama(model="llama3.2", temperature=0)

    # Initialize Python REPL
    if "repl" not in st.session_state:
        st.session_state.repl = PythonREPL()

    # Expose df to REPL globals
    globals()["df"] = df

    # Write uploaded file to disk for CSV agent
    if "csv_agent" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        st.session_state.csv_agent = create_csv_agent(
            llm=st.session_state.llm,
            path=tmp_path,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,  # ⛔ Required for CSV agent to work
            verbose=False
        )

    # Ask user for a question
    st.subheader("💬 Ask a question about your data")
    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("🤖 Generating analysis code..."):
            try:
                # Step 1: CSV agent generates code
                agent_response = st.session_state.csv_agent.run(user_query)

                # Step 2: Extract Python code block from response
                code_blocks = re.findall(r"```(?:python)?(.*?)```", agent_response, re.DOTALL)

                if code_blocks:
                    code = code_blocks[0].strip()
                    st.markdown("### 🧾 Generated Python Code:")
                    st.code(code, language="python")

                    # Step 3: Execute code in REPL
                    with st.spinner("⚙️ Executing code..."):
                        try:
                            execution_result = st.session_state.repl.run(code)
                            st.markdown("### ✅ Code Execution Output:")
                            st.write(execution_result)
                        except Exception as exec_err:
                            st.error(f"❌ Error while executing code:\n\n{exec_err}")
                            execution_result = None

                    # Step 4: Summarize result with LLM
                    if execution_result:
                        st.markdown("### 🧠 LLM Explanation:")
                        explanation_prompt = f"""
You are a helpful data analyst.

The user asked: {user_query}

Here is the output from the code execution:
{execution_result}

Summarize the result in clear language for the user.
"""
                        try:
                            summary = st.session_state.llm.invoke(explanation_prompt)
                            st.write(summary)
                        except Exception as e:
                            st.error(f"❌ LLM explanation error:\n\n{e}")
                else:
                    st.markdown("### 🤖 Agent Response (no code found):")
                    st.write(agent_response)

            except Exception as e:
                st.error(f"❌ CSV Agent Error:\n\n{e}")

