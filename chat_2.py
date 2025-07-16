# Version 2 of the application with enhanced One-shot mode prompt and logic
# Multi-turn mode and spreadsheet untouched
# Might need further optimisation
import pandas as pd
import re
import tempfile
import subprocess
import json
import os
import ast
import difflib
import streamlit as st
from mitosheet.streamlit.v1 import spreadsheet
from langchain_community.chat_models import ChatOllama

# === Utility Functions ===

def get_defined_and_used_variables(code_str):
    try:
        tree = ast.parse(code_str)
        defined_vars, used_vars = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
        return defined_vars, used_vars
    except Exception:
        return set(), set()

def extract_code_from_llm_response(llm_response):
    match = re.search(r"```(?:python)?(.*?)```", llm_response, re.DOTALL)
    return match.group(1).strip() if match else None

def fix_common_hallucinations(code):
    match = re.search(r"(\w+)\s*=\s*df\.groupby", code)
    if match:
        var = match.group(1)
        code = re.sub(r"\bmax_rating_tax\b", var, code)
        code = re.sub(r"\bmax_avg\b", var, code)
    return code

def auto_correct_undefined_vars(code):
    defined, used = get_defined_and_used_variables(code)
    excluded = {"df", "pd", "result", "json", "mean", "sum", "len", "min", "max",
                "idxmin", "idxmax", "groupby", "loc", "at", "print", "import",
                "in", "for", "if", "else", "elif", "try", "except", "range", "list",
                "set", "tuple", "dict"}
    undefined = used - defined - excluded
    for var in undefined:
        suggestion = difflib.get_close_matches(var, defined, n=1, cutoff=0.75)
        if suggestion:
            code = re.sub(rf"\b{re.escape(var)}\b", suggestion[0], code)
    return code

def execute_code(code, csv_string):
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            script_path = f.name
            helper = """
import pandas as pd, json
def to_safe_json(result):
    def convert(o):
        if hasattr(o, 'item'):
            return o.item()
        if isinstance(o, (pd.Series, pd.DataFrame)):
            return o.to_dict()
        return str(o)
    return json.dumps(result, default=convert)
"""
            full_script = (
                "import pandas as pd\nfrom io import StringIO\n"
                f"{helper}\n"
                f"df = pd.read_csv(StringIO(\"\"\"{csv_string}\"\"\"))\n"
                f"{code}\nprint(to_safe_json(result))\n"
            )
            f.write(full_script)
        output = subprocess.check_output(["python3", script_path], stderr=subprocess.STDOUT)
        return json.loads(output.decode()), None
    except subprocess.CalledProcessError as e:
        return None, e.output.decode()
    except json.JSONDecodeError:
        return None, "⚠️ JSON parsing error."
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

# === Streamlit App ===

st.set_page_config(page_title="Conversational Data Analyst", layout="wide")
st.title("💬 Conversational Data Analyst")
st.markdown("Upload your CSV and ask data questions using natural language.")

uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully!")
    spreadsheet(df)

    csv_string = uploaded_file.getvalue().decode("utf-8")
    columns = df.columns.tolist()
    columns_str = ", ".join(f"'{col}'" for col in columns)
    llm = ChatOllama(model="codegemma:latest", temperature=0)

    mode = st.radio("🧭 Choose mode:", ["Chat Mode (multi-turn)", "One-shot Mode"], horizontal=True)

    if mode == "One-shot Mode":
        user_query = st.text_input("❓ Ask your question about the dataset:")
        if user_query:
            st.markdown("---")

            prompt = f"""
You are a data scientist working with a pandas DataFrame named `df`, which contains the following columns:
{columns_str}

All monetary values are in rupees.

Your task is to write **correct, complete, and executable Python code** that answers the user's question below:
\"\"\"{user_query}\"\"\"

--- Guidelines for the generated code ---

✅ Always:
- Use only the column names **exactly as listed** above — no renaming, abbreviating, or re-typing.
- Use **explicit indexing** methods such as `.loc[]` or `.at[]` for filtering rows.
- Define **all variables** before using them — especially those used in the final `result` dictionary.
- Use **clear and semantically correct variable names** (e.g., `city_with_highest_avg_carbon`).
- Match logic with language:
  - If the user asks for the “highest,” use `.idxmax()`.
  - If the user asks for the “lowest,” use `.idxmin()`.

❌ Never:
- Use undefined or placeholder variables (e.g., `max_avg`, `city` if it hasn’t been declared).
- Include `print()` statements or any non-code commentary.
- Write partial, ambiguous, or syntactically invalid code.

--- Output formatting rules ---

The **last line** of the code **must be** the variable `result`, evaluated alone so it is returned.

The `result` should be a **Python dictionary** summarizing the answer, with keys that are human-readable and values that are already computed using earlier variables.

✅ Example of correct style:

```python
city = df.groupby('City')['CarbonScore'].mean().idxmin()
avg_cost = df.loc[df['City'] == city, 'HotelCost'].mean()
result = {{
    'CityWithLowestAverageCarbonScore': city,
    'AverageHotelCostInThisCity': avg_cost
}}

"""
            llm_response = llm.invoke(prompt).content
            code = extract_code_from_llm_response(llm_response)

            if code:
                code = fix_common_hallucinations(code)
                code = auto_correct_undefined_vars(code)

                # Double-check variables in result are defined
                defined, used = get_defined_and_used_variables(code)
                excluded = {"df", "pd", "result", "json", "mean", "sum", "len", "min", "max",
                            "idxmin", "idxmax", "groupby", "loc", "at", "print", "import",
                            "in", "for", "if", "else", "elif", "try", "except"}
                result_dict_match = re.search(r"result\s*=\s*{(.*?)}", code, re.DOTALL)
                if result_dict_match:
                    keys = set(re.findall(r"\b(\w+)\b", result_dict_match.group(1)))
                    if not keys.issubset(defined | excluded):
                        st.warning(f"⚠️ Some variables in result are not defined: {keys - defined}")

                result, error = execute_code(code, csv_string)

                if result:
                    st.success("✅ Final result:")
                    st.json(result)

                    explanation_prompt = f"""

You are a helpful data analyst assistant.

The user asked the following question:
\"\"\"{user_query}\"\"\"

The code executed successfully and produced the following result as a Python dictionary:

{result}

Please write a concise and natural-language answer to the user's question using this result.
Include specific values and units like rupees.
"""
                    explanation = llm.invoke(explanation_prompt).content.strip()
                    st.markdown(f"💡 Explanation: {explanation}")
                else:
                    st.error(f"❌ Execution failed:\n{error}")
            else:
                st.error("❌ Could not extract code block from LLM response.")

    elif mode == "Chat Mode (multi-turn)":
        st.markdown("### 💬 Multi-turn Chat with your data")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for i, (q, r) in enumerate(st.session_state.chat_history):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(q)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(r)

        user_query = st.chat_input("Type your question about the dataset...")

        if user_query:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_query)

            prompt = f"""
You are a data scientist working with a pandas DataFrame named `df`, which contains the following columns:
{columns_str}

All monetary values are in rupees.

The user asked the following question:
\"\"\"{user_query}\"\"\"

Write **complete, executable Python code** to compute the answer using the DataFrame `df`.

Ensure:
- No undefined variables.
- Use only columns exactly as listed above.
- The final line must be: result
- result should be a Python dictionary with clean key names.

Only return valid Python code inside a code block.
"""

            llm_response = llm.invoke(prompt).content
            code = extract_code_from_llm_response(llm_response)

            if code:
                code = fix_common_hallucinations(code)
                code = auto_correct_undefined_vars(code)

                result, error = execute_code(code, csv_string)

                if result:
                    explanation_prompt = f"""
You are a helpful data analyst assistant.

The user asked the following question:
\"\"\"{user_query}\"\"\"

The code executed successfully and produced the following result:

{result}

Please write a concise and natural-language answer to the user's question using this result. Mention rupees if relevant.
"""
                    explanation = llm.invoke(explanation_prompt).content.strip()
                    st.session_state.chat_history.append((user_query, explanation))

                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(explanation)
                else:
                    st.session_state.chat_history.append((user_query, f"❌ Error: {error}"))
                    with st.chat_message("assistant", avatar="🤖"):
                        st.error(f"❌ Execution failed:\n{error}")
            else:
                st.session_state.chat_history.append((user_query, "❌ Could not extract code block."))
                with st.chat_message("assistant", avatar="🤖"):
                    st.error("❌ Could not extract code block from LLM response.")

