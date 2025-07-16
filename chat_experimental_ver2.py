#Version 2 of the application with more modules and further optimisation
#Experimental
#Might need further optimisation
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

# ========== Utility Functions ==========

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
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", llm_response, re.DOTALL | re.IGNORECASE)
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

def ask_llm_to_breakdown_question(question, columns_str):
    prompt = f"""
You are a data scientist assistant helping to answer complex questions
about a pandas DataFrame named df, which has the following columns:
{columns_str}

The user asked:
\"\"\"{question}\"\"\"

Break this into simpler subquestions using only the column names above.
Return them as a numbered list. If already simple, return a one-item list.
"""
    response = llm.invoke(prompt).content.strip()
    return [re.match(r"^\s*\d+\.\s*(.+)$", l).group(1) for l in response.splitlines() if re.match(r"^\s*\d+\.", l)] or [question]

def has_dangerous_patterns(code: str) -> bool:
    return bool(re.search(r"df\['[^']+'\]\.idxmax\(\)", code)) and not "groupby" in code

def combine_subresults(subresults):
    combined = {}
    for i, res in enumerate(subresults):
        for k, v in res.items():
            combined[k if k not in combined else f"{k}_part{i+1}"] = v
    return combined

def combine_results_by_consensus(results_list):
    dict_results = [r['result'] for r in results_list if isinstance(r, dict) and 'result' in r]
    if not dict_results:
        return None
    keys = set().union(*[r.keys() for r in dict_results])
    combined = {}
    for k in keys:
        vals = [r.get(k) for r in dict_results]
        if len(set(json.dumps(v, sort_keys=True) for v in vals)) == 1:
            combined[k] = vals[0]
        else:
            combined[k] = vals
    return combined

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

# ========== Streamlit App ==========
st.set_page_config(page_title="Conversational Data Analyst", layout="wide")
st.title("💬 Conversational Data Analyst")
st.markdown("Upload your CSV and ask data questions using natural language.")

uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully!")
    spreadsheet(df)  # Show editable spreadsheet view

    csv_string = uploaded_file.getvalue().decode("utf-8")
    columns = df.columns.tolist()
    columns_str = ", ".join(f"'{col}'" for col in columns)
    llm = ChatOllama(model="codegemma:latest", temperature=0)

    mode = st.radio("🧭 Choose mode:", ["Chat Mode (multi-turn)", "One-shot Mode"], horizontal=True)

    if mode == "One-shot Mode":
        user_query = st.text_input("❓ Ask your question about the dataset:")
        if user_query:
            st.markdown("---")
            temperatures = [0, 0.3, 0.7]
            all_results, final_result = [], None

            for temp in temperatures:
                llm.temperature = temp
                code_prompt = f"""
You are a data scientist working with a pandas DataFrame named df
with columns: {columns_str}

Answer the following question using valid Python code:
\"\"\"{user_query}\"\"\"

Assign final result to a variable named `result`. Return only code.
"""
                llm_response = llm.invoke(code_prompt).content
                code = extract_code_from_llm_response(llm_response)

                if code:
                    code = fix_common_hallucinations(code)
                    code = auto_correct_undefined_vars(code)
                    if has_dangerous_patterns(code):
                        st.warning("⚠️ Unsafe pattern detected, skipping.")
                        continue
                    result, error = execute_code(code, csv_string)
                    if result:
                        all_results.append({"result": result})
                        break

            if not all_results:
                st.error("❗ Could not get a valid response. Try refining the question.")
            else:
                final_result = combine_results_by_consensus(all_results)
                st.success("✅ Final result:")
                st.json(final_result)

                explanation_prompt = f"""
The user asked:
\"{user_query}\"

The final result is:
{final_result}

Write a clear answer using this result.
"""
                explanation = llm.invoke(explanation_prompt).content.strip()
                st.markdown(f"💡 **Explanation:** {explanation}")

    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.chat_input("Ask a question about your data")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                current_question = user_query
                result = None
                for temp in [0, 0.3, 0.7]:
                    llm.temperature = temp
                    attempt = 0
                    while attempt < 2 and result is None:
                        attempt += 1
                        prompt = f"""
You are a data scientist working with a DataFrame `df` with columns:
{columns_str}

Answer:
\"\"\"{current_question}\"\"\"

Use correct column names and assign to `result`. Return only code block.
"""
                        response = llm.invoke(prompt).content
                        code = extract_code_from_llm_response(response)
                        if code:
                            code = fix_common_hallucinations(code)
                            code = auto_correct_undefined_vars(code)
                            if has_dangerous_patterns(code):
                                st.warning("⚠️ Unsafe code detected, skipping.")
                                break
                            result, error = execute_code(code, csv_string)
                            if result:
                                break

                if not result:
                    subqs = ask_llm_to_breakdown_question(current_question, columns_str)
                    subresults = []
                    for sq in subqs:
                        sub_prompt = f"""
Given df with columns: {columns_str}

Answer:
\"\"\"{sq}\"\"\"

Assign to `result` and return only code.
"""
                        try:
                            response = llm.invoke(sub_prompt).content
                            code = extract_code_from_llm_response(response)
                            if code:
                                code = fix_common_hallucinations(code)
                                code = auto_correct_undefined_vars(code)
                                if has_dangerous_patterns(code):
                                    continue
                                sub_result, _ = execute_code(code, csv_string)
                                if sub_result:
                                    subresults.append(sub_result)
                        except Exception:
                            continue
                    if subresults:
                        result = combine_subresults(subresults)

                if result:
                    explanation_prompt = f"""
User question:
\"{user_query}\"

Result:
{result}

Generate a natural language summary using this result only.
"""
                    explanation = llm.invoke(explanation_prompt).content.strip()
                    st.markdown(f"✅ **Answer:** {explanation}")
                    st.session_state.chat_history.append((user_query, explanation))
                else:
                    st.error("❗ Failed to produce result or explanation.")
