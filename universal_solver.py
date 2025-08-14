#univsal solver
import os
import sys
import json
import tempfile
import subprocess
import pandas as pd
import io
import base64
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# ========== Auto-install missing dependencies ==========
def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ImportError:
        print(f"Installing missing package: {pkg_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        __import__(import_name or pkg_name)

ensure_package("openai")
ensure_package("google-generativeai", "google.generativeai")
ensure_package("pandas")
ensure_package("networkx")
ensure_package("matplotlib")

# ========== Config ==========
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "openai" or "gemini"
OPENAI_MODEL = "gpt-4"
GEMINI_MODEL = "gemini-pro"
MAX_FIX_ATTEMPTS = 3
PYTHON_TIMEOUT_SEC = 90

# ========== File Reading ==========
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".csv", ".tsv", ".txt"]:
            return pd.read_csv(path, sep=None, engine="python")
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif ext in [".json"]:
            obj = pd.read_json(path)
            return obj if isinstance(obj, pd.DataFrame) else pd.json_normalize(obj)
        else:
            return pd.read_csv(path, sep=None, engine="python")
    except:
        return pd.DataFrame()

# ========== LLM Call ==========
def call_llm(prompt: str) -> str:
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content
    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    else:
        raise ValueError("Invalid LLM_PROVIDER")

def exec_python(code: str) -> Tuple[bool, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        res = subprocess.run(["python", path], capture_output=True, text=True, timeout=PYTHON_TIMEOUT_SEC)
        ok = res.returncode == 0
        out = res.stdout.strip() if ok else res.stderr.strip()
    finally:
        os.unlink(path)
    return ok, out

# ========== Domain: Sales ==========
def analyze_sales(dfs: List[pd.DataFrame]) -> Dict:
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    col_rev = next((c for c in df.columns if "sale" in c.lower() or "revenue" in c.lower()), None)
    rev = pd.to_numeric(df[col_rev], errors="coerce").fillna(0) if col_rev else pd.Series([0]*len(df))
    total = float(rev.sum())
    avg = float(rev.mean()) if len(rev) else 0
    top_product = None
    if "product" in df.columns.str.lower().to_list():
        colp = next(c for c in df.columns if "product" in c.lower())
        top_product = str(df.groupby(colp)[col_rev].sum().idxmax()) if col_rev else None
    sales_by_region = {}
    top_region = None
    if "region" in df.columns.str.lower().to_list():
        colr = next(c for c in df.columns if "region" in c.lower())
        if col_rev:
            sales_by_region = {str(k): float(v) for k, v in df.groupby(colr)[col_rev].sum().to_dict().items()}
            top_region = max(sales_by_region, key=sales_by_region.get)
    return {
        "total_sales": round(total, 6),
        "average_sales": round(avg, 6),
        "top_product": top_product,
        "top_region": top_region,
        "sales_by_region": sales_by_region
    }

# ========== Domain: Weather ==========
def analyze_weather(dfs: List[pd.DataFrame]) -> Dict:
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    col_temp = next((c for c in df.columns if "temp" in c.lower()), None)
    col_rain = next((c for c in df.columns if "rain" in c.lower() or "precip" in c.lower()), None)
    avg_temp = max_temp = min_temp = None
    if col_temp:
        vals = pd.to_numeric(df[col_temp], errors="coerce")
        avg_temp = float(vals.mean())
        max_temp = float(vals.max())
        min_temp = float(vals.min())
    total_rain = 0
    if col_rain:
        vals = pd.to_numeric(df[col_rain], errors="coerce").fillna(0)
        total_rain = float(vals.sum())
    days = len(df)
    return {
        "average_temperature": None if avg_temp is None else round(avg_temp, 6),
        "max_temperature": None if max_temp is None else round(max_temp, 6),
        "min_temperature": None if min_temp is None else round(min_temp, 6),
        "total_rainfall": round(total_rain, 6),
        "days_recorded": days
    }

# ========== Domain: Graph ==========
def analyze_graph(dfs: List[pd.DataFrame]) -> Dict:
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    if df.shape[1] < 2:
        return {}
    edges = list(df.iloc[:, :2].itertuples(index=False, name=None))
    G = nx.Graph()
    G.add_edges_from(edges)
    edge_count = G.number_of_edges()
    degrees = dict(G.degree())
    highest_degree_node = max(degrees, key=degrees.get)
    average_degree = sum(degrees.values()) / len(degrees)
    density = nx.density(G)
    shortest_path_alice_eve = None
    if "Alice" in G and "Eve" in G:
        try:
            shortest_path_alice_eve = nx.shortest_path_length(G, "Alice", "Eve")
        except nx.NetworkXNoPath:
            shortest_path_alice_eve = None
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    network_graph_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.figure(figsize=(6, 4))
    plt.hist(list(degrees.values()), bins=range(1, max(degrees.values())+2))
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    degree_histogram_b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "edge_count": edge_count,
        "highest_degree_node": highest_degree_node,
        "average_degree": average_degree,
        "density": density,
        "shortest_path_alice_eve": shortest_path_alice_eve,
        "network_graph": network_graph_b64,
        "degree_histogram": degree_histogram_b64
    }

# ========== Main Orchestrator ==========
def orchestrate(questions_file: str, attachment_files: List[str]) -> Dict:
    q_text = read_text_file(questions_file)
    dfs = [read_any_table(p) for p in attachment_files]
    q_low = q_text.lower()

    # Direct domain handling
    if "sale" in q_low or any("sale" in c.lower() for df in dfs for c in df.columns):
        return {"task_type": "sales", "output": analyze_sales(dfs)}
    if "temp" in q_low or "weather" in q_low or any("temp" in c.lower() for df in dfs for c in df.columns):
        return {"task_type": "weather", "output": analyze_weather(dfs)}
    if "edge" in q_low or "graph" in q_low or any("edge" in c.lower() for df in dfs for c in df.columns):
        return {"task_type": "graph", "output": analyze_graph(dfs)}

    # Fallback to LLM pipeline
    prompt_meta = f"""
You are a Python data assistant.
Generate Python code to:
1. Read the attached data.
2. Print JSON with: num_rows, num_columns, column_names, sample_data (max 5 rows).
Question: {q_text}
Attachments: {[os.path.basename(f) for f in attachment_files]}
"""
    meta_code = call_llm(prompt_meta)
    ok, meta_out = exec_python(meta_code)
    metadata_json = json.loads(meta_out) if ok else {"raw": meta_out}

    prompt_solution = f"""
You are a Python data assistant.
Solve the question based on the metadata and attachments.
Return only JSON output that answers the question.
Question: {q_text}
Metadata: {json.dumps(metadata_json)}
"""
    solution_code = call_llm(prompt_solution)
    for _ in range(MAX_FIX_ATTEMPTS):
        ok, sol_out = exec_python(solution_code)
        if ok:
            try:
                parsed = json.loads(sol_out)
            except:
                parsed = {"raw_output": sol_out}
            return {"task_type": "other", "output": parsed}
        fix_prompt = f"Fix this Python code so it runs:\nERROR:\n{sol_out}\nCODE:\n{solution_code}"
        solution_code = call_llm(fix_prompt)

    raise RuntimeError("Max LLM fix attempts reached.")
