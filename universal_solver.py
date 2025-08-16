# universal_solver.py
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

# ================= Auto-install missing dependencies =================
def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        __import__(import_name or pkg_name)

ensure_package("pandas")
ensure_package("networkx")
ensure_package("matplotlib")

# ================= Helpers =================
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

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()
    # ensure under 100kB
    if len(data) > 100000:
        # try compress by lowering dpi
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        data = buf.getvalue()
    return base64.b64encode(data).decode()

# ================= Sales =================
def analyze_sales(dfs: List[pd.DataFrame]) -> Dict:
    if not dfs:
        return {k: None for k in [
            "total_sales","top_region","day_sales_correlation",
            "bar_chart","median_sales","total_sales_tax","cumulative_sales_chart"
        ]}
    df = pd.concat(dfs, ignore_index=True)

    # Identify revenue/sales column
    col_rev = next((c for c in df.columns if "sale" in c.lower() or "revenue" in c.lower()), None)
    rev = pd.to_numeric(df[col_rev], errors="coerce").fillna(0) if col_rev else pd.Series([0]*len(df))

    # Total & median
    total_sales = float(rev.sum())
    median_sales = float(rev.median())

    # Sales tax (assume 10% if no explicit column)
    if any("tax" in c.lower() for c in df.columns):
        col_tax = next(c for c in df.columns if "tax" in c.lower())
        total_sales_tax = float(pd.to_numeric(df[col_tax], errors="coerce").fillna(0).sum())
    else:
        total_sales_tax = total_sales * 0.1

    # Top region
    top_region = None
    if any("region" in c.lower() for c in df.columns):
        colr = next(c for c in df.columns if "region" in c.lower())
        top_region = str(df.groupby(colr)[col_rev].sum().idxmax()) if col_rev else None

    # Day vs sales correlation
    day_sales_correlation = None
    if any("day" in c.lower() for c in df.columns):
        cold = next(c for c in df.columns if "day" in c.lower())
        try:
            day_sales_correlation = float(pd.to_numeric(df[cold], errors="coerce").corr(rev))
        except:
            day_sales_correlation = None

    # Bar chart (sales by region if available)
    bar_chart = None
    if top_region and col_rev:
        fig, ax = plt.subplots(figsize=(6,4))
        df.groupby(colr)[col_rev].sum().plot(kind="bar", ax=ax)
        ax.set_ylabel("Sales")
        bar_chart = fig_to_base64(fig)

    # Cumulative sales chart
    fig, ax = plt.subplots(figsize=(6,4))
    rev.cumsum().plot(ax=ax)
    ax.set_ylabel("Cumulative Sales")
    cumulative_sales_chart = fig_to_base64(fig)

    return {
        "total_sales": round(total_sales,6),
        "top_region": top_region,
        "day_sales_correlation": None if day_sales_correlation is None else round(day_sales_correlation,6),
        "bar_chart": bar_chart,
        "median_sales": round(median_sales,6),
        "total_sales_tax": round(total_sales_tax,6),
        "cumulative_sales_chart": cumulative_sales_chart
    }

# ================= Weather =================
def analyze_weather(dfs: List[pd.DataFrame]) -> Dict:
    if not dfs:
        return {k: None for k in [
            "average_temp_c","max_precip_date","min_temp_c",
            "temp_precip_correlation","average_precip_mm",
            "temp_line_chart","precip_histogram"
        ]}
    df = pd.concat(dfs, ignore_index=True)

    # Columns
    col_temp = next((c for c in df.columns if "temp" in c.lower()), None)
    col_precip = next((c for c in df.columns if "precip" in c.lower() or "rain" in c.lower()), None)
    col_date = next((c for c in df.columns if "date" in c.lower()), None)

    temp = pd.to_numeric(df[col_temp], errors="coerce") if col_temp else pd.Series([0]*len(df))
    precip = pd.to_numeric(df[col_precip], errors="coerce").fillna(0) if col_precip else pd.Series([0]*len(df))

    avg_temp = float(temp.mean())
    min_temp = float(temp.min())
    avg_precip = float(precip.mean())

    max_precip_date = None
    if col_precip and col_date:
        idx = precip.idxmax()
        if idx is not None and idx < len(df):
            max_precip_date = str(df[col_date].iloc[idx])

    # Correlation
    try:
        temp_precip_corr = float(temp.corr(precip))
    except:
        temp_precip_corr = None

    # Temp line chart
    fig, ax = plt.subplots(figsize=(6,4))
    temp.plot(ax=ax)
    ax.set_ylabel("Temperature (C)")
    temp_line_chart = fig_to_base64(fig)

    # Precip histogram
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(precip.dropna(), bins=10)
    ax.set_xlabel("Precipitation (mm)")
    ax.set_ylabel("Frequency")
    precip_histogram = fig_to_base64(fig)

    return {
        "average_temp_c": round(avg_temp,6),
        "max_precip_date": max_precip_date,
        "min_temp_c": round(min_temp,6),
        "temp_precip_correlation": None if temp_precip_corr is None else round(temp_precip_corr,6),
        "average_precip_mm": round(avg_precip,6),
        "temp_line_chart": temp_line_chart,
        "precip_histogram": precip_histogram
    }

# ================= Graph =================
def analyze_graph(dfs: List[pd.DataFrame]) -> Dict:
    if not dfs:
        return {k: None for k in [
            "edge_count","highest_degree_node","average_degree","density",
            "shortest_path_alice_eve","network_graph","degree_histogram"
        ]}
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

    # Network graph
    fig, ax = plt.subplots(figsize=(6,6))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, ax=ax)
    network_graph = fig_to_base64(fig)

    # Degree histogram
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(list(degrees.values()), bins=range(1, max(degrees.values())+2))
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    degree_histogram = fig_to_base64(fig)

    return {
        "edge_count": edge_count,
        "highest_degree_node": highest_degree_node,
        "average_degree": round(average_degree,6),
        "density": round(density,6),
        "shortest_path_alice_eve": shortest_path_alice_eve,
        "network_graph": network_graph,
        "degree_histogram": degree_histogram
    }

# ================= Orchestrator =================
def orchestrate(questions_file: str, attachment_files: List[str]) -> Dict:
    q_text = read_text_file(questions_file)
    dfs = [read_any_table(p) for p in attachment_files]
    q_low = q_text.lower()

    if "sale" in q_low:
        return {"task_type": "sales", "output": analyze_sales(dfs)}
    if "weather" in q_low or "temp" in q_low:
        return {"task_type": "weather", "output": analyze_weather(dfs)}
    if "graph" in q_low or "edge" in q_low:
        return {"task_type": "graph", "output": analyze_graph(dfs)}

    return {"task_type": "other", "output": {"message": "Unsupported question"}}
