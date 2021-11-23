from typing import Optional
import intervaltree
import ast
import pandas as pd

def parse_file_to_intervaltree(file_path: str) -> intervaltree.IntervalTree:
    """https://julien.danjou.info/finding-definitions-from-a-source-file-and-a-line-number-in-python"""

    def node_interval(node: ast.stmt):
        min_ = node.lineno
        max_ = node.lineno
        for node in ast.walk(node):
            if hasattr(node, "lineno"):
                min_ = min(min_, node.lineno)
                max_ = max(max_, node.lineno)
        return min_, max_ + 1
    
    with open(file_path, "r") as f:
        parsed = ast.parse(f.read())
    tree = intervaltree.IntervalTree()
    for item in ast.walk(parsed):
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)):
            interval_ = node_interval(item)
            tree[interval_[0]: interval_[1]] = item.name

    return tree

def get_perfomance_dataframe(scalene_json_data: dict, main_filename: str = "main.py", top: Optional[int] = 5):
    total_time = scalene_json_data["elapsed_time_sec"]
    top_lines = list()
    for file in scalene_json_data["files"]:
        if file == main_filename:
            continue
        tree = parse_file_to_intervaltree(file)
        df_view = pd.DataFrame(scalene_json_data["files"][file]["lines"])
        df_view["n_cpu_percent_all"] = df_view["n_cpu_percent_python"] + df_view["n_cpu_percent_c"] 
        df_view = df_view.sort_values(by="n_cpu_percent_all", ascending=False)
        df_view["file"] = file
        df_view["function"] = df_view.lineno.apply(lambda y: ".".join(i.data for i in sorted(tree[y], key=lambda x: x.begin)))
        df_view["function_n_cpu_percent_all"] = df_view.groupby("function").n_cpu_percent_all.transform("sum")
        df_view["function_n_copy_mb_s"] = df_view.groupby("function").n_copy_mb_s.transform("sum")
        if top:
            df_view = df_view.head(top)
        df_view["percent_cpu_time"] = scalene_json_data["files"][file]["percent_cpu_time"]
        df_view["total_time"] = total_time
        top_lines.append(df_view)
    
    top_lines = pd.concat(top_lines).sort_values(by=["percent_cpu_time", "function_n_cpu_percent_all", "n_cpu_percent_all"], ascending=False)
    top_lines = top_lines.set_index(["file", "total_time", "percent_cpu_time", "function", "function_n_cpu_percent_all", "function_n_copy_mb_s"])[["line", "n_cpu_percent_all", "n_cpu_percent_c", "n_cpu_percent_python", "n_copy_mb_s"]]
    return top_lines
