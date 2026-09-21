"""A3：CPLEX Studio cplex.exe LP 文件求解通路——突破 pip 社区版 1000 变量限制。

流程：docplex 模型 -> export LP -> cplex.exe 求解 -> 解析 .sol (XML) -> 解字典。
仅用于离线/评估场景（每次求解含进程+文件开销 ~0.5-2s），训练循环仍用 pip 引擎。

用法:
    from src.models.cplex_studio_solver import solve_via_studio
    sol = solve_via_studio(model, time_limit=60)
"""
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

CPLEX_EXE = Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe")


def solve_via_studio(wrapper, time_limit: float = 60.0) -> Optional[Dict]:
    """wrapper: CDW_LIRP_MultiModel（用其 .model 导出 LP，用 .y/.x/.I 回填解）。"""
    model = wrapper.model if hasattr(wrapper, "model") else wrapper
    if not CPLEX_EXE.exists():
        raise FileNotFoundError(f"CPLEX Studio not found: {CPLEX_EXE}")
    with tempfile.TemporaryDirectory() as td:
        lp = Path(td) / "model.lp"
        sol = Path(td) / "model.sol"
        model.export_as_lp(str(lp))
        cmds = str(CPLEX_EXE).replace("\\", "/")
        script = (
            f"read {str(lp).replace(chr(92), '/')}\n"
            f"set timelimit {int(time_limit)}\n"
            "optimize\n"
            f"write {str(sol).replace(chr(92), '/')}\n"
            "quit\n"
        )
        r = subprocess.run([str(CPLEX_EXE), "-c", script], capture_output=True,
                           text=True, timeout=time_limit + 120)
        if not sol.exists():
            return None
        return _parse_sol(sol, wrapper)


def _parse_sol(sol_path: Path, model) -> Optional[Dict]:
    ns = {"cpx": "http://www.cplex.com/xml" + "/formats2012/II/analyze/solution"}
    # .sol 命名空间随版本；宽松解析
    tree = ET.parse(sol_path)
    root = tree.getroot()
    header = root.find("header")
    if header is None or header.get("solutionStatusString", "").lower().startswith("infeasible"):
        return None
    obj = float(header.get("objectiveValue", "nan"))
    values = {}
    for v in root.iter():
        tag = v.tag.split("}")[-1]
        if tag == "variable" and v.get("name"):
            values[v.get("name")] = float(v.get("value", 0.0))
    # 回填到模型结构（wrapper 缺 .y 时仅返回 raw，供 SAA 扩展式等按变量名取值）
    result = {"objective": obj, "y": {}, "x": {}, "I": {}, "raw": values}
    ys = getattr(model, "y", None)
    if ys:
        for f, var in ys.items():
            result["y"][f] = values.get(var.name if hasattr(var, "name") else f"f_{f}", 0.0)
        for k, var in (getattr(model, "x", None) or {}).items():
            val = values.get(var.name, 0.0)
            if val > 1e-6:
                result["x"][k] = val
        for k, var in (getattr(model, "I", None) or {}).items():
            result["I"][k] = values.get(var.name, 0.0)
    return result
