"""Minimal zero-dependency YAML subset parser.

The managed runtime may not ship PyYAML; this fallback parses the small YAML
subset we actually use (nested maps, lists, scalars, comments, inline lists),
so the CLI works out of the box with no pip installs.
"""
import re


def _strip_comment(line):
    return re.sub(r"(\s|^)#.*$", "", line).rstrip()


def _scalar(s):
    s = s.strip()
    if not s:
        return ""
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        return s
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s.strip("\"'").strip()


def _inline_list(s):
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [_scalar(x) for x in inner.split(",")]


def loads(text):
    lines = text.splitlines()
    root = {}
    stack = [[-1, root, None, None]]
    for raw in lines:
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.startswith("- "):
            value = _scalar(content[2:].strip())
            while stack and stack[-1][0] >= indent:
                stack.pop()
            cont = stack[-1][1]
            if not isinstance(cont, list):
                parent = stack[-1][3]
                key = stack[-1][2]
                lst = []
                if parent is not None and key is not None:
                    parent[key] = lst
                stack[-1][1] = lst
                cont = lst
            cont.append(value)
            continue
        m = re.match(r"^(.*?):(?:\s+(.*))?$", content)
        if not m:
            continue
        key = m.group(1).strip()
        rest = (m.group(2) or "").strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        cont = stack[-1][1]
        if rest == "":
            child = {}
            if isinstance(cont, dict):
                cont[key] = child
                stack.append([indent, child, cont, key])
            elif isinstance(cont, list):
                cont.append(child)
                stack.append([indent, child, cont, key])
        else:
            if rest.startswith("[") and rest.endswith("]"):
                val = _inline_list(rest)
            elif rest.startswith("{") and rest.endswith("}"):
                val = rest
            else:
                val = _scalar(rest)
            if isinstance(cont, dict):
                cont[key] = val
            elif isinstance(cont, list):
                cont.append({key: val})
    return root


def parse(path):
    with open(path, "r", encoding="utf-8") as f:
        return loads(f.read())
