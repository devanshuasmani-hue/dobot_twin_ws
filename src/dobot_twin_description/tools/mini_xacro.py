#!/usr/bin/env python3
"""Minimal xacro subset expander (verification fallback, NOT a replacement for xacro).

Supports only what dobot_description uses: xacro:arg / property / include / macro
(with `name:=default` params) / if / unless, ${expr}, $(arg x), $(find pkg).
Use the real `xacro` whenever ROS is installed; this exists so the model can be
verified on machines without ROS (CI, sandboxes). Expressions run in a restricted
namespace (math functions + defined properties).
"""
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

XACRO_NS = "http://ros.org/wiki/xacro"
Q = "{%s}" % XACRO_NS

_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_MATH.update(abs=abs, min=min, max=max, round=round, float=float, int=int, str=str)


class MiniXacro:
    def __init__(self, package_roots, args=None):
        self.roots = package_roots          # {'dobot_description': '/path/to/pkg'}
        self.cli_args = dict(args or {})
        self.args = {}
        self.macros = {}

    # ---- substitution -------------------------------------------------
    @staticmethod
    def _literal(s):
        t = s.strip()
        if t in ("true", "True"):
            return True
        if t in ("false", "False"):
            return False
        try:
            return int(t)
        except ValueError:
            pass
        try:
            return float(t)
        except ValueError:
            return s

    def _ctx(self, scope):
        c = dict(_MATH)
        c.update(scope)
        return c

    def _subst(self, text, scope):
        if text is None:
            return text

        def sub_dollar_paren(m):
            kind, val = m.group(1), m.group(2).strip()
            if kind == "arg":
                return str(self.args[val])
            if kind == "find":
                return self.roots[val]
            raise ValueError("unsupported $(%s)" % kind)

        text = re.sub(r"\$\((arg|find)\s+([^)]+)\)", sub_dollar_paren, text)
        whole = re.fullmatch(r"\s*\$\{([^}]*)\}\s*", text)
        if whole:
            return eval(whole.group(1), {"__builtins__": {}}, self._ctx(scope))

        def sub_expr(m):
            v = eval(m.group(1), {"__builtins__": {}}, self._ctx(scope))
            return repr(float(v)) if isinstance(v, float) else str(v)

        return re.sub(r"\$\{([^}]*)\}", sub_expr, text)

    def _val(self, text, scope):
        v = self._subst(text, scope)
        return self._literal(v) if isinstance(v, str) else v

    # ---- expansion ----------------------------------------------------
    def expand_children(self, parent_el, scope, out):
        for ch in list(parent_el):
            self.expand(ch, scope, out)

    def expand(self, el, scope, out):
        tag = el.tag
        if not tag.startswith(Q):
            new = ET.SubElement(out, tag)
            for k, v in el.attrib.items():
                s = self._subst(v, scope)
                new.set(k, s if isinstance(s, str) else (repr(s) if isinstance(s, float) else str(s)))
            new.text = (el.text or "").strip() or None
            self.expand_children(el, scope, new)
            return
        name = tag[len(Q):]
        if name == "arg":
            a = el.get("name")
            self.args[a] = self.cli_args.get(a, self._subst(el.get("default"), scope))
        elif name == "property":
            scope[el.get("name")] = self._val(el.get("value"), scope)
        elif name == "include":
            path = self._subst(el.get("filename"), scope)
            root = ET.parse(path).getroot()
            self.expand_children(root, scope, out)
        elif name == "macro":
            params = []
            for p in el.get("params", "").split():
                if ":=" in p:
                    k, d = p.split(":=", 1)
                    params.append((k, d))
                else:
                    params.append((p, None))
            self.macros[el.get("name")] = (params, el)
        elif name in ("if", "unless"):
            cond = self._val(el.get("value"), scope)
            if isinstance(cond, str):
                raise ValueError("non-boolean condition %r" % cond)
            if bool(cond) == (name == "if"):
                self.expand_children(el, scope, out)
        elif name in self.macros:
            params, mel = self.macros[name]
            local = dict(scope)
            for k, d in params:
                if k in el.attrib:
                    local[k] = self._val(el.get(k), scope)
                elif d is not None:
                    local[k] = self._literal(d)
                else:
                    raise ValueError("macro %s missing param %s" % (name, k))
            self.expand_children(mel, local, out)
        else:
            raise ValueError("unsupported xacro element: " + name)

    def run(self, path):
        root = ET.parse(path).getroot()
        out = ET.Element(root.tag)
        for k, v in root.attrib.items():
            if not k.startswith("{") and not k.startswith("xmlns"):
                out.set(k, v)
        self.expand_children(root, {}, out)
        return out


def expand_file(xacro_path, pkg_root, **args):
    """Expand *.urdf.xacro -> ElementTree root of plain URDF."""
    m = MiniXacro({"dobot_description": pkg_root}, {k: str(v) for k, v in args.items()})
    return m.run(xacro_path)


if __name__ == "__main__":
    pkg = sys.argv[1]
    kv = dict(a.split(":=") for a in sys.argv[3:])
    root = expand_file(os.path.join(pkg, "model", sys.argv[2]), pkg, **kv)
    ET.indent(root)
    print(ET.tostring(root, encoding="unicode"))
