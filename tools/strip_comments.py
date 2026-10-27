"""Remove comments and docstrings from an exported model.py, leaving the code unchanged.

    python tools/strip_comments.py teams/exml_rl/model.py [--out OUT] [--check]

Comments are found with `tokenize`; docstrings and other bare string statements with
`ast`. Only the lines they occupy are touched, so every other byte (including the
embedded weight blob) is kept. The result is verified: its AST must equal the
original's AST with the bare string statements removed, and it must parse as
Python 3.9.
"""

import argparse
import ast
import io
import sys
import tokenize
from pathlib import Path

BODY_FIELDS = ("body", "orelse", "finalbody")


def _is_bare_string(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _without_bare_strings(tree):
    """Copy of `tree` with bare string statements removed (empty bodies become pass)."""
    class Strip(ast.NodeTransformer):
        def generic_visit(self, node):
            super().generic_visit(node)
            for field in BODY_FIELDS:
                body = getattr(node, field, None)
                if isinstance(body, list) and body and all(isinstance(x, ast.stmt) for x in body):
                    kept = [x for x in body if not _is_bare_string(x)]
                    if not kept and field == "body":
                        kept = [ast.Pass()]
                    setattr(node, field, kept)
            return node
    return Strip().visit(tree)


def strip(source):
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    drop = set()
    replace = {}
    parents = [n for n in ast.walk(tree) if any(isinstance(getattr(n, f, None), list) for f in BODY_FIELDS)]
    for node in parents:
        for field in BODY_FIELDS:
            body = getattr(node, field, None)
            if not (isinstance(body, list) and body and all(isinstance(x, ast.stmt) for x in body)):
                continue
            strings = [x for x in body if _is_bare_string(x)]
            for x in strings:
                first, last = x.lineno - 1, x.end_lineno - 1
                if lines[first][:x.col_offset].strip() or lines[last][x.end_col_offset:].strip():
                    raise ValueError(f"string statement shares a line with code at line {x.lineno}")
                drop.update(range(first, last + 1))
            if strings and len(strings) == len(body) and field == "body":
                x = strings[0]
                replace[x.lineno - 1] = " " * x.col_offset + "pass\n"
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start[0] - 1, tok.start[1]
        if row in drop:
            continue
        before = lines[row][:col]
        if before.strip():
            lines[row] = before.rstrip() + "\n"
        else:
            drop.add(row)
    out = []
    for i, line in enumerate(lines):
        if i in replace:
            out.append(replace[i])
        elif i not in drop:
            out.append(line)
    return "".join(out)


def verify(original, stripped):
    want = ast.dump(_without_bare_strings(ast.parse(original)))
    got = ast.dump(_without_bare_strings(ast.parse(stripped)))
    if want != got:
        raise AssertionError("stripped code is not AST-equivalent to the original")
    ast.parse(stripped, feature_version=(3, 9))
    leftover = [t for t in tokenize.generate_tokens(io.StringIO(stripped).readline) if t.type == tokenize.COMMENT]
    if leftover:
        raise AssertionError(f"{len(leftover)} comments left")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--out", help="default: rewrite in place")
    p.add_argument("--check", action="store_true", help="only verify, do not write")
    args = p.parse_args(argv)
    src = Path(args.model).read_text(encoding="utf-8")
    new = strip(src)
    verify(src, new)
    if not args.check:
        Path(args.out or args.model).write_text(new, encoding="utf-8")
    print(f"{args.model}: {len(src.splitlines())} -> {len(new.splitlines())} lines, "
          f"{len(src) / 1024:.1f} -> {len(new) / 1024:.1f} KiB, AST-equivalent, parses as 3.9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
