from pathlib import Path
from collections import Counter, defaultdict
import ast, csv, hashlib, json, os, re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SKIP = {'.git', '__pycache__', '.pytest_cache', '.deepeval', '.mypy_cache', '.ruff_cache', 'node_modules', 'wandb'}
TEXT = {'.py', '.md', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.txt', '.csv', '.log', '.xml', '.sha256'}
inventory, corpus, symbols, errors = [], {}, {}, []
for base, dirs, names in os.walk(ROOT):
    dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.venv') and not (Path(base) == ROOT / 'results' and d == 'codebase_review'))
    for name in sorted(names):
        p = Path(base) / name
        rel = p.relative_to(ROOT).as_posix()
        stat = p.stat()
        rec = {'path': rel, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        safe = (p.suffix in TEXT or name in {'.gitignore', '.gitattributes', '.env.example', '.gitkeep'}) and not (name.startswith('.env') and name != '.env.example')
        if rel.startswith('data/raw/') or stat.st_size > 10_000_000:
            safe = False
        if any(w in name.lower() for w in ['credentials', 'secret', 'token']) and name != '.env.example':
            safe = False
        if safe:
            try:
                raw = p.read_bytes()
                content = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
                rec['sha256'] = hashlib.sha256(raw).hexdigest()
                rec['read'] = True
                corpus[rel] = content
                if p.suffix == '.py':
                    tree = ast.parse(content, filename=rel)
                    imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
                    items = []
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            item = {'name': node.name, 'kind': type(node).__name__, 'line': node.lineno, 'end': node.end_lineno, 'doc': ast.get_docstring(node) or ''}
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                item.update(args=ast.unparse(node.args), returns=ast.unparse(node.returns) if node.returns else '', calls=sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}), exceptions=sorted({ast.unparse(n.exc) for n in ast.walk(node) if isinstance(n, ast.Raise) and n.exc}))
                            items.append(item)
                    symbols[rel] = {'imports': imports, 'symbols': items, 'doc': ast.get_docstring(tree) or '', 'lines': len(content.splitlines())}
            except Exception as exc:
                errors.append({'path': rel, 'error': str(exc)})
        inventory.append(rec)

summary = {'files': len(inventory), 'text_read': len(corpus), 'python_read': sum(k.endswith('.py') for k in corpus), 'active_python': sum(k.endswith('.py') and not k.startswith(('runs/', 'results/')) for k in corpus), 'active_scripts': sum(k.startswith('scripts/') and k.endswith('.py') for k in corpus), 'test_modules': sum(k.startswith('tests/test_') and k.endswith('.py') for k in corpus), 'configs': sum(k.endswith('.yaml') and k.startswith(('configs/', 'experiments/')) for k in corpus), 'extensions': dict(Counter(Path(r['path']).suffix for r in inventory)), 'errors': errors}
for name, value in [('inventory', inventory), ('corpus', corpus), ('symbols', symbols), ('scope', summary)]:
    (OUT / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('PYTHON BY ROOT', dict(Counter(k.split('/')[0] for k in symbols)))
print('TEXT BY ROOT', dict(Counter(k.split('/')[0] for k in corpus)))
print('SNAPSHOTS', '\n'.join(k for k in corpus if k.endswith('source_checksums.json')))
