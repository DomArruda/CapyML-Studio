"""CapyML Studio verification harness.

Checks the single-file app end to end, without a browser:
  1. py_compile the embedded Python engine, then import it and run
     real scikit-learn/scipy workloads (models, tests, graphs, pipeline).
  2. node --check every embedded <script> block.
  3. Cross-check every referenced DOM id exists in the HTML.

Usage:  python tools/verify.py
Exit code 0 = all checks passed.

Requires (dev machine / CI only): python with scikit-learn, scipy, pandas,
numpy; node on PATH. The app itself has no dependencies beyond a browser.
"""
import re, json, sys, subprocess, tempfile, pathlib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = ROOT / 'index.html'

def die(msg):
    print('FAILED: %s' % msg)
    sys.exit(1)

html = HTML.read_text(encoding='utf-8')

# 1. Python engine ------------------------------------------------------------
py = re.search(r'<script type="text/python" id="pysrc">(.*?)</script>', html, re.S)
if not py:
    die('pysrc block not found')
py = py.group(1)

with tempfile.TemporaryDirectory() as td:
    tdp = pathlib.Path(td)
    (tdp / 'pysrc.py').write_text(py, encoding='utf-8')
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(tdp / 'pysrc.py')],
                       capture_output=True, text=True)
    if r.returncode != 0:
        die('py_compile:\n' + r.stderr[:2000])

ns = {}
exec(compile(py, 'pysrc.py', 'exec'), ns)
sample, run, run_flow, set_df = ns['sample'], ns['run'], ns['run_flow'], ns['set_df']

def ok(r, tag):
    assert r.get('ok'), (tag, r)
    return r

def J(spec):
    return json.loads(run(json.dumps(spec)))

ok(json.loads(sample('churn')), 'sample churn')
ok(json.loads(sample('housing')), 'sample housing')

r = ok(J({'task': 'model', 'model': 'ols', 'target': 'price', 'features': ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'garage', 'lot_size', 'neighborhood'], 'test_size': 0.2, 'seed': 42, 'scale': False}), 'ols')
assert r['r2'] > 0.5
r = ok(J({'task': 'model', 'model': 'ridge', 'target': 'price', 'features': None, 'alpha': 1.0, 'scale': True}), 'ridge')
assert r['scaled'] is True
r = ok(J({'task': 'model', 'model': 'enet', 'target': 'price', 'features': ['sqft', 'neighborhood'], 'alpha': 0.5, 'l1_ratio': 0.5, 'scale': True}), 'enet')
r = ok(J({'task': 'model', 'model': 'poly', 'target': 'price', 'features': ['sqft'], 'degree': 2, 'scale': False}), 'poly')
assert r['curve'] and len(r['curve']['x']) == 60
r = ok(J({'task': 'model', 'model': 'gbt', 'target': 'price', 'features': None, 'n_estimators': 30, 'learning_rate': 0.1}), 'gbt reg')
assert r['r2'] > 0.5 and r['importances']
print('regression models OK')

json.loads(sample('churn'))
r = ok(J({'task': 'model', 'model': 'logreg', 'target': 'churn', 'features': None, 'C': 1.0, 'scale': True}), 'logreg')
assert 0.5 < r['accuracy'] <= 1 and r['auc'] is not None and r['coefs']
r = ok(J({'task': 'model', 'model': 'dt', 'target': 'churn', 'features': None, 'max_depth': 4}), 'dt')
assert r['rules'] and r['importances']
r = ok(J({'task': 'model', 'model': 'gbt', 'target': 'churn', 'features': None, 'n_estimators': 30, 'learning_rate': 0.2}), 'gbt clf')
assert r['cm'] and len(r['labels']) == 2
r = ok(J({'task': 'model', 'model': 'gbt', 'target': 'churn', 'features': None, 'n_estimators': 20, 'learning_rate': 0.2}), 'gbt clf tree')
assert r['gbt_tree'] and 'Tree 1 of 20' in r['gbt_tree']
print('classification models OK')

json.loads(sample('housing'))
r = ok(J({'task': 'corr', 'cols': None, 'method': 'pearson'}), 'corr')
assert 'price' in ''.join(r['names']) and r['pairs'][0][2] > 0.5
print('correlation OK')

for kind, spec in [
    ('bar', {'x': 'neighborhood', 'y': 'price', 'agg': 'mean'}),
    ('hist', {'x': 'price', 'bins': 20}),
    ('scatter', {'x': 'sqft', 'y': 'price', 'group': 'neighborhood'}),
    ('box', {'x': 'neighborhood', 'y': 'price'}),
]:
    r = ok(J(dict({'task': 'graph', 'kind': kind}, **spec)), kind)
assert len(json.loads(run(json.dumps({'task': 'graph', 'kind': 'scatter', 'x': 'sqft', 'y': 'price', 'group': 'neighborhood'})))['series']) == 3
print('graphs OK')

json.loads(sample('abtest'))
for t, spec in [
    ('ttest1', {'col': 'spend', 'mu': 60}),
    ('ttest2', {'col': 'spend', 'grp': 'group'}),
    ('ttestp', {'col': 'spend', 'colB': 'age'}),
    ('anova', {'col': 'spend', 'grp': 'device'}),
    ('chi2', {'col': 'group', 'colB': 'converted'}),
    ('mwu', {'col': 'spend', 'grp': 'group'}),
    ('kw', {'col': 'spend', 'grp': 'device'}),
]:
    r = ok(J({'task': 'test', 'test': t, **spec}), t)
assert r['p'] is not None
print('hypothesis tests OK')

json.loads(sample('housing'))
cv = ok(json.loads(ns['col_values']('sqft', 100, 42)), 'col_values')
assert len(cv['rows']) == 100
info = json.loads(ns['add_sentiment'](json.dumps([[cv['rows'][i][0], 'positive' if i % 2 else 'negative', 0.9] for i in range(50)])))
assert info['ok'] and [c for c in info['cols'] if c['name'] == 'sentiment_score'][0]['type'] == 'num'
r = ok(json.loads(ns['coerce']('sqft', 'str')), 'coerce str')
assert [c for c in r['cols'] if c['name'] == 'sqft'][0]['type'] == 'cat'
assert not json.loads(ns['coerce']('bathrooms', 'int'))['ok']
assert ns['df_csv']().strip()
assert len(json.loads(ns['df_json']())['records']) > 0
r = ok(json.loads(sample('reviews')), 'reviews')
assert r['rows'] == 160
print('data utils OK (col_values, sentiment merge, coerce, csv/json)')

flow = json.dumps({'nodes': [
    {'id': 'n1', 'type': 'input', 'params': {'source': 'abtest'}},
    {'id': 'n2', 'type': 'filter', 'params': {'col': 'group', 'op': '==', 'value': 'A'}},
    {'id': 'n3', 'type': 'transform', 'params': {'calc_name': 'spend_x2', 'calc_expr': 'spend*2', 'dropna_cols': [], 'onehot_cols': [], 'scale_cols': []}},
    {'id': 'n4', 'type': 'split', 'params': {'test_size': 0.25, 'seed': 42, 'stratify': True, 'strat_col': 'converted'}},
    {'id': 'n5', 'type': 'model', 'params': {'algo': 'logreg', 'target': 'converted', 'C': 1}},
    {'id': 'n6', 'type': 'evaluate', 'params': {}},
], 'edges': [
    {'from': 'n1', 'fromPort': 'out', 'to': 'n2', 'port': 'in'},
    {'from': 'n2', 'fromPort': 'out', 'to': 'n3', 'port': 'in'},
    {'from': 'n3', 'fromPort': 'out', 'to': 'n4', 'port': 'in'},
    {'from': 'n4', 'fromPort': 'train', 'to': 'n5', 'port': 'train'},
    {'from': 'n4', 'fromPort': 'test', 'to': 'n5', 'port': 'test'},
    {'from': 'n5', 'fromPort': 'scored', 'to': 'n6', 'port': 'in'},
]})
fr = ok(json.loads(run_flow(flow)), 'run_flow')
st = fr['nodes']
assert st['n1']['status'] == 'ok' and st['n1']['report']['ports']['out']['rows'] == 600
assert st['n3']['status'] == 'ok' and 'spend_x2' in st['n3']['report']['ports']['out']['cols']
assert st['n4']['status'] == 'ok'
assert st['n5']['status'] == 'ok' and st['n5']['report']['extra']['cm']
assert st['n6']['status'] == 'ok'
cyc = json.loads(run_flow(json.dumps({'nodes': [{'id': 'a', 'type': 'filter', 'params': {}}, {'id': 'b', 'type': 'filter', 'params': {}}], 'edges': [{'from': 'a', 'fromPort': 'out', 'to': 'b', 'port': 'in'}, {'from': 'b', 'fromPort': 'out', 'to': 'a', 'port': 'in'}]})))
assert not cyc['ok'] and 'Cycle' in cyc['error']
print('pipeline engine OK (+ cycle detection)')

# 2. JS syntax ---------------------------------------------------------------
blocks = re.findall(r'<script>(.*?)</script>', html, re.S)
with tempfile.TemporaryDirectory() as td:
    tdp = pathlib.Path(td)
    for i, b in enumerate(blocks):
        p = tdp / ('block%d.js' % i)
        p.write_text(b, encoding='utf-8')
        r = subprocess.run(['node', '--check', str(p)], capture_output=True, text=True)
        if r.returncode != 0:
            die('JS block %d:\n%s' % (i, r.stderr[:2000]))
print('JS syntax OK (%d blocks)' % len(blocks))

# 3. DOM id cross-check ------------------------------------------------------
js_all = '\n'.join(blocks)
refs = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", js_all)) | set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", js_all))
ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
missing = sorted(x for x in refs if x not in ids)
if missing:
    die('DOM ids referenced but not present: %s' % missing)
print('DOM id cross-check OK (%d ids)' % len(refs))

# 4. Structure ---------------------------------------------------------------
tabs = set(re.findall(r'data-ws="(\w+)"', html))
labs = set(re.findall(r'data-lab="(\w+)"', html))
assert tabs == {'data', 'viz', 'lab', 'flow'}, tabs
assert labs == {'lin', 'clf', 'nl', 'tests', 'sent'}, labs
for t in tabs:
    assert ('id="ws-%s"' % t) in html
for t in labs:
    assert ('id="lab-%s"' % t) in html
for t in ['ttest1', 'ttest2', 'ttestp', 'anova', 'chi2', 'mwu', 'kw']:
    assert ('id="t-%s"' % t) in html
print('structure cross-check OK')
print('ALL CHECKS PASSED')