import json
nb = json.load(open('solution.ipynb','r',encoding='utf-8'))
print(f"Notebook valid JSON: {len(nb['cells'])} cells, nbformat {nb['nbformat']}")
# Check each code cell has valid source
code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
print(f"Code cells: {len(code_cells)}")
md_cells = [c for c in nb['cells'] if c['cell_type']=='markdown']
print(f"Markdown cells: {len(md_cells)}")
print("All cells parsed OK")
