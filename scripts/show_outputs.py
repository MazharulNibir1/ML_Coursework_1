"""Print the text outputs of executed notebook cells (optionally only cells after index N)."""
import json, sys
nb = json.load(open("coursework.ipynb"))
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for i, c in enumerate(nb["cells"]):
    if i < start or c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            print(f"[{i}]", "".join(o["text"]))
        elif o.get("output_type") == "error":
            print(f"[{i}] ERROR", o["ename"], o["evalue"])
        elif "data" in o and "text/plain" in o["data"] and "image/png" not in o["data"]:
            print(f"[{i}]", "".join(o["data"]["text/plain"]))
