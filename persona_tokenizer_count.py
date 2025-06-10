import json, sys, pathlib, requests
txt = pathlib.Path("persona.txt").read_text()
r = requests.post("http://localhost:5000/tokenize",
                  json={"content": txt})
print("Token count:", len(r.json()["tokens"]))