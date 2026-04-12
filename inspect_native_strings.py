import json
from pathlib import Path

def main():
    path = r"C:\Users\troyh\.claude\projects\C--Users-troyh-Documents-dev-agent-migrator\4ae71411-a89d-4bf7-977b-dbcae83d9250.jsonl"
    results = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("type") == "user":
                    content = rec.get("message", {}).get("content")
                    if isinstance(content, str):
                        results.append(content)
                        if len(results) >= 10:
                            print(json.dumps(results, indent=2))
                            return
            except Exception:
                continue

if __name__ == "__main__":
    main()
