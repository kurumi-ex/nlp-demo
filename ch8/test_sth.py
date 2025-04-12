from pathlib import Path

file_path = Path(__file__).resolve()
root_path = file_path.parent

print(type(root_path))