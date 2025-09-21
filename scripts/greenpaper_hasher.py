import hashlib
with open('greenpaper.py', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
print(f"Updated SHA-256 Hash: {file_hash}")
