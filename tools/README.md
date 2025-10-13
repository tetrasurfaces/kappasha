Kappasha Tools
Simple utilities for copyright snapshots and hashing.
Run a copyright snapshot of your repo:

Export GITHUB_TOKEN=your_token_here
git clone github.com/tetrasurfaces/Kappasha
cd Kappasha/tools
python3 repo_audit.py yourusername yourrepo

Get a kappa hash for a file:python3 kappa_hash.py path/to/myfile
No extension for snapshot, no commit message needed. That's it.
