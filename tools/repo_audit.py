#!/usr/bin/env python3
# Copyright 2025 xAI
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import datetime
import os

def get_all_commits(owner, repo):
    base_url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=100"
    url = base_url
    commits = []
    while url:
        response = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"})
        data = response.json()
        commits.extend(data)
        url = None
        if 'link' in response.headers:
            links = response.headers['link']
            for link in links.split(','):
                link = link.strip()
                if 'rel="next"' in link:
                    url = link.split(';')[0].strip('<> ')
                    break
    return commits

def format_commit_history(commits):
    history_entries = []
    for commit in commits:
        date_str = commit['commit']['author']['date']
        author = commit['commit']['author']['name']
        message = commit['commit']['message'].strip()
        history_entries.append((date_str, f"{date_str} - {author}: {message}"))
    history_entries.sort(key=lambda x: datetime.datetime.fromisoformat(x[0]))
    return "\n".join(entry[1] for entry in history_entries)

def get_repo_tree(owner, repo, branch):
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    response = requests.get(tree_url, headers={"Accept": "application/vnd.github.v3+json"})
    return response.json()['tree']

def fetch_file_contents(owner, repo, branch, tree):
    contents = []
    for item in tree:
        if item['type'] == 'blob':
            path = item['path']
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
            response = requests.get(raw_url)
            if response.status_code == 200:
                content = response.text
                contents.append(f"----- {path} -----\n{content}\n----- END {path} -----\n")
    contents.sort(key=lambda x: x.split('\n')[0])
    return "\n".join(contents)

def main(owner, repo, branch="main", output_file="repo_snapshot.txt"):
    commits = get_all_commits(owner, repo)
    history = format_commit_history(commits)
    tree = get_repo_tree(owner, repo, branch)
    files_content = fetch_file_contents(owner, repo, branch, tree)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Commit History (Oldest to Newest) ===\n")
        f.write(history + "\n\n")
        f.write("=== File Contents ===\n")
        f.write(files_content)
    print(f"Successfully compiled repo data into {output_file}")

if __name__ == "__main__":
    main(owner="yourusername", repo="yourrepo")  # Replace with your repo
