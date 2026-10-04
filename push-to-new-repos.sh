#!/usr/bin/env bash
# Pushes raftkv, quarry and lens to their own GitHub repositories.
#
# Prerequisite: create three EMPTY repos (no README, no .gitignore, no licence)
# under your account first:
#
#   https://github.com/new  ->  raftkv
#   https://github.com/new  ->  quarry
#   https://github.com/new  ->  lens
#
# Then run this script. It turns each project directory into its own git
# repository with a single initial commit and pushes it.
set -euo pipefail

GH_USER="${GH_USER:-Khwahishd}"
cd "$(dirname "$0")"

for project in raftkv quarry lens; do
  echo
  echo "=== $project -> https://github.com/$GH_USER/$project"

  if [[ ! -d "$project" ]]; then
    echo "   skipping: directory not found"
    continue
  fi

  # Work on a copy so the monorepo checkout is left untouched.
  work="$(mktemp -d)/$project"
  mkdir -p "$work"
  # Copy only what git tracks, so build artifacts never reach the new repo.
  git ls-files -z -- "$project" | while IFS= read -r -d '' path; do
    rel="${path#"$project"/}"
    mkdir -p "$work/$(dirname "$rel")"
    cp "$path" "$work/$rel"
  done

  (
    cd "$work"
    git init -q -b main
    git add -A
    git -c user.email="${GIT_EMAIL:-khwahish.daudani@zii.aero}" \
        -c user.name="${GIT_NAME:-Khwahish Daudani}" \
        commit -q -m "$(head -1 "$OLDPWD/$project/README.md" | sed 's/^# //')

See README.md for the design, the measurements and the known limitations."
    git remote add origin "https://github.com/$GH_USER/$project.git"
    git push -u origin main
  )
  echo "   pushed $(cd "$work" && git ls-files | wc -l) files"
done

echo
echo "done. All three repositories are live."
