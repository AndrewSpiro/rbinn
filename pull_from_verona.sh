git fetch verona
git merge --squash -s subtree verona/main --allow-unrelated-histories
git commit -m "Update VERONA subtree to latest remote version"