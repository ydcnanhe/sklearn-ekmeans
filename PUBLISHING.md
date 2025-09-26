# Publishing sklekmeans to PyPI

This project is configured for PyPI Trusted Publishing via GitHub Actions.

## One-time setup

1. Create a new release in GitHub with a tag matching the version in `pyproject.toml` (e.g. `v0.1.0`).
2. In the PyPI project `sklekmeans`, add GitHub as a trusted publisher:
   - Go to PyPI > Your projects > sklekmeans > Manage > Collaborators and maintainers > Add a new publisher.
   - Select `Owner: ydcnanhe`, `Repository: sklearn-ekmeans`, `Workflow: publish.yml`.
   - Save.

No PyPI token is needed; authentication uses OIDC.

## Release steps

1. Update `pyproject.toml` version.
2. Update `CHANGELOG.md` (optional) and `README.md` if needed.
3. Commit and push to main (or a release branch) and create a GitHub Release with the same tag (e.g., `v0.1.1`).
4. The `Publish to PyPI` workflow will build and upload `sdist` and `wheel` to PyPI.

## Local test build (optional)

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Troubleshooting

- Ensure the tag version matches `[project].version` in `pyproject.toml`.
- Verify Trusted Publishing is configured on PyPI for this repo and workflow.
- Check the GitHub Actions logs (`Publish to PyPI`) for build or metadata errors.
