from pathlib import Path


def test_workspace_is_gitignored():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    content = gitignore.read_text()
    assert "/workspace/" in content
