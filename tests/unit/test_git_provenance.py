# ruff: noqa: S603, S607
"""The deploy harness's git provenance: the checkout's HEAD commit and its dirty state (P1.1 audit G11)."""

import shutil
import subprocess
from pathlib import Path

import pytest

import scripts.run_deploy_harness as harness


def _git(tmp_path: Path, head: str) -> Path:
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head + "\n", encoding="utf-8")
    return git


def test_head_commit_reads_a_loose_branch_ref(tmp_path: Path) -> None:
    """HEAD -> refs/heads/<branch>, with a slash in the branch name."""
    git = _git(tmp_path, "ref: refs/heads/feat/x")
    (git / "refs" / "heads" / "feat").mkdir()
    (git / "refs" / "heads" / "feat" / "x").write_text("abc123\n", encoding="utf-8")
    assert harness._head_commit(git) == "abc123"


def test_head_commit_reads_a_packed_ref_and_a_detached_head(tmp_path: Path) -> None:
    """A ref only in packed-refs (comments and peeled lines skipped), then a detached HEAD."""
    git = _git(tmp_path, "ref: refs/heads/main")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled\nfff000 refs/heads/other\ndef456 refs/heads/main\n^999 \n", encoding="utf-8"
    )
    assert harness._head_commit(git) == "def456"
    (git / "HEAD").write_text("0123abcd\n", encoding="utf-8")
    assert harness._head_commit(git) == "0123abcd"


def test_git_sha_prefers_the_checkout_over_a_stale_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkout's HEAD wins over the install-time version; the version is the fallback; else ``None``."""
    monkeypatch.setattr(harness, "version", lambda _: "0.1.0+g1a2b3c")  # an editable install's frozen version
    git = _git(tmp_path, "ref: refs/heads/main")
    (git / "refs" / "heads" / "main").write_text("feed42\n", encoding="utf-8")
    assert harness._git_sha(tmp_path) == "feed42"
    assert harness._git_sha(tmp_path / "nowhere") == "g1a2b3c"
    monkeypatch.setattr(harness, "version", lambda _: "0.1.0")  # the container's image install
    assert harness._git_sha(tmp_path / "nowhere") is None


def _repo(tmp_path: Path) -> Path:
    """A real one-commit git checkout with one tracked file."""
    tmp_path.mkdir(parents=True)

    def run(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    run("init", "-q")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    run("add", "a.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return tmp_path


@pytest.mark.skipif(shutil.which("git") is None, reason="needs the git binary")
def test_git_changes_reads_a_clean_tree_then_edits_and_untracked_files(tmp_path: Path) -> None:
    """Clean is ``[]``; an edit and a new file are both listed; outside a checkout git cannot tell (``None``)."""
    repo = _repo(tmp_path / "repo")
    assert harness._git_changes(repo) == []
    (repo / "a.txt").write_text("b\n", encoding="utf-8")
    (repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert sorted(harness._git_changes(repo) or []) == ["a.txt", "new.py"]
    (tmp_path / "plain").mkdir()
    assert harness._git_changes(tmp_path / "plain") is None


@pytest.mark.skipif(shutil.which("git") is None, reason="needs the git binary")
def test_provenance_records_the_commit_and_the_dirty_state(tmp_path: Path) -> None:
    """The manifest fields: the HEAD commit, the dirty flag, and the changed paths (capped)."""
    repo = _repo(tmp_path / "repo")
    clean = harness._provenance(repo)
    assert clean == {"git_sha": harness._head_commit(repo / ".git"), "git_dirty": False, "git_dirty_paths": []}
    for name in ("x.py", "y.py", "z.py"):
        (repo / name).write_text("\n", encoding="utf-8")
    dirty = harness._provenance(repo, max_paths=2)
    assert dirty["git_dirty"] is True and len(dirty["git_dirty_paths"]) == 2


def test_provenance_fails_without_a_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No checkout and no commit in the version: the run refuses to start."""
    monkeypatch.setattr(harness, "version", lambda _: "0.1.0")
    with pytest.raises(RuntimeError, match="no git provenance"):
        harness._provenance(tmp_path / "nowhere")
