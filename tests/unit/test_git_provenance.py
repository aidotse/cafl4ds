"""The deploy harness's git provenance: the checkout's HEAD commit, read without a subprocess (P1.1 audit G11)."""

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
