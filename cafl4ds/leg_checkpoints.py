"""Leg-boundary checkpoints — save the full method at the end of every leg of a drive (P1.1.0).

The deploy corpus logs label-free and frozen-probe readings every step, but some reads are too costly
to run inline — P1.3's fine-tune probe chief among them (Phase 0 read MAE forgetting only through
frozen probes, which can understate it). :class:`CheckpointingStream` wraps a stream so the method is
saved (encoder *and* head, via :func:`cafl4ds.warmup.save_well`) at a fixed set of steps, for any such
read to run post hoc.

It wraps the stream rather than the loop, so the Phase-0 :class:`~cafl4ds.loop.StreamingLoop` runs
untouched: the loop trains on a batch only after the stream yields it, so saving *just before*
yielding the batch at step ``s`` captures the state after every step ``< s`` — the end of the previous
leg when ``s`` opens a new one. A final checkpoint is saved once the stream is exhausted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from cafl4ds.data.streams import StreamBatch
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.warmup import save_well


class CheckpointingStream:
    """A stream wrapper that saves the full method before the batches at ``steps``, and at the end."""

    def __init__(
        self, stream: Iterable[StreamBatch], method: SSLMethod, steps: Iterable[int], out_dir: str | Path
    ) -> None:
        """Wrap ``stream``.

        Args:
            stream: The stream the loop iterates (its length, if any, is passed through).
            method: The method the loop trains (the object saved — not a copy).
            steps: The steps to checkpoint *before* (e.g. the walk's leg-boundary steps).
            out_dir: Directory the ``step<NNNNN>.pt`` / ``final.pt`` checkpoints and their
                ``index.json`` land in.
        """
        self.stream = stream
        self.method = method
        self.steps = sorted({int(s) for s in steps})
        self.out_dir = Path(out_dir)
        self.saved: list[dict[str, int | str]] = []

    def __len__(self) -> int:
        """The wrapped stream's batch count."""
        return len(self.stream)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[StreamBatch]:
        """Yield the wrapped stream's batches, checkpointing before each configured step and at the end."""
        pending = set(self.steps)
        last_step = -1
        for batch in self.stream:
            if batch.step in pending:
                self._save(f"step{batch.step:05d}.pt", completed_steps=batch.step)
            last_step = batch.step
            yield batch
        self._save("final.pt", completed_steps=last_step + 1)
        (self.out_dir / "index.json").write_text(json.dumps(self.saved, indent=2), encoding="utf-8")

    def _save(self, filename: str, *, completed_steps: int) -> None:
        """Save the method and record which checkpoint covers how many completed steps."""
        save_well(self.method, self.out_dir / filename)
        self.saved.append({"file": filename, "completed_steps": completed_steps})
