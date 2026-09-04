"""A time series that stays small without losing its peaks.

A run can last hours. Sampling once a second and keeping every point makes the
memory cost a function of how long someone left the simulator open, and plotting
tens of thousands of points into a chart a few hundred pixels wide throws almost
all of them away anyway -- but it throws them away *arbitrarily*, so a one-second
stall lands on a pixel or it does not.

This keeps a fixed number of buckets. Each bucket carries the mean of the samples
that fell in it and the smallest and largest of them, so the extremes survive
however many times the series is compressed. When the buffer fills, adjacent
buckets are merged pairwise: the running sums add, and the minima and maxima take
the narrower and wider of the pair. Halving is cheap and amortises to a constant
cost per sample.

Plotted, that is a line through the means inside a shaded band between the minima
and maxima -- a *min/max envelope*. It is the same idea as M4 aggregation in the
time-series literature, and the same idea as the peak envelope an audio editor
draws for a waveform it cannot fit on screen: the shape of the signal is the
mean, and the fact that something spiked is the band.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """One bucket, ready to plot."""

    x: float          #: midpoint of the bucket, in the series' x units
    x_start: float
    x_end: float
    mean: float
    minimum: float
    maximum: float
    count: int        #: raw samples behind this bucket

    @property
    def collapsed(self) -> bool:
        """Whether this bucket is a single sample, so band and line coincide."""
        return self.count == 1


class EnvelopeSeries:
    """Fixed-memory series of ``(x, y)`` keeping mean, min and max per bucket.

    ``capacity`` is the most buckets held; the series halves when it reaches
    that, so it uses bounded memory for a run of any length while never losing
    a peak.
    """

    def __init__(self, capacity: int = 2000) -> None:
        if capacity < 2 or capacity % 2:
            raise ValueError("capacity must be an even number of at least 2")
        self.capacity = int(capacity)
        self.samples = 0          #: raw samples added, before any compression
        self.compressions = 0     #: how many times the series has halved
        # Parallel arrays rather than objects: a bucket is merged far more
        # often than it is read, and this keeps the merge a plain arithmetic
        # pass with no allocation per bucket.
        self._start: list[float] = []
        self._end: list[float] = []
        self._total: list[float] = []
        self._count: list[int] = []
        self._min: list[float] = []
        self._max: list[float] = []

    def __len__(self) -> int:
        return len(self._count)

    def add(self, x: float, y: float) -> None:
        x, y = float(x), float(y)
        self._start.append(x)
        self._end.append(x)
        self._total.append(y)
        self._count.append(1)
        self._min.append(y)
        self._max.append(y)
        self.samples += 1
        if len(self._count) >= self.capacity:
            self._compress()

    def _compress(self) -> None:
        """Merge adjacent buckets pairwise, halving the series.

        Sums and counts are carried rather than means, so a mean taken after
        any number of compressions is the true mean of every raw sample behind
        it -- averaging means of unequal buckets would drift.
        """
        pairs = len(self._count) // 2
        for i in range(pairs):
            a, b = i * 2, i * 2 + 1
            self._start[i] = self._start[a]
            self._end[i] = self._end[b]
            self._total[i] = self._total[a] + self._total[b]
            self._count[i] = self._count[a] + self._count[b]
            self._min[i] = min(self._min[a], self._min[b])
            self._max[i] = max(self._max[a], self._max[b])
        # An odd tail has no partner; carry it forward unmerged.
        odd = len(self._count) % 2
        if odd:
            last = len(self._count) - 1
            self._start[pairs] = self._start[last]
            self._end[pairs] = self._end[last]
            self._total[pairs] = self._total[last]
            self._count[pairs] = self._count[last]
            self._min[pairs] = self._min[last]
            self._max[pairs] = self._max[last]
        kept = pairs + odd
        for column in (self._start, self._end, self._total, self._count,
                       self._min, self._max):
            del column[kept:]
        self.compressions += 1

    def spans(self) -> list[Span]:
        return [Span(x=(self._start[i] + self._end[i]) / 2.0,
                     x_start=self._start[i], x_end=self._end[i],
                     mean=self._total[i] / self._count[i],
                     minimum=self._min[i], maximum=self._max[i],
                     count=self._count[i])
                for i in range(len(self._count))]

    def plot_arrays(self) -> tuple[list[float], list[float], list[float], list[float]]:
        """``(x, mean, minimum, maximum)``, ready for a line and a filled band."""
        spans = self.spans()
        return ([s.x for s in spans], [s.mean for s in spans],
                [s.minimum for s in spans], [s.maximum for s in spans])

    def latest(self) -> float | None:
        """The most recent bucket's mean, or ``None`` before anything is added."""
        if not self._count:
            return None
        return self._total[-1] / self._count[-1]

    def extremes(self) -> tuple[float, float] | None:
        """The smallest and largest sample ever added, across all compressions."""
        if not self._count:
            return None
        return min(self._min), max(self._max)
