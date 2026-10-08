"""Bounded, callback-safe playback of resident Piper speech."""

import math
import re
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import pedalboard
import sounddevice as sd


class PlaybackControl:
    """One-shot control handle for a single play_text call."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ring = None
        self._stream = None
        self._bound = False
        self._ended = False
        self.paused = False
        self.stopped = False

    def _bind(self, ring):
        with self._lock:
            if self._bound:
                raise RuntimeError("playback control is already in use")
            self._bound = True
            self._ring = ring
            if self.stopped:
                ring.cancel()
            elif self.paused:
                ring.set_paused(True)

    def _bind_stream(self, stream):
        with self._lock:
            self._stream = stream
            stopped = self.stopped
        if stopped:
            self._abort(stream)

    @staticmethod
    def _abort(stream):
        try:
            stream.abort()
        except (AttributeError, sd.PortAudioError):
            # The callback also observes stopped and aborts on its next call.
            pass

    def pause(self):
        with self._lock:
            if not self.stopped and not self._ended:
                self.paused = True
                if self._ring is not None:
                    self._ring.set_paused(True)

    def resume(self):
        with self._lock:
            self.paused = False
            if self._ring is not None:
                self._ring.set_paused(False)

    def stop(self):
        """Silence output and cancel buffering without waiting for inference."""
        with self._lock:
            if self._ended:
                return
            self.stopped = True
            self.paused = False
            ring, stream = self._ring, self._stream
        if ring is not None:
            ring.cancel()
        if stream is not None:
            self._abort(stream)

    def _end(self):
        with self._lock:
            self._ended = True
            self._stream = None
            self._ring = None


def text_segments(text: str, limit: int = 300):
    """Split near word boundaries without losing source characters."""
    if limit < 1:
        raise ValueError("segment limit must be positive")
    current = ""
    for match in re.finditer(r"\s+|\S+", text):
        token = match.group()
        if current and not token.isspace() and len(token) <= limit and len(current) + len(token) > limit:
            yield current
            current = ""
        while token:
            room = limit - len(current)
            if room == 0:
                yield current
                current = ""
                room = limit
            part, token = token[:room], token[room:]
            current += part
    if current:
        yield current


class AudioRing:
    """Preallocated mono float32 ring; callbacks never wait for the producer."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("ring capacity must be positive")
        self.data = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.read_at = self.write_at = self.size = self.peak = 0
        self.lock = threading.Lock()
        self.changed = threading.Event()
        self.eof = False
        self.error: Exception | None = None
        self.cancelled = False
        self.paused = False
        self.starvations = 0
        self.output_underflows = 0
        self.played_frames = 0
        self.first_submit_time: float | None = None
        self.first_dac_time: float | None = None

    def put(self, samples: np.ndarray):
        offset = 0
        while offset < len(samples):
            with self.lock:
                if self.cancelled:
                    return
                free = self.capacity - self.size
                count = min(free, len(samples) - offset)
                if count:
                    first = min(count, self.capacity - self.write_at)
                    self.data[self.write_at:self.write_at + first] = samples[offset:offset + first]
                    self.data[:count - first] = samples[offset + first:offset + count]
                    self.write_at = (self.write_at + count) % self.capacity
                    self.size += count
                    self.peak = max(self.peak, self.size)
                    offset += count
                    self.changed.set()
            if not count:
                self.changed.wait(0.02)
                self.changed.clear()

    def take(self, output: np.ndarray, status=None, dac_delay: float | None = None):
        """Fill the entire callback buffer; return True after natural EOF."""
        output.fill(0)
        if status is not None and getattr(status, "output_underflow", False):
            self.output_underflows += 1
        if not self.lock.acquire(blocking=False):
            self.starvations += 1
            return False
        try:
            if self.cancelled:
                return True
            if self.paused:
                return False
            count = min(len(output), self.size)
            if count:
                first = min(count, self.capacity - self.read_at)
                output[:first] = self.data[self.read_at:self.read_at + first]
                output[first:count] = self.data[:count - first]
                self.read_at = (self.read_at + count) % self.capacity
                self.size -= count
                self.played_frames += count
                if self.first_submit_time is None:
                    self.first_submit_time = time.perf_counter()
                    self.first_dac_time = self.first_submit_time + dac_delay if dac_delay is not None else None
            if count < len(output) and not self.eof:
                self.starvations += 1
            return self.eof and self.size == 0
        finally:
            self.lock.release()

    def finish(self, error: Exception | None = None):
        with self.lock:
            if self.cancelled:
                return
            self.error = error
            self.eof = True
            self.changed.set()

    def set_paused(self, paused: bool):
        with self.lock:
            self.paused = paused

    def cancel(self):
        with self.lock:
            self.cancelled = True
            self.size = 0
            self.changed.set()


@dataclass
class PlaybackMetrics:
    """Callback submission counts; Stop may discard device-buffered frames."""

    sample_rate: int
    speed: float
    volume: float
    normal_frames: int = 0
    stretched_frames: int = 0
    stretch_seconds: float = 0.0
    first_submitted_seconds: float | None = None
    first_dac_estimate_seconds: float | None = None
    peak_ring_frames: int = 0
    ring_capacity_frames: int = 0
    software_starvations: int = 0
    portaudio_output_underflows: int = 0
    played_frames: int = 0
    stopped: bool = False

    def to_dict(self):
        result = asdict(self)
        result["normal_duration_seconds"] = self.normal_frames / self.sample_rate
        result["stretched_duration_seconds"] = self.stretched_frames / self.sample_rate
        result["played_duration_seconds"] = self.played_frames / self.sample_rate
        return result


def play_text(voice, text: str, *, speed: float = 1.5, volume: float = 1.0,
              stream_factory=sd.OutputStream, control: PlaybackControl | None = None) -> PlaybackMetrics:
    """Synthesize on a worker, stretch there, and drain a fixed-size ring."""
    if not math.isfinite(speed) or not 1 <= speed <= 2:
        raise ValueError("speed must be finite and between 1 and 2")
    if not math.isfinite(volume) or not 0 <= volume <= 1:
        raise ValueError("volume must be finite and between 0 and 1")
    if not text.strip():
        raise ValueError("text must contain speech")
    rate = int(voice.voice.config.sample_rate)
    control = control if control is not None else PlaybackControl()
    ring = AudioRing(rate * 2)
    control._bind(ring)
    metrics = PlaybackMetrics(rate, speed, volume, ring_capacity_frames=ring.capacity)
    started = time.perf_counter()

    def produce():
        try:
            for segment in text_segments(text):
                if control.stopped:
                    return
                for chunk in voice.synthesize(segment):
                    if control.stopped:
                        return
                    if chunk.sample_rate != rate:
                        raise RuntimeError("voice sample rate changed during playback")
                    normal = np.asarray(chunk.audio_float_array, dtype=np.float32)
                    metrics.normal_frames += len(normal)
                    stretch_start = time.perf_counter()
                    processed = (pedalboard.time_stretch(normal, rate, stretch_factor=speed,
                                 pitch_shift_in_semitones=0.0) if speed != 1 else normal)
                    metrics.stretch_seconds += time.perf_counter() - stretch_start
                    processed = np.asarray(processed, dtype=np.float32).reshape(-1)
                    if not np.isfinite(processed).all():
                        raise RuntimeError("speed processing produced non-finite audio")
                    metrics.stretched_frames += len(processed)
                    ring.put(processed * volume)
                    if ring.cancelled:
                        return
        except Exception as exc:
            ring.finish(exc)
            return
        ring.finish()

    if control.stopped:
        metrics.stopped = True
        control._end()
        return metrics
    worker = threading.Thread(target=produce, name="clipspeak-synthesis", daemon=True)
    try:
        worker.start()
    except BaseException:
        control._end()
        raise
    stream = None
    try:
        while True:
            with ring.lock:
                if ring.size or ring.eof or control.stopped:
                    break
            if not worker.is_alive():
                raise RuntimeError("speech worker exited unexpectedly")
            time.sleep(0.02)
        with ring.lock:
            if ring.error and not control.stopped:
                raise RuntimeError(f"speech processing failed ({type(ring.error).__name__})") from None

        if control.stopped:
            metrics.stopped = True
            return metrics

        def callback(outdata, frames, time_info, status):
            if control.stopped:
                outdata.fill(0)
                raise sd.CallbackAbort
            dac = getattr(time_info, "outputBufferDacTime", None)
            current = getattr(time_info, "currentTime", None)
            delay = max(0.0, dac - current) if dac is not None and current is not None else None
            if ring.take(outdata[:, 0], status, delay):
                raise sd.CallbackStop

        stream = stream_factory(samplerate=rate, channels=1, dtype="float32", callback=callback)
        with stream:
            control._bind_stream(stream)
            while stream.active:
                if control.stopped:
                    control._abort(stream)
                    break
                if ring.error:
                    raise RuntimeError(f"speech processing failed ({type(ring.error).__name__})") from None
                if not worker.is_alive() and not ring.eof:
                    raise RuntimeError("speech worker exited unexpectedly")
                time.sleep(0.02)
        if ring.error and not control.stopped:
            raise RuntimeError(f"speech processing failed ({type(ring.error).__name__})") from None
        if (ring.size or not ring.eof) and not control.stopped:
            raise RuntimeError("audio stream stopped before speech finished")
        metrics.stopped = control.stopped
        metrics.first_dac_estimate_seconds = ring.first_dac_time - started if ring.first_dac_time else None
        metrics.first_submitted_seconds = ring.first_submit_time - started if ring.first_submit_time else None
        metrics.peak_ring_frames = ring.peak
        metrics.software_starvations = ring.starvations
        metrics.portaudio_output_underflows = ring.output_underflows
        metrics.played_frames = ring.played_frames
        return metrics
    finally:
        ring.cancel()
        control._end()
        worker.join()
