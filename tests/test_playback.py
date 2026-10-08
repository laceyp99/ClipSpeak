import math
import time
from types import SimpleNamespace

import numpy as np
import pedalboard
import pytest
import sounddevice as sd

from clipspeak.playback import AudioRing, PlaybackControl, play_text, text_segments


def test_segments_preserve_whitespace_and_long_tokens():
    source = "  hi.\n" + "x" * 701 + "  end! "
    segments = list(text_segments(source))
    assert "".join(segments) == source
    assert max(map(len, segments)) <= 300


def test_segments_keep_words_whole_when_they_fit_the_limit():
    assert list(text_segments("first second third", limit=12)) == ["first second", " third"]


@pytest.mark.parametrize("speed", [1.0, 1.5, 2.0])
def test_stretch_preserves_tone_pitch(speed):
    rate = 22050
    tone = np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate).astype(np.float32) * 0.3
    output = pedalboard.time_stretch(tone, rate, stretch_factor=speed,
                                     pitch_shift_in_semitones=0).reshape(-1)
    assert abs(len(output) / rate - 2 / speed) < 0.06
    middle = output[len(output) // 4:len(output) // 2]
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
    frequency = np.fft.rfftfreq(len(middle), 1 / rate)[np.argmax(spectrum)]
    assert abs(frequency - 440) < 10


def test_ring_backpressure_and_order():
    import threading
    ring = AudioRing(4)
    payload = np.arange(10, dtype=np.float32)
    producer = threading.Thread(target=ring.put, args=(payload,))
    producer.start()
    received = []
    for _ in range(5):
        output = np.empty(2, dtype=np.float32)
        deadline = time.monotonic() + 2
        while ring.size < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        ring.take(output)
        received.extend(output)
    producer.join(timeout=2)
    assert not producer.is_alive()
    assert received == list(payload)
    assert ring.peak <= ring.capacity


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.active = False
        self.output = []

    def __enter__(self):
        import threading
        self.active = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        return self

    def run(self):
        while self.active:
            output = np.empty((64, 1), dtype=np.float32)
            timing = SimpleNamespace(currentTime=1.0, outputBufferDacTime=1.02)
            try:
                self.callback(output, 64, timing, SimpleNamespace(output_underflow=False))
                self.output.extend(output[:, 0])
            except sd.CallbackStop:
                self.output.extend(output[:, 0])
                self.active = False
            except sd.CallbackAbort:
                self.active = False
            time.sleep(0.0005)

    def __exit__(self, *_):
        self.active = False
        self.thread.join(timeout=2)

    def abort(self):
        self.active = False


def test_playback_drains_and_reports_frames(monkeypatch):
    from clipspeak import playback
    monkeypatch.setattr(playback.pedalboard, "time_stretch", lambda samples, *_args, **_kwargs: samples)
    class Voice:
        voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
        def synthesize(self, segment):
            yield SimpleNamespace(audio_float_array=np.arange(150, dtype=np.float32) / 150,
                                  sample_rate=100)
    streams = []
    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream
    metrics = play_text(Voice(), "hello", stream_factory=factory)
    assert metrics.played_frames == 150
    assert metrics.stretched_frames == 150
    assert np.allclose(streams[0].output[:150], np.arange(150) / 150)
    assert metrics.peak_ring_frames <= 200
    assert metrics.first_dac_estimate_seconds is not None


def test_processing_error_propagates_without_hang():
    class Voice:
        voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
        def synthesize(self, segment):
            raise ValueError("copied private text")
            yield
    with pytest.raises(RuntimeError, match="speech processing failed \\(ValueError\\)") as error:
        play_text(Voice(), "hello", stream_factory=FakeStream)
    assert "private" not in str(error.value)


def test_volume_scales_samples_and_failed_device_releases_producer():
    class Voice:
        voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
        def synthesize(self, segment):
            yield SimpleNamespace(audio_float_array=np.full(500, 0.4, dtype=np.float32),
                                  sample_rate=100)
    streams = []
    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream
    metrics = play_text(Voice(), "hello", speed=1, volume=0.5, stream_factory=factory)
    audible = np.asarray(streams[0].output)
    assert np.allclose(audible[audible != 0], 0.2)
    assert metrics.played_frames == 500
    assert metrics.first_submitted_seconds <= metrics.first_dac_estimate_seconds

    def unavailable(**kwargs):
        raise sd.PortAudioError("output unavailable")
    started = time.monotonic()
    with pytest.raises(sd.PortAudioError, match="output unavailable"):
        play_text(Voice(), "hello", speed=1, stream_factory=unavailable)
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("speed,volume", [(math.nan, 1), (2.1, 1), (1.5, math.inf)])
def test_rejects_invalid_settings(speed, volume):
    with pytest.raises(ValueError):
        play_text(None, "hello", speed=speed, volume=volume)


def test_pause_preserves_cursor_and_bounded_buffer():
    import threading
    control = PlaybackControl()
    ring = AudioRing(4)
    control._bind(ring)
    ring.put(np.array([1, 2, 3, 4], dtype=np.float32))
    producer = threading.Thread(target=ring.put, args=(np.array([5, 6], dtype=np.float32),))
    producer.start()
    control.pause()
    for _ in range(3):
        output = np.empty(2, dtype=np.float32)
        ring.take(output)
        assert output.tolist() == [0, 0]
        assert ring.size == 4
        assert ring.peak <= ring.capacity
    assert producer.is_alive()
    control.resume()
    collected = []
    for _ in range(3):
        deadline = time.monotonic() + 2
        while ring.size < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        output = np.empty(2, dtype=np.float32)
        ring.take(output)
        collected.extend(output)
    producer.join(timeout=2)
    assert not producer.is_alive()
    assert collected == [1, 2, 3, 4, 5, 6]


def test_pause_before_playback_silences_callback_until_resume():
    import threading
    class Voice:
        voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
        def synthesize(self, segment):
            yield SimpleNamespace(audio_float_array=np.full(300, 0.4, dtype=np.float32),
                                  sample_rate=100)
    control = PlaybackControl()
    control.pause()
    streams = []
    results = []
    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream
    caller = threading.Thread(target=lambda: results.append(play_text(
        Voice(), "hello", speed=1, stream_factory=factory, control=control)))
    caller.start()
    deadline = time.monotonic() + 2
    while (not streams or len(streams[0].output) < 128) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert streams and len(streams[0].output) >= 128
    assert not any(streams[0].output)
    assert control._ring.size <= control._ring.capacity
    assert control._ring.played_frames == 0
    control.resume()
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert results[0].played_frames == 300
    assert not results[0].stopped


@pytest.mark.parametrize("paused", [False, True])
def test_stop_aborts_stream_and_clears_buffer(paused):
    import threading
    entered = threading.Event()
    class Voice:
        voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
        def synthesize(self, segment):
            yield SimpleNamespace(audio_float_array=np.full(2000, 0.4, dtype=np.float32),
                                  sample_rate=100)
    class Stream(FakeStream):
        def run(self):
            entered.set()
            super().run()
    control = PlaybackControl()
    if paused:
        control.pause()
    streams = []
    result = []
    def factory(**kwargs):
        stream = Stream(**kwargs)
        streams.append(stream)
        return stream
    caller = threading.Thread(target=lambda: result.append(play_text(
        Voice(), "hello", speed=1, stream_factory=factory, control=control)))
    caller.start()
    assert entered.wait(2)
    started = time.monotonic()
    control.stop()
    control.stop()
    control.resume()
    assert time.monotonic() - started < 0.2
    caller.join(timeout=2)
    assert not caller.is_alive()
    assert result[0].stopped
    assert not streams[0].active


def test_stop_during_inference_rejects_late_audio_and_serializes_voice_reuse():
    import threading
    inference_started = threading.Event()
    release = threading.Event()
    class Voice:
        voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
        def synthesize(self, segment):
            if segment == "first":
                inference_started.set()
                assert release.wait(3)
            yield SimpleNamespace(audio_float_array=np.full(100, 0.5, dtype=np.float32),
                                  sample_rate=100)
    voice = Voice()
    control = PlaybackControl()
    first = []
    first_caller = threading.Thread(target=lambda: first.append(play_text(
        voice, "first", speed=1, stream_factory=FakeStream, control=control)))
    first_caller.start()
    assert inference_started.wait(2)
    started = time.monotonic()
    control.stop()
    assert time.monotonic() - started < 0.2
    assert first_caller.is_alive()
    release.set()
    first_caller.join(timeout=3)
    assert not first_caller.is_alive()
    assert first[0].stopped and first[0].played_frames == 0
    second = play_text(voice, "second", speed=1, stream_factory=FakeStream)
    assert second.played_frames == 100


def test_control_rejects_second_playback():
    class Voice:
        voice = SimpleNamespace(config=SimpleNamespace(sample_rate=100))
        def synthesize(self, segment):
            yield SimpleNamespace(audio_float_array=np.full(100, 0.5, dtype=np.float32),
                                  sample_rate=100)
    control = PlaybackControl()
    play_text(Voice(), "first", speed=1, stream_factory=FakeStream, control=control)
    with pytest.raises(RuntimeError, match="already in use"):
        play_text(Voice(), "second", speed=1, stream_factory=FakeStream, control=control)
