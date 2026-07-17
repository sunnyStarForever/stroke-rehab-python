import threading
import time
import unittest

from rehab_engine._stub import VoiceConfig
from rehab_engine.voice import VoiceAssistant


class _FakeEngine:
    def __init__(self, fail=False):
        self.properties = {}
        self.pending = ""
        self.spoken = []
        self.fail = fail

    def setProperty(self, key, value):
        self.properties[key] = value

    def say(self, text):
        self.pending = text

    def runAndWait(self):
        if self.fail:
            raise RuntimeError("audio device failed")
        self.spoken.append(self.pending)

    def stop(self):
        pass


class VoiceAssistantTests(unittest.TestCase):
    def test_priority_dedup_and_non_blocking_delivery(self):
        config = VoiceConfig(cooldown_seconds=10, queue_size=4)
        engine = _FakeEngine()
        release = threading.Event()

        def factory():
            release.wait(2)
            return engine

        voice = VoiceAssistant(config, engine_factory=factory)
        started = time.monotonic()
        self.assertTrue(voice.speak("普通提示", key="normal", priority=20))
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertFalse(voice.speak("重复提示", key="normal", priority=20))
        self.assertTrue(voice.speak("重要提示", key="important", priority=1))
        release.set()
        deadline = time.monotonic() + 2
        while len(engine.spoken) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        voice.stop()
        self.assertEqual(engine.spoken, ["重要提示", "普通提示"])
        self.assertEqual(engine.properties["rate"], 175)
        self.assertEqual(engine.properties["volume"], 0.9)

    def test_backend_failure_disables_voice_without_raising_to_caller(self):
        config = VoiceConfig()
        engine = _FakeEngine(fail=True)
        statuses = []
        voice = VoiceAssistant(config, engine_factory=lambda: engine)
        voice.on_status = statuses.append
        self.assertTrue(voice.speak("测试", force=True))
        deadline = time.monotonic() + 2
        while voice.available is not False and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(voice.available)
        self.assertIn("audio device failed", voice.last_error)
        self.assertTrue(any("continuing silently" in item for item in statuses))
        voice.stop()

    def test_disabled_configuration_never_starts_worker(self):
        voice = VoiceAssistant(VoiceConfig(enabled=False), engine_factory=_FakeEngine)
        self.assertFalse(voice.start())
        self.assertFalse(voice.speak("不会播放"))
        self.assertFalse(voice.available)


if __name__ == "__main__":
    unittest.main(verbosity=2)
