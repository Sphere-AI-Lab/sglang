"""The radix key must separate KV computed under different weight versions of the
SAME adapter. Under RL the trainer re-pushes one adapter id every step, so an
id-only key would let a prefix cached at version k be reused at k+1."""

import unittest

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_batch import _extend_oft_extra_key as maybe_extend_extra_key
from sglang.srt.sampling.sampling_params import SamplingParams


class TestExtraKeyBuilder(unittest.TestCase):
    def test_version_is_rendered(self):
        self.assertEqual(maybe_extend_extra_key(None, "abc", 7), "|oft:abc:v7")
        self.assertEqual(maybe_extend_extra_key("pre", "abc", 2), "pre|oft:abc:v2")

    def test_missing_version_degrades_to_v0(self):
        # Callers that cannot supply a version still get a well-formed key.
        self.assertEqual(maybe_extend_extra_key(None, "abc", None), "|oft:abc:v0")

    def test_versions_do_not_collide(self):
        self.assertNotEqual(
            maybe_extend_extra_key(None, "abc", 1),
            maybe_extend_extra_key(None, "abc", 2),
        )

    def test_base_request_is_untouched(self):
        # No adapter id -> no oft segment at all, so base prefixes stay shareable.
        self.assertIsNone(maybe_extend_extra_key(None, None, 5))


class TestReqCarriesVersion(unittest.TestCase):
    """The builder was always correct; the bug was that Req never fed it a version."""

    def _req(self, rid, **kw):
        return Req(rid, "hello", [1, 2, 3], SamplingParams(), **kw)

    def test_same_adapter_different_version_keys_differ(self):
        r1 = self._req("r1", adapter_id="A", adapter_version=1)
        r2 = self._req("r2", adapter_id="A", adapter_version=2)
        self.assertIn("v1", r1.extra_key)
        self.assertIn("v2", r2.extra_key)
        self.assertNotEqual(r1.extra_key, r2.extra_key)

    def test_same_adapter_same_version_keys_match(self):
        a = self._req("r1", adapter_id="A", adapter_version=3)
        b = self._req("r2", adapter_id="A", adapter_version=3)
        self.assertEqual(a.extra_key, b.extra_key)

    def test_different_adapters_keys_differ(self):
        a = self._req("r1", adapter_id="A", adapter_version=1)
        b = self._req("r2", adapter_id="B", adapter_version=1)
        self.assertNotEqual(a.extra_key, b.extra_key)

    def test_base_request_has_no_adapter_segment(self):
        base = self._req("r1")
        self.assertTrue(base.extra_key is None or "oft" not in base.extra_key)

    def test_version_is_stored_on_the_req(self):
        self.assertEqual(self._req("r1", adapter_id="A", adapter_version=9).adapter_version, 9)


if __name__ == "__main__":
    unittest.main()
