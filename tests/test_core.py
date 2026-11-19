import unittest
from lib.models.qirtrack.trigger import low_quality
from lib.models.qirtrack.core.vlm.online_codebook import SemanticCodebookPolicy,semantic_label
from lib.models.qirtrack.core.vlm.identity_memory import fused_references
from lib.models.qirtrack.mllm.prompts import require_prompts
class CoreTests(unittest.TestCase):
 def test_trigger(self):
  self.assertTrue(low_quality(.39,.4));self.assertFalse(low_quality(.4,.4))
 def test_semantics(self):
  self.assertEqual(semantic_label({'status':'ok','conditional_probs_uncalibrated':{'YES':.7,'NO':.3}}),'high_yes')
  self.assertEqual(semantic_label({'status':'failed'}),'uncertain')
 def test_memory_limits(self):
  p=SemanticCodebookPolicy();self.assertEqual((p.max_active,p.max_pending,p.support_per_mode),(8,8,16))
 def test_fused(self):
  z=[1.]+[0.]*511;obs=[dict(sequence='s',frame=i,vector=z,crop_sha256=str(i)) for i in range(1,10)]
  rows,audit=fused_references(obs,z,10);self.assertLessEqual(len(rows),6);self.assertTrue(all(r['frame']<10 for r in rows))
 def test_empty_prompts_fail(self):
  with self.assertRaises(RuntimeError):require_prompts()
