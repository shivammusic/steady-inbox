import math
import unittest
import numpy as np
from app import Engine, SAMPLE, group_vectors, validate_request


class InputTests(unittest.TestCase):
    def test_blank_lines_are_ignored_without_losing_order(self):
        self.assertEqual(validate_request({'text':'  first \n\n second '})[0], ['first','second'])

    def test_invalid_notes(self):
        for data in [None, [], {}, {'text':''}, {'text':5}, {'text':'x\n'*101}, {'text':'x'*1501}]:
            with self.subTest(data=str(data)[:50]), self.assertRaises(ValueError):
                validate_request(data)

    def test_invalid_threshold(self):
        for value in [True, None, '0.76', .1, .99, math.nan, math.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_request({'text':'hello','threshold':value})

    def test_complete_link_prevents_similarity_chaining(self):
        # A-B and B-C clear 0.8, but A-C does not. Do not collapse all three.
        angles = np.radians([0,30,60])
        vectors = np.array([[np.cos(x),np.sin(x)] for x in angles])
        self.assertEqual(group_vectors(vectors,.8), [[0,1],[2]])

    def test_unrelated_items_remain_separate(self):
        self.assertEqual(group_vectors(np.eye(3),.76), [[0],[1],[2]])


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def test_real_embeddings_are_finite_normalized_and_384_dimensions(self):
        vectors, truncated = self.engine.embed(['Please review the pull request.','Buy milk.'])
        self.assertEqual(vectors.shape,(2,384))
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
        self.assertEqual(truncated,[False,False])

    def test_semantic_paraphrases_group_and_unrelated_note_does_not(self):
        report = self.engine.digest({'text':'Remember to buy milk on the way home.\nPick up some milk on your way back home.\nDeploy the database migration tonight.','threshold':.76})
        self.assertEqual(report['group_count'],2)
        self.assertEqual(report['groups'][0]['source_ids'],[1,2])
        self.assertEqual(report['groups'][1]['source_ids'],[3])

    def test_every_source_survives_once_and_representatives_are_extracts(self):
        report = self.engine.digest({'text':'\n'.join(SAMPLE)})
        ids = [i for group in report['groups'] for i in group['source_ids']]
        self.assertEqual(sorted(ids),list(range(1,len(SAMPLE)+1)))
        for group in report['groups']:
            self.assertIn(group['representative'],group['messages'])
            self.assertEqual(group['messages'],[SAMPLE[i-1] for i in group['source_ids']])
        self.assertGreater(report['repeated_count'],0)

    def test_long_message_retains_full_source_and_flags_truncation(self):
        note='hello '*240
        report=self.engine.digest({'text':note})
        # 240 wordpieces plus special tokens fits; 280 exceeds the 256-token limit.
        self.assertEqual(report['groups'][0]['messages'],[note.strip()])
        note='hi '*280
        report=self.engine.digest({'text':note})
        self.assertEqual(report['truncated_count'],1)
        self.assertEqual(report['groups'][0]['messages'],[note.strip()])

    def test_duplicate_source_messages_keep_distinct_ids(self):
        report=self.engine.digest({'text':'Review this PR.\nReview this PR.'})
        self.assertEqual(report['group_count'],1)
        self.assertEqual(report['groups'][0]['source_ids'],[1,2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
