import unittest

from workspace.vqvae.evaluate_downstream import answer


class AnswerTests(unittest.TestCase):
    def test_arc_numeric_and_fifth_choice(self):
        self.assertEqual(answer('[4]', 'ARC-c'), 'D')
        self.assertEqual(answer('[E]', 'ARC-e'), 'E')
        self.assertEqual(answer('The answer is B.', 'ARC-e'), 'B')

    def test_boolean(self):
        self.assertEqual(answer('[False]', 'BoolQ'), 'FALSE')
        self.assertEqual(answer('true', 'BoolQ'), 'TRUE')

    def test_no_character_count_guess(self):
        self.assertIsNone(answer('Because this is unclear.', 'PIQA'))
        self.assertIsNone(answer('I do not know.', 'ARC-c'))
        self.assertIsNone(answer('', 'BoolQ'))


if __name__ == '__main__':
    unittest.main()
