import unittest
import data.process_data as process_data

class ETLTestCase(unittest.TestCase):
    def setUp(self):
        categories_path = '../data/disaster_categories.csv'
        messages_path = '../data/disaster_messages.csv'
        self.df = process_data.load_data(messages_filepath=messages_path, categories_filepath=categories_path)

    def test_inner_join(self):
        """ 
        test_inner_join tests that the expected number of rows are returned in the inner join
        """

        expected = 26248
        actual = self.df.shape[0]
        error_msg = "\n\nThe expected number of rows ({}) did not equal the actual number of rows ({})".format(expected, actual)
        self.assertEqual(expected, actual, msg=error_msg)

    def test_no_duplicates(self):
        """
        test_no_duplicates tests that no duplicate rows exists
        """
        expected_truth_count = 0
        actual_truth_count = self.df.duplicated().sum()
        msg = "\n\nThere were {} duplicate rows in your dataset after the inner join".format(actual_truth_count)
        self.assertEqual(expected_truth_count, actual_truth_count, msg)

    def tearDown(self):
        del self.df

if __name__ == '__main__':
    unittest.main()