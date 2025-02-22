import unittest
import data.process_data as process_data
import pandas as pd

class ETLTestCase(unittest.TestCase):
    def setUp(self):
        categories_path = '../data/disaster_categories.csv'
        messages_path = '../data/disaster_messages.csv'
         # import and define data types for categories
        categories_dtypes = {'id':  'Int64', 'categories': 'string'}
        self.categories = pd.read_csv("data/disaster_categories.csv", dtype=categories_dtypes)

        # import and define data types for messages and remove original field
        messages_dtypes = {'id':  'Int64', 'message': 'string', 'genre': 'string'}
        messages_usecols = ['id', 'message', 'genre']
        self.messages = pd.read_csv('data/disaster_messages.csv', dtype=messages_dtypes, usecols=messages_usecols)
        self.df = process_data.load_data(messages_filepath=messages_path, categories_filepath=categories_path)

    def test_inner_join(self):
        """ 
        test_inner_join tests that the expected ids are returned in the inner join.  This is tested by finding the intersection of each dataset's ids
        """

        expected = set(self.messages.id).intersection(self.categories.id)
        actual = set(self.df.id)
        error_msg = "The inner join did not have the correct set of ids"
        self.assertEqual(expected, actual, msg=error_msg)

    def test_no_duplicates(self):
        """
        test_no_duplicates tests that no duplicate rows exists
        """
        expected_truth_count = 0
        actual_truth_count = self.df.duplicated().sum()
        msg = "\n\nThere were {} duplicate rows in your dataset after the inner join".format(actual_truth_count)
        self.assertEqual(expected_truth_count, actual_truth_count, msg)

    def test_correct_nbr_of_category_columns(self):
        """
        test_correct_nbr_of_category_columns tests that there is a column for each category
        """
        
        expected = set([
                        'id', 'categories', 'message', 'genre', 'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                        'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                        'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'
                        ])
        actual = set(self.df.columns)
        msg = "\n\nThe expected columns were not created in your dataset"
        self.assertEqual(expected, actual, msg)
    
    def tearDown(self):
        del self.df

if __name__ == '__main__':
    unittest.main()