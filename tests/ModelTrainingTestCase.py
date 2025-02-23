import unittest
from models.train_classifier import load_data
import pandas as pd

class ModelTrainingTestCase(unittest.TestCase):
    def setUp(self):
        path = 'data/DisasterResponse.db'
        self.X, self.Y, self.category_names = load_data(database_filepath=path)
    
    def test_correct_columns_X(self):
        """
        This test determines that the correct columns are in the feature set X
        """

        expected = set(['id', 'message', 'genre'])
        actual = set(self.X.columns)
        msg = "\nThe correct columns were not provided to X"
        self.assertEqual(expected, actual, msg=msg)

    def test_correct_columns_Y(self):
        """
        This test determines that the correct columns are in the target set Y
        """

        expected = set([
                        'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                        'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                        'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'
                        ])
        actual = set(self.Y.columns)
        msg = "\nThe correct columns were not provided to Y"
        self.assertEqual(expected, actual, msg=msg)

    def test_correct_category_names(self):
        """
        This test determines that the the correct category names were obtained in load_data's category_names output
        """

        expected = set([
                'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'
                ])
        actual = set(self.category_names)
        msg = "\nThe correct columns were not provided to category_names"
        self.assertEqual(expected, actual, msg=msg)


    def tearDown(self):
        del self.X
        del self.Y
        del self.category_names