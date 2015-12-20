import unittest
from hagelslag.data.MRMSGrid import MRMSGrid
from datetime import datetime

class TestMRMSGrid(unittest.TestCase):
    def setUp(self):
        self.path = "../../testdata/mrms_3km/"
        self.variable = "MESH_Max_60min_00.50"
        self.start_date = datetime(2015, 5, 1, 18, 0)
        self.end_date = datetime(2015, 5, 2, 15, 0)
        self.mrms = MRMSGrid(self.start_date, self.end_date, self.variable, self.path)
 
    def test_constructor(self):
        self.assertEquals(self.mrms.all_dates.size, 22, "Number of dates is wrong")
        self.assertIsNone(self.mrms.data, "Data already loaded")         
        self.assertIsNone(self.mrms.valid_dates, "Valid dates already loaded")

    #def test_loading(self):
        #self.mrms.load_data()
        #self.assertEquals(self.mrms.data.shape[0], self.mrms.valid_dates.shape[0], "Data and valid dates unequal length")
        #self.assertEquals(self.mrms.all_dates.shape[0], self.mrms.valid_dates.shape[0], "All dates were not loaded")

        
