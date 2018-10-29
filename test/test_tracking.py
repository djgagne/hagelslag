import unittest
import numpy as np
from datetime import datetime, timedelta
from hagelslag.data.ModelOutput import ModelOutput
from hagelslag.processing.tracker import label_storm_objects, extract_storm_objects


class TestTracking(unittest.TestCase):
    def setUp(self):
        self.model_path = "testdata/spring2015_unidata/"
        self.ensemble_name = "SSEF"
        self.member ="wrf-s3cn_arw"
        self.run_date = datetime(2015, 6, 4)
        self.variable = "uh_max"
        self.start_hour = 18
        self.end_hour = 24
        self.start_date = self.run_date + timedelta(hours=self.start_hour)
        self.end_date = self.run_date + timedelta(hours=self.end_hour)
        self.model_grid = ModelOutput(self.ensemble_name,
                                 self.member,
                                 self.run_date,
                                 self.variable,
                                 self.start_date,
                                 self.end_date,
                                 self.model_path,
                                 "mapfiles/ssef2015.map",
                                 single_step=False)
        self.model_grid.load_data()
        self.model_grid.load_map_info("mapfiles/ssef2015.map")

    def test_model_grid_loading(self):

        self.assertEquals(self.model_grid.lat.shape, self.model_grid.lon.shape,
                          "Lat and Long grids are not equal shaped.")
        self.assertEquals(self.model_grid.data.shape[1:], self.model_grid.lat.shape,
                          "Data shape does not match map shape.")
        self.assertEquals(self.end_hour - self.start_hour + 1, self.model_grid.data.shape[0],
                          "Number of hours loaded does not match specified hours.")

    def test_object_identification(self):
        min_thresh = 10
        max_thresh = 50
        label_grid = label_storm_objects(self.model_grid.data[0], "hyst",
                                         min_thresh, max_thresh, min_area=2, max_area=100)
        label_points = self.model_grid.data[0][label_grid > 0]
        self.assertGreaterEqual(label_points.min(), min_thresh, "Labeled points include those below minimum threshold")
        storm_objs = extract_storm_objects(label_grid, self.model_grid.data[0], self.model_grid.x,
                                           self.model_grid.y, np.array([0]))
        self.assertEquals(len(storm_objs[0]), label_grid.max(), "Storm objects do not match number of labeled objects")
