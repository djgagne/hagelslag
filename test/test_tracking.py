import unittest
import numpy as np
from datetime import datetime, timedelta
from hagelslag.data.ModelOutput import ModelOutput
from hagelslag.processing.tracker import label_storm_objects, extract_storm_objects
from hagelslag.processing.TrackProcessing import TrackProcessor
from hagelslag.processing.ObjectMatcher import shifted_centroid_distance, centroid_distance, time_distance
import os

class TestTracking(unittest.TestCase):
    def setUp(self):
        current_dir_files = os.listdir("./")
        if "testdata" not in current_dir_files:
            start_path = "../"
        else:
            start_path = "./"
        self.model_path = start_path + "testdata/spring2015_unidata/"
        self.ensemble_name = "SSEF"
        self.member ="wrf-s3cn_arw"
        self.run_date = datetime(2015, 6, 4)
        self.variable = "uh_max"
        self.start_hour = 18
        self.end_hour = 24
        self.start_date = self.run_date + timedelta(hours=self.start_hour)
        self.end_date = self.run_date + timedelta(hours=self.end_hour)
        self.map_file = start_path + "mapfiles/ssef2015.map"
        self.model_grid = ModelOutput(self.ensemble_name,
                                 self.member,
                                 self.run_date,
                                 self.variable,
                                 self.start_date,
                                 self.end_date,
                                 self.model_path,
                                 self.map_file,
                                 single_step=False)
        self.model_grid.load_data()
        self.model_grid.load_map_info(self.map_file)

    def test_model_grid_loading(self):
        print(self.model_grid.lat.shape, self.model_grid.lon.shape)
        self.assertEqual(self.model_grid.lat.shape[0], self.model_grid.lon.shape[0],
                          "Lat and Long grids are not equal shaped. {")
        self.assertEqual(self.model_grid.data.shape[1:], self.model_grid.lat.shape,
                          "Data shape does not match map shape.")
        self.assertEqual(self.end_hour - self.start_hour + 1, self.model_grid.data.shape[0],
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
        self.assertEqual(len(storm_objs[0]), label_grid.max(), "Storm objects do not match number of labeled objects")

    def test_track_processing(self):
        ws_params = (25, 50)
        object_matcher_params = ([shifted_centroid_distance], np.array([1.0]),
                             np.array([24000]))
        track_matcher_params = ([centroid_distance, time_distance],
                                     np.array([80000, 2]))
        size_filter = 10
        gaussian_window = 1
        tp = TrackProcessor(self.run_date, self.start_date, self.end_date, self.ensemble_name,
                            self.member, self.variable, self.model_path, self.map_file, ws_params,
                            object_matcher_params, track_matcher_params, size_filter, gaussian_window,
                            segmentation_approach="ws", single_step=False
                            )
        track_model_patch_objects = tp.find_model_patch_tracks()
        self.assertGreater(len(track_model_patch_objects), 0, "No patch objects found")
        #track_model_objects = tp.find_model_tracks()
        #self.assertGreater(len(track_model_objects), 0, "No objects found")

