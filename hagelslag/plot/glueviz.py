from glue.viewers.custom.qt import CustomViewer
from glue.core import Data
from glue.config import data_factory
import json
import numpy as np
from mpl_toolkits.basemap import Basemap


class TrackPolyMap(CustomViewer):
    name = "Track Poly Map"
    x = "att(x)"
    y = "att(y)"
    color = "red"

    def plot_data(self, axes, x, y, color, state):
        for p in range(len(x)):
            axes.fill(x[p], y[p], color)

    def plot_subset(self, axes, x, y, style, state):
        for p in range(len(x)):
            mx, my = state.m(x[p], y[p])
            axes.fill(mx, my,
                      alpha=style.alpha,
                      facecolor=style.color,
                      edgecolor=style.color)

    def setup(self, axes, state):
        bmap = Basemap(projection="cyl")
        bmap.drawstates()
        bmap.drawcoastlines()
        bmap.drawcountries()
        state.m = bmap

    def select(self, roi, latitude, longitude, state):
        mx, my = state.m(longitude, latitude)
        return roi.contains(mx, my)


def is_json(filename, **kwargs):
    return filename.endswith(".json")


@data_factory('Hagelslag geoJSON loader', is_json)
def read_json(file_name):
    json_file = open(file_name)
    data = json.load(json_file)
    json_file.close()
    track_steps = []
    for feature in data['features']:
        feature_grids = {}
        for grid in feature['properties']['attributes'].keys():
            feature_grids[grid] = np.array(feature['properties']['attributes'])
        feature_grids["i"] = feature["properties"]["i"]

        track_steps.append(Data(label=data['properties']["id"], **feature_grids))
    return track_steps
