import numpy as np


def melting_layer_height(height_surface, height_700mb, height_500mb, temperature_700mb, temperature_500mb):
    melting_layer_height_asl = -temperature_500mb * (height_700mb - height_500mb) / \
                               (temperature_700mb - temperature_500mb) + height_500mb
    return melting_layer_height_asl - height_surface


def relative_humidity_pressure_level(temperature, specific_humidity, pressure):
    saturation_vapor_pressure = 611.2 * np.exp((17.67 * temperature) / (temperature + 243.5))
    vapor_pressure = specific_humidity / 1000.0 * pressure / 0.622
    return vapor_pressure / saturation_vapor_pressure * 100.0