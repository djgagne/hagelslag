import pandas as pd

from hagelslag.data.ZarrModelGrid import ZarrModelGrid


class HRRRZarrModelGrid(ZarrModelGrid):

    def __init__(self, run_date, variable, start_date, end_date, path, frequency="1H"):
        self.run_date = pd.Timestamp(run_date)
        self.variable = variable
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.frequency = frequency
        self.path = path

        super(HRRRZarrModelGrid, self).__init__(path, run_date, start_date, end_date, variable)
