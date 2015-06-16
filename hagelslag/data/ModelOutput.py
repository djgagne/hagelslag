from SSEFModelGrid import SSEFModelGrid
from NCARModelGrid import NCARModelGrid

class ModelOutput(object):
    def __init__(self, 
                 ensemble_name, 
                 member_name, 
                 run_date, 
                 variable, 
                 start_date, 
                 end_date,
                 path,
                 **kwargs):
        self.ensemble_name = ensemble_name
        self.member_name = member_name
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.valid_dates = None
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def load_data(self):
        if self.ensemble_name.upper() == "SSEF":
            with SSEFModelGrid(self.member_name,
                               self.run_date,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step) as mg:
                self.data = mg.load_data()
        elif self.ensemble_name.upper() == "NCAR":
            with NCARModelGrid(self.member_name,
                               self.run_date,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step) as mg:
                self.data = mg.load_data()
