class ModelOutput(object):
    def __init__(self, ensemble_name, member_name, run_date, variable):
        self.ensemble_name = ensemble_name
        self.member_name = member_name
        self.run_date = run_date
        self.variable = variable
        self.data = None
        self.valid_dates = None

    def get_data(self, data):
        self.data = data

    def get_valid_dates(self, valid_dates):
        self.valid_dates = valid_dates
