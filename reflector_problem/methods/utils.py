class History(object):
    """History: Object that allows to save state of optimization

    Args:
        frequency: Frequency of saves
        variables_to_save: List of name of variables to save
    """

    def __init__(self, variables_to_save, frequency=1):
        self.frequency = frequency
        self.variables_to_save = variables_to_save
        self.step_numbers = []
        self.step_history = []
        self.var_history = dict()

    def save_step(self, step_number, **kwargs):
        if step_number % self.frequency == 0:
            self.step_numbers.append(step_number)
            self.step_history.append({key: value for key, value in kwargs.items(
            ) if self.variables_to_save is None or key in self.variables_to_save})

    def save_vars(self, **kwargs):
        self.var_history.update(kwargs)

    def get_step_history(self):
        return self.step_history

    def get_var_history(self):
        return self.var_history

    def get_var_history_variable(self, variable_name):
        return self.var_history[variable_name]

    def get_step_history_variable(self, variable_name):
        return [step_dict[variable_name] for step_dict in self.step_history]
