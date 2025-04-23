from modules.module_base import Module

class Pipeline:
    """
    Stellt eine modulare Pipeline zusammen, in der verschiedene Verarbeitungsschritte
    (Klassen mit einer process()-Methode) sequentiell ausgeführt werden.
    """
    def __init__(self, input_data: dict = {}):
        self.stages: list[Module] = []
        self.data: dict = input_data
    
    def add_stage(self, stage):
        self.stages.append(stage)
    
    def _check_condition(self, module: Module):
        for condition in module.get_preconditions():
            if self.data.get(condition, None) is None:
                raise Exception(f"Precondition of ${module.module_key} not fulfilled")

    def run(self, input_data = None):
        self.data['input'] = input_data
        for module in self.stages:
            self._check_condition(module)

            self.data[module.module_key] = module.process(self.data)

        # Output is result of last stage
        return self.data[self.stages[-1].module_key]