import torch

from modules.module_base import Module

class Pipeline:
    """
    Stellt eine modulare Pipeline zusammen, in der verschiedene Verarbeitungsschritte
    (Klassen mit einer process()-Methode) sequentiell ausgeführt werden.
    """
    def __init__(self, input_data: dict | None = None):
        self.stages: list[Module] = []
        self.data: dict = {}
        if input_data is not None:
            self.data = input_data
    
    def add_stage(self, stage):
        """
        Add module to pipeline
        """
        self.stages.append(stage)
    
    def _check_condition(self, module: Module):
        """
        Check if all precondition of current module in pipeline is fulfilled
        """
        for condition in module.get_preconditions():
            if self.data.get(condition, None) is None:
                raise Exception(f"Precondition of ${module.module_key} not fulfilled")

    def run(self, input_data = None):
        """
        Executes pipeline
        """
        self.data['input'] = input_data
        for module in self.stages:
            self._check_condition(module)

            # Init of models should be moved there
            if hasattr(module, '_warmup'):
                module._warmup()

            self.data[module.module_key] = module.process(self.data)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Output is result of last stage
        return self.data[self.stages[-1].module_key]
