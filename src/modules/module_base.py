import typing

class Module:
    """ 
    Base Module Interface
    """
    module_key=None
    
    def __init__(self, module_key = None):
        if not module_key:
            raise Exception("Please declare a module key")
        
        self.module_key = module_key

    def get_preconditions(self) -> list[str]:
        """
        Returns preconditions of current Module
        """
        raise Exception("Please define get_preconditions")
    
    def process(self, data: dict) -> typing.Any:
        """
        Executes current Module
        """
        raise Exception("Please define process")
