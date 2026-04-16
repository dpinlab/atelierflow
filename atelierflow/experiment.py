import logging
from typing import List, Any, Dict, Optional, Literal, get_args, get_origin, TypeVar
from .core.step import Step

class Experiment:
    """
    Orchestrates the execution of a machine learning pipeline.
    Ensures data integrity between steps through type validation.
    """

    def __init__(
        self, 
        name: str,
        device: str = 'cpu',
        logging_level: Literal['NOTSET', 'DEBUG', 'INFO', 'WARNING'] = 'INFO',
        tags: Optional[Dict[str, Any]] = None,
        enable_caching: bool = False
    ):
        self.name = name
        self.steps: List[Step] = []
        self.config = {
            'device': device,
            'logging_level': logging_level.upper(),
            'tags': tags or {},
            'enable_caching': enable_caching
        }
        self._setup_logging()

    def _setup_logging(self):
        """Sets up the global logging configuration for the experiment."""
        log_level = self.config.get('logging_level')
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f"Experiment '{self.name}' initialized on {self.config['device']}")

    def _get_step_types(self, instance: Any):
        """
        Extracts T_in and T_out types from the generic class hierarchy.
        This is used to inspect the 'contract' defined by the researcher.
        """
        # Look into the class's original bases to find the specialized 'Step'
        for base in instance.__class__.__orig_bases__:
            if get_origin(base) is Step:
                args = get_args(base)
                return args if len(args) == 2 else (None, None)
        return None, None

    def add_step(self, step: Step):
        """
        Adds a step to the pipeline and validates type compatibility 
        between the new step and the previous one.
        
        This prevents 'Pipeline Jungles' by checking input/output contracts.
        """
        if self.steps:
            last_step = self.steps[-1]
            
            # Extract type contracts (T_in, T_out)
            _, last_out = self._get_step_types(last_step)
            current_in, _ = self._get_step_types(step)

            if last_out is not None and current_in is not None:
                # If either type is a TypeVar (e.g., T), allow the connection (Generic Pass-through)
                is_generic_in = isinstance(current_in, TypeVar)
                is_generic_out = isinstance(last_out, TypeVar)

                if not is_generic_in and not is_generic_out:
                    # If both are concrete types and they differ
                    if current_in is not type(None) and current_in != last_out:
                        in_name = current_in.__name__ if hasattr(current_in, "__name__") else str(current_in)
                        out_name = last_out.__name__ if hasattr(last_out, "__name__") else str(last_out)
                        
                        error_msg = (
                            f"\n{'!'*40}\n"
                            f"Incorrect Pipeline Construction!\n"
                            f"{'-'*40}\n"
                            f"Incompatible Connection:\n"
                            f"  From:  {last_step.__class__.__name__} -> returns [{out_name}]\n"
                            f"  To:    {step.__class__.__name__} -> expects [{in_name}]\n"
                            f"{'!'*40}"
                        )
                        logging.error(error_msg)
                        raise TypeError(error_msg)

        self.steps.append(step)
        logging.debug(f"Step '{step.__class__.__name__}' added to pipeline.")

    def run(self, initial_input: Any = None) -> Any:
        """
        Executes all steps in sequence, passing the result of one as input to the next.
        """
        if not self.steps:
            raise ValueError("Cannot run experiment: no steps have been added.")

        logging.info(f"--- Starting Experiment: '{self.name}' ---")
        current_result = initial_input

        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            logging.info(f"---> Step {i}/{len(self.steps)}: Executing '{step_name}'...")
            
            try:
                current_result = step.run(
                    input_data=current_result, 
                    experiment_config=self.config
                )
            except Exception as e:
                logging.error(f"Error in step '{step_name}': {e}", exc_info=True)
                raise

        logging.info(f"--- Experiment '{self.name}' Finished Successfully ---")
        return current_result