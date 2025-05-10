from __future__ import annotations

import functools
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Callable, Concatenate, cast, TypeVar

from common.config import NUMBER_SECTION_DEL

if TYPE_CHECKING:
    from pipelines.pipeline import TrackedPipeline

_TP = TypeVar("_TP", bound="TrackedPipeline")
_TPMethod = Callable[Concatenate[_TP, ...], _TP, ]


def timer_pipeline(name: str) -> Callable[[_TPMethod[_TP]], _TPMethod[_TP]]:
    """Print the runtime of the decorated function"""
    def decorator_timer(func: _TPMethod[_TP]) -> _TPMethod[_TP]:
        @functools.wraps(func)
        def wrapper_timer(pipeline: _TP, **kwargs: Any) -> _TP:

            _name = name
            if "name" in kwargs:
                _name += f"_{kwargs['name']}"

            print(f"{_name}: ", end = '\0')
            start_time = timer()
            res: _TP = func(pipeline, **kwargs)
            end_time = timer()
            run_time = end_time - start_time
            print(f"Lasted {run_time:.2f} seconds.")
            print("-"*NUMBER_SECTION_DEL)

            pipeline.add_time(_name, run_time)
            return res
        return cast(_TPMethod[_TP], wrapper_timer)
    return decorator_timer