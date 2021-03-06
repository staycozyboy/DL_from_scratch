is_simple_core = False

if is_simple_core:
    from dezero.core_simple import Variable, Function
    from dezero.core_simple import using_config, no_grad
    from dezero.core_simple import as_array, as_variable
    from dezero.core_simple import setup_variable

else:
    from dezero.core import Variable, Function, Parameter
    from dezero.core import using_config, no_grad
    from dezero.core import as_array, as_variable
    from dezero.core import setup_variable
    from dezero.layers import Layer
    from dezero.models import Model

    import dezero.optimizers
    import dezero.functions
    import dezero.layers
    import dezero.utils

setup_variable()