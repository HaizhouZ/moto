from moto import var
import casadi as cs
import types


class symvar(cs.SX):
    def __init__(self, var_value: var):

        #         # We still create the internal delegate object
        self.sym_handle: var = var_value
        super().__init__(var_value.sx)  # Initialize the SX part with the inner SX

    def __getattribute__(self, name):
        """
        Intercepts EVERY attribute access to control the lookup order.
        """
        if name == "sym_handle":
            print("found sym_handle")
            return object.__getattribute__(self, name)

        # 2. Try to get the delegate. If it doesn't exist yet (during __init__),
        # this will raise an AttributeError and we'll fall back.
        try:
            delegate = object.__getattribute__(self, "sym_handle")
            print("delegate:", delegate)
            return getattr(delegate, name)
        except AttributeError:
            print("no delegate")
            # This happens if 'sym_handle' isn't set yet, OR if the attribute
            # isn't found on the delegate. Fall back to the parent class (cs.SX).
            return super().__getattribute__(name)

    @staticmethod
    def create(var_value: var):
        """
        Factory to create a cs.SX instance with custom attribute delegation.

        Args:
            var_value: The 'var' object to delegate attribute lookups to.

        Returns:
            A genuine cs.SX instance whose behavior is controlled by DelegatingSX.
        """
        # 1. Create a GENUINE, normal cs.SX instance.
        #    Its underlying C++ type is casadi::SX, which is what Pinocchio needs.
        # sx_instance = cs.SX(var_value.sx)
        sx_instance = symvar(var_value)

        # 2. Attach the delegate object needed by our custom __getattribute__.
        #    We use object.__setattr__ to bypass any logic.
        # object.__setattr__(sx_instance, "sym_handle", var_value)
        # object.__setattr__(sx_instance, "__getattribute__", types.MethodType(symvar.generic__getattribute__, sx_instance))
        sx_instance.__class__ = cs.SX
        sx_instance.__getattribute__ = types.MethodType(symvar.__getattribute__, sx_instance)
        # print(type(sx_instance))
        return sx_instance
