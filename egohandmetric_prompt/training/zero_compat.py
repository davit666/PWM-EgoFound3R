"""Compatibility for DeepSpeed's byte-view broadcast of scalar gate parameters.

The original model has scalar gates. A zero-dimensional bf16 tensor cannot be
viewed as uint8 in PyTorch. Only during initial broadcast, expose a length-one
view and restore the original shape before optimizer initialization. No model
state names, parameter objects, values or exported checkpoint shapes change.
"""
from contextlib import contextmanager


@contextmanager
def broadcastable_scalars(module):
    scalars=[p for p in module.parameters() if p.ndim==0]
    try:
        for parameter in scalars:
            parameter.data=parameter.data.reshape(1)
        yield
    finally:
        for parameter in scalars:
            parameter.data=parameter.data.reshape(())


def install_scalar_broadcast_compat():
    from deepspeed.runtime.engine import DeepSpeedEngine
    original=DeepSpeedEngine._broadcast_model
    if getattr(original,'_pwm_scalar_compat',False):
        return
    def broadcast(self,*args,**kwargs):
        with broadcastable_scalars(self.module):
            return original(self,*args,**kwargs)
    broadcast._pwm_scalar_compat=True
    DeepSpeedEngine._broadcast_model=broadcast
