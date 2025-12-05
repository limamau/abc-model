import jax
import jax.numpy as jnp
from simple_pytree import Pytree, static_field

class MyState(Pytree):
    x: jax.Array
    y: float
    name: str = static_field()

    def __init__(self, x, y, name="test"):
        self.x = x
        self.y = y
        self.name = name

def test_simple_pytree():
    s = MyState(jnp.array([1.0, 2.0]), 3.0, "foo")
    print(f"Original: {s}")
    
    leaves, treedef = jax.tree_util.tree_flatten(s)
    print(f"Leaves: {leaves}")
    
    s_new = jax.tree_util.tree_unflatten(treedef, leaves)
    print(f"Restored: {s_new}")
    print(f"Name preserved: {s_new.name}")

    # Test with JIT
    @jax.jit
    def f(state):
        return state.x * state.y

    res = f(s)
    print(f"JIT result: {res}")

if __name__ == "__main__":
    test_simple_pytree()
