"""Tools to help handle coordinate systems and transformations.

TODO:
- Use unxt and coordinax where possible.
"""

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

__all__ = ["get_u_vec", "get_tangent_basis"]


def get_u_vec(lon: ArrayLike, lat: ArrayLike) -> jax.Array:
    """Construct a unit vector pointing in the direction of a given sky coordinate.

    Given two sky coordinates at a longitude and latitude (e.g., RA, Dec), return a unit
    vector that points in the direction of that sky position. The inputs can be scalar
    or arrays. The output will be `(3, *shape)` where `shape` is the broadcasted shape
    of each input.
    """
    return jnp.array(
        [jnp.cos(lon) * jnp.cos(lat), jnp.sin(lon) * jnp.cos(lat), jnp.sin(lat)]
    )


def get_tangent_basis(lon: ArrayLike, lat: ArrayLike) -> jax.Array:
    """Construct the tangent-space basis vectors at a given sky coordinate.

    Given two sky coordinates at a longitude and latitude (e.g., RA, Dec), return the
    tangent-space basis vectors at that sky position. The output shape is `(3, 3,
    *shape)`, where the first dimension indexes the three basis vectors, the second
    dimension indexes the x, y, z components of each basis vector, and `shape` is the
    broadcasted shape of each input.
    """
    return jnp.array(
        [
            [-jnp.sin(lon), jnp.cos(lon), 0.0],
            [-jnp.sin(lat) * jnp.cos(lon), -jnp.sin(lat) * jnp.sin(lon), jnp.cos(lat)],
            [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)],
        ]
    )
