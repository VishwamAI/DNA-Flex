"""Attention call argument specifications.

Attention argument specifications used by users of the library.
They are the most important test cases, and also cases for optimize
performance of via autotuning.
"""

from typing import Any

import jax

ShapedArray = jax.ShapeDtypeStruct


def _make_argspec(
    *,
    q_shape,
    dtype,
    k_shape=None,
    v_shape=None,
    bias_shape=None,
    mask_shape=None,
    **kwargs,
) -> dict[str, Any]:
  """Make argspec from shapes and kwargs."""
  if k_shape is None:
    k_shape = q_shape
  if v_shape is None:
    v_shape = k_shape

  return dict(
      query=ShapedArray(q_shape, dtype),
      key=ShapedArray(k_shape, dtype),
      value=ShapedArray(v_shape, dtype),
      bias=ShapedArray(bias_shape, dtype) if bias_shape is not None else None,
      mask=ShapedArray(mask_shape, 'bool_') if mask_shape is not None else None,
      **kwargs,
  )


# A subset of the full set of argument specifications. Useful for tap-tests and
# microbenchmarks.
CALL_ARG_SPECS = dict(
    vanilla_f32=_make_argspec(q_shape=(8, 1024, 4, 128), dtype='float32'),
    vanilla_bf16=_make_argspec(q_shape=(8, 1024, 4, 128), dtype='bfloat16'),
    alphafold=_make_argspec(
        q_shape=(384, 384, 4, 32),
        bias_shape=(1, 4, 384, 384),
        mask_shape=(384, 1, 1, 384),
        dtype='bfloat16',
    ),
)