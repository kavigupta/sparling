Baselines
=========

Simple activation-based layers that do not enforce a target sparsity level,
useful as baselines or when sparsity is controlled externally (e.g. via an
L1 or KL loss).

.. autoclass:: sparling.NoSparsity
   :members:

.. autoclass:: sparling.SparsityForL1
   :members:

.. autoclass:: sparling.ChangingSparsityForL1
   :members:

.. autoclass:: sparling.SparsityForKL
   :members:

.. autoclass:: sparling.NoiseRatherThanSparsity
   :members:
