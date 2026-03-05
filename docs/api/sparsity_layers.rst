Sparsity Layers
===============

.. note::

   In practice you want to use :class:`sparling.SparseLayerWithBatchNorm`, as without this,
   the Sparling algorithm does not work.

Base Class
----------

.. autoclass:: sparling.Sparsity
   :no-members:
   :show-inheritance:

   .. autoattribute:: sparsity
   .. autoattribute:: density
   .. automethod:: notify_sparsity

Per-Channel Enforcement
-----------------------

.. autoclass:: sparling.EnforceSparsityPerChannel
   :members:

.. autoclass:: sparling.EnforceSparsityPerChannelAccumulated
   :members:

Dimensional Wrappers
--------------------

.. autoclass:: sparling.EnforceSparsityPerChannel2D
   :members:

.. autoclass:: sparling.EnforceSparsityPerChannel1D
   :members:

Universal Enforcement
---------------------

.. autoclass:: sparling.EnforceSparsityUniversally
   :members:

Combinators
-----------

.. autoclass:: sparling.SparseLayerWithBatchNorm
   :members:

.. autoclass:: sparling.ParallelSparsityLayers
   :members:

Accumulation Stop Strategies
----------------------------

.. autoclass:: sparling.StopAtFixedNumberElements
   :members:

.. autoclass:: sparling.StopAtFixedNumberMotifs
   :members:
