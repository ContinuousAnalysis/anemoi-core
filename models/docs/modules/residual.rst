######################
 Residual connections
######################

Residual connections are a key architectural feature in Anemoi's
encoder-processor-decoder models, enabling more effective information
flow and gradient propagation across network layers. Residual
connections help mitigate issues such as vanishing gradients and support
the training of deeper, and more expressive models.

In Anemoi, the type of residual connection used in a model is specified
under the `residual` key in the model configuration YAML. This modular
approach allows users to select and customize the residual strategy best
suited for their forecasting task, whether it be a standard skip
connection, no connection, or a truncated connection.

The following classes implement the available residual connection types
in Anemoi.

*****************
 Skip Connection
*****************

.. autoclass:: anemoi.models.layers.residual.SkipConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

**********************
 Truncated Connection
**********************

.. autoclass:: anemoi.models.layers.residual.TruncatedConnection
   :members:
   :no-undoc-members:
   :show-inheritance:

****************
 Configuration
****************

The residual connection type is configured under the ``residual`` key in
the model config. In the training config, residual presets are stored
under ``model/residual/`` and referenced via Hydra defaults.

Skip connection (default):

.. code:: yaml

   residual:
     _target_: anemoi.models.layers.residual.SkipConnection
     step: -1

Graph-based truncated connection:

.. code:: yaml

   residual:
     _target_: anemoi.models.layers.residual.TruncatedConnection
     data_nodes: data
     truncation_nodes: "int"
     edge_weight_attribute: gauss_weight

File-based truncated connection:

.. code:: yaml

   residual:
     _target_: anemoi.models.layers.residual.TruncatedConnection
     truncation_down_file_path: /path/to/n320_to_o96.npz
     truncation_up_file_path: /path/to/o96_to_n320.npz

For multi-dataset training, different residual connections can be
specified per dataset. See the training documentation for details on
per-dataset configuration.
