Settings
==========

.. note::
   At the package init by default etna checks availability of all packages
   and warns you if some of them are not available.

All available installation options:

.. code-block:: console

        pip install etna
        pip install etna[prophet]
        pip install etna[pytorch]
        pip install etna[wandb]
        pip install etna[all]

.. _etna:

.. currentmodule:: etna

Details and available algorithms
--------------------------------

See the API documentation for further details on setting up your etna environment:

.. currentmodule:: etna

.. moduleautosummary::
   :toctree: api/
   :template: custom-module-template.rst
   :recursive:

   etna.settings