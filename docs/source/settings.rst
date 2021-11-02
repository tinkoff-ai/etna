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

.. note::
   You also may want to make sure, that your etna project always has
   necessary dependencies installed. In order to do that, you need to create `.etna`
   file in the project directory. This way you will get error if any of
   the dependencies are not present.

Example `.etna` file:

.. code-block:: console

        [etna]
        torch_required = false
        prophet_required = true
        wandb_required = false

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