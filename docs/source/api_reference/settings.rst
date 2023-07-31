.. _settings:

Settings
==========

.. automodule:: etna.settings
    :no-members:
    :no-inherited-members:

.. note::
   At the package init by default etna checks availability of all packages
   and warns you if some of them are not available.

All available installation options:

.. code-block:: console

        pip install etna
        pip install etna[prophet]
        pip install etna[torch]
        pip install etna[wandb]
        pip install etna[auto]
        pip install etna[classification]
        pip install etna[statsforecast]
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

API details
-----------

.. currentmodule:: etna.settings

.. autosummary::
   :toctree: api/
   :template: class.rst

   Settings

There is global object :code:`SETTINGS` that can be imported.
