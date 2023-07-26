.. Template is based on github.com/jdb78/pytorch-forecasting/tree/v0.9.2/docs/source under MIT License

{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% set allowed_methods = methods.copy() %}

   {% if "__init__" in members %}
   {% set idx_pop = allowed_methods.index("__init__") %}
   {% set tmp = allowed_methods.pop(idx_pop) %}
   {% endif %}

   {% if "__call__" in members %}
   {% set tmp = allowed_methods.append("__call__") %}
   {% endif %}

   {% if allowed_methods %}
   .. rubric:: {{ _('Methods') }}
   .. autosummary::
   {% for item in allowed_methods %}
      ~{{ name }}.{{ item }}
   {% endfor %}
   {% endif %}

   {% endblock %}

   {% block attributes %}
   {% set allowed_attributes = [] %}

   {% for item in attributes %}
   {% if not item.startswith("_") %}
   {% set tmp = allowed_attributes.append(item) %}
   {% endif %}
   {% endfor %}

   {% if allowed_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in allowed_attributes %}
      ~{{ name }}.{{ item }}
   {% endfor %}
   {% endif %}

   {% endblock %}
