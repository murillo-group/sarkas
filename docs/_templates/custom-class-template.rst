{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: {{ name }}_methods

   {% for item in methods %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
