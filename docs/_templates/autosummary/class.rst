{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block attributes %}
{% if attributes %}

.. rubric:: Attributes

.. autosummary::
    :toctree:
    :template: autosummary/base.rst
    {% for item in all_attributes %}
    {%- if not item.startswith('_') %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}

{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}

.. rubric:: Methods

.. autosummary::
    :toctree:
    :template: autosummary/base.rst
    :nosignatures:
    {% for item in all_methods %}
    {%- if not item.startswith('_') or item in ['__call__'] %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}

{% endif %}
{% endblock %}
