
UML diagrams
==============

- To generate UML diagrams of the project:

    .. code:: bash

        pyreverse ./src -A -o dot -d docs/_static/

- To render the generated files to svg:

    .. code:: bash

        dot docs/_static/classes.dot -Tsvg > docs/_static/img/classes.svg
        dot docs/_static/packages.dot -Tsvg > docs/_static/img/packages.svg

Modifying the diagrams
------------------------------

- Use the `Graphviz Interactive Preview`_ VS Code extension for rendering the ``.dot`` files interactively.

.. _`Graphviz Interactive Preview`: https://marketplace.visualstudio.com/items?itemName=tintinweb.graphviz-interactive-preview

- You can remove the isolated modules or the extra classes.

- To change the layout of the diagrams, change the order of the nodes or change the layout engine.

- The theme of the diagrams could be changed. Check out `Graphviz Gallery`_ to find out how.

.. _`Graphviz Gallery`: https://graphviz.org/gallery/

Themes
------------------------------

- You can use these settings for the package dependencies diagram:

    .. code::

        fontname = "Helvetica,Arial,sans-serif"
        graph [
            label = "Package dependencies\n\n"
            labelloc = t
            fontsize = 20
            rankdir = LR
            newrank = true
        ]
        node [
            style=filled
            pencolor="#00000044"
            shape=ellipse
        ]
        edge [
            arrowsize=0.5
            labeldistance=3
            labelfontcolor="#00000080"
            penwidth=2
            style=dotted
        ]

- You can use these settings for the class UML diagram:

    .. code::

        fontname="Helvetica,Arial,sans-serif"
        graph [
            label = "Class UML diagram\n\n"
            labelloc = t
            fontname = "Helvetica,Arial,sans-serif"
            fontsize = 20
            rankdir = LR
            newrank = true
        ]
        node [
            style=filled
            fillcolor="#ffd034ab"
            pencolor="#4a0000"
        ]
        edge [
            arrowsize=1
            labeldistance=3
        ]
