Testing
===============

- The tests are located in ``tests/``. They are unused and not necessarily up to date.

- If you plan to add new tests, first get familiar with the
  `pytest library <https://docs.pytest.org>`_.

- To do all the tests:

    .. code:: bash

        pytest tests

- To do the integration tests:

    .. code:: bash

        pytest tests/integration

- To do the unit tests:

    .. code:: bash

        pytest tests/unit

- To skip the slow tests:

    .. code:: bash

        pytest tests -m 'not slow'
