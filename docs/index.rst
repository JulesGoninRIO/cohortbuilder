:notoc:

.. Cohort Builder documentation master file

.. module:: cb

Cohort Builder
==========================================================

**Date**: |today| **Version**: |version|

**Useful links**:
`Discovery (FHV) <http://soinsrv01.fhv.ch/login>`__ |
`Discovery (FHV) queues <http://10.128.24.77:7000/>`__ |
`SLIMS (FHV) <http://sfhvbiob01/slims/login>`__

:mod:`Cohort Builder <cb>` handles the data extraction and data preparation pipeline of the plateforme RIO
at the Jules-Gonin hospital.
The general aim of the software is to programmatically extract medical scans and patient data
from databases and image pool servers, treat the files, anonymize them, upload them to an external
web-based software (RetinAI Discovery), and download the images and analyses in a configurable way.
The final results will be used by other projects to train deep learning networks or test specific
scientific hypotheses.

---------------------------

.. image:: _static/img/graphs-overview.drawio.png
    :width: 80 %
    :align: center

---------------------------

The above figure shows an overview of the pipeline and the interactions with external sources at each
step. The external sources at the Jules-Gonin hospital are:

- Billing Records: Opale databases
- Medical Records: MediSIGHT databases
- Patient Consent: SLIMS database
- Raw Images: Heyex servers
- Data Management System: RetinAI Discovery
- Viewer: RetinAI Discovery
- Workspace: Swiss Ophthalmic Image Network (SOIN)

---------------------------

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting started
        :img-top: _static/img/index_getting_started.svg
        :class-card: intro-card
        :shadow: md

        Get started with the basic concepts and typical usages of Cohort Builder.

        +++

        .. button-ref:: gettingstarted
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the getting started guides

    .. grid-item-card::  User Guide
        :img-top: _static/img/index_user_guide.svg
        :class-card: intro-card
        :shadow: md

        You have mastered the Getting Started guides and are ready to explore all functionalities of Cohort Builder?
        Use the contributing guides to learn how to tailor the commands and arguments to your needs.

        +++

        .. button-ref:: userguide
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the user guide

    .. grid-item-card::  API reference
        :img-top: _static/img/index_api.svg
        :class-card: intro-card
        :shadow: md

        The reference guide contains a detailed description of the Cohort Builder API.
        The reference describes how the methods work and which parameters can
        be used. It assumes that you have a great understanding of the key concepts.

        +++

        .. button-ref:: api
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide

    .. grid-item-card::  Developer Guide
        :img-top: _static/img/index_contribute.svg
        :class-card: intro-card
        :shadow: md

        The contributing guidelines will guide you through the process of
        improving Cohort Builder.

        +++

        .. button-ref:: development
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the developper guide


:ref:`genindex`
.. :ref:`modindex`
.. :ref:`search`

.. toctree::
   :maxdepth: 2
   :hidden:

   source/gettingstarted/index
   source/userguide/index
   source/api/index
   source/development/index
