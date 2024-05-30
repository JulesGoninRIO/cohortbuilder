"""
This module includes queries that can be sent to the Discovery servers.
"""

from io import StringIO

#: Query for getting the available projects
Q_PROJECTS: str = """
query ProfileQuery {
  profile {
    projects {
      uuid
      title
    }
  }
}
"""

#: Query for fetching the workbooks in a project
Q_PROJECT_WORKBOOKS: str = """
query {
    workbooks (
        projectUuid: "%s",
        first: %d,
        after: %d
    ) {
        totalCount
        totalPages
        edges{
            cursor
            node{
                uuid
                name
                currentPermission{
                    read
                    write
                    edit
                }
            }
        }
    }
}
"""

#: Query for fetching the patients in a workbook
Q_WORKBOOK_PATIENTS: str = """
query {
    patientSearchFeed(
        search: {workbookUuid: "%s"},
        first: %d,
        after: %d
    ) {
    totalCount
    totalPages
    edges {
      cursor
      node {
        uuid
        name
        surname
        patientId
        birthdate
        sex
        studies(first: 1, workbookUuid: "%s") {
            totalCount
        }
      }
    }
  }
}
"""

#: Query for copying all the studies of a patient to another workbook
Q_PATIENT_MOVE: str = """
mutation {
    addAllPatientDatasetsToWorkbook(
        input: {
            patientUuids: ["%s"],
            patientWorkbookUuid: "%s",
            workbookUuid: "%s"
        }
    ) {
        uuid
    }
}
"""

# UNUSED
# NOTE: The response will be an error if the same patient (pid, name, surname) exists in the same workbook.
# NOTE: You should merge this patient with an existing one in the case above
#: Query for editing a patient
Q_PATIENT_EDIT: str = """
mutation {
  editPatient(input: {
		birthdate: "%s",
		middleName: "%s",
		name: "%s",
		patientId: "%s",
		sex: "%s",
		surname: "%s",
		uuid: "%s",
		workbookUuid: "%s"
}
  }) {
    uuid
  }
}
"""

# UNUSED
# NOTE: The sourceUuids will be absorbed by uuid.
# NOTE: The patient information of the patient with uuid will remain.
#: Query for merging multiple patients into a single patient
Q_PATIENT_MERGE: str = """
mutation {
  mergePatient(
    uuid: "%s",
    sourceUuids: ["%s"]
  ) {
    uuid
  }
}
"""

#: Query for unlinking a patient from a workbook
Q_PATIENT_UNLINK: str = """
mutation RemoveAllPatientDatasetsFromWorkbook($input: RemoveAllPatientDatasetsFromWorkbookInput!) {
  removeAllPatientDatasetsFromWorkbook(input: $input) {
    uuid
  }
}
"""

#: Query for deleting a patient
#: A deleted Patient will not be returned by queries and cannot be edited.
#: Note: use the undeletePatient mutation to restore a soft deleted patient. Requires authentication.
#: input SoftDeletePatientInput {uuid: String!, workbookUuid: String!}
Q_PATIENT_DELETE: str = """
mutation SoftDeletePatient($input: SoftDeletePatientInput!) {
  softDeletePatient(input: $input)
}
"""

#: Query for fetching the studies of a patient in a workbook
Q_PATIENT_STUDIES: str = """
query {
    patient(uuid: "%s", workbookUuid: "%s") {
        studies(
            first: %d,
            after: %d,
            workbookUuid: "%s"
        ) {
            edges {
                cursor
                node {
                    uuid
                    studyId
                    studyDatetime
                    modalities
                }
            }
        }
    }
}
"""


# NOTE: It will contain child datasets (e.g., segmentations) as well
#: Query for fetching the info of datasets of a patient
Q_PATIENT_DATASETS_SHORT: str = """
query {
    patient(uuid: "%s", workbookUuid: "%s") {
        datasets(
            first: %d,
            after: %d,
            workbookUuid: "%s"
        ) {
            totalCount
            totalPages
            edges {
                cursor
                node {
                    uuid
                    status
                    purpose
                    parentFile{
                        uuid
                        filename
                        extension
                        signedUrl
                    }
                }
            }
        }
    }
}
"""

#: Query for copying all the datasets of a study to another workbook
Q_STUDY_MOVE: str = """
mutation {
    addAllStudyDatasetsToWorkbook (
        input: {
            studyUuids: ["%s"],
            studyWorkbookUuid: "%s",
            workbookUuid: "%s"
        }
    ) {
        uuid
    }
}
"""

#: Query for deleting a study
#: The softDeleteStudy mutation marks a study as deleted.
#: A soft-deleted study is not visible to users, but can be restored using the undeleteStudy mutation. Requires authentication.
#: input SoftDeleteStudyInput {uuid: String!, workbookUuid: String!}
Q_STUDY_DELETE: str = """
mutation DeleteStudyMutation($input: SoftDeleteStudyInput!) {
  softDeleteStudy(input: $input)
}
"""

# Input of type
# {"input": {"studyUuids": [String!], "workbookUuid": String!}}
Q_STUDY_UNLINK: str = """
mutation RemoveAllStudyDatasetsFromWorkbook($input: RemoveAllStudyDatasetsFromWorkbookInput!) {
  removeAllStudyDatasetsFromWorkbook(input: $input) {
    uuid
  }
}
"""

#: Query for fetching the datasets in a study
Q_STUDY_DATASETS: str = """
query {
    study(uuid: "%s", workbookUuid: "%s") {
        datasets(
            datasetType: %s,
            first: %d,
            after: %d,
            workbookUuid: "%s",
            leavesOnly: false
        ) {
            totalCount
            totalPages
            edges {
                cursor
                node {
                    uuid
                    purpose
                    status
                    owner{
                        uuid
                        firstName
                        lastName
                    }
                    createdAt
                    updatedAt
                    laterality
                    tags
                    device
                    manufacturer
                    acquisitionDatetime
                    seriesDatetime
                    signedUrl
                    thumbnail
                    parentFile{
                        uuid
                        filename
                        extension
                        signedUrl
                    }
                    children(workbookUuid: "%s") {
                        uuid
                        layers{
                            uuid
                            name
                            layerType
                            biomarkers {
                                title
                                volume
                            }
                            content
                        }
                    }
                    layers {
                        uuid
                        name
                        layerType
                        scanType
                        scanVariant
                        shape
                        scale
                        spacing
                        content
                    }
                }
            }
        }
    }
}
"""

#: Query for fetching the URLs of a dataset
Q_DATASET_REFRESH: str = """
query {
    dataset(uuid: "%s", workbookUuid: "%s") {
        signedUrl
        thumbnail
        parentFile{
            signedUrl
        }
        children(workbookUuid: "%s") {
            layers{
                uuid
                content
            }
        }
        layers {
            uuid
            content
        }
    }
}
"""

#: Query for copying a dataset from a workbook to another workbook
Q_DATASET_MOVE: str = """
mutation {
    addDatasetToWorkbook (
        input: {
            datasetUuids: ["%s"],
            workbookUuid: "%s",
            fromWorkbookUuid: "%s"
        }
    ) {
        uuid
    }
}
"""

#: Query for unlinking a dataset from a workbook
#: Example input = {'workbookUuid': wb.uuid, 'datasetUuids': [ds1.uuid, ds2.uuid]}
Q_DATASET_UNLINK: str = """
mutation RemoveDatasetMutation($input: AddDatasetInput!) {
  removeDatasetFromWorkbook(input: $input)
}
"""

#: Query for deleting a dataset
#: A soft-deleted dataset is not visible to users, but can be restored using the undeleteDataset mutation. Requires authentication.
#: input SoftDeleteDatasetInput {uuid: String!, workbookUuid: String!}
Q_DATASET_DELETE: str = """
mutation SoftDeleteDataset($input: SoftDeleteDatasetInput!) {
  softDeleteDataset(input: $input)
}
"""

# UNUSED
#: Query for getting all the files that are registered in Discovery
Q_FILES: str = """
query files {
    files(first: %d, after: %d) {
        totalCount
        totalPages
        edges{
            cursor
            node{
                uuid
                filename
                status
                createdAt
                datasets(first: 1, after: 0) {
                    totalCount
                }
            }
        }
    }
}
"""

Q_FILE_BY_NAME = '''
query FilesQuery(
    $first: Int,
    $after: Int,
    $status: FileStatus,
    $input: FindFilesInput
) {
  files(first: $first, after: $after, status: $status, input: $input) {
    totalPages
    pageInfo {
      hasNextPage
    }
    edges {
      cursor
      node {
        uuid
        filename
        size
        createdAt
        deletedAt
        status
        datasetCount
        owner {
          uuid
          firstName
          lastName
        }
      }
    }
  }
}
'''

#: Query for uploading a local file
Q_FILE_UPLOAD: str = """
mutation UploadMutation(
    $file: Upload!,
    $tags: [String!],
    $workbookUuid: String,
    $overwrite: JSON
) {
    upload(
        file: $file,
        tags: $tags,
        workbookUuid: $workbookUuid,
        overwrite: $overwrite
    ) {
        uuid,
        isDuplicate,
        status
    }
}
"""

#: Query for getting the status of a file and its jobs
Q_FILE_INFO: str = """
query {
    file(uuid: "%s") {
        uuid
        filename
        status
        createdAt
        datasets(first: 1, after: 0) {
            totalCount
        }
    }
}
"""

#: Query for getting the datasets of a file
Q_FILE_ACQUISITIONS: str = """
query {
  file(uuid: "%s") {
    datasets(first: %d, after: %d) {
        totalCount
        totalPages
        edges {
            cursor
            node {
                uuid
                status
            }
        }
    }
  }
}
"""

#: Query for deleting a file from Discovery (clears up space)
#: This mutation soft-deletes a file from the system. Requires authentication.
#: A soft-deleted file is not visible to the user, but can be restored using the undeleteDataset mutation. Requires authentication.
Q_FILE_DELETE: str = """
mutation SoftDeleteFile($uuid: String!) {
  softDeleteFile(uuid: $uuid)
}
"""

#: Query for getting the jobs in a dataset of a file
Q_ACQUISITION_JOBS: str = """
query {
    jobs(
      datasetUuid: "%s",
      fileUuid: "%s",
      includeChildDatasetJobs: true,
      workbookUuid: null
    ) {
        uuid
        name
        status
        startedAt
        createdAt
        tasks {
            uuid
            path
            status
            processor {
                uuid
                name
                version
            }
            message
            io {
                status
                message
                inputFile {
                    uuid
                }
                inputDataset {
                    uuid
                }
                outputDataset {
                    uuid
                }
            }
        }
    }
}
"""

Q_ACQUISITION_STARTJOB = """
mutation {
  startSimpleJob(
    datasetUuid: "%s",
    processorUuid: "%s",
    withPostprocessor: true
  ) {
    uuid
  }
}
"""

#: Query for getting the processor information of a process
Q_PROCESSOR =  """
query {
    processors(input: { name: "%s" }) {
        uuid
        name
        version
    }
}
"""

class QueryBuilder:
    """
    Class for automatically building a query from a configuration file.

    This class is not used in the current version.
    Consider reusing it to get the minimum needed queries instead
    of the full query.
    """

    Q_START =\
    """    query {
            study(uuid: "%s", workbookUuid: "%s") {
                datasets(first: %d, workbookUuid: "%s") {
                    totalCount
                    edges {
                        node {
                            uuid
                            laterality
                            acquisitionDatetime """

    Q_CHILDREN = """
                            children(workbookUuid: "%s") {
                                layers{
                                    uuid
                                    name
                                    content
                                    biomarkers{
                                        title
                                        volume
                                    }
                                }
                            }"""

    Q_LAYERS = """
                            layers {
                                uuid
                                name
                                content
                                shape
                                scale
                                spacing
                                type
                                scanType
                                scanVariant
                            }"""

    Q_END = """
                        }
                    }
                }
            }
        } """

    def __init__(self):
        self.query = StringIO()
        self.query.write(self.Q_START)
        self.query.write(self.Q_LAYERS)

    def _add_children(self) -> None:
        self.query.write(self.Q_CHILDREN)

    def _add_arg(self, arg: str) -> None:
        # NOTE: It is not necessary to have this structure,
        # but it makes it easier to read.
        arg_f = """
                        {}""".format(arg)
        self.query.write(arg_f)

    def _close_query(self) -> None:
        self.query.write(self.Q_END)

    def build(self, configs: dict) -> None:
        """
        Builds the query and stores it in the attribute.

        Args:
            configs: Configurations.
        """

        self._add_arg('device')
        self._add_arg('manufacturer')

        if any([
            configs['types']['biomarkers'],
            configs['types']['thicknesses'],
            configs['types']['volumes'],
            configs['types']['segmentation'],
            configs['types']['projection_images'],
            configs['types']['thickness_images'],
        ]):
            self._add_children()

        if configs['types']['thumbnail']:
            self._add_arg('thumbnail')

        if configs['types']['h5']:
            self._add_arg('signedUrl')

        if configs['types']['dicom'] or configs['types']['e2e']:
            parentFile = '''
                        parentFile{
                            filename
                            extension
                            signedUrl
                        } '''
            self.query.write(parentFile)

        self._close_query()

    def read_query(self) -> str:
        """
        Reads the current query.

        Returns:
            The stored query.
        """

        return self.query.getvalue()

def get_raw(query: str, n_spaces: int = 4) -> str:
    """Removes tabs, line breaks, and spaces from a given query.

    Args:
        query: The given query.
        n_spaces: If given, spaces that are repeated n times (instead of tab)
          will also be removed. Defaults to 4.

    Returns:
        The query without tabs and line breaks.
    """

    query = ''.join(query.split('\n'))
    query = ''.join(query.split('\t'))
    query = ''.join(query.split(' ' * n_spaces))

    return query
