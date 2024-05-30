"""
This module includes the classes that manage the connection to a Discovery server.

.. note::
    Consider combining this module with `src.discovery.manager`.
"""

from __future__ import annotations

import io
import json
import pathlib
from functools import wraps
from time import sleep
from typing import Callable, TypeVar, Union

import requests
from requests.exceptions import RequestException
from loguru import logger
from typing_extensions import ParamSpec
from threading import Lock

from src.cohortbuilder.discovery.exceptions import (BadRequestException,
                                      InvalidCredentialsException,
                                      NoUrlPassedException,
                                      RequestExpiredException,
                                      TokenTimeoutException,
                                      RequestMaxAttemptsReached,
                                      TokenRefreshMaxAttemptsReached,
                                      UrlNotFoundException,
                                      UnknownStatusCodeException,
                                      GatewayTimeoutException,
                                      BadGatewayException,
                                      TokenAllCredentialsInvalid
                                      )
from src.cohortbuilder.discovery.queries import Q_FILE_UPLOAD, Q_FILES, get_raw
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.utils.helpers import log_errors

T = TypeVar('T')
P = ParamSpec('P')

class Token:
    """
    Class for managing the access token to Discovery.
    The access token is needed for sending requests to the Discovery API.

    Args:
        settings: The settings of the Discovery instance
          (``url``, ``url_dataset``, ``timeout``, ``anonymize``).
    """

    def __init__(self, settings: dict):
        # Instantiate the attributes
        #: API settings fetched from the global settings file
        self.settings: dict = settings
        #: URL for getting a new token
        self.url: str = self.settings['url'] + '/oauth/token'
        #: Discovery access token
        self.access_token: str = None
        #: Discovery refresh token
        self.refresh_token: str = None
        #: Login request status
        #: (``'OK'`` means that the token is ready,
        #: ``'INVALID'`` means that the token is invalid,
        #: ``None`` means that the token is not set yet.)
        self.status: str = None
        # Always start with attempting the first token
        self.token_index = 0
        # Number of different login pairs which failed to auth in a row. If this is ever equal to the number of login pairs we have, we fail.
        self.successive_invalid_logins = 0

        # Trun off CA verification warning
        # CHECK: Consider removing it
        requests.packages.urllib3.disable_warnings()

        # Refresh the token
        # TODO: Do the retries in a wrapper
        while self.status != 'OK':
            for attempt in range(Parser.settings['general']['token_refresh_max_attempts']):
                try:
                    self.refresh()
                    if self.status == 'OK':
                        self.successive_invalid_logins = 0
                    break
                except InvalidCredentialsException:
                    logger.trace(f"Attempt {attempt + 1} failed for refreshing the token with login {self.settings['login_pairs'][self.token_index]['login']} from {self.url}.")
                    sleep(1)
            else:
                logger.warning(f"Maximum attempts reached for refreshing the token with login {self.settings['login_pairs'][self.token_index]['login']} from {self.url}.")
                self.switch_creds()

    def switch_creds(self) -> None:
        """
        Switches to the next login pair, while doing checks for if we have tried all login pairs.
        """

        self.token_index = (self.token_index + 1) % len(self.settings['login_pairs']) # Cycle through the login pairs, while looping back if we reach the end.
        self.successive_invalid_logins += 1
        if self.successive_invalid_logins >= len(self.settings['login_pairs']):
            raise TokenAllCredentialsInvalid


    def refresh(self) -> None:
        """
        Initializes the token by sending a login request and updates its attributes.
        API URL, user authentication, and API client authentication information are fetched
        from the settings.
        """

        if self.is_valid():
            logger.trace('Skipped refreshing token: is already valid again.')
            self.successive_invalid_logins = 0
            self.status = 'OK' # Make sure the status is set to OK if the token is valid again.
            return # Do nothing if this token has already been refreshed while waiting for lock acquisition.

        active_creds = self.settings['login_pairs'][self.token_index]

        try:
            # Send the login request
            data = {
                'grant_type': 'password',
                'username': active_creds['login'],
                'password': active_creds['password']
            }
            response = requests.post(
                url=self.url,
                data=data,
                verify=False,
                auth=(self.settings['client_authentication'], self.settings['client_password']),
            ).json()

            # Update the attributes
            try:
                self.access_token = response['access_token']
                self.refresh_token = response['refresh_token']
                self.status = 'OK'
                self.successive_invalid_logins = 0
            except:
                self.reset()
                self.status = 'INVALID'

            # Catch MFA
            if 'challenge' in response:
                self.reset()
                self.status = 'MFA'

        except Exception as e:
            self.reset()
            self.status = 'INVALID'
            msg = f'{type(e).__name__}'
            logger.error(msg)
            logger.exception(f'{msg}: {e}')

        # Log the response status
        if self.status == 'OK':
            logger.info(f'{active_creds["login"]} at {self.settings["url"]}: Credentials accepted.')
        elif self.status == 'INVALID':
            logger.trace(f'Invalid credentials for user: {active_creds["login"]} at {self.settings["url"]}.')
            raise InvalidCredentialsException
        elif self.status == 'MFA':
            logger.debug('MFA')  # CHECK: Multi-factor authentication: What does this status mean?

    def is_valid(self) -> bool:
        '''
        Execute a super simple query (get version number) to check if we can authenticate to Discovery with this token.
        '''
        url = self.settings['url'] + '/api'
        headers = {
            'Authorization': 'Bearer %s' % self.access_token,
            'Content-Type': 'application/json'
        }

        # Send request and check the response
        response = requests.request(
            method='POST',
            url=url,
            headers=headers,
            json={'query': "query Metadata {\n  metadata {\n    version\n  }\n}\n", 'variables': dict(), 'operationName': 'Metadata'},
            verify=False,
        )

        return response.ok and 'error' not in response.json() # Can get status 200 with errors in response. This eliminates this possibility.

    def is_valid(self) -> bool:
        '''
        Execute a super simple query (get version number) to check if we can authenticate to Discovery with this token.
        '''
        url = self.settings['url'] + '/api'
        headers = {
            'Authorization': 'Bearer %s' % self.access_token,
            'Content-Type': 'application/json'
        }

        # Send request and check the response
        response = requests.request(
            method='POST',
            url=url,
            headers=headers,
            json={'query': "query Metadata {\n  metadata {\n    version\n  }\n}\n", 'variables': dict(), 'operationName': 'Metadata'},
            verify=False,
        )

        return response.ok and 'error' not in response.json() # Can get status 200 with errors in response. This eliminates this possibility.

    # UNUSED
    def refresh_old(self) -> None:
        """Tries to refresh the access token using the refresh token."""

        try:
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
            }

            response = requests.post(
                self.url,
                data=data,
                verify=False,
                auth=(self.settings['client_authentication'], self.settings['client_password'])
            ).json()
            self.access_token = response['access_token']
            # self.status = 'OK'
        except:
            self.access_token = None
            self.refresh_token = None
            self.status = 'INVALID'

    def reset(self) -> None:
        """Sets the attributes of the token class to None."""

        self.access_token = None
        self.refresh_token = None
        self.status = None

    # UNUSED
    def get_mfa(self, mfa_code: str) -> None:
        """
        Gets an access token associated to the credentials and a multifactor
        -authentication code.

        Args:
            mfa_code: Multi-factor authentication code.
        """

        try:
            data = {
                'grant_type': 'password',
                'username': self.settings['login'],
                'password': self.settings['password'],
                "mfa_code": mfa_code,
            }

            response = requests.post(
                self.url,
                json=data,
                verify=False,
                auth=(self.settings['client_authentication'], self.settings['client_password'])
            ).json()
            try:
                self.access_token = response['access_token']
                self.refresh_token = response['refresh_token']
                self.status = 'OK'
            except:
                self.reset()
                self.status = 'INVALID'
        except:
            pass

    def __repr__(self) -> str:
        return self.access_token

class Discovery:
    """
    Manages the communications with a Discovery instance.

    Args:
        instance: The name of the Discovery instance.
        gettoken: If ``True``, initializes a Discovery access token.

    Examples:
        >>> from src.parser import Parser
        >>> from src.discovery.discovery import Discovery
        >>>
        >>> ... # Get the settings and the arguments
        >>> Parser.store(args=args, settings=settings)
        >>>
        >>> discovery = Discovery(instance='fhv_jugo')
        >>> r = discovery.send_query(query=...)

    .. seealso::
        `DiscoveryManager <src.discovery.manager.DiscoveryManager>`
    """

    def __init__(self, instance: str):

        #: The name of the Discovery instance
        self.instance: str = instance
        #: The settings fetched from the global settings file
        self.settings: dict = Parser.settings['api'][instance]
        #: The current Discovery access token
        self.token: Token = Token(settings=self.settings)
        #: The multithreading lock so that the token is not refreshed excessively
        self.token_threading_lock = Lock()

    def __locked_token_refresh(self) -> None:
        with self.token_threading_lock:
            self.token.refresh()

    def __locked_credential_switch(self) -> None:
        with self.token_threading_lock:
            self.token.switch_creds()

    def _refresh_token_if_expired(func: Callable[P, T]) -> Callable[P, T]:
        """
        Decorator for allowing seamless refreshing of the access token if it expires
        before the call of the function. Calls the function once, if it raises the custom
        TokenTimeoutException exception, refreshes the token before calling the function
        a second time. If it fails again, re-raises the exception.
        """

        @wraps(func)
        def wrapper(self: Discovery, *args, **kwargs) -> T:
            try:
                return func(self, *args, **kwargs)
            except TokenTimeoutException:
                self.token.status = 'INVALID'
                while self.token.status != 'OK':
                    for attempt in range(Parser.settings['general']['token_refresh_max_attempts']):
                        try:
                            self.__locked_token_refresh()
                            return func(self, *args, **kwargs)
                        except InvalidCredentialsException:
                            logger.trace(f'Attempt {attempt + 1} failed for refreshing the token of Discovery {self}.')
                            sleep(1)
                        except TokenTimeoutException:
                            logger.trace(f'Attempt {attempt + 1} failed for refreshing the token of Discovery {self}.')
                            sleep(1)
                    else:
                        logger.trace(f'Switching logins for refreshing token of Discovery {self}.')
                        self.__locked_credential_switch()

        return wrapper

    def _handle_exceptions(func: Callable[P, T]) -> Callable[P, T]:
        """
        Decorator for handling request-related exceptions.
        """

        @wraps(func)
        def wrapper(self: Discovery, *args, **kwargs) -> T:
            for attempt in range(Parser.settings['general']['discovery_request_max_attempts']):
                try:
                    return func(self, *args, **kwargs)
                except BadRequestException as e:
                    logger.trace(f'Attempt {attempt+1} for {func.__name__} failed: {type(e).__name__}')
                    msg = f'Bad request encountered.'
                    logger.debug(msg)
                    logger.exception(f'{msg}: {e}')
                    break
                except TokenTimeoutException as e:
                    logger.trace(f'Attempt {attempt+1} for {func.__name__} failed: {type(e).__name__}')
                    msg = 'Unexpected TokenTimeoutException occured.'
                    logger.debug(msg)
                    logger.exception(f'{msg}: {e}')
                    self.token.refresh()
                except BadGatewayException as e:
                    logger.trace(f'Attempt {attempt+1} for {func.__name__} failed: {type(e).__name__}')
                    sleep(5)
                except GatewayTimeoutException as e:
                    logger.trace(f'Attempt {attempt+1} for {func.__name__} failed: {type(e).__name__}')
                    sleep(20)
                except UnknownStatusCodeException as e:
                    logger.trace(f'Attempt {attempt+1} for {func.__name__} failed: {type(e).__name__}')
                    msg = 'Unexpected response error occured.'
                    logger.debug(msg)
                    logger.exception(f'{msg}: {e}')
                except RequestException as e:
                    logger.trace(f'Attempt {attempt+1} for {func.__name__} failed: {type(e).__name__}')
                    sleep(2)

            else:
                raise RequestMaxAttemptsReached(f'Maximum attempts reached for {func.__name__}')

        return wrapper

    @staticmethod
    def _check_response(r: requests.Response):
        """
        Checks a request response by its status code.

        Args:
            r: Response to be checked.

        Raises:
            BadRequestException: If status code is 400.
            TokenTimeoutException: If status code is 401.
            RequestExpiredException: If status code is 403.
            UrlNotFoundException: If status code is 404.
            BadGatewayException: If status code is 502.
            GatewayTimeoutException: If status code is 504.
            UnknownStatusCodeException: If status code is 400 or above and not defined.
        """

        if r.ok:
            return

        elif r.status_code == 400:
            if r.request.method == 'GET':
                message = f'Bad request (ERROR {r.status_code}) for {r.url}: (REASON: {r.reason})'
            elif r.request.method == 'POST':
                message = f'Bad request for (ERROR {r.status_code}): (REASON: {r.reason}), (BODY: {r.request.body})'
            raise BadRequestException(message)

        elif r.status_code == 401:
            message = f'Token timed out (ERROR {r.status_code}) for {r.url}: (REASON: {r.reason})'
            raise TokenTimeoutException(message)

        elif r.status_code == 403:
            if r.request.method == 'GET':
                message = f'Request has expired (ERROR {r.status_code}) for {r.url}: (REASON: {r.reason})'
            elif r.request.method == 'POST':
                message = f'Request has expired (ERROR {r.status_code}): (REASON: {r.reason}), (BODY: {r.request.body})'
            raise RequestExpiredException(message)

        elif r.status_code == 404:
            if r.request.method == 'GET':
                message = f'URL not found (ERROR {r.status_code}) for {r.url}: (REASON: {r.reason})'
            elif r.request.method == 'POST':
                message = f'URL not found (ERROR {r.status_code}): (REASON: {r.reason}), (BODY: {r.request.body})'
            raise UrlNotFoundException(message)

        elif r.status_code == 502:
            if r.request.method == 'GET':
                message = f'No response from the server (ERROR {r.status_code}) for {r.url}: (REASON: {r.reason})'
            elif r.request.method == 'POST':
                message = f'No response from the server (ERROR {r.status_code}): (REASON: {r.reason}), (BODY: {r.request.body})'
            raise BadGatewayException(message)

        elif r.status_code == 504:
            if r.request.method == 'GET':
                message = f'No response from the server (ERROR {r.status_code}) for {r.url}: (REASON: {r.reason})'
            elif r.request.method == 'POST':
                message = f'No response from the server (ERROR {r.status_code}): (REASON: {r.reason}), (BODY: {r.request.body})'
            raise GatewayTimeoutException(message)

        else:
            if r.request.method == 'GET':
                message = f'Unknown response error (ERROR {r.status_code}) for {r.url}: (REASON: {r.reason})'
            elif r.request.method == 'POST':
                message = f'Unknown response error (ERROR {r.status_code}): (REASON: {r.reason}), (BODY: {r.request.body})'
            raise UnknownStatusCodeException(message)

    @log_errors
    @_handle_exceptions
    @_refresh_token_if_expired
    def send_query(self, query: str, variables: dict = None, name: str = None) -> dict:
        """
        Sends a query to the API and receives the response.

        Args:
            query: Query to be sent.
            variables: Variables to be sent with the query.
            name: Name of the operation.

        Returns:
            The response in JSON format.
        """

        url = self.settings['url'] + '/api'
        headers = {
            'Authorization': 'Bearer %s' % self.token.access_token,
            'Content-Type': 'application/json'  # MODIFY: Consider removing
        }

        # Send request and check the response
        response = requests.request(
            method='POST',
            url=url,
            headers=headers,
            json={'query': query, 'variables': variables, 'operationName': name},
            verify=False,
        )
        self._check_response(r=response)

        return response.json()

    @log_errors
    @_handle_exceptions
    @_refresh_token_if_expired
    def upload(self, file: Union[pathlib.Path, str], uuid: str) -> dict:
        """
        Uploads a file to Discovery through API.

        Args:
            file: The path of the file to be uploaded.
            uuid: UUID of the workbook that the file has to be uploaded to.

        Returns:
            The response of the request in JSON format.
        """

        # Get path object of the file
        file = pathlib.Path(file)
        assert file.exists()

        # Define the request settings
        url = self.settings['url'] + '/api'
        headers = {
            'Authorization': 'Bearer %s' % self.token.access_token,
        }
        operations = {
            'query': get_raw(Q_FILE_UPLOAD),
            'variables': {
                'file': None,
                'workbookUuid': uuid,
                'tags': [],
                'overwrite': None,
            },
            'operationName': 'UploadMutation',
        }
        map_ = {
            'sourcefile': ['variables.file']
        }

        # Send a request and check the response
        with open(file, 'rb') as f:
            response = requests.request(
                method='POST',
                url=url,
                headers=headers,
                data={
                    'operations': json.dumps(operations),
                    'map': json.dumps(map_)
                },
                files={
                    'sourcefile': (file.name, f, 'application/octet-stream')
                },
            )
        self._check_response(r=response)

        return response.json()

    @_handle_exceptions
    @_refresh_token_if_expired
    def download(self, url: Union[str, None], out: Union[pathlib.Path, str] = None) -> Union[None, io.BytesIO]:
        """
        Downloads content from a URL and returns the downloaded file or its path.

        Args:
            url: The URL to be downloaded. If ``None``, returns ``None``.
            out: The path of the output file. Defaults to ``None`` .

        Raises:
            NoUrlPassedException: If the passed URL is ``None``.

        Returns:
            ``None`` if an output file is passed, a buffered stream otherwise.
        """

        # Check the URL
        if not url:
            raise NoUrlPassedException

        # Send request and check the response
        response = requests.get(
            url=url,
            verify=False,
            timeout=Parser.settings['general']['request_timeout'],
            )
        self._check_response(r=response)

        # Write the response content
        if out:
            # In a file and return the file path
            with open(out, 'wb') as f:
                f.write(response.content)
            return out
        else:
            # In a bytes object and return it
            f = io.BytesIO()
            f.write(response.content)
            f.seek(0)
            return f

    # TODO: Return File(Entity) objects
    def get_files(self, n: int = None) -> list[str]:
        """Gets the last ``n`` files registered in Discovery."""

        if not n:
            n = self.send_query(query=(Q_FILES % (1, 0)))['data']['files']['totalCount']

        response = self.send_query(query=(Q_FILES % (n, 0)))
        uuids = [edge['node']['uuid'] for edge in response['data']['files']['edges']]

        return uuids

    def __str__(self):
        return self.instance
