"""
This module includes custom exceptions for handling different scenarios related
to the Discovery API.
"""

class BadRequestException(Exception):
    """Exception for handling unvalid queries (ERROR 400)."""
    pass

class TokenTimeoutException(Exception):
    """Exception for handling access token expiry (ERROR 401)."""
    pass

class InvalidCredentialsException(Exception):
    """Exception for handling the invalid credentials problem."""
    pass

class TokenRefreshMaxAttemptsReached(Exception):
    """Exception for handling the cases where refreshing the token fails."""
    pass

class TokenAllCredentialsInvalid(Exception):
    '''Exception for handling the case where all attempted credentials are invalid.
    '''
    pass

class RequestMaxAttemptsReached(Exception):
    """Exception for handling the cases where sending a request fails."""
    pass

class RequestExpiredException(Exception):
    """Exception for handling expired requests (ERROR 403)."""
    pass

class UrlNotFoundException(Exception):
    """Exception for handling not found urls (ERROR 404)."""
    pass

class BadGatewayException(Exception):
    """Exception for handling bad gateway error (ERROR 502)."""
    pass

class GatewayTimeoutException(Exception):
    """Exception for handling gateway timeout error (ERROR 504)."""
    pass

class UnknownStatusCodeException(Exception):
    """Exception for handling unknown response status codes."""
    pass

class NoUrlPassedException(Exception):
    """Exception for handling ``None`` values passed for URL."""

class UnauthorizedCallException(Exception):
    """Exception for handling unauthorized calls to methods."""
