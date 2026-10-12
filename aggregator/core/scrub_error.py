"""The scrubber's fail-closed error, in a module of its own.

Its own module so it stays the same class when tests ``importlib.reload``
``aggregator.core.scrub``: callers catch it by identity.
"""


class ScrubUnavailableError(Exception):
    """The secret scan did not complete: the text must not be stored.

    The message names the failure only, never the text."""
