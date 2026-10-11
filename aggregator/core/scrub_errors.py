"""The scrubber's fail-closed error, in a module of its own (see scrub.py)."""


class ScrubUnavailableError(RuntimeError):
    """The secret scanner could not run, so the text is NOT known to be clean.

    FAIL-CLOSED CONTRACT: a caller that gets this must neither store nor
    return the text it asked about. Ingest lets it propagate, which rolls the
    write back and fails the source loudly (see ``imports/runner.py``, which
    treats it as a broken sink rather than a poison record); the MCP surface
    returns an error instead of results. The message names the failure only —
    never any of the scanned text.
    """
