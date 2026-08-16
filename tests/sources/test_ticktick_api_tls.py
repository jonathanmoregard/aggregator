"""TickTick's HTTPS calls must carry trust anchors of their own.

THE INCIDENT. 2026-08-15 21:59:01, first live timer run:
``CERTIFICATE_VERIFY_FAILED - self-signed certificate in certificate chain``
against ``api.ticktick.com``. Not a MITM and not a code bug at the time — the
systemd unit set neither ``SSL_CERT_FILE`` nor ``NIX_SSL_CERT_FILE``, so
OpenSSL fell back to an absent ``/etc/ssl/cert.pem`` and a non-hashed
``/etc/ssl/certs``, and ``ssl.create_default_context()`` came up with
``{'x509': 0, 'x509_ca': 0}`` — zero trusted CAs. TickTick is the only source
that speaks HTTPS from Python (github shells out to ``gh``), which is why it
alone broke.

The unit's environment is fixed on a separate track. This is the
belt-and-braces half: the process carries its own CA bundle, so the source
keeps working on a machine whose trust store is empty or misconfigured.
"""
from __future__ import annotations

import ssl
import urllib.request

from aggregator.sources import ticktick_api


def test_module_ssl_context_actually_has_trust_anchors():
    """The measured failure was ZERO loaded CAs. Assert the number, not the call.

    ``create_default_context()`` returns a perfectly valid object on a machine
    with no trust store at all — it just verifies nothing successfully. So the
    assertion has to be on ``cert_store_stats``, which is the number that was
    0 on the failing run and is what actually decides whether the handshake can
    succeed.
    """
    context = ticktick_api._ssl_context()
    assert isinstance(context, ssl.SSLContext)
    assert context.cert_store_stats()["x509_ca"] > 0


def test_module_ssl_context_still_verifies():
    """Carrying our own CAs must not become a way to stop checking them."""
    context = ticktick_api._ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_the_opener_uses_that_context():
    """The context has to be wired into the opener the module actually calls.

    A context built and never installed is the shape of a fix that passes its
    own unit test and changes nothing in production, so this asserts the object
    identity on the handler ``_OPENER`` will use.
    """
    https = [
        h
        for h in ticktick_api._OPENER.handlers
        if isinstance(h, urllib.request.HTTPSHandler)
    ]
    assert https, "the module opener has no HTTPSHandler at all"
    assert any(
        getattr(h, "_context", None) is ticktick_api._SSL_CONTEXT for h in https
    )


def test_https_only_redirect_handler_survives_the_extra_handler():
    """Adding an HTTPSHandler must not displace the hardening already there."""
    assert any(
        isinstance(h, ticktick_api._HttpsOnlyRedirectHandler)
        for h in ticktick_api._OPENER.handlers
    )
