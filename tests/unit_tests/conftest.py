import os
import signal


def pytest_sessionfinish(session, exitstatus):
    if exitstatus != 0:
        # Violently terminate process
        os.kill(os.getpid(), signal.SIGTERM)
