"""Checks for the Rubin Science Pipelines and SLSim environment."""


def dependency_paths() -> dict[str, str]:
    """Return import locations of the main dependencies."""
    import slsim
    import lsst.afw.image
    import lsst.daf.butler

    return {
        "slsim": str(slsim.__file__),
        "lsst.afw.image": str(lsst.afw.image.__file__),
        "lsst.daf.butler": str(lsst.daf.butler.__file__),
    }
