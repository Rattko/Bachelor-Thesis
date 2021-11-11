class NoModelError(Exception):
    """
    Exception to raise when AutoSklearn is unable
    to train a model within a given time limit.
    """
