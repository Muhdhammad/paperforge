import uuid

def get_uuid():
    return str(uuid.uuid4())

def batch_iterate(lst, batch_size):
    """Yield batches from a list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i: i + batch_size]