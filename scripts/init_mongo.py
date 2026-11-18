"""Create the dedicated database and indexes used by live monitoring."""
from __future__ import annotations

import json

from ship_fault.mongo_store import MongoStore
from ship_fault.settings import RuntimeSettings


def main():
    settings = RuntimeSettings.from_env()
    store = MongoStore(settings.mongo_uri, settings.mongo_database)
    try:
        print(json.dumps(store.info(), indent=2))
    finally:
        store.close()


if __name__ == '__main__':
    main()
