"""Create the dedicated external training collection without inserting samples."""
from pathlib import Path
import sys

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ship_fault.external_training import load_source_config


def main() -> None:
    source = load_source_config(Path(__file__).resolve().parents[1])
    client = MongoClient(source['uri'], serverSelectionTimeoutMS=3000)
    try:
        client.admin.command('ping')
        database = client[source['database']]
        if source['collection'] not in database.list_collection_names():
            database.create_collection(source['collection'])
        database[source['collection']].create_index([
            (source['fields']['session'], 1), (source['fields']['created_at'], 1)])
        print(f"Ready: {source['database']}.{source['collection']}")
    finally:
        client.close()


if __name__ == '__main__':
    main()
