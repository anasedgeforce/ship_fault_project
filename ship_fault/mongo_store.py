"""Store live readings and fault candidates in two MongoDB collections."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from bson import Binary, ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from .settings import CHANNELS


class MongoStore:
    LIVE_COLLECTION = 'live_data'
    FAULT_COLLECTION = 'live_faults'

    def __init__(self, uri: str, database: str):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=2000,
                                  connectTimeoutMS=2000, socketTimeoutMS=5000)
        try:
            self.ping()
        except Exception:
            self.client.close()
            raise
        self.database = database
        self.db = self.client[database]
        for name in (self.LIVE_COLLECTION, self.FAULT_COLLECTION):
            if name not in self.db.list_collection_names():
                self.db.create_collection(name)
        self.db[self.LIVE_COLLECTION].create_index([('session_id', ASCENDING),
                                                    ('start_timestamp_seconds', ASCENDING)], unique=True)
        self.db[self.LIVE_COLLECTION].create_index([('stream_id', ASCENDING), ('created_at', DESCENDING)])
        self.db[self.LIVE_COLLECTION].create_index([('equipment_id', ASCENDING), ('created_at', DESCENDING)])
        self.db[self.LIVE_COLLECTION].create_index([('condition_status', ASCENDING), ('created_at', DESCENDING)])
        self.db[self.FAULT_COLLECTION].create_index([('session_id', ASCENDING),
                                                     ('window_index', ASCENDING)], unique=True)
        self.db[self.FAULT_COLLECTION].create_index([('stream_id', ASCENDING), ('created_at', DESCENDING)])
        self.db[self.FAULT_COLLECTION].create_index([('equipment_id', ASCENDING), ('created_at', DESCENDING)])
        self.db[self.FAULT_COLLECTION].create_index([('event_type', ASCENDING)])

    def close(self):
        self.client.close()

    def ping(self):
        self.client.admin.command('ping')

    def store_batch(self, session_id: str, stream_id: str, timestamps, channels, source: dict | None) -> dict:
        times = np.asarray(timestamps, dtype='<f8')
        values = np.asarray(channels, dtype='<f4')
        if values.shape != (len(times), len(CHANNELS)):
            raise ValueError('Mongo batch must have six channels')
        document = {
            'stream_id': stream_id,
            'session_id': session_id,
            'equipment_id': source.get('equipment_id') if source else None,
            'created_at': datetime.now(timezone.utc),
            'start_timestamp_seconds': float(times[0]),
            'end_timestamp_seconds': float(times[-1]),
            'recording_duration_seconds': float(times[-1] - times[0] + 1 / 4000),
            'recording_time_basis': 'sensor_seconds',
            'fs': 4000,
            'total_samples': int(len(times)),
            'dtype': 'float32_le',
            'model_input_order': list(CHANNELS),
            'declared_channel_names': source.get('channel_names') if source else None,
            'declared_channel_units': source.get('channel_units') if source else None,
            'mapping_status': source.get('mapping_status') if source else 'unverified',
            'cycle_key': source.get('cycle_key') if source else None,
            'live_label': source.get('live_label') if source else None,
            'record_type': source.get('record_type') if source else None,
            'source_timestamp_ms': source.get('source_timestamp_ms') if source else None,
            'condition_status': 'pending',
            'processing_state': 'saved',
            'recovery_model': source.get('_recovery_model') if source else None,
            'window_results': [],
            'faults_detected': [],
            'timestamps_bin': Binary(times.tobytes()),
            'channel_data_bin': {name: Binary(values[:, index].tobytes())
                                 for index, name in enumerate(CHANNELS)},
        }
        try:
            document_id = self.db[self.LIVE_COLLECTION].insert_one(document).inserted_id
        except DuplicateKeyError:
            existing = self.db[self.LIVE_COLLECTION].find_one(
                {'session_id': session_id, 'start_timestamp_seconds': float(times[0])})
            compared_fields = ('stream_id', 'equipment_id', 'end_timestamp_seconds', 'fs',
                               'total_samples', 'dtype', 'model_input_order',
                               'declared_channel_names', 'declared_channel_units',
                               'cycle_key', 'live_label', 'record_type', 'source_timestamp_ms',
                               'timestamps_bin', 'channel_data_bin')
            if (not existing or existing.get('condition_status') != 'pending'
                    or any(existing.get(key) != document.get(key) for key in compared_fields)):
                raise ValueError('duplicate live batch differs from the saved pending batch')
            document_id = existing['_id']
        return {'database': self.database, 'collection': self.LIVE_COLLECTION,
                'document_id': str(document_id), 'fs': 4000,
                'total_samples': int(len(times)), 'dtype': 'float32_le'}

    def load_batch(self, reference: dict, session_id: str, stream_id: str):
        """Decode the saved sensor batch used by live inference."""
        if (reference.get('database') != self.database
                or reference.get('collection') != self.LIVE_COLLECTION):
            raise ValueError('saved batch reference belongs to another collection')
        document = self.db[self.LIVE_COLLECTION].find_one(
            {'_id': ObjectId(reference['document_id']),
             'session_id': session_id, 'stream_id': stream_id})
        if document is None:
            raise ValueError('saved live batch was not found')
        count = document['total_samples']
        if (not isinstance(count, int) or count < 1 or document['fs'] != 4000
                or document['dtype'] != 'float32_le'
                or document['model_input_order'] != list(CHANNELS)):
            raise ValueError('saved live batch has an incompatible sensor schema')
        raw_times = document['timestamps_bin']
        if len(raw_times) != count * 8:
            raise ValueError('saved live batch has damaged timestamps')
        times = np.frombuffer(raw_times, dtype='<f8').copy()
        axes = []
        for name in CHANNELS:
            raw_axis = document['channel_data_bin'].get(name)
            if raw_axis is None or len(raw_axis) != count * 4:
                raise ValueError(f'saved live batch has damaged {name} samples')
            axes.append(np.frombuffer(raw_axis, dtype='<f4'))
        values = np.column_stack(axes)
        if not np.isfinite(times).all() or not np.isfinite(values).all():
            raise ValueError('saved live batch contains nonfinite samples')
        return times, values

    def store_results(self, reference: dict, session_id: str, stream_id: str,
                      predictions: list[dict], alerts: list[dict]):
        if (reference.get('database') != self.database
                or reference.get('collection') != self.LIVE_COLLECTION):
            raise ValueError('live result reference does not belong to this MongoDB collection')
        document_id = ObjectId(reference['document_id'])
        alerts_by_window = {alert['window_index']: alert for alert in alerts}
        for item in predictions:
            family = item.get('family')
            detail = item.get('detailed_condition')
            if not family or not detail:
                continue
            family_name, subtype = family['label'], detail['label']
            independent = item.get('independent_detailed_condition')
            candidate_source = 'family_constrained_prediction'
            if family_name == 'normal' and subtype.split('__', 1)[0] == 'normal':
                if independent and independent['label'].split('__', 1)[0] != 'normal':
                    family_name, subtype = independent['label'].split('__', 1)[0], independent['label']
                    candidate_source = 'independent_detailed_condition'
            is_fault = family_name != 'normal' or subtype.split('__', 1)[0] != 'normal'
            alert = alerts_by_window.get(item['window_index'])
            if alert and not is_fault:
                raise ValueError('normal prediction cannot create a fault alert')
            if alert and (alert['fault_type'] != family_name or alert['fault_subtype'] != subtype):
                raise ValueError('persistent alert does not match the fault candidate')
            event = {
                'stream_id': stream_id, 'session_id': session_id,
                'equipment_id': item.get('equipment_id'),
                'window_index': item['window_index'],
                'timestamp_seconds': item['end_seconds'],
                'event_type': 'fault_candidate' if is_fault else 'normal_prediction',
                'is_fault': is_fault, 'condition_label': subtype,
                'fault_family': family_name, 'fault_type': family_name, 'fault_subtype': subtype,
                'decision': 'needs_review' if candidate_source == 'independent_detailed_condition' else item['decision'],
                'review_reasons': sorted(set(item.get('review_reasons', []) +
                    (['family_detail_disagree'] if candidate_source == 'independent_detailed_condition' else []))),
                'candidate_source': candidate_source,
                'primary_family': family, 'primary_detailed_condition': detail,
                'independent_detailed_condition': independent,
                'family_score': family['score'] if candidate_source != 'independent_detailed_condition' else None,
                'subtype_score': independent['score'] if candidate_source == 'independent_detailed_condition' else detail['score'],
                'persistent_alert': bool(alert),
                'consecutive_windows': alert['consecutive_windows'] if alert else None,
                'mongo_references': item.get('mongo_references', []),
                'family_model_sha256': item.get('family_model_sha256'),
                'detail_model_sha256': item.get('detail_model_sha256'),
                'source_timestamp_ms': (item.get('source') or {}).get('source_timestamp_ms'),
                'created_at': datetime.now(timezone.utc),
            }
            self.db[self.FAULT_COLLECTION].replace_one(
                {'session_id': session_id, 'window_index': item['window_index']},
                event, upsert=True)
        results_by_batch = {document_id: []}
        for item in predictions:
            for source_ref in item.get('mongo_references', []):
                if (source_ref.get('database') != self.database
                        or source_ref.get('collection') != self.LIVE_COLLECTION):
                    raise ValueError('window references another MongoDB collection')
                results_by_batch.setdefault(ObjectId(source_ref['document_id']), []).append(item)
        for batch_id, new_results in results_by_batch.items():
            query = {'_id': batch_id, 'session_id': session_id, 'stream_id': stream_id}
            existing = self.db[self.LIVE_COLLECTION].find_one(query)
            if existing is None:
                raise ValueError('saved live batch was not found for its results')
            merged = {item['window_index']: item for item in existing.get('window_results', [])}
            merged.update({item['window_index']: item for item in new_results})
            windows = [merged[key] for key in sorted(merged)]
            faults = {(item['fault_type'], item['fault_subtype']) for item in windows
                      if item['decision'] == 'classified' and item['fault_type'] not in (None, 'normal')}
            if faults:
                condition_status = 'fault'
            elif any(item['decision'] == 'invalid_input' for item in windows):
                condition_status = 'invalid_input'
            elif any(item['decision'] == 'needs_review' for item in windows):
                condition_status = 'needs_review'
            elif any(item['decision'] == 'classified' and item['fault_type'] == 'normal' for item in windows):
                condition_status = 'normal'
            else:
                condition_status = 'pending'
            updated = self.db[self.LIVE_COLLECTION].update_one(
                query, {'$set': {'condition_status': condition_status,
                                 'window_results': windows,
                                 'faults_detected': [{'fault_type': family, 'fault_subtype': subtype}
                                                     for family, subtype in sorted(faults)],
                                 'processed_at': datetime.now(timezone.utc)}})
            if updated.matched_count != 1:
                raise ValueError('saved live batch disappeared while saving its results')
        self.db[self.LIVE_COLLECTION].update_one(
            {'_id': document_id, 'session_id': session_id, 'stream_id': stream_id},
            {'$set': {'processing_state': 'processed'}})

    def active_stream_ids(self):
        return self.db[self.LIVE_COLLECTION].distinct('stream_id')

    def close_session(self, session):
        latest = self.db[self.LIVE_COLLECTION].find_one(
            {'session_id': session.session_id, 'stream_id': session.stream_id},
            sort=[('start_timestamp_seconds', DESCENDING)])
        if latest is not None:
            self.db[self.LIVE_COLLECTION].update_one(
                {'_id': latest['_id']}, {'$set': {'session_closed': True}})

    def recover_stream(self, session):
        """Rebuild an unclosed session from raw data; writes are idempotent by window ID."""
        latest = self.db[self.LIVE_COLLECTION].find_one(
            {'stream_id': session.stream_id}, sort=[('created_at', DESCENDING), ('_id', DESCENDING)])
        if latest is None or latest.get('session_closed'):
            return False
        query = {'stream_id': session.stream_id, 'session_id': latest['session_id']}
        # Check the complete session before overwriting any results. Old batches
        # without a pinned model need explicit operator review, not guessed replay.
        for batch in self.db[self.LIVE_COLLECTION].find(query, {'recovery_model': 1}):
            if batch.get('recovery_model') != session.recovery_model:
                raise ValueError('Saved stream model or verification settings differ; recovery needs review')
        session.session_id = latest['session_id']
        for batch in self.db[self.LIVE_COLLECTION].find(query).sort('start_timestamp_seconds', ASCENDING):
            reference = {'database': self.database, 'collection': self.LIVE_COLLECTION,
                         'document_id': str(batch['_id']), 'fs': batch['fs'],
                         'total_samples': batch['total_samples'], 'dtype': batch['dtype']}
            times, values = self.load_batch(reference, session.session_id, session.stream_id)
            source = {'equipment_id': batch.get('equipment_id'),
                      'channel_names': batch.get('declared_channel_names'),
                      'channel_units': batch.get('declared_channel_units'),
                      'mapping_status': batch.get('mapping_status'),
                      'sample_rate_hz': batch['fs'], 'mongo': reference}
            for field in ('cycle_key', 'live_label', 'record_type', 'source_timestamp_ms'):
                source[field] = batch.get(field)
            result = session.push_batch(times, values, source)
            self.store_results(reference, session.session_id, session.stream_id,
                               result['predictions'], result['alerts'])
        session.recovered_from_mongo = True
        return True

    def info(self) -> dict:
        return {'database': self.database,
                'collections': [self.LIVE_COLLECTION, self.FAULT_COLLECTION],
                'live_data_count': self.db[self.LIVE_COLLECTION].estimated_document_count(),
                'live_fault_count': self.db[self.FAULT_COLLECTION].count_documents(
                    {'event_type': {'$ne': 'normal_prediction'}}),
                'live_normal_count': self.db[self.FAULT_COLLECTION].count_documents(
                    {'event_type': 'normal_prediction'})}

    def history(self, stream_id: str, limit: int) -> dict:
        batches = list(self.db[self.LIVE_COLLECTION]
                       .find({'stream_id': stream_id}, {'window_results': 1, 'session_id': 1})
                       .sort([('created_at', DESCENDING), ('_id', DESCENDING)]).limit(limit))
        by_window = {}
        for batch in reversed(batches):
            for prediction in batch.get('window_results', []):
                session_id = prediction.get('session_id', batch.get('session_id'))
                key = (session_id, prediction['window_index'])
                by_window[key] = {**prediction, 'session_id': session_id}
        predictions = list(by_window.values())[-limit:]
        candidates = list(self.db[self.FAULT_COLLECTION]
                      .find({'stream_id': stream_id, 'event_type': {'$ne': 'normal_prediction'}},
                            {'_id': 0, 'created_at': 0})
                      .sort([('created_at', DESCENDING), ('_id', DESCENDING)]).limit(limit))
        normal_predictions = list(self.db[self.FAULT_COLLECTION]
                      .find({'stream_id': stream_id, 'event_type': 'normal_prediction'},
                            {'_id': 0, 'created_at': 0})
                      .sort([('created_at', DESCENDING), ('_id', DESCENDING)]).limit(limit))
        # Older entries were created only for persistent alerts and lack this field.
        alerts = list(self.db[self.FAULT_COLLECTION]
                      .find({'stream_id': stream_id, '$or': [
                          {'persistent_alert': True}, {'persistent_alert': {'$exists': False}}]},
                          {'_id': 0, 'created_at': 0})
                      .sort([('created_at', DESCENDING), ('_id', DESCENDING)]).limit(limit))
        return {'predictions': predictions, 'normal_predictions': normal_predictions[::-1],
                'fault_candidates': candidates[::-1], 'alerts': alerts[::-1]}
