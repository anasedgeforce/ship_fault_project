"""HTTP boundary for manual recordings and direct vibration batches."""
from __future__ import annotations

import hmac
import hashlib
import json
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .catalog import configured_data_root, public_catalog
from .manual import parse_manual_payload
from .models import InferenceEngine
from .settings import RuntimeSettings
from .signal import parse_csv_recording
from .source import normalize_source
from .stream import StreamManager

STREAM_ID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


class StorageUnavailable(Exception):
    pass


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    settings = settings or RuntimeSettings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.engine = InferenceEngine(settings)
        application.state.streams = StreamManager(application.state.engine)
        application.state.mongo = None
        application.state.mongo_error = None
        application.state.recovery_errors = {}
        if settings.mongo_enabled:
            try:
                from .mongo_store import MongoStore
                application.state.mongo = MongoStore(settings.mongo_uri, settings.mongo_database)
            except Exception as exc:
                application.state.mongo_error = f'{type(exc).__name__}: MongoDB connection failed'
            def restore(session):
                store = connected_store(None)
                if store is None:
                    raise StorageUnavailable('MongoDB is required to recover saved streams')
                try:
                    recovered = store.recover_stream(session)
                    application.state.recovery_errors.pop(session.stream_id, None)
                    return recovered
                except Exception as exc:
                    application.state.recovery_errors[session.stream_id] = type(exc).__name__
                    raise StorageUnavailable('Saved stream recovery failed; check model compatibility and stored readings') from exc

            def close_session(session):
                store = connected_store(None)
                if store is None:
                    raise StorageUnavailable('MongoDB is required to save the stream reset')
                try:
                    store.close_session(session)
                except Exception as exc:
                    raise StorageUnavailable('Could not save the stream reset') from exc

            application.state.streams.restore = restore
            application.state.streams.close_session = close_session
            if application.state.mongo is not None:
                try:
                    saved_streams = application.state.mongo.active_stream_ids()
                except Exception:
                    # Manual mode remains usable during an outage. Watching a
                    # stream or sending its next batch retries recovery later.
                    saved_streams = []
                    application.state.mongo_error = 'MongoDB stream discovery unavailable'
                for stream_id in saved_streams:
                    try:
                        application.state.streams.get(stream_id)
                    except (StorageUnavailable, ValueError):
                        application.state.recovery_errors.setdefault(stream_id, 'recovery_failed')
        yield
        if application.state.mongo is not None:
            application.state.mongo.close()

    application = FastAPI(title='Ship Fault Inference', version='0.1.0', lifespan=lifespan)
    web_dir = Path(__file__).with_name('web')
    mongo_lock = threading.RLock()

    @application.exception_handler(StorageUnavailable)
    async def unavailable_handler(request, exc):
        return JSONResponse(status_code=503, content={'detail': str(exc)})

    def connected_store(request):
        # Also used during lifespan recovery before an HTTP request exists.
        state = application.state
        if not settings.mongo_enabled:
            return None
        with mongo_lock:
            try:
                if state.mongo is None:
                    from .mongo_store import MongoStore
                    state.mongo = MongoStore(settings.mongo_uri, settings.mongo_database)
                else:
                    state.mongo.ping()
                state.mongo_error = None
                return state.mongo
            except Exception as exc:
                state.mongo_error = f'{type(exc).__name__}: MongoDB connection unavailable'
                return None

    def flush_pending_results(session, stream_id, store):
        pending = session.pending_persistence
        if pending is None:
            return None
        try:
            store.store_results(pending['batch_reference'], session.session_id,
                                stream_id, pending['predictions'], pending['alerts'])
        except Exception as exc:
            raise StorageUnavailable('MongoDB result save is pending; retry saving before sending more samples') from exc
        pending['persistence'] = 'saved'
        pending.pop('persistence_error', None)
        session.pending_persistence = None
        return pending

    def authorize(request: Request):
        if settings.api_key and not hmac.compare_digest(request.headers.get('X-API-Key', ''),
                                                         settings.api_key):
            raise HTTPException(status_code=401, detail='valid X-API-Key required')

    def require_mode(mode: str):
        if settings.input_mode not in ('both', mode):
            raise HTTPException(status_code=409, detail=f'{mode} input is disabled; active mode is {settings.input_mode}')

    async def limited_body(request: Request) -> bytes:
        try:
            claimed = int(request.headers.get('Content-Length', '0'))
        except ValueError:
            raise HTTPException(status_code=400, detail='invalid Content-Length')
        if claimed > settings.max_request_bytes:
            raise HTTPException(status_code=413, detail='request too large')
        chunks = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > settings.max_request_bytes:
                raise HTTPException(status_code=413, detail='request too large')
            chunks.append(chunk)
        return b''.join(chunks)

    async def json_body(request: Request):
        try:
            body = json.loads((await limited_body(request)).decode('utf-8'))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail='expected UTF-8 JSON') from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail='expected JSON object')
        return body

    def valid_stream_id(stream_id: str):
        if not STREAM_ID.fullmatch(stream_id):
            raise HTTPException(status_code=422, detail='stream_id must be 1-64 letters, digits, _ or -')

    @application.get('/', include_in_schema=False)
    def index():
        return FileResponse(web_dir / 'index.html', media_type='text/html', headers={'Cache-Control': 'no-store'})

    @application.get('/assets/app.css', include_in_schema=False)
    def web_css():
        return FileResponse(web_dir / 'app.css', media_type='text/css')

    @application.get('/assets/app.js', include_in_schema=False)
    def web_js():
        return FileResponse(web_dir / 'app.js', media_type='text/javascript', headers={'Cache-Control': 'no-store'})

    @application.get('/assets/twin-map.js', include_in_schema=False)
    def twin_map_js():
        return FileResponse(web_dir / 'twin-map.js', media_type='text/javascript', headers={'Cache-Control': 'no-store'})

    @application.get('/assets/twin-view.js', include_in_schema=False)
    def twin_view_js():
        return FileResponse(web_dir / 'twin-view.js', media_type='text/javascript', headers={'Cache-Control': 'no-store'})

    @application.get('/assets/digital-twin.html', include_in_schema=False)
    def digital_twin_html():
        return FileResponse(web_dir / 'digital_twin.html', media_type='text/html',
                            headers={'Cache-Control': 'no-store'})

    @application.get('/v1/demo-inputs/{name}')
    def demo_input(name: str):
        if name not in {'synthetic_unseen_smooth.csv', 'synthetic_unseen_impulsive.csv'}:
            raise HTTPException(status_code=404, detail='Demo recording was not found.')
        path = settings.project_root / 'demo_inputs' / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Demo recording was not found.')
        return FileResponse(path, media_type='text/csv', headers={'Cache-Control': 'no-store'})

    @application.get('/health/live')
    def live():
        return {'status': 'ok'}

    @application.get('/health/ready')
    def ready(request: Request):
        store = connected_store(request)
        storage_failed = settings.mongo_enabled and store is None
        return {'status': 'degraded' if storage_failed or request.app.state.recovery_errors else 'ready', 'input_mode': settings.input_mode,
                'equipment_id': settings.equipment_id,
                'stream_recovery_errors': request.app.state.recovery_errors,
                'models': request.app.state.engine.model_info,
                'live_storage': ('ready' if store is not None else
                                 'unavailable' if settings.mongo_enabled else 'disabled'),
                'continuous_stream_validated': False}

    @application.get('/v1/models')
    def models(request: Request):
        authorize(request)
        return request.app.state.engine.model_info

    @application.get('/v1/labels')
    def labels(request: Request):
        authorize(request)
        return request.app.state.engine.model_info['label_catalog']

    @application.get('/v1/storage')
    def storage(request: Request):
        authorize(request)
        store = connected_store(request)
        if store is None:
            return {'status': 'unavailable' if settings.mongo_enabled else 'disabled',
                    'database': settings.mongo_database,
                    'error': request.app.state.mongo_error}
        try:
            return {'status': 'ready', **store.info()}
        except Exception as exc:
            return {'status': 'unavailable', 'database': settings.mongo_database,
                    'error': f'{type(exc).__name__}: MongoDB status could not be read'}

    @application.get('/v1/recordings')
    def recordings(request: Request):
        authorize(request)
        require_mode('manual')
        try:
            days, _ = public_catalog(settings.project_root, request.app.state.engine.manifest)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=500, detail=f'Could not list recordings: {exc}') from exc
        reference = settings.project_root / 'config/saved_recordings.json'
        metadata = json.loads(reference.read_text(encoding='utf-8')) if reference.is_file() else {}
        return {'days': days, **metadata}

    @application.post('/v1/predict/saved')
    async def predict_saved(request: Request):
        authorize(request)
        require_mode('manual')
        data = await json_body(request)
        selected = data.get('id')
        if not isinstance(selected, str):
            raise HTTPException(status_code=422, detail='Select a saved recording.')
        try:
            _, allowed = public_catalog(settings.project_root, request.app.state.engine.manifest)
            entry = allowed.get(selected)
            if entry is None:
                raise HTTPException(status_code=404, detail='Saved recording was not found. Refresh the list.')
            path, expected_hash = entry
            base = configured_data_root(settings.project_root)
            if not path.resolve().is_relative_to(base.resolve()):
                raise HTTPException(status_code=404, detail='Saved recording was not found.')
            if path.stat().st_size > settings.max_request_bytes:
                raise HTTPException(status_code=413, detail='Recording is too large.')
            raw = await run_in_threadpool(path.read_bytes)
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise HTTPException(status_code=409, detail='Recording changed since evaluation. Refresh the list.')
            timestamps, channels = await run_in_threadpool(parse_manual_payload, raw, settings)
            return await run_in_threadpool(request.app.state.engine.predict_recording, timestamps, channels)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post('/v1/predict/input')
    async def predict_input(request: Request):
        authorize(request)
        require_mode('manual')
        try:
            timestamps, channels = await run_in_threadpool(parse_manual_payload, await limited_body(request), settings)
            return await run_in_threadpool(request.app.state.engine.predict_recording, timestamps, channels)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post('/v1/predict/csv')
    async def predict_csv(request: Request):
        authorize(request)
        require_mode('manual')
        try:
            timestamps, channels = await run_in_threadpool(parse_csv_recording, await limited_body(request), settings)
            return await run_in_threadpool(request.app.state.engine.predict_recording, timestamps, channels)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post('/v1/predict/recording')
    async def predict_recording(request: Request):
        authorize(request)
        require_mode('manual')
        data = await json_body(request)
        try:
            return await run_in_threadpool(request.app.state.engine.predict_recording,
                                           data['timestamps'], data['channels'])
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f'missing field {exc.args[0]}') from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post('/v1/streams/{stream_id}/samples')
    async def ingest_samples(stream_id: str, request: Request):
        authorize(request)
        require_mode('live')
        valid_stream_id(stream_id)
        data = await json_body(request)
        samples = data.get('samples')
        if not isinstance(samples, list) or not samples:
            raise HTTPException(status_code=422, detail='samples must be a nonempty list')
        if len(samples) > settings.max_batch_samples:
            raise HTTPException(status_code=413, detail='sample batch too large')
        try:
            timestamps = [sample['timestamp'] for sample in samples]
            channels = [sample['channels'] for sample in samples]
            source = normalize_source(data.get('source'), settings.sample_rate_hz)
            if settings.equipment_id:
                if source and source.get('equipment_id') not in (None, settings.equipment_id):
                    raise ValueError(f'equipment_id must be {settings.equipment_id} for this application')
                source = {**(source or {}), 'equipment_id': settings.equipment_id}
            store = await run_in_threadpool(connected_store, request)
            if settings.mongo_enabled and store is None:
                raise HTTPException(status_code=503, detail='MongoDB storage is unavailable; live samples were not accepted')
            session = request.app.state.streams.get_or_create(stream_id)
            def persist(times, values, metadata):
                try:
                    return store.store_batch(session.session_id, stream_id, times, values,
                                             {**(metadata or {}), '_recovery_model': session.recovery_model})
                except Exception as exc:
                    raise StorageUnavailable('MongoDB did not save the batch; no samples were accepted') from exc

            def fetch_saved(reference, expected_times, expected_values):
                try:
                    saved_times, saved_values = store.load_batch(reference, session.session_id, stream_id)
                    if (not np.array_equal(saved_times, expected_times)
                            or not np.array_equal(saved_values, expected_values)):
                        raise ValueError('saved samples differ from the accepted batch')
                    return saved_times, saved_values
                except Exception as exc:
                    raise StorageUnavailable(
                        'Batch was saved but could not be verified from MongoDB; prediction was not performed') from exc

            def process_batch():
                # Keep the read, prediction, and result updates ordered for this stream.
                with session.lock:
                    if store is not None:
                        flush_pending_results(session, stream_id, store)
                    result = session.push_batch(timestamps, channels, source,
                                                persist if store is not None else None,
                                                fetch_saved if store is not None else None)
                    if store is not None:
                        session.pending_persistence = result
                        try:
                            flush_pending_results(session, stream_id, store)
                        except StorageUnavailable as exc:
                            result['persistence'] = 'result_save_failed'
                            result['persistence_error'] = f'{type(exc).__name__}: live result or fault save failed'
                    return result

            result = await run_in_threadpool(process_batch)
            return {'stream_id': stream_id, **result}
        except (KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail='each sample needs timestamp and six channels') from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StorageUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post('/v1/streams/{stream_id}/retry-storage')
    def retry_storage(stream_id: str, request: Request):
        authorize(request)
        require_mode('live')
        valid_stream_id(stream_id)
        session = request.app.state.streams.get(stream_id)
        if session is None:
            raise HTTPException(status_code=404, detail='stream not found')
        store = connected_store(request)
        if store is None:
            raise HTTPException(status_code=503, detail='MongoDB storage is unavailable')
        with session.lock:
            try:
                result = flush_pending_results(session, stream_id, store)
            except StorageUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return {'stream_id': stream_id, 'session_id': session.session_id,
                    'persistence': 'saved' if result is not None else 'up_to_date'}

    @application.get('/v1/streams/{stream_id}')
    def stream_status(stream_id: str, request: Request, after: int = -1):
        authorize(request)
        require_mode('live')
        valid_stream_id(stream_id)
        if after < -1:
            raise HTTPException(status_code=422, detail='after must be -1 or a window index')
        session = request.app.state.streams.get(stream_id)
        if session is None:
            raise HTTPException(status_code=404, detail='stream not found')
        return {'stream_id': stream_id, **session.status(after)}

    @application.get('/v1/streams/{stream_id}/diagnostics')
    def stream_diagnostics(stream_id: str, request: Request):
        authorize(request)
        require_mode('live')
        valid_stream_id(stream_id)
        session = request.app.state.streams.get(stream_id)
        if session is None:
            raise HTTPException(status_code=404, detail='stream not found')
        return {'stream_id': stream_id, **session.diagnostics()}

    @application.get('/v1/streams/{stream_id}/history')
    def stream_history(stream_id: str, request: Request, limit: int = 100):
        authorize(request)
        require_mode('live')
        valid_stream_id(stream_id)
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail='limit must be 1-500')
        store = connected_store(request)
        if store is None:
            raise HTTPException(status_code=503, detail='MongoDB storage is unavailable')
        try:
            return {'stream_id': stream_id, **store.history(stream_id, limit)}
        except Exception as exc:
            raise HTTPException(status_code=503, detail='MongoDB history could not be read') from exc

    @application.delete('/v1/streams/{stream_id}')
    def reset_stream(stream_id: str, request: Request):
        authorize(request)
        require_mode('live')
        valid_stream_id(stream_id)
        try:
            if not request.app.state.streams.reset(stream_id):
                raise HTTPException(status_code=404, detail='stream not found')
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {'stream_id': stream_id, 'reset': True}

    return application


app = create_app()
