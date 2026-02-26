# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import multiprocessing
import threading
import queue
import time
from websockets.sync.server import serve, Server
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

_frontend_connection = None
_connection_lock = threading.Lock()

_request_configs = {}

def get_frontend_connection():
    with _connection_lock:
        return _frontend_connection

def websocket_worker_process(master_addr: str, port: int, rank: int, data_queue: multiprocessing.Queue, shutdown_event: multiprocessing.Event):
    uri = f"ws://{master_addr}:{port}"
    print(f"Rank {rank} (Worker): Connecting to Hub at {uri}...", flush=True)

    while not shutdown_event.is_set():
        try:
            with connect(uri, max_size=None) as websocket:
                print(f"Rank {rank} (Worker): Connected.", flush=True)

                while not shutdown_event.is_set():
                    try:
                        name_tuple, report_args, tensor_data = data_queue.get(timeout=1.0)

                        payload = {
                            "type": "worker_forward",
                            "data": {
                                "type": "update",
                                "update_type": name_tuple[1].value,
                                "layer_id": name_tuple[0],
                                "args": report_args,
                                "result": tensor_data.tolist()
                            }
                        }

                        websocket.send(json.dumps(payload))
                    except queue.Empty:
                        continue
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
        except Exception as e:
            print(f"Rank {rank} (Worker): Error: {e}, retrying...", flush=True)
            time.sleep(2)

def websocket_server_process(port: int, data_queue: multiprocessing.Queue, config_queue: multiprocessing.Queue, start_event: multiprocessing.Event, shutdown_event: multiprocessing.Event, training_args: dict):
    global _frontend_connection

    def _local_data_sender():
        print("Rank 0 (Server): Local data sender started.", flush=True)
        while not shutdown_event.is_set():
            try:
                name_tuple, report_args, tensor_data = data_queue.get(timeout=1.0)

                payload = {
                    "type": "update",
                    "update_type": name_tuple[1].value,
                    "layer_id": name_tuple[0],
                    "args": report_args,
                    "result": tensor_data.tolist()
                }

                ws = get_frontend_connection()
                if ws:
                    try:
                        ws.send(json.dumps(payload))
                    except ConnectionClosed:
                        pass
                    except Exception as e:
                        print(f"Rank 0 (Server): Error sending data: {e}", flush=True)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Rank 0 (Server): Unexpected error in sender: {e}", flush=True)

    def _websocket_handler(websocket):
        global _frontend_connection
        is_frontend = False

        try:
            for message in websocket:
                try:
                    msg_obj = json.loads(message)
                    msg_type = msg_obj.get("type")
                    if msg_type == "worker_forward":
                        forward_data = msg_obj.get("data")
                        frontend = get_frontend_connection()
                        if frontend:
                            frontend.send(json.dumps(forward_data))
                    elif msg_type == "run_training_step":
                        print("Rank 0 (Server): Frontend connected and assumed control.", flush=True)
                        with _connection_lock:
                            _frontend_connection = websocket
                        is_frontend = True

                        _request_configs['visualization_flags'] = msg_obj.get("visualization_flags", {})
                        _request_configs['compressor_config'] = msg_obj.get("compressor_config", {})
                        config_queue.put(_request_configs)

                        start_payload = {
                            "type": "start",
                            "micro_batch_size": training_args.get("micro_batch_size"),
                            "seq_length": training_args.get("seq_length"),
                            "num_layers": training_args.get("num_layers")
                        }
                        websocket.send(json.dumps(start_payload))
                        start_event.set()
                except Exception as e:
                    print(f"Rank 0 (Server): Error processing message: {e}", flush=True)
        except ConnectionClosed:
            print("Rank 0 (Server): Connection handler closed.", flush=True)
        finally:
            if is_frontend:
                print("Rank 0 (Server): Frontend disconnected.", flush=True)
                with _connection_lock:
                    _frontend_connection = None
                start_event.clear()

    sender_thread = threading.Thread(target=_local_data_sender, daemon=True)
    sender_thread.start()

    server: Server = None

    def shutdown_handler():
        shutdown_event.wait()
        print("Rank 0 (Server): Shutdown event received, stopping server...", flush=True)
        if server:
            server.shutdown()

    shutdown_thread = threading.Thread(target=shutdown_handler, daemon=True)
    shutdown_thread.start()

    print(f"Rank 0 (Server): Starting server on ws://0.0.0.0:{port}", flush=True)
    
    try:
        with serve(
            _websocket_handler, "0.0.0.0", port, 
            ping_interval=None, reuse_port=True,
            max_size=None,
        ) as server_instance:
            server = server_instance
            server.serve_forever()
    except Exception as e:
        print(f"Rank 0 (Server): Server crashed with an error: {e}", flush=True)
    finally:
        sender_thread.join(timeout=1.0)
        config_queue.close()
        print("Rank 0 (Server): Server has shut down.", flush=True)
