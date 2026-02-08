# Copyright 2025 Suanzhi Future Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import json
import multiprocessing
import threading
import queue
import time
from websockets.sync.server import serve, Server
from websockets.exceptions import ConnectionClosed

_websocket_connection = None
_websocket_lock = threading.Lock()

_request_configs = {}

def get_websocket():
    with _websocket_lock:
        return _websocket_connection

def websocket_server_process(port: int, data_queue: multiprocessing.Queue, config_queue: multiprocessing.Queue, start_event: multiprocessing.Event, shutdown_event: multiprocessing.Event, training_args: dict):
    global _websocket_connection

    def _data_sender_thread_inner():
        print("Rank 0 (WS Process): Data sender thread started.", flush=True)
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
                
                ws = get_websocket()
                if ws:
                    try:
                        ws.send(json.dumps(payload))
                    except ConnectionClosed:
                        pass
                    except Exception as e:
                        print(f"Rank 0 (WS Process): Error sending data: {e}", flush=True)

            except queue.Empty:
                continue
            except (BrokenPipeError, EOFError):
                print("Rank 0 (WS Process): Data queue connection broken, sender thread exiting.", flush=True)
                break
            except Exception as e:
                print(f"Rank 0 (WS Process): Unexpected error in sender: {e}", flush=True)
        print("Rank 0 (WS Process): Data sender thread exiting.", flush=True)

    def _websocket_handler(websocket):
        global _websocket_connection
        print("Rank 0 (WS Process): Frontend connected.", flush=True)
        with _websocket_lock:
            _websocket_connection = websocket
        
        try:
            for message in websocket:
                try:
                    request = json.loads(message)
                    if request.get("type") == "run_training_step":
                        print("Rank 0 (WS Process): Received 'run_training_step' command.", flush=True)
                        _request_configs['visualization_flags'] = request.get("visualization_flags", {})
                        _request_configs['compressor_config'] = request.get("compressor_config", {})
                        config_queue.put(_request_configs)
                        try:
                            start_payload = {
                                "type": "start",
                                "micro_batch_size": training_args.get("micro_batch_size"),
                                "seq_length": training_args.get("seq_length"),
                                "num_layers": training_args.get("num_layers")
                            }
                            websocket.send(json.dumps(start_payload))
                            print("Rank 0 (WS Process): Sent 'start' message to frontend.", flush=True)
                        except Exception as e:
                            print(f"Rank 0 (WS Process): Failed to send 'start' message: {e}", flush=True)
                        start_event.set()
                except Exception as e:
                    print(f"Rank 0 (WS Process): Error processing message: {e}", flush=True)
        except ConnectionClosed:
            print("Rank 0 (WS Process): Connection handler closed as expected.", flush=True)
        finally:
            with _websocket_lock:
                _websocket_connection = None
            start_event.clear()
            print("Rank 0 (WS Process): Frontend disconnected.", flush=True)

    sender_thread = threading.Thread(target=_data_sender_thread_inner, daemon=True)
    sender_thread.start()

    server: Server = None

    def shutdown_handler():
        shutdown_event.wait()
        print("Rank 0 (WS Process): Shutdown event received, stopping server...", flush=True)
        if server:
            server.shutdown()

    shutdown_thread = threading.Thread(target=shutdown_handler, daemon=True)
    shutdown_thread.start()

    print(f"Rank 0 (WS Process): Starting server on ws://0.0.0.0:{port}", flush=True)
    
    try:
        with serve(
            _websocket_handler, "0.0.0.0", port, 
            ping_interval=None, reuse_port=True
        ) as server_instance:
            server = server_instance
            server.serve_forever()
    except Exception as e:
        print(f"Rank 0 (WS Process): Server crashed with an error: {e}", flush=True)
    finally:
        sender_thread.join(timeout=1.0)
        config_queue.close()
        print("Rank 0 (WS Process): Server has shut down.", flush=True)
