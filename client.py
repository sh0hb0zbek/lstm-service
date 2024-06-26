import socket
import asyncio
import websockets
import os
import base64
from datetime import datetime
import json
from time import time


# get the local ip address
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.connect(('8.8.8.8', 53))
    WS = s.getsockname()[0]

PORT_NUM = 5500
FORMAT = 'utf-8'
DIR = '.data'


async def start_client(websocket, port_num):
    url = f'ws://{websocket}:{port_num}'
    start_time = time()
    try:
        connection = await websockets.connect(url)
        if os.path.exists(DIR):
            datalist = os.listdir(DIR)
            for item in datalist:
                item_path = os.path.join(DIR, item)
                with open(item_path, 'rb') as f:
                    encoded_sting = base64.b64encode(f.read()).decode(FORMAT)
                    data_json = dict()
                    data_json = {'data': encoded_sting,
                                 'timestamp': str(datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))}
                    json.dumps(data_json, indent=4)
                    await connection.send(encoded_sting)
        end_time = time() - start_time
        print(f'End-to-End time: {end_time:.3f}')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(start_client(WS, PORT_NUM))
    asyncio.get_event_loop().run_forever()
