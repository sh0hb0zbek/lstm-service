import socket
import asyncio
import websockets


# get the local ip address
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.connect(('8.8.8.8', 53))
    WS = s.getsockname()[0]

PORT_NUM = 5500


async def register_client(websocket, port_num):
    async for message in websocket:
        print(message)


if __name__ == '__main__':
    start_server = websockets.serve(register_client, WS, 5500)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
