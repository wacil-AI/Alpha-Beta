

import time
from client import ClientSocket
from argparse import ArgumentParser

# config.py
SERVER_IP = "127.0.0.1"  # L'IP du serveur (en local pour les tests)
SERVER_PORT = 5555       # Le port utilisé par le serveur








import socket
from typing import List
import time
import random
from argparse import ArgumentParser

# --- Exceptions ---
class EndException(Exception): pass
class ByeException(Exception): pass

# --- Utils ---
def bytes_to_int(data: bytes) -> int:
    return int.from_bytes(data, "little")

# --- ClientSocket ---
class ClientSocket:
    def __init__(self, ip: str = 'localhost', port: int = 5555):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ip = ip
        self._port = port
        self._connected = False
        self.connect_to_server()

    def connect_to_server(self):
        if not self._connected:
            self._socket.connect((self._ip, self._port))
            self._connected = True
            print(f"Connected to server {self._ip}:{self._port}")

    def _get_command(self) -> str:
        data = bytes()
        while len(data) < 3:
            data += self._socket.recv(3 - len(data))
        return data.decode()

    def _get_message(self, length: int) -> int:
        data = bytes()
        while len(data) < length:
            data += self._socket.recv(length - len(data))
        return bytes_to_int(data)

    def _parse_message(self) -> List:
        command = self._get_command()
        if command == "END": raise EndException()
        if command == "BYE": raise ByeException()
        if command not in ["SET", "HUM", "HME", "MAP", "UPD"]:
            raise ValueError(f"Unknown command: {command}")

        if command == "SET":
            return ["set", [self._get_message(1), self._get_message(1)]]

        if command == "HUM":
            humans = []
            nb = self._get_message(1)
            for _ in range(nb):
                humans.append([self._get_message(1), self._get_message(1)])
            return ["hum", humans]

        if command == "HME":
            return ["hme", [self._get_message(1), self._get_message(1)]]

        if command == "MAP":
            map_data = []
            nb = self._get_message(1)
            for _ in range(nb):
                map_data.append([self._get_message(1) for _ in range(5)])
            return ["map", map_data]

        if command == "UPD":
            upd_data = []
            nb = self._get_message(1)
            for _ in range(nb):
                upd_data.append([self._get_message(1) for _ in range(5)])
            return ["upd", upd_data]

    def get_message(self) -> List:
        try:
            return self._parse_message()
        except OSError:
            return None
        except IOError as e:
            print(e)
        except EndException:
            print("Game ended.")
            raise
        except ByeException:
            print("Server closed connection.")
            raise

    def send_nme(self, name: str):
        self._socket.send("NME".encode() + bytes([len(name)]) + name.encode())

    def send_mov(self, nb_moves: int, moves: List):
        message = bytes([nb_moves])
        for move in moves:
            for val in move:
                message += bytes([val])
        self._socket.send("MOV".encode() + message)























# --- AI Logic ---
import math
from typing import Tuple, List, Set, Optional

# Game state structure
GAME_STATE = {
    'grid_size': (0, 0),  # (n_rows, m_cols)
    'humans': [],  # List of (x, y) coordinates
    'home': (0, 0),  # My starting position
    'board': {},  # {(x, y): [humans, vampires, werewolves]}
    'my_species': None,  # 'vampire' or 'werewolf' (detected from MAP)
    'enemy_species': None,
    'my_positions': set(),  # Set of (x, y) where I have units
    'enemy_positions': set(),
    'move_history': []  # Recent positions to avoid loops
}

def UPDATE_GAME_STATE(message):
    """Update game state from server messages"""
    msg_type, data = message
    
    if msg_type == "set":
        GAME_STATE['grid_size'] = tuple(data)
        print(f"Grid size: {data}")
    
    elif msg_type == "hum":
        GAME_STATE['humans'] = [tuple(pos) for pos in data]
        print(f"Human positions: {len(data)} houses")
    
    elif msg_type == "hme":
        GAME_STATE['home'] = tuple(data)
        print(f"My home: {data}")
    
    elif msg_type in ["map", "upd"]:
        # Preserve existing board or initialize
        if msg_type == "map":
            board = {}
        else:
            board = GAME_STATE['board'].copy()
        
        # Update only the cells mentioned in the message
        for cell_data in data:
            x, y, humans, vampires, werewolves = cell_data
            pos = (x, y)
            board[pos] = [humans, vampires, werewolves]
            
            # Detect species on first MAP
            if msg_type == "map" and GAME_STATE['my_species'] is None:
                if pos == GAME_STATE['home']:
                    if vampires > 0:
                        GAME_STATE['my_species'] = 'vampire'
                        GAME_STATE['enemy_species'] = 'werewolf'
                        print(f"Detected: I am VAMPIRE")
                    elif werewolves > 0:
                        GAME_STATE['my_species'] = 'werewolf'
                        GAME_STATE['enemy_species'] = 'vampire'
                        print(f"Detected: I am WEREWOLF")
        
        # Rebuild position sets from entire board
        my_positions = set()
        enemy_positions = set()
        
        for pos, (humans, vampires, werewolves) in board.items():
            if GAME_STATE['my_species'] == 'vampire' and vampires > 0:
                my_positions.add(pos)
            elif GAME_STATE['my_species'] == 'werewolf' and werewolves > 0:
                my_positions.add(pos)
            
            if GAME_STATE['my_species'] == 'vampire' and werewolves > 0:
                enemy_positions.add(pos)
            elif GAME_STATE['my_species'] == 'werewolf' and vampires > 0:
                enemy_positions.add(pos)
        
        GAME_STATE['board'] = board
        GAME_STATE['my_positions'] = my_positions
        GAME_STATE['enemy_positions'] = enemy_positions
        
        # Detailed debug info
        my_total = sum(get_my_count(pos) for pos in my_positions)
        enemy_total = sum(get_enemy_count(pos) for pos in enemy_positions)
        human_total = sum(counts[0] for counts in board.values())
        
        print(f"Board updated: {len(my_positions)} my groups ({my_total} units), {len(enemy_positions)} enemy groups ({enemy_total} units), {human_total} humans")
        print(f"My positions: {my_positions}")
        print(f"Enemy positions: {enemy_positions}")


def get_my_count(cell):
    """Get my population at a cell"""
    if cell not in GAME_STATE['board']:
        return 0
    humans, vampires, werewolves = GAME_STATE['board'][cell]
    if GAME_STATE['my_species'] == 'vampire':
        return vampires
    return werewolves


def get_enemy_count(cell):
    """Get enemy population at a cell"""
    if cell not in GAME_STATE['board']:
        return 0
    humans, vampires, werewolves = GAME_STATE['board'][cell]
    if GAME_STATE['my_species'] == 'vampire':
        return werewolves
    return vampires


def get_human_count(cell):
    """Get human population at a cell"""
    if cell not in GAME_STATE['board']:
        return 0
    return GAME_STATE['board'][cell][0]


def chebyshev_distance(pos1, pos2):
    """Chebyshev distance (8-directional grid)"""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def get_neighbors(pos):
    """Get all valid adjacent cells (8 directions)"""
    x, y = pos
    n_rows, m_cols = GAME_STATE['grid_size']
    neighbors = []
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < m_cols and 0 <= ny < n_rows:
                neighbors.append((nx, ny))
    
    return neighbors


def is_cell_empty(pos):
    """Check if cell is empty (no units of any kind)"""
    if pos not in GAME_STATE['board']:
        return True
    humans, vampires, werewolves = GAME_STATE['board'][pos]
    return humans == 0 and vampires == 0 and werewolves == 0


def find_valid_targets(group_pos, group_count):
    """
    Find all valid targets for a group:
    - Must satisfy: group_count >= 1.4 * target_count
    - Target can be human or enemy
    Returns list of (target_pos, target_count, distance, target_type)
    """
    targets = []
    
    for pos, (humans, vampires, werewolves) in GAME_STATE['board'].items():
        # Check enemies
        if GAME_STATE['my_species'] == 'vampire':
            enemy_count = werewolves
        else:
            enemy_count = vampires
        
        if enemy_count > 0:
            if group_count >= 2 * enemy_count:
                distance = chebyshev_distance(group_pos, pos)
                targets.append((pos, enemy_count, distance, 'enemy'))
        
        # Check humans
        if humans > 0:
            if group_count >= 1.1 * humans:
                distance = chebyshev_distance(group_pos, pos)
                targets.append((pos, humans, distance, 'human'))
    
    # Sort by distance (closest first)
    targets.sort(key=lambda x: x[2])
    return targets


def find_path_to_target(start_pos, target_pos, group_count):
    """
    Find next step toward target.
    Only move on:
    1. Empty cells (green cells)
    2. The target cell itself
    Returns the next position to move to, or None if blocked
    """
    if start_pos == target_pos:
        return target_pos
    
    # Get all neighbors
    neighbors = get_neighbors(start_pos)
    
    # Find neighbors that get us closer to target
    best_next = None
    best_distance = chebyshev_distance(start_pos, target_pos)
    
    for neighbor in neighbors:
        new_distance = chebyshev_distance(neighbor, target_pos)
        
        # Check if we can move to this cell
        can_move = False
        
        # Can always move to target
        if neighbor == target_pos:
            can_move = True
        # Can move to empty cells
        elif is_cell_empty(neighbor):
            can_move = True
        # Can move to cells with only our units (merge)
        elif get_enemy_count(neighbor) == 0 and get_human_count(neighbor) == 0:
            can_move = True
        
        if can_move and new_distance < best_distance:
            best_distance = new_distance
            best_next = neighbor
    
    return best_next


def COMPUTE_NEXT_MOVE(game_state):
    """
    Simple strategy:
    1. Maximum 2 groups allowed
    2. For each group, find closest valid target (where group_count >= 1.4 * target_count)
    3. Move toward that target (only on empty cells or target cell)
    4. If can't reach target, move to merge with other group
    """
    board = GAME_STATE['board']
    my_species = GAME_STATE['my_species']
    
    if not my_species:
        print("ERROR: Species not detected yet!")
        return 0, []
    
    if not GAME_STATE['my_positions']:
        print(f"ERROR: No positions found for {my_species}!")
        return 0, []
    
    print(f"\n=== Computing move for {my_species} ===")
    
    my_groups = list(GAME_STATE['my_positions'])
    moves = []
    
    # Process each group
    for group_pos in my_groups:
        group_count = get_my_count(group_pos)
        
        print(f"\nGroup at {group_pos} with {group_count} units:")
        
        # Find valid targets for this group
        valid_targets = find_valid_targets(group_pos, group_count)
        
        if valid_targets:
            # Get closest valid target
            target_pos, target_count, distance, target_type = valid_targets[0]
            print(f"  Closest valid target: {target_type} at {target_pos} (distance {distance}, count {target_count})")
            
            # Find next step toward target
            next_pos = find_path_to_target(group_pos, target_pos, group_count)
            
            if next_pos and next_pos != group_pos:
                print(f"  Moving toward target: {group_pos} -> {next_pos}")
                move = (group_pos[0], group_pos[1], group_count, next_pos[0], next_pos[1])
                moves.append(move)
            else:
                print(f"  Cannot find valid path to target")
        else:
            print(f"  No valid targets (not strong enough)")
            
            # Try to merge with another group if we have 2 groups
            if len(my_groups) == 2:
                other_group = [g for g in my_groups if g != group_pos][0]
                print(f"  Attempting to merge with group at {other_group}")
                
                # Move toward other group
                next_pos = find_path_to_target(group_pos, other_group, group_count)
                if next_pos and next_pos != group_pos:
                    print(f"  Moving to merge: {group_pos} -> {next_pos}")
                    move = (group_pos[0], group_pos[1], group_count, next_pos[0], next_pos[1])
                    moves.append(move)
    
    # Return moves
    if moves:
        print(f"\nReturning {len(moves)} move(s)")
        for move in moves:
            print(f"  Move: ({move[0]}, {move[1]}) -> ({move[3]}, {move[4]}) with {move[2]} units")
        return len(moves), moves
    
    # Fallback: make a simple move
    print("\nNo strategic moves found, making fallback move")
    pos = list(GAME_STATE['my_positions'])[0]
    count = get_my_count(pos)
    neighbors = get_neighbors(pos)
    
    # Try to find an empty neighbor
    for neighbor in neighbors:
        if is_cell_empty(neighbor):
            return 1, [(pos[0], pos[1], count, neighbor[0], neighbor[1])]
    
    # Just move to first neighbor
    if neighbors:
        return 1, [(pos[0], pos[1], count, neighbors[0][0], neighbors[0][1])]
    
    return 0, []







# --- Main loop ---
def play_game(args):
    try:
        print(f"[MAIN] Connecting to {args.ip}:{args.port} ...")
        client = ClientSocket(args.ip, args.port)
        print("[MAIN] Connected")

        client.send_nme("MY_AI")
        print("[MAIN] Name sent")

        # Handshake: SET → HUM → HME → MAP
        UPDATE_GAME_STATE(client.get_message())
        UPDATE_GAME_STATE(client.get_message())
        UPDATE_GAME_STATE(client.get_message())
        UPDATE_GAME_STATE(client.get_message())
        print("[MAIN] Setup complete. Game loop...")

        while True:
            try:
                time.sleep(0.5)
                message = client.get_message()
                if message is None:
                    print("[MAIN] Connection lost")
                    break

                UPDATE_GAME_STATE(message)

                if message[0] == "upd":
                    nb, moves = COMPUTE_NEXT_MOVE(GAME_STATE)
                    client.send_mov(nb, moves)
            except EndException:
                print("[MAIN] END → new game expected")
                # ici on pourrait réinitialiser l'état si plusieurs parties s’enchaînent
                continue
            except ByeException:
                print("[MAIN] BYE → closing")
                break

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('ip', type=str, help='IP address')
    # parser.add_argument('port', type=int, help='Port')
    # args = parser.parse_args()
    play_game(args=type('', (), {'ip': '127.0.0.1', 'port': 5555})())


    
