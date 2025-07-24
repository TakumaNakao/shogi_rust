import shogi

def convert_sfen(input_usi_command):
    """
    Converts a USI position command string to standard SFEN.
    """
    parts = input_usi_command.split(' ')
    board = shogi.Board()

    if parts[1] == "startpos":
        # Initial position is already set by default shogi.Board()
        move_start_index = 3 # "position startpos moves"
    else:
        # Handle custom SFEN string (not expected for this problem, but good practice)
        board.set_sfen_str(parts[1])
        move_start_index = 4 # "position <sfen> moves"

    if len(parts) > move_start_index and parts[move_start_index - 1] == "moves":
        for move_str in parts[move_start_index:]:
            try:
                move = shogi.Move.from_usi(move_str)
                board.push(move)
            except ValueError:
                print(f"Warning: Could not parse move {move_str}. Skipping.")
                break
    return board.sfen()

input_file = "records2016_10818.sfen"
output_file = "converted_records2016_10818.sfen"

try:
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if line:
                # Prepend "position " to the line to make it a valid USI position command
                usi_command = f"position {line}"
                converted_sfen = convert_sfen(usi_command)
                outfile.write(converted_sfen + '\n')
                print(f"Converted: {usi_command} -> {converted_sfen}")
    print(f"Converted SFENs saved to {output_file}")
except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")