import os
import csv
import json
from closest_adder import find_closest_element_by_transformed_vector,new_load_csv
import requests
import math
import traceback
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)

# Initial state
# pyramid = [
#     [' ', ' ', ' ', '1', ' ', ' ', ' '],
#     [' ', ' ', '1', '1', '1', ' ', ' '],
#     [' ', '1', '1', '1', '1', '1', ' '],
#     ['1', '1', '1', '1', '1', '1', 'X']
# ]

# ─────────── new helper ───────────
def generate_pyramid(bit_width: int) -> list[list[str]]:
    """
    Build a centred triangular pyramid for an N‑bit partial‑product tree.

        top row has      1  '1'
        next row has     3  '1's
        …
        (N‑1)th row has (2N‑3) '1's
        bottom row has  (2N‑2) '1's then an 'X'

    All rows are padded with spaces so every row length = (2·bit_width – 1)
    """
    if bit_width < 2:
        raise ValueError("Need at least 2 bits")

    width = 2 * bit_width - 1          # final rectangle width
    rows  = []

    # rows 0 … N‑2
    for r in range(bit_width - 1):
        ones = 2 * r + 1
        left = (width - ones) // 2
        row  = [' '] * left + ['1'] * ones + [' '] * (width - left - ones)
        rows.append(row)

    # bottom row: all 1s then X
    rows.append(['1'] * (width - 1) + ['X'])
    return rows


DEFAULT_BITS = 4         # keep 4 as the default so nothing else breaks
pyramid = generate_pyramid(DEFAULT_BITS)



log = []
custom_adders = []
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Constructing the file path
file_path = os.path.join(os.getcwd(), 'lookuptable.csv')

def add_to_log(message):
    log.append(message)
    print(message)

# Load custom adders from CSV
def load_csv(file_path):
    result = []
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []

        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            lines = list(reader)

        if len(lines) < 2:
            print('CSV file is empty or has no data rows')
            return []

        for i, line in enumerate(lines[1:], start=1):
            if len(line) == 9:
                data = list(map(int, line[:5]))
                data = [num for num in data if num != 0][::-1]
                cost = int(line[8])
                result.append({'id': i, 'data': data, 'cost': cost})
        add_to_log(f"Loaded {len(result)} custom adders from CSV.")
    except Exception as e:
        print(f'Error reading CSV: {e}')
    return result

custom_adders = load_csv(file_path)

def visualize_pyramid(p):
    """
    Return a single string showing the pyramid rows top-to-bottom,
    each row centered with leading spaces based on the width of the bottom row.
    """
    if not p:
        return ""
    
    # print("list of list of visualize pyramid", p)
    max_width = len(''.join(p[-1]))
    lines = []
    for row in p:
        row_str = ''.join(row)
        spaces = ' ' * ((max_width - len(row_str)) // 2)
        lines.append(spaces + row_str)
    
    return "\n".join(lines)



def modify_selected_best_adder(selected_adder, pyramid):
    """
    Modify the selected best adder and compute the carry bits for it.

    Args:
        selected_adder (dict): The selected adder containing 'data', 'row', and 'col'.
        pyramid (list of lists): The current pyramid state.

    Returns:
        int: Number of bits required to represent the total sum.
    """
    heights = selected_adder['data']  # Extract the heights of the selected adder
    max_height = max(heights)  # Find max height to determine rows
    carry_bits = []  # Store carry bits
    total_sum = 0

    # Process each row from the top to the bottom
    for row in range(max_height):
        binary_row = []

        # Determine if each column contributes to the current row
        for col_height in heights:
            if col_height > row:
                binary_row.append('1')
            else:
                binary_row.append('0')

        # Convert the binary row to a decimal number
        row_decimal = int(''.join(binary_row), 2)
        total_sum += row_decimal

        # Store carry bits (extra bits when summing at each level)
        carry_bits.append(f"Row {row}: {''.join(binary_row)} -> {row_decimal}")

    # Calculate the number of bits required
    num_bits_required = math.ceil(math.log2(total_sum + 1))

    # Print the carry bits information
    print("\nCarry Bits for the Selected Adder:")
    for carry in carry_bits:
        print(carry)
    
    print(f"\nNumber of Bits Required: {num_bits_required}")

    return carry_bits, num_bits_required


def apply_adder_and_shrink_pyramid(pyramid, adder_data, row, col):
    """
    Drop one adder, create the right number of carry bits, let gravity settle the
    bits, and return the new pyramid plus a success flag.
    """
    # ─────────────────────────────────────────────────────────────────────────
    new_pyramid = [r[:] for r in pyramid]          # deep copy
    applied_successfully = True

    # 1. ── stamp the adder (turn matched 1s to 0s) ──────────────────────────
    add_to_log(f"Custom adder used: {adder_data}") 
    adder_data = adder_data[::-1]                  # walk right → left
    start_col  = col

    for i, height in enumerate(adder_data):
        c = start_col - i
        for j in range(height):
            r = row - j
            if r < 0 or c < 0 or c >= len(new_pyramid[r]) or new_pyramid[r][c] != '1':
                applied_successfully = False
                break
            new_pyramid[r][c] = '0'
        if not applied_successfully:
            break

    if not applied_successfully:
        return pyramid, False
    
    add_to_log("\n① Pyramid with adder placed (0 = consumed 1):\n" +
           visualize_pyramid(new_pyramid))
    
    

    # 2. ── compute how many carry bits we need for *this* adder ──────────────
    selected   = {'data': adder_data, 'row': row, 'col': col}
    carry_bits, num_bits_required = modify_selected_best_adder(selected, pyramid)

    add_to_log(f"\nCarry‑bit budget for this adder: {num_bits_required}")
    for line in carry_bits:            # optional: show the per‑row breakdown
        add_to_log("   " + line)

    # ── 3.  mark every 0 in the *bottom row of THIS ADDER* as '*' ──────────
    carry_row_idx = row                          # “row” is the adder’s bottom
    adder_cols    = [start_col - i               # columns touched in that row
                     for i, h in enumerate(adder_data) if h > 0]

    for c in adder_cols:
        if new_pyramid[carry_row_idx][c] == '0':
            new_pyramid[carry_row_idx][c] = '*'

    add_to_log("\n② Pyramid after replacing 0 with * in adder row:\n" +
               visualize_pyramid(new_pyramid))

    # 4. ── turn every remaining 0 in the whole pyramid into blank space ─────
    for r in range(len(new_pyramid)):
        for c in range(len(new_pyramid[r])):
            if new_pyramid[r][c] == '0':
                new_pyramid[r][c] = ' '

    # 5. ── gravity: pull each column of 1s straight down, ignoring * and X ──
    rows, cols = len(new_pyramid), len(new_pyramid[-1])
    for c in range(cols):
        stack = [new_pyramid[r][c] for r in range(rows) if new_pyramid[r][c] == '1']
        for r in range(rows):
            if new_pyramid[r][c] not in ('*', 'X'):
                new_pyramid[r][c] = ' '
        r = rows - 1
        while stack:
            if new_pyramid[r][c] == ' ':
                new_pyramid[r][c] = stack.pop(0)
            r -= 1

    # 6. ── ensure *exactly* num_bits_required stars on the adder row ───────
    new_pyramid = make_pyramid_rectangular(new_pyramid)
    stars = [i for i, v in enumerate(new_pyramid[carry_row_idx]) if v == '*']
    missing = max(0, num_bits_required - len(stars))

    for _ in range(missing):
        target = (min(stars) if stars else new_pyramid[carry_row_idx].index('X')) - 1
        if target < 0:                           # need a brand‑new column
            for r in range(len(new_pyramid)):
                new_pyramid[r].insert(0, ' ')
            target = 0
            stars  = [i + 1 for i in stars]

        # # If shifting would drop a useful bit, add a blank row on top first
        # if new_pyramid[0][target] != ' ':
        #     new_pyramid.insert(0, [' '] * len(new_pyramid[0]))
        #     carry_row_idx += 1                   # adder row slid down one

        rows = len(new_pyramid)

        # 6‑A.  Make room at the very top if the topmost cell is occupied
        if new_pyramid[0][target] != ' ':
            new_pyramid.insert(0, [' '] * len(new_pyramid[0]))
            carry_row_idx += 1               # the adder row slid down
            rows += 1

        # 6‑B.  Copy the slice we intend to move
        upper_limit = rows if carry_row_idx == rows - 1 else carry_row_idx + 1
        col_slice   = [new_pyramid[r][target] for r in range(upper_limit)]

        # 6‑C.  Shift that slice up by one row
        for r in range(upper_limit - 1):
            new_pyramid[r][target] = col_slice[r + 1]

        # 6‑D.  Drop the new carry bit (‘*’) at the adder row
        new_pyramid[carry_row_idx][target] = '*'
        stars.append(target)

        # ------------------------------------------------------------------
        #  re‑run gravity **only on the rows above the star**
        # ------------------------------------------------------------------
        stack = [new_pyramid[r][target]
                 for r in range(carry_row_idx) if new_pyramid[r][target] == '1']

        for r in range(carry_row_idx):                # blank rows 0 … carry‑1
            if new_pyramid[r][target] not in ('*', 'X'):
                new_pyramid[r][target] = ' '

        r = carry_row_idx - 1                         # pack the 1s downward
        while stack:
            if new_pyramid[r][target] == ' ':
                new_pyramid[r][target] = stack.pop(0)
            r -= 1

    add_to_log("\n③ Pyramid after inserting all required carry bits (*):\n" +
               visualize_pyramid(new_pyramid))

    # 7. ── turn every '*' in the adder row into a real '1' ──────────────────
    for c, v in enumerate(new_pyramid[carry_row_idx]):
        if v == '*':
            new_pyramid[carry_row_idx][c] = '1'


    # 8. ── logs & return ────────────────────────────────────────────────────
    add_to_log("\nUpdated pyramid after adder + gravity + carry bits:\n" +
               visualize_pyramid(new_pyramid))
    return new_pyramid, True



#***** ⬇️⬇️⬇️converting the pyramid to a matrix form 
def make_pyramid_rectangular(pyramid):
    """
    Given a list of lists (the pyramid), returns a new list of lists
    where each row is the same length, and the pyramid content
    is centered within that length.
    """
    # 1) Compute the max width of the final pyramid rows
    max_width = 0
    for row in pyramid:
        row_length = len(''.join(row))  # Join row into string
        if row_length > max_width:
            max_width = row_length

    # 2) Build the new list of lists, each row of length = max_width
    rectangular_pyramid = []
    for row in pyramid:
        row_str = ''.join(row)
        current_length = len(row_str)
        # Compute left_spaces to center the row
        left_spaces = (max_width - current_length) // 2
        right_spaces = max_width - current_length - left_spaces
        # Build the padded string
        padded_str = (' ' * left_spaces) + row_str + (' ' * right_spaces)
        # Convert to list of characters
        rectangular_pyramid.append(list(padded_str))

    return rectangular_pyramid




def is_pyramid_covered(pyramid):
    return all(bit == '0' or bit == 'X' or bit == ' ' for row in pyramid for bit in row)

def find_valid_start_positions(pyramid):
    """
    Return the coordinates of *all* 1‑bits that are legal landing spots.
    We still skip the right‑most “X” column.
    """
    positions = []
    for r in range(len(pyramid) - 1, -1, -1):           # bottom → top
        for c in range(len(pyramid[r]) - 2, -1, -1):    # right  → left
            if pyramid[r][c] == '1':
                positions.append({'row': r, 'col': c})
    return positions           # may be hundreds, we’ll prune later





### BEGIN PATCH 1 – _encode_adders  ##########################################
def _encode_adders(adders):
    """
    Return a '\\n'-separated block, one candidate per line:

        <id>:<row>,<col>:<heights_digits_right‑to‑left>:<cost>:S=<score>

    Example:
        17:3,4:321:6:S=1.500
        ↑  ↑  ↑   ↑   └── score = covered_ones / cost  (rounded to 3 dp)
        │  │  │   └──── cost of this adder
        │  │  └──────── heights list [3,2,1]  (right‑most column first)
        │  └─────────── column and row where the adder will land
        └────────────── stable ID from the CSV
    """
    out = []
    for a in adders:
        heights = ''.join(str(h) for h in a['data'])
        span    = sum(a['data'])          # how many ‘1’s this adder consumes
        score   = round(span / a['cost'], 3)
        out.append(
            f"{a['adderIndex']}:{a['row']},{a['col']}:{heights}:{a['cost']}:S={score}"
        )
    return '\n'.join(out)
### END PATCH 1 ##############################################################





def select_best_adder(current_pyramid, available_adders, used_ids):
    """
    Select the best adder and its position using OpenAI API.
    """
    try:
        # Step 1: Find valid positions
        valid_positions = find_valid_start_positions(current_pyramid)
        if not valid_positions:                     # nothing to land on
            return None

        # ───────────────────────────────────────────────────────────────
        # 2) decide whether we *must* allow an already‑used ID
        # ───────────────────────────────────────────────────────────────
        remaining = [a for a in available_adders if a['id'] not in used_ids]

        if remaining:                 # at least one brand‑new ID still free
            skip_ids = used_ids       # keep banning the spent ones
        else:                         # everything is spent → allow reuse
            remaining = available_adders[:]   # full catalogue
            skip_ids  = set()                 # empty set = skip nothing

        # keep only the cheapest copy of each ID
        catalogue = {}
        for a in available_adders:                # ← no more ‘remaining’ / ‘skip’
            keep = catalogue.get(a['id'])
            if keep is None or a['cost'] < keep['cost']:
                catalogue[a['id']] = a
        base_adders = list(catalogue.values())

        # 3) enumerate every legal landing of every adder
        candidate_adders = filter_valid_adders(
            current_pyramid,
            valid_positions,
            base_adders,
            skip_ids,           # we pass the full set so the helper
        )     
        
        
        if not candidate_adders and skip_ids:
            candidate_adders = filter_valid_adders(
                current_pyramid, valid_positions, base_adders, set())
            skip_ids = set()                 # make downstream logic consistent                  # can *prefer* fresh IDs but still keep the rest

        if not candidate_adders:
            return None

        

        # Log the current state
        add_to_log('Sending request to OpenAI API...')
        add_to_log('Current pyramid:\n' + visualize_pyramid(current_pyramid))
        # add_to_log('Valid start positions: ' + json.dumps(valid_positions))
        add_to_log(f'Valid adders this turn: {len(candidate_adders)}')

        # Step 2: Prepare the prompt for OpenAI API
        system_message = {
            "role": "system",
            "content": (
                "You are an AI assistant that selects the best custom adder and its position for a given partial product pyramid. "
                "The goal is to cover the most 1s in the pyramid efficiently."
                "Respond only with a valid JSON object."
            )
        }

        # ----------------------------------------------------------------------
        # 👇 new ultra‑compact prompt (fits well under 128 k tokens)
        # ----------------------------------------------------------------------
        adder_block = _encode_adders(candidate_adders)


        user_message = {
            "role": "user",
            "content": (
                "You are given the current partial-product pyramid (2-D list), the "
                "list of valid start positions, and ALL valid custom adders.\n\n"
                f"Current pyramid (raw JSON):\n{json.dumps(current_pyramid)}\n\n"
                f"Valid start positions (raw JSON):\n{json.dumps(valid_positions)}\n\n"
                "Available adders - ONE PER LINE, format  <id>:<row>,<col>:<heights>:<cost>\n"
                "(Heights are digits, rightmost column first; e.g. 321 → [3,2,1]).\n"
                f"{adder_block}\n\n"
                "USED adder IDs (avoid re-using these unless no unused "
                "adder will fit): " + json.dumps(sorted(list(used_ids))) + "\n\n"
                "Rules to follow strictly:\n"
                "- Choose **one line from the list above verbatim.**\n"
                "- Never overlap the 'X'.\n"
                "- Compute **S = covered_ones / cost** from the trailing “S=” in each line.\n"
                "- Select the line with the **largest S**. If several lines share that S,\n"
                "  break ties by the larger covered_ones, then by the lower cost.\n\n"
                "Respond ONLY with a JSON object:\n"
                "{\n"
                "  \"adderIndex\": <id>,\n"
                "  \"row\": <bottom row index>,\n"
                "  \"col\": <rightmost column index>\n"
                "}"
            )
        }





        # Step 3: Make the request to OpenAI API
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [system_message, user_message],
            "temperature": 0.2,
            "max_tokens": 256
        }

        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data
        )

        # Handle the response
        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            print("Full response:", response_data)  # debug
            raise ValueError("No response from OpenAI API")

        api_response = response_data['choices'][0]['message']['content']
        add_to_log('Raw API Response: ' + api_response)

        # Step 4: Parse the JSON response
        cleaned_response = api_response.replace('```json\n', '').replace('\n```', '')
        selected_adder_info = json.loads(cleaned_response)

        # ── NEW: keep only the numeric id, discard the rest if GPT copied the whole line
        raw_id = str(selected_adder_info.get('adderIndex', ''))
        if ':' in raw_id:                      # e.g.  "1626:3,3:3:11"
            raw_id = raw_id.split(':', 1)[0]   # keep "1626"

        try:
            selected_adder_info['adderIndex'] = int(raw_id)
            selected_adder_info['row']        = int(selected_adder_info['row'])
            selected_adder_info['col']        = int(selected_adder_info['col'])
        except (KeyError, ValueError, TypeError):
            raise ValueError("API response has invalid adderIndex / row / col")

        # Validate the response
        if 'adderIndex' not in selected_adder_info:
            raise ValueError("API response missing 'adderIndex'")

        stable_id = selected_adder_info['adderIndex']

        # ① lookup the chosen adder
        try:
            adder_rec = next(a for a in candidate_adders if a['adderIndex'] == stable_id)
        except StopIteration:
            raise ValueError("Unknown adderIndex from GPT")

         # ① verify that GPT copied a line that really existed in the prompt
        r, c = selected_adder_info['row'], selected_adder_info['col']
        matches = [
            a for a in candidate_adders
            if (a['adderIndex'] == stable_id and
                a['row']        == r          and
                a['col']        == c)
        ]
        if not matches:
            raise ValueError("GPT chose an adder/position that isn’t valid now")
        adder_rec = matches[0]                    # the vetted record

        # ② (optional safety) make sure the placement still fits
        if not can_apply_adder(current_pyramid, adder_rec['data'], r, c):
            raise ValueError("Verified coordinates no longer fit the pyramid")

        # ③ assemble the object our downstream code expects
        selected_adder = {
            'adderIndex': adder_rec['adderIndex'],
            'data'      : adder_rec['data'],
            'cost'      : adder_rec['cost'],
            'row'       : r,
            'col'       : c,
        }
        return selected_adder

    except Exception as error:
        add_to_log(f'Error in select_best_adder: {str(error)}')
        return None



def can_apply_adder(pyramid, adder_data, row, col):
    """
    Check if an adder can be applied at (row, col) on the current pyramid.
    """
    start_col = col
    num_rows = len(pyramid)
    
    adder_data = adder_data[::-1]  # Reverse like apply_adder

    for i, height in enumerate(adder_data):
        current_col = start_col - i
        if current_col < 0:
            return False  # Out of bounds
        for j in range(height):
            current_row = row - j
            if current_row < 0:
                return False  # Out of bounds
            if current_col >= len(pyramid[current_row]):
                return False
            if pyramid[current_row][current_col] != '1':
                return False  # Should be '1'
    return True

def filter_valid_adders(pyramid, start_positions, adders, banned_ids):
    """
    Filter adders that can be applied from available adders.
    """
    valid_choices = []
    for adder in adders:
        # skip IDs that are already spent
        if adder['id'] in banned_ids:
            continue

        heights = adder['data']
        for pos in start_positions:
            if can_apply_adder(pyramid, heights, pos['row'], pos['col']):
                valid_choices.append({
                    'adderIndex': adder['id'],
                    'data'      : heights,
                    'cost'      : adder['cost'],
                    'row'       : pos['row'],
                    'col'       : pos['col'],
                })
    
    # ── A.  keep only the *cheapest* landing per adderIndex ───────────────
    unique = {}
    for choice in valid_choices:
        key   = (choice['adderIndex'], choice['row'], choice['col'])
        keep  = unique.get(key)
        if keep is None or choice['cost'] < keep['cost']:
            unique[key] = choice
    valid_choices = list(unique.values())

    # ── B.  hard‑cap so the GPT prompt never explodes ─────────────────────
    MAX_FOR_GPT = 150
    u = banned_ids                          # just a short alias

    valid_choices = sorted(
        valid_choices,
        key=lambda d: (
            d['adderIndex'] in u,              # ❶ fresh before reused
            -(sum(d['data']) / d['cost']),     # ❷ score
            -sum(d['data']),                   # ❸ coverage
            d['cost']                          # ❹ cheaper
        )
    )[:MAX_FOR_GPT]

    return valid_choices          # ← ← ← put this line back




def find_ones_heights(pyramid):
    """
    Calculate the heights of all diagonal columns of '1's in the pyramid,
    starting from the rightmost diagonal and moving left.

    Args:
        pyramid (list of lists): The pyramid structure.

    Returns:
        list: Heights of '1's in each diagonal from the rightmost to the leftmost.
    """
    num_rows = len(pyramid)
    heights = []

    # Start from the bottom row and traverse from right to left
    for start_col in range(len(pyramid[-1]) - 1, -1, -1):  # Start from the last column
        height = 0
        row, col = num_rows - 1, start_col  # Start at the bottom-most position

        # Traverse the diagonal upwards-left
        while row >= 0 and col >= 0:
            if col < len(pyramid[row]) and pyramid[row][col] == '1':
                height += 1
                row -= 1  # Move up
                col -= 1  # Move left
            else:
                break  # Stop if the cell is not '1' or out of bounds

        # Append the height for the current diagonal
        if height>0:
            heights.append(height)

    return heights



def remove_empty_top_rows(pyramid):
    """
    Remove any top rows that are completely empty (only spaces).
    """
    while pyramid and all(c == ' ' for c in pyramid[0]):
        pyramid.pop(0)
    return pyramid



def calculate_total_cost(closest_roww, all_selected_adders):
    """
    Calculate the total cost of the closest adder and all selected adders.

    Args:
        closest_roww (list): Closest adder returned by find_closest_element_by_transformed_vector.
        all_selected_adders (list): List of adders applied so far.

    Returns:
        int: Total cost of all adders.
    """
    total_cost = 0

    # Add the cost of the closest_roww adder (if applicable)
    if closest_roww and 'cost' in closest_roww:
        total_cost += closest_roww['cost']

    # Add the cost of all selected adders
    for adder in all_selected_adders:
        if 'cost' in adder:
            total_cost += adder['cost']

    return total_cost



def collapse_pyramid_to_one_row(pyramid):
    """
    Collapse all '1's from the pyramid into a single flat row, right before the 'X'.
    """
    total_ones = 0
    x_found = False

    # Count all '1's and check if 'X' exists
    for row in pyramid:
        for c in row:
            if c == '1':
                total_ones += 1
            if c == 'X':
                x_found = True

    if not x_found:
        raise ValueError("Pyramid must contain exactly one 'X'")

    # Now, create the final flat row
    final_row = ['1'] * total_ones + ['X']

    return [final_row]




def fetch_custom_adder() -> dict:
    """
    Keep calling the OpenAI “best–adder” helper (plus a local fallback) until
    either the pyramid is fully covered or no further progress is possible.
    Returns a dict with the final pyramid and the list of adders that were used.
    """
    global pyramid, custom_adders

    if not custom_adders:
        add_to_log("No custom adders available")
        return {"error": "No custom adders available"}

    try:
        current_pyramid       = [row[:] for row in pyramid]   # working copy
        all_selected_adders   = []
        remaining_adders      = custom_adders[:]

        # ────────────────────────────────────────────────────────────────
        #  Main loop – one iteration = “try to drop exactly one adder”.
        # ────────────────────────────────────────────────────────────────
        while not is_pyramid_covered(current_pyramid):
            used_ids = {a['adderIndex'] for a in all_selected_adders}


            # ----- progress snapshot (for the safety‑net at bottom) -----
            snapshot = [row[:] for row in current_pyramid]

            # -----------------------------------------------------------
            #  ① Ask OpenAI for the best valid adder
            # -----------------------------------------------------------
            best_adder = select_best_adder(current_pyramid,
                               remaining_adders,
                               used_ids)

            # -----------------------------------------------------------
            #  ② If OpenAI returned something usable, apply it
            # -----------------------------------------------------------
            if best_adder:
                new_pyramid, ok = apply_adder_and_shrink_pyramid(
                    current_pyramid,
                    best_adder['data'],
                    best_adder['row'],
                    best_adder['col']
                )

                if ok:                                          # success
                    current_pyramid   = remove_empty_top_rows(new_pyramid)
                    pyramid           = current_pyramid
                    all_selected_adders.append(best_adder)

                    # remove just the exact dict we used this turn
                    
                    # remaining_adders = [
                    #     a for a in remaining_adders
                    #     if a['id'] != best_adder['adderIndex']          #  ← key point
                    # ]
                else:
                    add_to_log(f"Adder could not be applied: {best_adder}")

            # -----------------------------------------------------------
            #  ③ If nothing from OpenAI was usable, fall back to “closest”
            # -----------------------------------------------------------
            else:
                heights_now  = find_ones_heights(current_pyramid)
                lookup_rows  = new_load_csv(file_path, len(heights_now))
                closest_roww = find_closest_element_by_transformed_vector(
                                   heights_now, lookup_rows)

                add_to_log(f"Closest adder: {closest_roww}")
                add_to_log(f"Selected adders so far: {all_selected_adders}")

                # ---- NORMALISE KEY so later code can rely on 'adderIndex' ----
                if closest_roww and 'adderIndex' not in closest_roww:
                    closest_roww['adderIndex'] = closest_roww.get('id')

                total_cost = calculate_total_cost(closest_roww,
                                                  all_selected_adders)
                add_to_log(f"Total cost of adders: {total_cost}")

                if closest_roww:
                    # ––– find a legal landing spot for this pattern –––
                    start_pos = find_valid_start_positions(current_pyramid)
                    if not start_pos:               # nowhere to land
                        break                       # leave the loop

                    closest_roww['row'] = start_pos[0]['row']
                    closest_roww['col'] = start_pos[0]['col']

                    new_pyr, ok = apply_adder_and_shrink_pyramid(
                        current_pyramid,
                        closest_roww['data'],
                        closest_roww['row'],
                        closest_roww['col']
                    )
                    if ok:
                        current_pyramid   = remove_empty_top_rows(new_pyr)
                        pyramid           = current_pyramid
                        all_selected_adders.append(closest_roww)
                        
                        # remaining_adders = [
                        #         a for a in remaining_adders
                        #         if a['id'] != closest_roww['adderIndex']
                        #     ]
                       
                        # no “continue” → fall through to safety‑net
                    else:
                        add_to_log("No valid adder found; terminating loop")
                        break

            # ───────────────────────────────────────────────────────────
            #  ④ SAFETY‑NET – if an iteration didn’t change the pyramid,
            #     bail out to avoid an infinite loop.
            # ───────────────────────────────────────────────────────────
            if current_pyramid == snapshot:
                add_to_log("No adder fitted this turn; exiting to avoid "
                           "infinite loop.")
                break

        # ────────────────────────────────────────────────────────────────
        #  Post‑processing
        # ────────────────────────────────────────────────────────────────
        pyramid = make_pyramid_rectangular(pyramid)
        pyramid = collapse_pyramid_to_one_row(pyramid)
        add_to_log("\nCollapsed pyramid to single row:\n"
                   + visualize_pyramid(pyramid))

        return {
            "success": bool(all_selected_adders),
            "selectedAdders": all_selected_adders,
            "pyramid": current_pyramid
        }

    except Exception as err:
        add_to_log(f"Error in fetch_custom_adder: {err}")
        traceback.print_exc()
        return {"error": f"Failed to fetch custom adder: {err}"}




# Flask routes
@app.route('/')
def index():
    return render_template(
        'dashboard.html',
        pyramid=pyramid,          # ← current global pyramid
        adder=None,               # or your “current adder” object
        selected_adders=[],       # or the list you keep
        error=None,
        bits=len(pyramid)         # lets the input show the last size
    )


@app.route('/load_adders', methods=['GET'])
def load_adders():
    return jsonify(custom_adders)

@app.route('/pyramid', methods=['GET'])
def get_pyramid():
    return jsonify({"pyramid": pyramid})

@app.route('/apply_adder', methods=['POST'])
def apply_adder():
    global pyramid
    data = request.json
    adder = data.get('adder', [])
    row = data.get('row', 0)
    col = data.get('col', 0)
    
    new_pyramid, success = apply_adder_and_shrink_pyramid(pyramid, adder, row, col)
    if success:
        pyramid = new_pyramid
        add_to_log(f"Applied adder: {adder} at row {row}, col {col}")
        return jsonify({"success": True, "pyramid": pyramid})
    else:
        return jsonify({"success": False, "error": "Adder could not be applied"})

@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify(log)

@app.route('/visualize', methods=['GET'])
def get_visualization():
    visualization = visualize_pyramid(pyramid)
    return jsonify({"visualization": visualization})

@app.route('/is_covered', methods=['GET'])
def check_coverage():
    covered = is_pyramid_covered(pyramid)
    return jsonify({"covered": covered})

@app.route('/find_positions', methods=['GET'])
def find_positions():
    positions = find_valid_start_positions(pyramid)
    return jsonify({"positions": positions})

@app.route('/select_best_adder', methods=['POST'])
def select_best_adder_route():
    data = request.json
    current_pyramid = data.get('currentPyramid', [])
    available_adders = data.get('availableAdders', [])

    result = select_best_adder(current_pyramid, available_adders, set())

    if result:
        return jsonify({"success": True, "selectedAdder": result})
    else:
        return jsonify({"success": False, "error": "Failed to select the best adder"}), 500
    

# FETCH CUSTOM ADDER

@app.route('/fetch_custom_adder', methods=['GET'])
def fetch_custom_adder_route():
    try:
        result = fetch_custom_adder()
        
        if 'error' in result:
            add_to_log(result['error'])
            return jsonify({"success": False, "error": result['error']}), 500

        return jsonify({
            "success": True,
            "selectedAdders": result['selectedAdders'],
            "pyramid": result['pyramid']
        })
    except Exception as e:
        add_to_log(f"Unexpected error: {str(e)}")
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500


# ─────────────────────────  put this in place of your current route  ─────────────────────────
@app.route('/init_pyramid', methods=['POST'])
def init_pyramid_route():
    """
    Reset the global pyramid to the requested bit‑width, clear the log,
    run the adder‑placement loop, and return everything the page needs.
    """
    global pyramid, log
    try:
        bits = int(request.json.get('bits', DEFAULT_BITS))
        if bits < 2:
            return jsonify({"error": "Need at least 2 bits"}), 400

        # build fresh triangle
        pyramid = generate_pyramid(bits)
        log.clear()
        add_to_log(f"Initialised {bits}-bit pyramid:\n" + visualize_pyramid(pyramid))

        # run the normal algorithm (calls OpenAI etc.)
        adder_result = fetch_custom_adder()

        return jsonify({
            "bits": bits,
            "pyramid": pyramid,           # final collapsed 111…X form
            "adderRun": adder_result,     # success flag & selected adders list
            "log": log                    # full step‑by‑step log for the UI
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# ──────────────────────────────────────────────────────────────────────────────────────────────





if __name__ == '__main__':
    app.run(debug=True)
