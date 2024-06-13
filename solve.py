import cv2
import sys
import numpy as np
from tqdm import trange
import string
from matplotlib import pyplot as plt

method = 1
if len(sys.argv) > 2:
    method = int(sys.argv[2])
assert method in set([1, 2])

video_cap = cv2.VideoCapture("v2.mp4")

length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
last_pos = None
count = 0

positions = [
    ('a', 74.5, 202.5),
    ('b', 118.5, 172.5),
    ('c', 161.5, 155.5),
    ('d', 202.5, 139.5),
    ('e', 247.5, 135.5),
    ('f', 289.5, 128.5),
    ('g', 329.5, 124.5),
    ('h', 382.5, 130.5),
    ('i', 420.5, 137.5),
    ('j', 453.5, 140.5),
    ('k', 496.5, 153.5),
    ('l', 545.5, 174.5),
    ('m', 600.5, 196.5),
    ('n', 65.5, 293.5),
    ('o', 116.5, 270.5),
    ('p', 146.5, 241.5),
    ('q', 196.5, 222.5),
    ('r', 243.5, 207.5),
    ('s', 289.5, 201.5),
    ('t', 333.5, 193.5),
    ('u', 376.5, 202.5),
    ('v', 423.5, 207.5),
    ('w', 472.5, 222.5),
    ('x', 521.5, 244.5),
    ('y', 557.5, 267.5),
    ('z', 601.5, 300.5),
    ('caps', 151.5, 60.5),
    ('1', 141.5, 334.5),
    ('2', 181.5, 336.5),
    ('3', 222.5, 340.5),
    ('4', 267.5, 335.5),
    ('5', 318.5, 342.5),
    ('6', 358.5, 337.5),
    ('7', 400.5, 338.5),
    ('8', 441.5, 338.5),
    ('9', 488.5, 334.5),
    ('0', 535.5, 344.5),
    ('/', 522.5, 63.5),
    ('+', 266.5, 400.5),
    ('=', 435.5, 406.5),
]

def dist2(p, q):
    px, py = p
    qx, qy = q
    return (px - qx) ** 2 + (py - qy) ** 2

caps = True
def get_text(pos, tick):
    global caps

    log = False

    mini, minch = None, None
    for ch, x, y in positions:
        dist = dist2(pos, (x, y))
        if mini is None or dist < mini:
            mini = dist
            minch = ch
    
    if mini > 200.0:
        print(f"tick {tick} is wrong: {mini}, {pos}, {minch}")
        return None, True
    elif mini > 110.0:
        print(f"tick {tick} is weird: {mini}, {pos}, {minch}")
        log = True

    if minch == 'caps':
        caps = not caps
        return "caps", log
    else:
        return minch.lower(), log

    if caps:
        return minch.lower(), log
    else:
        return minch.upper(), log

# f = open('output.txt', 'w')

q = []

prev_character = None

track_positions = []
track = open(sys.argv[1], "r").read().replace("(", "").replace(")", "").strip().split("\n")
for i in track:
    temp = i.split(",")
    track_positions.append(tuple([int(j.strip()) for j in temp]))

pos_idx = 0

def dist2(p, q):
    px, py = p
    qx, qy = q
    return (px - qx) ** 2 + (py - qy) ** 2

recovered = []

count = 0
last_pos = (0, 0)
for tick in trange(length):
    found = False
    # print(tick, last_pos)
    if pos_idx < len(track_positions) and track_positions[pos_idx][0] == tick:
        found = True
        next_pos = track_positions[pos_idx][1:]
        next_pos = (next_pos[0] + next_pos[2] // 2, next_pos[1] + next_pos[3] // 2)
        # next_pos = (next_pos[0] + next_pos[2] // 2, next_pos[1] + 60)
        pos_idx += 1

    if found and next_pos is not None:
        if method == 2:
            dist = dist2(next_pos, last_pos)
            if dist <= 90.0:
                count += 1
                if count == 10:
                    ch, log = get_text(last_pos, tick)
                    if ch is not None:
                        recovered.append(ch)
            else:
                # if count == 5:
                #     print(f"check frame_{tick - 1}")
                count = 0
            last_pos = next_pos

        if method == 1:
            if prev_character is None or dist2(next_pos, prev_character) > 150:
                q.append(next_pos)
                prev_character = None
            if len(q) > 8:
                q.pop(0)
            all_close = True
            for i in range(len(q)):
                d1 = dist2(q[i], q[0])
                d2 = dist2(q[i], q[-1])
                if d1 > 70 or d2 > 70:
                    all_close = False
            if all_close:
                if len(q) == 8:
                    prev_character = q[-1]
                    ch, log = get_text(last_pos, tick)
                    q = []
                    if ch is not None:
                        recovered.append(ch)
                    #     f.write(ch)
                    #     f.flush()
                    # if log:
                    #     cv2.circle(frame, (int(next_pos[0]), int(next_pos[1])), int(next_r), (0, 255, 0), 2)
                    #     cv2.imwrite(f"frames/frame{tick}.png", frame)
                    #     with open(f"frames/frame{tick}.txt", "w") as lol:
                    #         lol.write(f"{next_pos[0]},{next_pos[1]},{next_r}")
                # count += 1
                # if count == 8:
                #     ch = get_text(last_pos, tick)
                #     if ch is None:
                #         print(f"Tick {tick} is something wrong, {last_pos}")
                #     else:
                #         f.write(ch)
                #         f.flush()
            last_pos = next_pos

def lcs(X, Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
 
    # declaring the array for storing the dp values 
    L = [[None]*(n + 1) for i in range(m + 1)] 
 
    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n]
 
flag = open("flag1.b64", "r").read().strip()
ground_truth = []
caps = False
for ch in flag:
    if (ch in string.ascii_lowercase and caps) or (ch in string.ascii_uppercase and not caps):
        ground_truth.append("caps")
        caps = not caps
    if ch.lower() in string.ascii_lowercase:
        ground_truth.append(ch.lower())
    else:
        ground_truth.append(ch)

longest_length = lcs(recovered, ground_truth)
print(longest_length, len(ground_truth), len(recovered), longest_length / len(ground_truth))

# Do plotting
