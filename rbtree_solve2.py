import cv2
import time
import numpy as np
from tqdm import trange

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
    ('m', 595.5, 200.5),
    ('n', 65.5, 297.5),
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
    ('+', 256.5, 409.5),
    ('+', 266.5, 400.5),
    ('=', 435.5, 406.5),
]

def subtract_and_clamp(image1, image2):
    # Perform subtraction
    result = image1.astype(np.int16) - image2.astype(np.int16)
    # Clamp pixel values to ensure they stay within [0, 255]
    result_clamped = np.clip(result, 0, 255).astype(np.uint8)
    return result_clamped

def dist2(p, q):
    px, py = p
    qx, qy = q
    return (px - qx) ** 2 + (py - qy) ** 2

caps = False
def get_text(pos, tick):
    global caps

    mini, minch = None, None
    for ch, x, y in positions:
        dist = dist2(pos, (x, y))
        if mini is None or dist < mini:
            mini = dist
            minch = ch
    
    if mini > 180.0:
        print(f"tick {tick} is wrong: {mini}, {pos}, {minch}")
        return True, None
    elif mini > 120.0:
        print(f"tick {tick} is weird: {mini}, {pos}, {minch}")
        flag = True
    else:
        flag = False

    if minch == 'caps':
        caps = not caps
        return flag, ""

    if caps:
        return flag, minch.lower()
    else:
        return flag, minch.upper()

f = open('output.txt', 'w')
result = open("rbtree2_results.txt", "w")

bg = cv2.imread("bg.png")

prev_frame = None

THRESHOLD_N = 8
THRESHOLD = 200000
history = []

pause = False

seq = []

for tick in trange(length):
    # print(tick, last_pos)
    success, frame = video_cap.read()

    frame = frame[116:580, 300:953]
    frame = subtract_and_clamp(frame, bg)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_blurred = cv2.blur(gray, (3, 3)) 
    gray_blurred = (gray * 2.5).astype(np.uint8)
    gray_rgb = cv2.cvtColor(gray_blurred,cv2.COLOR_BGR2RGB)
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 70, 
                param2 = 40, minRadius = 30, maxRadius = 55) 

    if detected_circles is None:
        continue

    detected_circles = detected_circles[0, :]
    if last_pos is None:
        # assert len(detected_circles) == 1
        x, y, r = detected_circles[0]
        last_pos = (x, y)
        continue

    cl, next_pos = None, None
    for x, y, r in detected_circles:
        if x <= 65.0 and y <= 60.0: # Sun
            continue
        if x >= 580.0 and y <= 75.0: # Moon
            continue

        dist = dist2((x, y), last_pos)

        cv2.circle(gray_rgb, (int(x), int(y)), int(r), (0, 255, 0), 2)

        if cl is None or cl > dist:
            cl = dist
            next_pos = x, y
    # if prev_frame is not None:
    #     subtracted = subtract_and_clamp(gray_blurred, prev_frame)
    #     flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_blurred,  
    #                                    None, 
    #                                    0.5, 1, 15, 3, 7, 1.5, 0) 
    #     magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
    #     history.append(np.sum(subtracted))
    #     if len(history) > 10:
    #         history.pop(0)

    #     cv2.imshow("Frame", gray_blurred)
    #     key = cv2.waitKey(800)
    #     if key == "q":
    #         break

    # prev_frame = gray_blurred

    if next_pos is not None:
        result.write(str((tick, int(next_pos[0]), int(next_pos[1]), 0, 0)) + "\n")

        # if len(history) > 0 and history[-1] >= THRESHOLD:
        #     pause = False

        # if all([i < THRESHOLD for i in history[-THRESHOLD_N:]]) and not pause:
        #     flag, ch = get_text(last_pos, tick)
        #     if ch is not None:
        #         f.write(ch)
        #         f.flush()
        #         pause = True

        dist = dist2(next_pos, last_pos)
        if dist <= 90.0:
            count += 1
            if count == 5:
                cv2.imwrite(f"h2.png", gray_rgb)
            if count == 10:
                flag, ch = get_text(last_pos, tick)
                if flag:
                    # print (f"Tick {tick} is something wrong, {last_pos}")
                    # cv2.imwrite(f"frames/frame_{tick}.jpg", frame)
                    pass
                else:
                    f.write(ch)
                    f.flush()
        else:
            # if count == 5:
            #     print(f"check frame_{tick - 1}")
            count = 0
        last_pos = next_pos

f.close()
result.close()

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
print(longest_length, len(ground_truth), len(recovered))
