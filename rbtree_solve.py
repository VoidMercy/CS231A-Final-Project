import cv2
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

def dist2(p, q):
    px, py = p
    qx, qy = q
    return (px - qx) ** 2 + (py - qy) ** 2

caps = True
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
result = open("rbtree_results.txt", "w")

for tick in trange(length):
    # print(tick, last_pos)
    success, frame = video_cap.read()

    # if tick <= 8185:
    #     continue

    frame = frame[116:580, 300:953]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray_blurred = cv2.blur(gray, (3, 3)) 
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                param2 = 30, minRadius = 30, maxRadius = 55) 

    if detected_circles is None:
        continue

    detected_circles = detected_circles[0, :]
    if last_pos is None:
        assert len(detected_circles) == 1
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

        cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)

        if cl is None or cl > dist:
            cl = dist
            next_pos = x, y
    
    if next_pos is not None:
        result.write(str((tick, int(next_pos[0]), int(next_pos[1]), 0, 0)) + "\n")
        dist = dist2(next_pos, last_pos)
        if dist <= 90.0:
            count += 1
            if count == 5:
                cv2.imwrite(f"h1.png", frame)
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