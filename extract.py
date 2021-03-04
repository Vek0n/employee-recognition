import cv2

def read_file():
    with open('cam1.coords', 'r') as reader:
        raw_data = reader.readlines()
    lines = []
    for i in raw_data:
        line = list(map(int, i.split()))
        lines.append(line)
    return lines


def get_boundingbox_coords(line):
    w = line[2] - line[0]
    h = line[3] - line[1]
    y = line[1]
    x = line[0]
    return x,y,w,h


# cap = cv2.VideoCapture("http://192.168.0.115:8081/video")
cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    count += 1
    
    if ret:
        if count % 15 == 0:
            lines = read_file()
            i = len(lines)
            for l in lines:
                x,y,w,h = get_boundingbox_coords(l)
                crop_img = frame[y:y+h, x:x+w]
                cv2.imwrite('frame%d.jpg'%i, crop_img)
                i-=1
            count = 0
    else:
        cap.release()
        break