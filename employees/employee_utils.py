def read_coords_file():
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