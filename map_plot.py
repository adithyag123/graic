import matplotlib.pyplot as plt
with open('center_points.txt') as f:
    lines = f.readlines()
    xlist = []
    ylist = []
    for num, loc in enumerate(lines):
        pos = float(loc.split(' ')[1][:-1])
        if num%3 == 0:
            xlist.append(pos)
        elif num%3 == 1:
            ylist.append(pos)


    plt.plot(xlist, ylist)
    plt.show()
