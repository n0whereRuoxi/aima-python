import inspect
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle
import numpy as np

def plotTest11(seed=None):
    # Before pickle was used to save data, I printed the data to terminal
    # Thus, in order to save the results, I have manually copy-pasted results
    # for a 20x20 environment with teams of 20 agents, while varying
    # sensor radius from 2 to 10.
    results = [(2, -0.0, 221.4), (2, 0.28639695711595625, 248.5), (2, 0.4689955935892812, 252.7), (2, 0.6098403047164004, 267.4), (2, 0.7219280948873623, 297.5), (2, 0.8112781244591328, 289.4), (2, 0.8812908992306927, 264.0), (2, 0.934068055375491, 273.5), (2, 0.9709505944546686, 281.1), (2, 0.9927744539878084, 310.3), (2, 1.0, 421.6), (2, 0.9927744539878084, 446.8), (2, 0.9709505944546686, 387.9), (2, 0.934068055375491, 387.6), (2, 0.8812908992306927, 544.1), (2, 0.8112781244591328, 442.4), (2, 0.7219280948873623, 723.0), (2, 0.6098403047164004, 1105.4), (2, 0.4689955935892812, 890.0), (2, 0.28639695711595625, 1596.5), (3, -0.0, 206.0), (3, 0.28639695711595625, 241.1), (3, 0.4689955935892812, 274.3), (3, 0.6098403047164004, 233.6), (3, 0.7219280948873623, 232.4), (3, 0.8112781244591328, 252.6), (3, 0.8812908992306927, 226.1), (3, 0.934068055375491, 218.6), (3, 0.9709505944546686, 265.6), (3, 0.9927744539878084, 163.0), (3, 1.0, 193.0), (3, 0.9927744539878084, 214.3), (3, 0.9709505944546686, 216.6), (3, 0.934068055375491, 185.2), (3, 0.8812908992306927, 387.6), (3, 0.8112781244591328, 146.2), (3, 0.7219280948873623, 298.3), (3, 0.6098403047164004, 160.1), (3, 0.4689955935892812, 426.8), (3, 0.28639695711595625, 483.9), (4, -0.0, 176.5), (4, 0.28639695711595625, 243.6), (4, 0.4689955935892812, 252.6), (4, 0.6098403047164004, 185.7), (4, 0.7219280948873623, 220.7), (4, 0.8112781244591328, 213.0), (4, 0.8812908992306927, 244.3), (4, 0.934068055375491, 253.5), (4, 0.9709505944546686, 167.8), (4, 0.9927744539878084, 197.7), (4, 1.0, 157.2), (4, 0.9927744539878084, 141.2), (4, 0.9709505944546686, 138.4), (4, 0.934068055375491, 195.0), (4, 0.8812908992306927, 116.6), (4, 0.8112781244591328, 154.1), (4, 0.7219280948873623, 145.4), (4, 0.6098403047164004, 127.5), (4, 0.4689955935892812, 174.2), (4, 0.28639695711595625, 556.9), (5, -0.0, 207.5), (5, 0.28639695711595625, 263.8), (5, 0.4689955935892812, 258.6), (5, 0.6098403047164004, 260.2), (5, 0.7219280948873623, 180.7), (5, 0.8112781244591328, 108.6), (5, 0.8812908992306927, 171.7), (5, 0.934068055375491, 145.9), (5, 0.9709505944546686, 171.4), (5, 0.9927744539878084, 176.7), (5, 1.0, 140.1), (5, 0.9927744539878084, 64.4), (5, 0.9709505944546686, 110.5), (5, 0.934068055375491, 89.6), (5, 0.8812908992306927, 138.1), (5, 0.8112781244591328, 92.0), (5, 0.7219280948873623, 140.8), (5, 0.6098403047164004, 133.5), (5, 0.4689955935892812, 216.2), (5, 0.28639695711595625, 314.6), (6, -0.0, 204.2), (6, 0.28639695711595625, 257.0), (6, 0.4689955935892812, 202.1), (6, 0.6098403047164004, 239.7), (6, 0.7219280948873623, 158.0), (6, 0.8112781244591328, 214.3), (6, 0.8812908992306927, 114.7), (6, 0.934068055375491, 114.0), (6, 0.9709505944546686, 69.2), (6, 0.9927744539878084, 99.1), (6, 1.0, 88.3), (6, 0.9927744539878084, 96.5), (6, 0.9709505944546686, 71.8), (6, 0.934068055375491, 92.1), (6, 0.8812908992306927, 139.6), (6, 0.8112781244591328, 134.5), (6, 0.7219280948873623, 100.7), (6, 0.6098403047164004, 171.1), (6, 0.4689955935892812, 181.1), (6, 0.28639695711595625, 336.0), (7, -0.0, 220.4), (7, 0.28639695711595625, 280.2), (7, 0.4689955935892812, 274.4), (7, 0.6098403047164004, 148.8), (7, 0.7219280948873623, 135.4), (7, 0.8112781244591328, 140.3), (7, 0.8812908992306927, 76.8), (7, 0.934068055375491, 89.4), (7, 0.9709505944546686, 90.4), (7, 0.9927744539878084, 102.3), (7, 1.0, 52.7), (7, 0.9927744539878084, 84.5), (7, 0.9709505944546686, 57.1), (7, 0.934068055375491, 61.8), (7, 0.8812908992306927, 81.5), (7, 0.8112781244591328, 87.0), (7, 0.7219280948873623, 106.1), (7, 0.6098403047164004, 124.2), (7, 0.4689955935892812, 170.3), (7, 0.28639695711595625, 465.3), (8, -0.0, 170.7), (8, 0.28639695711595625, 241.1), (8, 0.4689955935892812, 189.1), (8, 0.6098403047164004, 104.4), (8, 0.7219280948873623, 100.4), (8, 0.8112781244591328, 93.7), (8, 0.8812908992306927, 123.4), (8, 0.934068055375491, 98.8), (8, 0.9709505944546686, 96.9), (8, 0.9927744539878084, 65.5), (8, 1.0, 51.9), (8, 0.9927744539878084, 55.8), (8, 0.9709505944546686, 57.6), (8, 0.934068055375491, 103.7), (8, 0.8812908992306927, 76.0), (8, 0.8112781244591328, 86.7), (8, 0.7219280948873623, 97.9), (8, 0.6098403047164004, 124.0), (8, 0.4689955935892812, 174.5), (8, 0.28639695711595625, 467.0), (9, -0.0, 186.3), (9, 0.28639695711595625, 245.4), (9, 0.4689955935892812, 175.1), (9, 0.6098403047164004, 141.8), (9, 0.7219280948873623, 87.2), (9, 0.8112781244591328, 86.3), (9, 0.8812908992306927, 78.6), (9, 0.934068055375491, 47.0), (9, 0.9709505944546686, 44.9), (9, 0.9927744539878084, 49.5), (9, 1.0, 50.3), (9, 0.9927744539878084, 52.6), (9, 0.9709505944546686, 56.0), (9, 0.934068055375491, 68.6), (9, 0.8812908992306927, 77.3), (9, 0.8112781244591328, 91.9), (9, 0.7219280948873623, 100.4), (9, 0.6098403047164004, 127.9), (9, 0.4689955935892812, 179.9), (9, 0.28639695711595625, 481.8), (10, -0.0, 222.7), (10, 0.28639695711595625, 217.8), (10, 0.4689955935892812, 129.1), (10, 0.6098403047164004, 121.0), (10, 0.7219280948873623, 79.0), (10, 0.8112781244591328, 37.8), (10, 0.8812908992306927, 58.5), (10, 0.934068055375491, 54.8), (10, 0.9709505944546686, 41.3), (10, 0.9927744539878084, 49.3), (10, 1.0, 47.8), (10, 0.9927744539878084, 55.3), (10, 0.9709505944546686, 58.5), (10, 0.934068055375491, 64.1), (10, 0.8812908992306927, 76.1), (10, 0.8112781244591328, 86.9), (10, 0.7219280948873623, 100.8), (10, 0.6098403047164004, 119.2), (10, 0.4689955935892812, 172.7), (10, 0.28639695711595625, 821.5)]
    environment_width = 20
    environment_height = 20
    team_size = 20

    # Result lists for plotting
    s = [] # sensor radii
    e = [] # simple social entropy
    c = [] # average completion time

    for result in results:
        s.append(result[0])
        e.append(result[1])
        c.append(result[2])

    # 3D Scatterplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(s, e, c, marker='o')
    ax.set_title('%s x %s Environment of %s(), seed=%s, team size=%s, agent types=2, averaged over %s runs each' \
              % (environment_width, environment_height, inspect.stack()[0][3], seed, team_size, 10))
    ax.set_xlabel('Sensor Radius')
    ax.set_ylabel('Simple Social Entropy')
    ax.set_zlabel('Average Completion Time')
    plt.show()

def plotTest11PickleData(environment_width, environment_height, team_size, runs_to_average, max_steps, sensor_radius):
    filename = "50_50_30_20_3000_5to15/test11_%s_%s_%s_%s_%s_iter%s.p" % (environment_width, environment_height, team_size, runs_to_average, max_steps, sensor_radius)
    test_11_data = pickle.load(open(filename, "rb"))

    s = test_11_data['s']
    e = test_11_data['e']
    c = test_11_data['c']

    min_sensor_radius = 5

    # Create color linspace
    team_linspace = np.linspace(0, 1, team_size)
    unit_linspace = np.linspace(0, 1, team_size)
    for i in range(sensor_radius - min_sensor_radius):
        team_linspace = np.append(team_linspace, unit_linspace)
    colors = cm.get_cmap('cool', team_size)

    # 3D Scatterplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(s, e, c, marker='o', c=colors(team_linspace))
    ax.set_title('%s x %s Environment, seed=%s, team size=%s, agent types=2, max_steps=%s, averaged over %s runs each' \
              % (environment_width, environment_height, None, team_size, max_steps, runs_to_average))
    ax.set_xlabel('Sensor Radius')
    ax.set_ylabel('Simple Social Entropy')
    ax.set_zlabel('Average Completion Time')
    plt.show()

def plotTest11PickleDataRatio(environment_width, environment_height, team_size, runs_to_average, max_steps, sensor_radius):
    filename = "50_50_30_20_3000_5to15/test11_%s_%s_%s_%s_%s_iter%s.p" % (environment_width, environment_height, team_size, runs_to_average, max_steps, sensor_radius)
    test_11_data = pickle.load(open(filename, "rb"))

    s = test_11_data['s']
    e = test_11_data['e']
    c = test_11_data['c']

    min_sensor_radius = 5

    # Create color linspace
    team_linspace = np.linspace(0, 1, team_size)
    unit_linspace = np.linspace(0, 1, team_size)
    for i in range(sensor_radius - min_sensor_radius):
        team_linspace = np.append(team_linspace, unit_linspace)
    colors = cm.get_cmap('cool', team_size)

    ratios = [(team_size-i)/team_size for i in range(0, team_size)] * (sensor_radius - min_sensor_radius + 1)

    # for i in range(20):
    #     min_avg_completion = min(c)
    #     min_index = c.index(min_avg_completion)
    #     print(f"{i+1}. (Sensor radius: {s[min_index]}, Ratio of Roomba: {ratios[min_index]}, Avg completion: {c[min_index]})")
    #
    #     del s[min_index]
    #     del ratios[min_index]
    #     del c[min_index]

    # 3D Scatterplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(s, ratios, c, marker='o', c=colors(team_linspace))
    ax.set_title('%s x %s Environment, seed=%s, team size=%s, \nagent types=2, max_steps=%s, averaged over %s runs each' \
              % (environment_width, environment_height, None, team_size, max_steps, runs_to_average))
    ax.set_xlabel('Sensor Radius')
    ax.set_ylabel('Ratio of Roomba in Team')
    ax.set_zlabel('Average Completion Time')
    plt.show()

def plotBestTeams(environment_width, environment_height, team_size, runs_to_average, max_steps, sensor_radius):
    filename = "../50_50_30_100_3000_9to15/test11_%s_%s_%s_%s_%s_iter%s.p" % (environment_width, environment_height, team_size, runs_to_average, max_steps, sensor_radius)
    test_11_data = pickle.load(open(filename, "rb"))

    s = test_11_data['s']
    c = test_11_data['c']

    min_sensor_radius = 9
    ratios = [(team_size-i)/team_size for i in range(0, team_size)] * (sensor_radius - min_sensor_radius + 1)

    # Plot Avg Completion Time vs Roomba Ratio
    # smoothWindow = 7 # Should be odd
    sensor_radii_to_plot = [15]#[15, 14, 13, 12, 11, 10, 9]
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 0.9, len(sensor_radii_to_plot)))))
    for j, sensor_radii in enumerate(sensor_radii_to_plot):
        start_index_of_data = (sensor_radii - min_sensor_radius) * team_size
        current_avg_completion_data = c[start_index_of_data:start_index_of_data + team_size]
        current_ratio_data = ratios[start_index_of_data:start_index_of_data + team_size]

        x = current_ratio_data#smoothData(current_ratio_data, smoothWindow)
        y = current_avg_completion_data#smoothData(current_avg_completion_data, smoothWindow)

        # calculate polynomial
        z = np.polyfit(x, y, 2)
        f = np.poly1d(z)

        # calculate new x's and y's
        x_new = np.linspace(x[0], x[-1], 50)
        y_new = f(x_new)

        # annot_min(np.array(x),np.array(y))
        plt.plot(x, y, 'o', x_new, y_new)


        ax = plt.gca()
        ax.text(0, 1030, f"y={z[0]}x^2+({z[1]})x + {z[2]}")
        ax.text(0, 950, "Minimum at {:.3f}".format(-z[1]/(2*z[0])))

    legend = ["sensor_radius="+str(x) for x in sensor_radii_to_plot]
    plt.legend(legend, loc='upper left')
    plt.title('Roomba/Drone Teams of Size %s for varying drone sensor_radius (%sx%s env, average over %s samples each)' % (team_size, environment_height, environment_width, runs_to_average))
    plt.xlabel('Ratio of Roomba')
    plt.ylabel('Avg Completion Time')
    plt.show()

def plotBestTeamsWithTupleData(environment_width, environment_height, team_size, runs_to_average, max_steps, min_sensor_radius, max_sensor_radius):
    # Assumes files are spread out across multiple files with distinct data
    test_11_data = []
    for i in range(min_sensor_radius, max_sensor_radius + 1):
        filename = "../../test11_%s_%s_%s_%s_%s_iter%s.p" % (environment_width, environment_height, team_size, runs_to_average, max_steps, i)
        test_11_data += pickle.load(open(filename, "rb"))

    # Plot Avg Completion Time vs Roomba Ratio
    sensor_radii_to_plot = list(range(min_sensor_radius, max_sensor_radius + 1))
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 0.9, len(sensor_radii_to_plot)))))
    for j, sensor_radii in enumerate(sensor_radii_to_plot):
        filtered_data = list(filter(lambda tup: tup[0] == sensor_radii, test_11_data))

        # Extract sensor radius, ratio, and completion time
        s = [tup[0] for tup in filtered_data]
        r = [tup[1] for tup in filtered_data]
        c = [np.average(tup[2]) for tup in filtered_data]

        # Annotate the minimum for min/max sensor radius
        if sensor_radii == min_sensor_radius or sensor_radii == max_sensor_radius:
            annot_min(np.array(r),np.array(c))
        plt.plot(r, c)


    legend = ["sensor_radius=" + str(x) for x in sensor_radii_to_plot]
    plt.legend(legend, loc='upper left')
    plt.title('Roomba/Drone Teams of Size %s for varying drone sensor_radius (%sx%s env, average over %s samples each)' % (team_size, environment_height, environment_width, runs_to_average))
    plt.xlabel('Ratio of Roomba')
    plt.ylabel('Avg Completion Time')
    plt.show()

def smoothData(data, windowSize):
    # Window size should be odd
    smoothedData = []
    sideWindow = int((windowSize-1) / 2)
    for i in range(sideWindow, len(data)-sideWindow):
        avg = sum(data[i-sideWindow:i+sideWindow+1]) / windowSize
        smoothedData.append(avg)
    return smoothedData

def annot_min(x,y, ax=None):
    xmax = x[np.argmin(y)]
    ymax = y.min()
    text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=160")
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax+.5,ymax+5), **kw)

def main():
    plotBestTeamsWithTupleData(70,70,40,100,3000,9,13)

if __name__ == "__main__":
    # execute only if run as a script
    main()
