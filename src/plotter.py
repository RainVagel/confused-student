import matplotlib.pyplot as plt

def title_maker(title):
    title = title.replace(" ", "_")
    return title + ".png"


def reproduce_plotter(path, performance_old, performance, labels, x_label, y_label, title):
    y_pos = range(len(labels))
    # print(y_pos)
    plt.bar(y_pos, performance_old, width=0.25, label="Original")
    y_pos = [x + 0.25 for x in y_pos]
    plt.bar(y_pos, performance, width=0.25, label="Reproduced")
    plt.xticks(y_pos, labels)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()

    plt.savefig(path+title_maker(title))
    plt.close()
