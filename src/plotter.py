import matplotlib.pyplot as plt


def title_maker(title):
    title = title.replace(" ", "_")
    return title + ".png"


# def reproduce_plotter(path, performance_old, performance, labels, x_label, y_label, title):
#     y_pos = range(len(labels))
#     # print(y_pos)
#     plt.bar(y_pos, performance_old, width=0.25, label="Original")
#     y_pos = [x + 0.25 for x in y_pos]
#     plt.bar(y_pos, performance, width=0.25, label="Reproduced")
#     plt.xticks(y_pos, labels)
#     plt.ylabel(y_label)
#     plt.xlabel(x_label)
#     plt.title(title)
#     plt.legend()
#
#     plt.savefig(path+title_maker(title))
#     plt.close()


def reproduce_plotter(path, performances, labels, x_label, y_label, title):
    y_pos = range(len(labels))
    for performance in performances:
        plt.bar(y_pos, performance[1], width=0.25, label=performance[0])
        y_pos = [x + 0.25 for x in y_pos]
    plt.xticks(y_pos, labels)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.savefig(path+title_maker(title))
    plt.close()


def formatter(scores, title):
    avg = sum(scores.values()) / float(len(scores.values()))
    temp = [avg] + list(scores.values())
    final = [title, temp]
    return final
