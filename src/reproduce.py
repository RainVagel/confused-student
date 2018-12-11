from sklearn.naive_bayes import GaussianNB
from src.reader import reader, student_splitter, video_splitter


def student_dependent(data):
    clf = GaussianNB()
    students = student_splitter(data)


def student_independent(data):
    pass


def main():
    df = reader("EEG_data.csv")
    student_dependent(df)


if __name__ == '__main__':
    main()
