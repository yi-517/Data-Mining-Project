import jieba
import numpy
from matplotlib import pyplot


def get_stop_words(file_name):
    with open(file_name, 'r', encoding='UTF-8', errors='ignore') as f:
        return [line.strip() for line in f]


def get_content(file_name):
    return numpy.loadtxt(file_name, delimiter=',', dtype=str, usecols=1, encoding='UTF-8')


def split(sentences, stop_words):
    words = []
    for sentence in sentences:
        sentence = sentence.replace(' ', '')
        for word in jieba.cut(sentence):
            if word not in stop_words:
                words.append(word)
    return words


def get_length(sentences):
    length = []
    for sentence in sentences:
        length.append(len(sentence))
    return length


def get_tf(words, top):
    tf_dictionary = {}
    for word in words:
        tf_dictionary[word] = tf_dictionary.get(word, 0) + 1
    return sorted(tf_dictionary.items(), key=lambda x: x[1], reverse=True)[:top]


def bar_plot(tf, topic, file_name):
    x = []
    y = []
    for item in tf:
        x.append(item[0])
        y.append(item[1])
    pyplot.rcParams['font.sans-serif'] = ['KaiTi']
    pyplot.bar(x, y, color='yellow', edgecolor='gray', alpha=0.6)
    pyplot.title(topic)
    pyplot.savefig(file_name + '-bar.png')
    pyplot.show()


def histogram_plot(data, topic, file_name):
    pyplot.rcParams['font.sans-serif'] = ['KaiTi']
    pyplot.hist(data, color='blue', edgecolor='gray', alpha=0.4)
    pyplot.title(topic)
    pyplot.savefig(file_name + '-histogram.png')
    pyplot.show()


if __name__ == "__main__":
    files = ['preprocess_东奥瞬间.csv', 'preprocess_步法.csv', 'preprocess_冬奥热点.csv', 'preprocess_跳跃.csv']
    for file in files:
        titles = get_content('./Dataset/' + file)
        bar_plot(
            tf=get_tf(words=split(titles, stop_words=get_stop_words('stop_word.txt')), top=10),
            topic='标题词频统计',
            file_name=file.split('_')[1].split('.')[0]
        )
        histogram_plot(
            data=get_length(titles),
            topic='标题长度分布',
            file_name=file.split('_')[1].split('.')[0]
        )
