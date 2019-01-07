import gensim.models as w2v

def read_data(filename):

    google_stuff = open(filename)
    data = []
    for line in google_stuff.readlines():
        data.append(line.split())
    return data

data_path = "/home/xiexk/machineLearning/data_files/"
novel_name = 'text8'
isTrain = False

if isTrain:
    seg_stuff = read_data(data_path + "text8.txt")
    sentences = []
    for i in range(0, len(seg_stuff[0]), 20):
        sentences.append(seg_stuff[0][i: i+20])
    print('Begin to train...')
    model = w2v.Word2Vec(sentences=sentences, size=200, window=5, min_count=5, sg=1)
    model.save('skip-gram.model')
else:
    model = w2v.Word2Vec.load('skip-gram.model')


def find_relation(a, b, c):
    d, _ = model.most_similar(positive=[b, c], negative=[a])[0]
    return d


def load_test_exams(testfilename):
    examples = []
    testfile = open(testfilename)
    for line in testfile.readlines():
        line = line.lower()
        temp = line.split()
        if temp[0] != ':':
            examples.append(temp)
    return examples

if novel_name == 'text8':
    fileTest = False
    if fileTest:
        examples = load_test_exams('google_exams.txt')
        print(len(examples))
        print(examples[0])
        count = 0
        for i in range(len(examples)):
            try:
                result = find_relation(examples[i][0], examples[i][1], examples[i][2])
            except:
                pass
            if result == examples[i][3]:
                count = count + 1
        accuracy = count / len(examples)
        print(accuracy)

    print('Most close word to \'beijing\' is: ', model.most_similar('beijing', topn=5))
    print('Most close word to \'but\' is: ', model.most_similar('but', topn=5))
    print('Most close word to \'three\' is: ', model.most_similar('three', topn=5))

    print('Man to woman is king to: ', find_relation('man', 'king', 'woman'))
    print('Germany to France is to Berlin: ', find_relation('germany', 'berlin', 'france'))
    print('Spain to France is to Madrid to: ', find_relation('spain', 'madrid', 'france'))
    print('Boy to son is girl to: ', find_relation('boy', 'girl', 'son'))
    print('Quick to quicker is slow to: ', find_relation('quick', 'slow', 'quicker'))
    print('Walking to walked is to running to: ', find_relation('walking', 'running', 'walked'))


