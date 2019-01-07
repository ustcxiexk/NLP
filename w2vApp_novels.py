import jieba
import gensim.models as w2v

data_path = "/home/xiexk/machineLearning/data_files/"
novel_name = "ldj.txt"
stop_words_file = open(data_path + "stop_words.txt", 'r')
stop_words = list()

novel = open(data_path + novel_name, 'r')
print("Waiting for %s ..." % novel)
seg_novel = []
line = novel.readline()

names_file = open(data_path + 'names.txt')
kongfu_file = open(data_path + 'kongfu.txt')
menpai_file = open(data_path + 'menpai.txt')

names = []
for line in names_file.readlines():
    line = line.strip()
    jieba.add_word(line)
    names.append(line)
print(len(names))

kongfu = []
for line in kongfu_file.readlines():
    line = line.strip()
    jieba.add_word(line)
    kongfu.append(line)
print(len(kongfu))

menpai = []
for line in menpai_file.readlines():
    line = line.strip()
    jieba.add_word(line)
    menpai.append(line)
print(len(menpai))

for line in stop_words_file.readlines():
    line = line.strip()
    stop_words.append(line)
stop_words_file.close()
print('Number of stop words:', len(stop_words))

while line:
    line_1 = line.strip()
    outstr = ''
    line_seg = jieba.cut(line_1, cut_all=False)
    for word in line_seg:
        if word not in stop_words:
            if word != '\t':
                outstr += word
                outstr += ' '
    if len(str(outstr.strip())) != 0:
        seg_novel.append(str(outstr.strip()).split())
    line = novel.readline()
print("finished with {} Row".format(len(seg_novel)))

all_words = []
for line in seg_novel:
    all_words.extend(line)
print(len(all_words))

print('Begin to train...')
model = w2v.Word2Vec(sentences=seg_novel, size=300, window=5, min_count=5, sg=1)
model.save(data_path + 'CBOW.model')

def find_relation(a, b, c):
    d, _ = model.most_similar(positive=[c, b], negative=[a])[0]
    print("%s and %s is to %s and %s" %(a, b, c, d))


if novel_name == 'threeBody.txt':

    print("汪淼出现的次数:", all_words.count('汪淼'))
    print("罗辑出现的次数:", all_words.count('罗辑'))
    print("大史(史强)出现的次数:", all_words.count('大史') + all_words.count('史强'))
    print("程心出现的次数:", all_words.count('程心'))
    print("天明出现的次数:", all_words.count('天明'))
    print("维德出现的次数:", all_words.count('维德'))

    print('The similarity of 程心 and 维德 is :', model.similarity('程心','维德'))
    print('The similarity of 程心 and AA is :', model.similarity('程心','AA'))
    print('The similarity of 程心 and 罗辑 is :', model.similarity('程心','罗辑'))
    print('The similarity of 程心 and 云天明 is :', model.similarity('程心', '云天明'))
    print('The similarity of 程心 and 关一帆 is :', model.similarity('程心', '关一帆'))

    print('Most close word to 程心 is: ', model.most_similar('程心', topn=10))
    print('Most close word to 罗辑 is: ', model.most_similar('罗辑', topn=10))
    print('Most close word to 云天明 is: ', model.most_similar('云天明', topn=10))
    print('Most close word to 维德 is: ', model.most_similar('维德', topn=10))
    print('Most close word to 章北海 is: ', model.most_similar('章北海', topn=10))
    print('Most close word to 汪淼 is: ', model.most_similar('汪淼', topn=10))
    print('Most close word to 面壁 is: ', model.most_similar('面壁', topn=10))

if novel_name == 'ordinary_world.txt':
    print('Most close word to 少安 is: ', model.most_similar('少安', topn=10))
    print('Most close word to 少平 is: ', model.most_similar('少平', topn=10))
    print('Most close word to 润叶 is: ', model.most_similar('润叶', topn=10))
    print('Most close word to 晓霞 is: ', model.most_similar('晓霞', topn=10))
    print('Most close word to 秀莲 is: ', model.most_similar('秀莲', topn=10))
    print(find_relation('少安', '秀莲', '少平'))
    print(find_relation('少安', '秀莲', '向前'))


if novel_name == 'yttlj.txt':
    print('Most close word to 张无忌 is: ', model.most_similar('张无忌', topn=10))
    print('Most close word to 赵敏 is: ', model.most_similar('赵敏', topn=10))
    print('Most close word to 周芷若 is: ', model.most_similar('周芷若', topn=10))
    print('Most close word to 峨嵋派 is: ', model.most_similar('峨嵋派', topn=10))
    print('Most close word to 明教 is: ', model.most_similar('明教', topn=10))
    print('The similarity of 无忌 and 赵敏 is :', model.similarity('张无忌', '赵敏'))
    print('The similarity of 无忌 and 周芷若 is :', model.similarity('张无忌', '周芷若'))
    print('The similarity of 无忌 and 小昭 is :', model.similarity('张无忌', '小昭'))
    print('The similarity of 无忌 and 明教 is :', model.similarity('张无忌', '明教'))
    print(find_relation('峨嵋', '武当', '张三丰'))

if novel_name == 'tlbb.txt':
    print('Most close word to 乔峰 is: ', model.most_similar('乔峰', topn=10))
    print('Most close word to 段誉 is: ', model.most_similar('段誉', topn=10))
    print('Most close word to 丐帮 is: ', model.most_similar('丐帮', topn=10))
    print('The similarity of 段誉 and 王语嫣 is :', model.similarity('段誉', '王语嫣'))

if novel_name == 'ldj.txt':
    print('Most close word to 韦小宝 is: ', model.most_similar('韦小宝', topn=10))
    print('The similarity of 韦小宝 and 双儿 is :', model.similarity('韦小宝', '双儿'))
    print('The similarity of 韦小宝 and 阿珂 is :', model.similarity('韦小宝', '阿珂'))
    print('The similarity of 韦小宝 and 建宁公主 is :', model.similarity('韦小宝', '建宁公主'))
    print('The similarity of 韦小宝 and 苏荃 is :', model.similarity('韦小宝', '苏荃'))
    print('The similarity of 韦小宝 and 沐剑屏 is :', model.similarity('韦小宝', '沐剑屏'))
    print('The similarity of 韦小宝 and 曾柔 is :', model.similarity('韦小宝', '曾柔'))
    print('The similarity of 韦小宝 and 方怡 is :', model.similarity('韦小宝', '方怡'))
    print('The similarity of 韦小宝 and 苏菲亚 is :', model.similarity('韦小宝', '苏菲亚'))

