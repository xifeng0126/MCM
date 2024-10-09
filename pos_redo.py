import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# 确保已经下载了NLTK的数据和模型
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# 读取xlsx文件
df = pd.read_excel("Problem_C_Data_Wordle.xlsx")

# 定义一个函数来标注词性
def tag_pos(word):
    # 使用NLTK的word_tokenize来分割单词，然后使用pos_tag进行词性标注
    tokens = word_tokenize(word)
    tagged = nltk.pos_tag(tokens)
    # 只取第一个单词的词性标签
    return tagged[0][1] if tagged else 'N/A'

# 应用函数并创建新列
df['POS'] = df['Word'].apply(tag_pos)

# 将DataFrame保存回xlsx文件
df.to_excel("updated_file.xlsx", index=False)
