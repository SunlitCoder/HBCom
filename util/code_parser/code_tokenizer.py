#coding=utf-8
import nltk
import re
from nltk.tokenize import MWETokenizer
# from typing import List
from nltk import WordPunctTokenizer
from util.nl_parser.en_parser import punc_str
from copy import deepcopy
import numpy as np
# from nltk.corpus import wordnet
from ..nl_parser.en_parser import EnWordCheck,SP_ABBR_DICT

def _lemmatize_token(token):
    '''
    Lemmatize the word
    Note that, the short word like "as" will be lemmatized to "a".
    So only a word of which the length is more than 2 will be lemmatized
    :param token:
    :return: 
    '''
    error_lemmatized_token_dict={'is':'be','am':'be','does':'do','has':'have','as':'as','was':'be','parses':'parse','cls':'cls','tokenizing':'tokenize'}
    if token in error_lemmatized_token_dict.keys():
        return error_lemmatized_token_dict[token]
    elif len(token) > 2:
        lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
        return lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos='n'), pos='v'), pos='a')
    return token


class CompoundWordSplitter(object):
    def __init__(self, user_words=None, exclude_words=None, word2weight=None):
        '''
        Segment a string of compound word using the pyenchant vocabulary.
        Keeps most possible words that account for all characters,
        and returns list of segmented words.
        首先将可能存在snake_case和CamelCase分开，然后对其中分开的每个单词再进行分割
        基本逻辑为，提取尽可能所有的word组合列表，然后根据规则选择最合适的，具体规则如下：
        (1)首先，选择(noise) string个数最少的
        (2)否则，选择选择rword平均长度减去noise string平均长度最大的
        (3)否则，选择new word序列最短的
        #(3)否则，选择noise string总长度最小的
        (4)否则，根据word2weight选择new word序列平均weight最大的
        (5)否则，选择包含user word较多的
        #(6)否则，合并子word序列后与原word_str相比变化最小的,去掉lemmatize后不需要了
        (7)否则，选择word序列中word长度方差最小的
        (8)否则，选择第一个

        例如执行如下测试代码:
        ws=['Thisisatest','eelist','dockerid','extrance','importerrorapp','oversample','pathurl','webassets','machid','folderstatus','impacket','booleanvalue']
        cw_splitter=CompoundWordSplitter(user_words=['boolean','url','path'],exclude_words=['imp'],word2weight={'a':0.4,'test':0.1,'eat':0.01,'est':0.034})
        for w in ws:
            seg_words1=cw_splitter.split(w,lemmatize=True,need_noise_str=False)
            print(w,'-->',seg_words1)
        结果为:
        Thisisatest --> ['This', 'is', 'a', 'test']
        eelist --> ['ee', 'list']
        dockerid --> ['docker', 'id']
        extrance --> ['ex', 'trance']
        importerrorapp --> ['import', 'error', 'app']
        oversample --> ['over', 'sample']
        pathurl --> ['path', 'url']
        webassets --> ['web', 'asset']
        machid --> ['mach', 'id']
        folderstatus --> ['folder', 'status']
        impacket --> ['im', 'packet']
        booleanvalue --> ['boolean', 'value']
        :param user_words: 用户字典，有些词语比如url,boolean等在代码中的key word，pyenchant无法识别
        :param exclude_words: 有些词语pyenchant识别成word但需要被排除
        :param word2weight: 
        '''
        self.user_words = set(user_words) if user_words else set()
        self.exclude_words = set(exclude_words) if exclude_words else set()
        self.word2weight = word2weight
        self.word_checker=EnWordCheck(user_words=user_words,exclude_words=exclude_words)
        # self.dynamic_vocab = set(user_words)  # 动态词库会在分割时动态更新，减少eng_dict.check的时间
        # self.en_uk_dict = enchant.Dict("en_UK")  # 初始化一个UK检测器
        # self.en_us_dict = enchant.Dict("en_US")  # 初始化一个US检测器

    def _lemmatize_word(self, word):
        '''
        Lemmatize the word
        Note that， the short word like "as" will be lemmatized to "a".
        So only a word of which the length is more than 2 will be lemmatized
        :param word: 
        :return: 
        '''
        # if len(word) > 2:
        #     lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
        #     return lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='n'), pos='v'), pos='a')
        return _lemmatize_token(word)

    def _split_str_with_digit(self, string):
        '''
        split each string in the string list with digits if there are digits in the string.
        The new string tokens are put in a new token list as the result
        :param strs: 
        :return: 
        '''
        # text = ' '.join(strs)
        digits = re.findall(r'\d+', string)  # find all the digits
        digits = sorted(list(set(digits)), key=len, reverse=True)  # sort the digits by length reversely
        # digit_str = ''
        for digit in digits:
            string = string.replace(digit, ' ' + digit + ' ')  # 将数字用空格分开

        # 将被错误分割成多段的数字重新拼接，保证数字是整体的
        tokens = string.strip().split()
        if len(tokens) > 0:
            new_tokens = [tokens[0]]
            for token in tokens[1:]:
                if new_tokens[-1].isdigit() and token.isdigit():
                    new_tokens[-1] += token
                else:
                    new_tokens.append(token)
            tokens = new_tokens
        return tokens

    def _is_en(self, string):
        '''
        判断string是否为英文单词
        :param string:
        :return:
        '''
        # if string in self.dynamic_vocab or \
        #         (string not in self.exclude_words and
        #          (self.en_us_dict.check(string) or self.en_uk_dict.check(string) or
        #           (len(string) > 2 and wordnet.synsets(string)))):
        #     self.dynamic_vocab.add(string)
        #     return True
        # return False
        return self.word_checker.check(string)

    def _split_cw(self, cw_str):
        # 首先，将cw_str以其中的每个字符为单位构建一个自前向后的有向图，图中一个节点i到起邻居节点j的边表示cw_str[i:j]为一个可识别单词
        # 首先对建立一个列表，最终长度为cw_str的长度，每个位置i的元素为一个邻居节点列表
        neighbor_idss = []  # 初始化邻居节点列表的空列表
        for i in range(len(cw_str)):  # 遍历到最后一个字符，注意是最后一个字符
            neighbor_idss.append([])  # i位置的字符存入往后能和该位置形成word的位置游标,即邻居节点
            for j in range(len(cw_str), i + 1, -1):  # 从后往前遍历,最小到i+1，注意是i+1
                if self._is_en(cw_str[i:j]):  # 如果cw_str[i:j]被识别为一个单词，j作为i的邻居节点加入neighbor_idss[i]
                    neighbor_idss[i].append(j)
            # if not neighbor_idss[i]:   #如果nwords[i]为空，将当前字母作为一个单独的word
            neighbor_idss[i].append(i + 1)  # 每个i+1都强制作为i的邻居节点，必须加
        neighbor_idss.append([])  # 最后加一个空[]，作为编号len(cw_str)的邻居节点列表，用于下面路径搜寻用

        # 将0,1,...,len(cw_str)视为图中节点，neighbor_idss视为边
        # 将0和len(cw_str)视为图中的源节点和目标节点，利用图中两点之间所有路径搜索的原理，找出所有可能的分割单词new word列表
        # 算法参考refer to https://www.cnblogs.com/rednodel/p/12504837.html
        nwordss, rwordss, stringss = [], [], []  # 初始化new words列表，有效词语right words列表，noise strings列表
        main_stack = [0]  # 初始化一个主栈，主栈用于存储路径节点
        aux_stack = [deepcopy(neighbor_idss[0])]  # 初始化一个辅栈，辅栈用于存储路径中当前节点（对应到主栈中的元素）的邻居节点列表，注意要深拷贝
        # 主栈和辅栈长度始终要保持一致
        while main_stack:  # 如果主栈未空
            neighbor_ids = aux_stack.pop()  # 获取辅栈栈顶的邻居节点列表，并将其从辅栈弹出
            if neighbor_ids:  # 如果邻居节点列表非空
                first_neighor_id = neighbor_ids.pop(0)  # 取出第一个邻居节点，并将其从邻居节点列表中弹出
                assert first_neighor_id not in main_stack  # 防止回路，虽然在这里不可能出现
                main_stack.append(first_neighor_id)  # 并将其存入主栈
                aux_stack.append(neighbor_ids)  # 剩余的邻居节点（列表）压入辅栈
                aux_stack.append(deepcopy(neighbor_idss[first_neighor_id]))  # 主栈栈顶对应点的邻居节点列表压入辅栈
            else:  # 如果邻居节点列表为空
                main_stack.pop()  # 削栈，主栈弹出栈顶，此时辅栈已经弹出过栈顶，不需要再削辅栈了
                continue  # 继续下一个循环
            # note，主栈中栈顶为len(cw_str)时能够运行到下面

            # 如果主栈栈顶==目标节点，并且len(main_stack)<len(cw_str)
            # 这里需要len(main_stack)<len(cw_str)+1的原因是去除'alist'->'a','l','i','s','t'的情况，避免后续规则过滤出错
            # Note，这里是len(cw_str)+1，因为主栈第一个元素是0，最后一个元素是len(cw_str)
            if main_stack[-1] == len(cw_str) and len(main_stack) < len(cw_str) + 1:
                nwords, rwords, strings = [], [], []  # 初始化三个列表
                last_chars_noise = False  # 初始化上一个word长度为0
                for sid, eid in zip(main_stack[:-1], main_stack[1:]):  # 遍历主栈，分别从0和1开始
                    chars = cw_str[sid:eid]  # 提取字符出串
                    if eid - sid == 1 and eid!=len(cw_str) and sid!=0:  # 如果中间某处只有一个字符，正常情况下将一个字符（字母）识别为noise string,单独的"a"除外
                        if last_chars_noise:  # 如果上一个word长度为1
                            nwords[-1] += chars  # 与上一个new word拼接
                            strings[-1] += chars  # 与上一个noise string拼接
                        else:  # 否则
                            nwords.append(chars)  # 加入new word列表
                            strings.append(chars)  # 加入noise string列表
                        last_chars_noise=True
                    else:  # 否则
                        if len(strings) > 0 and strings[-1] == 'a':  # 如果noise string最后一个是'a'
                            # 也就是说出现了比如"This is a test"的情况，中加出现了一个单独的"a"，该"a"被识别为一个单词
                            rwords.append(strings[-1])  # 将之前的string加入right words
                            strings.pop()  # 对strings 削栈
                        nwords.append(chars)  # 将字符串chars加入new word序列
                        rwords.append(nwords[-1])  # 将字符串chars加入new word序列
                        last_chars_noise=False
                    # last_word_len = eid - sid  # 更新last_word_len

                nwordss.append(nwords)  # 加入
                rwordss.append(rwords)  # 加入
                stringss.append(strings)  # 加入
                main_stack.pop()  # 削栈
                aux_stack.pop()  # 削栈
                if len(nwordss)>10000:
                    break

        if not nwordss:  # 如果没有找到new word
            return [cw_str], [cw_str]  # cw_str作为整体输出，noise string也为cw_str

        candidates = list(zip(nwordss, rwordss, stringss))  # 候选集

        # if len(candidates)>1: #首先，选择

        if len(candidates) > 1:  # 首先，选择noise string个数最少的
            str_nums = [len(strings) for strings in stringss]
            min_str_num = min(str_nums)
            candidates = [candidate for candidate, str_num in zip(candidates, str_nums) if str_num == min_str_num]
        if len(candidates) > 1:  # 否则,选择rword平均长度减去noise string平均长度最大的
            _, rwordss, stringss = zip(*candidates)
            # rword_mean_lens=[sum([len(rword) for rword in rwords])/len(rwords) for rwords in rwordss]
            # string_mean_lens=[sum([len(string) for string in strings])/len(strings) for strings in stringss]
            rn_weights = [round(sum([len(rword) for rword in rwords]) / (len(rwords) + 1e-10) -
                                sum([len(string) for string in strings]) / (len(strings) + 1e-10))
                          for rwords, strings in zip(rwordss, stringss)]
            max_rn_weight = max(rn_weights)
            candidates = [candidate for candidate, rn_weight in zip(candidates, rn_weights) if
                          rn_weight == max_rn_weight]
        if len(candidates) > 1:  # 否则，选择new word序列最短的
            nwordss, _, _ = zip(*candidates)
            nword_nums = [len(nwords) for nwords in nwordss]
            min_nword_num = min(nword_nums)
            candidates = [candidate for candidate, nword_num in zip(candidates, nword_nums) if
                          nword_num == min_nword_num]
        # if len(candidates)>1:   # 否则，选择noise string 总长度最小的
        #     _,_,stringss=zip(*candidates)
        #     str_lens=[len(''.join(strings)) for strings in stringss]
        #     min_str_len=min(str_lens)
        #     candidates=[candidate for candidate,str_len in zip(candidates,str_lens) if str_len==min_str_len]
        #     # candidates=list(filter(lambda x:len(x[2])==min_str_len,candidates))
        if len(candidates) > 1 and self.word2weight is not None:  # 否则，根据word2weight选择new word序列平均weight最大的
            nwordss, _, _ = zip(*candidates)
            nword_weights = [sum([self.word2weight.get(nword, 0) for nword in nwords]) / len(nwords) for nwords in
                             nwordss]
            max_nword_weight = max(nword_weights)
            candidates = [candidate for candidate, nword_weight in zip(candidates, nword_weights) if
                          nword_weight == max_nword_weight]
        if len(candidates) > 1 and self.user_words:  # 否则，选择包含user word较多的
            _, rwordss, _ = zip(*candidates)
            user_word_nums = [len(list(filter(lambda x: x.lower() in self.user_words, rwords))) for rwords in rwordss]
            max_user_word_num = max(user_word_nums)
            candidates = [candidate for candidate, user_word_num in zip(candidates, user_word_nums) if
                          user_word_num == max_user_word_num]
            # candidates=list(filter(lambda rwords: len(list(filter(lambda x:x in self.user_words, rwords)))
        # if len(candidates) > 1:  # 否则，合并子word序列后与原word_str相比变化最小的
        #     nwordss, _, _ = zip(*candidates)
        #     diffs = [np.abs(len(''.join(nwords)) - len(cw_str)) + int(''.join(nwords) != cw_str) for nwords in
        #              nwordss]  # 长度变化+词形是否变化
        #     min_diff = min(diffs)
        #     candidates = [candidate for candidate, diff in zip(candidates, diffs) if diff == min_diff]
        if len(candidates) > 1:  # 否则，选择word序列中word长度方差最小的
            nwordss, _, _ = zip(*candidates)
            vars = [np.var([len(word) for word in nwords]) for nwords in nwordss]
            min_var = min(vars)
            candidates = [candidate for candidate, var in zip(candidates, vars) if var == min_var]
        if len(candidates) > 1:  # 否则，选择第一个
            candidates = candidates[:1]
        # if need_noise_str:
        nwords=[candidates[0][0][0]]
        for i in range(1,len(candidates[0][0])):
            if len(candidates[0][0][i-1])==1 and candidates[0][0][i-1].isalpha() and len(candidates[0][0][i])==1 and candidates[0][0][i].isalpha():
                nwords[-1]+=candidates[0][0][i]
            else:
                nwords.append(candidates[0][0][i])
        return nwords, candidates[0][2]
        # return candidates[0][0]

    def split(self, cw_str, lemmatize=False, need_noise_str=False):
        # try:
        # self.lemmatize=lemmatize
        # self.need_noise_str=need_noise_str
        # Pynsist -> # Pynsist |  installDir -> install Dir
        cw_str=cw_str.strip()
        m=re.match(r'[A-Z][a-z]+', cw_str, flags=re.S)
        if self._is_en(cw_str) or cw_str.lower() in self.user_words or cw_str.isdigit() or (m is not None and m.group()==cw_str):
            nwords, nstrs = [cw_str], []
        else:
            split_strs = set(re.findall(r'[^A-Z _][A-Z]', cw_str, re.S))  #
            for split_str in split_strs:
                cw_str = cw_str.replace(split_str, ' '.join(split_str))
            if len(cw_str)>4:
                split_strs = set(re.findall(r'[A-Z][A-Z][A-Z][a-z]', cw_str, re.S))  #RevHTTPClient
                for split_str in split_strs:
                    cw_str = cw_str.replace(split_str, '{} {}'.format(split_str[:2],split_str[2:]))
            cw_str=cw_str.strip()
            cw_str_token_num = len(cw_str.split())

            # cw_str1=cw_str.replace('_', ' ').strip()
            # if cw_str1=='':
            #     nwords, nstrs = [cw_str], []
            # else:
            #     cw_str1_tokens=cw_str1.split()
            #     if len(cw_str1_tokens)>1 or '_' in cw_str:
            #         nwords, nstrs = [self._lemmatize_word(token) for token in cw_str1_tokens] if lemmatize else cw_str1_tokens, []
            #         self.user_words |= set(' '.join(nwords).lower().split())
            #     else:
            #         nwords, nstrs = [], []
            #         cw_strs = self._split_str_with_digit(cw_str1)
            #         for sub_cw_str in cw_strs:
            #             if self._is_en(sub_cw_str):  # 数字也可以判定为word
            #                 nwords.append(self._lemmatize_word(sub_cw_str) if lemmatize else sub_cw_str)
            #             else:
            #                 sub_nwords, sub_nstrs = self._split_cw(sub_cw_str, lemmatize)
            #                 nwords.extend(sub_nwords)
            #                 nstrs.extend(sub_nstrs)

            if cw_str.replace('_', ' ').strip()=='':
                nwords, nstrs = [cw_str], []
            else:
                cw_str = cw_str.replace('_', ' ').strip()  # p_dep_id -> p dep id
                # cw_str_token_num=len(cw_str.split())
                nwords,nstrs=[],[]
                cw_strs=self._split_str_with_digit(cw_str)

                for sub_cw_str in cw_strs:
                    if cw_str_token_num>1 or self._is_en(sub_cw_str) or sub_cw_str.isdigit() or len(set(sub_cw_str))==1 or sub_cw_str.lower() in self.user_words: #如果cw_str有多个token，或者当前str被识别为word或者数字(数字也可以判定为word)
                        nwords.append(sub_cw_str)
                        self.user_words.add(sub_cw_str.lower()) #添加到词库,必须切换成小写
                        # self.word_checker.dynamic_vocab.add(sub_cw_str.lower()) #添加到检查器中的动态词库
                    else:
                        sub_nwords, sub_nstrs = self._split_cw(sub_cw_str)
                        nwords.extend(sub_nwords)
                        nstrs.extend(sub_nstrs)
        
        # #args -> arg s -> arg 去掉复数后面的s
        # if len(nwords)>1 and nwords[-1]=='s':
        #     nwords=nwords[:-1]
        
        if lemmatize:
            nwords=[self._lemmatize_word(word) for word in nwords]
        if need_noise_str:
            return nwords, nstrs
        return nwords


def _tokenize_code_line(code_line,lower=True,keep_punc=True,lemmatize=True,punc_str=punc_str,user_words=None,
                        operators=None,rev_dic=None):
    '''
    对代码行进行分词
    :param code_line: 代码行
    :param lower: 是否转为小写
    :param keep_punc: 是否保留标点
    :param lemmatize: 是否lemmatize
    :param punc_str: 标点符号字符串
    :param user_words: 用户辞典
    :param operators: 代码操作符
    :param rev_dic: 错误单词字典，用于纠正错误单词
    :return:
    '''
    if user_words is None:
        user_words=[]
    if operators is None:
        operators=[]
    user_words = sorted(user_words, key=len, reverse=True)
    indent_len=0
    indent=re.search(r'(^ +)[^ ]',code_line)    #查找缩进
    if bool(indent):
        indent_len=len(indent.group(1))

    #That's I'll I've I'am 这类缩写词引入特殊分词，变成 That is, I will等
    sp_abbrs=SP_ABBR_DICT.keys()
    abbrs=re.findall(r"{}".format('|'.join(sp_abbrs)),code_line,flags=re.S)
    for abbr in abbrs:
        code_line=code_line.replace(abbr,' '+SP_ABBR_DICT[abbr]+' ')

    code_line=code_line.strip()
    # if code_line.isdigit():
    #     return [code_line],indent_len
    code_line = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", r'<url>', code_line, flags=re.S)
    for punc in ['\\','.','^','$','*','+','-','?','[',']','(',')']: #\\必须放在第一个
        punc_str=punc_str.replace(punc,'\\{}'.format(punc))
    # print(punc_str)
    # code_line = code_line.replace('\\\\', ' ').replace('\n',' ').replace('\\"',' ').replace("\\'",' ')
    # code_line =code_line.replace('\n',' ')
    puncs=set()
    if punc_str:
        puncs = set(re.findall(r'[{}]+'.format(punc_str), code_line, flags=re.S))
    diff_puncs = filter(lambda x: len(x) > 1 and len(set(x)) > 1, puncs)
    same_puncs = filter(lambda x: len(x) > 3 and len(set(x)) == 1, puncs)
    for punc in same_puncs: #顺序不能颠倒
        if punc not in operators:
            code_line=code_line.replace(punc,' '+punc[0]+' ')
    for punc in diff_puncs:
        if punc not in operators:
            code_line=code_line.replace(punc,' '+' '.join(punc)+' ')
    for operator in operators:
        code_line=code_line.replace(operator,' '+operator+' ')
    # print(user)
    tokenizer = MWETokenizer([('<', 'url', '>'),('"','"','"'),("'","'","'"),('\\','n'),('e','.','g','.'),('i','.','e','.'),('-','>'),("'","s"),("'","t"),("'","d")]+user_words, separator='')
    # for user_word in user_words:
    #     nl_parser.add_mwe(user_word)
    tokens = tokenizer.tokenize(WordPunctTokenizer().tokenize(code_line.strip()))
    # tokens=' '.join(tokens).replace('\\ "','\\"').replace("\\ '","\\'").split() #避免引号混乱，代码解析出错
    # tokens=[tokens[0],' '+' '.join(tokens[])]

    # if len(tokens)>2:
    #     # mid_text=' ' + ' '.join(tokens[1:-1])
    #     # mid_text=mid_text.replace('\\ "','\\"').replace("\\ '","\\'").replace(' "',' \\"').replace(" '"," \\'")
    #     mid_text=' '.join(tokens[1:-1])
    #     mid_text=mid_text.replace('\\ "',' ').replace("\\ '"," ").replace('"',' ').replace("'"," ")
    #     tokens = [tokens[0]]+mid_text.strip().split()+[tokens[-1]]   #字符串里的引号问题
    
    # if seg_con:
    #     tokens=[' '.join(seg_conti_word(token)) for token in tokens]

    # code_line=' '.join(tokens)
    # if seg_var:
    #     # #Pynsist -> # Pynsist |  installDir -> install Dir
    #     seg_tokens=set(re.findall(r'[^A-Z ][A-Z]',code_line,re.S))    #
    #     for seg_token in seg_tokens:
    #         code_line=code_line.replace(seg_token,' '.join(seg_token))
    #     code_line = code_line.replace('_', ' ')  # p_dep_id -> p dep id

    if rev_dic:
        tokens=' '.join([rev_dic.get(token,token) for token in tokens]).split()

    if lower:
        tokens=' '.join(tokens).lower().split()
        if rev_dic:
            tokens=' '.join([rev_dic.get(token,token) for token in tokens]).split()

    if lemmatize:
        tokens=[_lemmatize_token(token) for token in tokens]
        if rev_dic:
            tokens=' '.join([rev_dic.get(token,token) for token in tokens]).split()
    # if rev_dic:
    #     tokens=[rev_dic.get(token,token) for token in tokens]
    #     code_line=' '.join(tokens).strip()

    if not keep_punc:
        code_line=' '+' '.join(tokens)+' '
        puncs=[]
        if punc_str:
            puncs=re.findall(r" [{}]+ ".format(punc_str),code_line,re.S)
            puncs=sorted(set(puncs)-set(operators),key=len,reverse=True)
        # puncs=sorted(set(filter(lambda x: x not in operators,puncs)),key=len,reverse=True)
        for punc in puncs:
            code_line=code_line.replace(punc,' ')
        tokens=code_line.strip().split()
        if rev_dic:
            tokens=' '.join([rev_dic.get(token,token) for token in tokens]).split()

        # lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
        # tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
        #          for word in tokens]

    poses = [j + indent_len for j in range(len(tokens))]
    return tokens,poses

def tokenize_code_str(code,lower=True,keep_punc=True,lemmatize=True,punc_str=punc_str,user_words=None,operators=None,
                  pos_tag=False,rev_dic=None):
    '''
    对python代码分词
    :param code: 代码
    :param lower: 是否转为小写
    :param keep_punc: 是否保留标点符号
    :param punc_str: 标点符号串
    :param user_words: 用户字典 list(tuple)
    :param operators: 操作符 + - /，这些都保留
    :param pos_tag: 是否标记位置
    :return:
    '''
    if user_words is None:
        user_words=[]
    if operators is None:
        operators=[]
    if pos_tag:
        tokens,poses=[],[]
        code_lines=code.split('\n')
        for i,code_line in enumerate(code_lines):
            line_tokens,line_poses=_tokenize_code_line(code_line,lower=lower,keep_punc=keep_punc,lemmatize=lemmatize,
                                                       punc_str=punc_str,user_words=user_words,operators=operators,
                                                       rev_dic=rev_dic)
            line_poses=list(zip([i]*len(line_poses),line_poses))
            tokens.extend(line_tokens)
            poses.extend(line_poses)
        return tokens,poses
    else:
        tokens,_=_tokenize_code_line(code,lower=lower,keep_punc=keep_punc,lemmatize=lemmatize,punc_str=punc_str,
                                    user_words=user_words,operators=operators,rev_dic=rev_dic)
        return tokens

if __name__=='__main__':
    s=''' <url> """she Does "not' want+= to did-it****--- inter-me_diate is went fully <number> <LOLFACE>. https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb '''
    tokenizer = MWETokenizer()
    tokens = WordPunctTokenizer().tokenize(s)
    print(tokens)


#     print(tokenize_english(s,user_words=[("want",'to')]))
#     s = "'''This is \n a test; e; a= a+ ''', This is not \n a test''', \"not a test\" a=b+c"
#     tokens,poses=tokenize_python(s,keep_punc=False,pos_tag=True)
#     print(tokens)
#     print(poses)
#     s='''
# class Test{
# int countOccurrences(String str, char ch) {
#     String url = props.getProperty("db.url");
#     return num;
# }
# }
#     '''
#     tokens, poses = tokenize_python(s, pos_tag=True)
#     print(tokens)
#     print(poses)
