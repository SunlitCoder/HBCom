import codecs
import json
import logging
import pickle
import re
from collections import Counter
from copy import deepcopy

from util.code_parser.code_parser import SitParser
from util.code_parser.code_tokenizer import (CompoundWordSplitter, tokenize_code_str)
from util.nl_parser.en_parser import EnWordCheck
from tqdm import tqdm

from config import *


class MySitter(SitParser):
    @property
    def code_strs(self):
        code_strs = []
        for node_id in self.ast_edges[0, :]:
            if self.ast_node_poses[node_id][-1] == -1 and self.ast_nodes[node_id][0] in ['"', "'", "'''", '"""']:
                parent_id = self.ast_edges[1, self.ast_edges[0, :].tolist().index(node_id)]
                if 'str' in self.ast_nodes[parent_id]:
                    if self.seg_attr:
                        str_edge_ids = np.argwhere(self.ast_edges[1, :] == parent_id)
                        code_strs.append([self.ast_nodes[self.ast_edges[0, idx[0]]] for idx in sorted(str_edge_ids)])
                    else:
                        assert self.ast_nodes[node_id][-1] == self.ast_nodes[node_id][0]
                        code_strs.append(self.ast_nodes[node_id])
        return code_strs


def make_rev_dic(train_raw_data_path, valid_raw_data_path, test_raw_data_path, tech_term_path, rev_dic_path,
                 noise_token_path):
    logging.info('Start making segmented word dictionary.')
    user_words = set()
    if os.path.exists(tech_term_path):
        with open(tech_term_path, 'r') as f:
            for line in f:
                user_words.add(line.strip())
    exclude_words = set(['nga', 'st', 'opp', 'pf', 'butt', 'gr'])
    path_dir = os.path.dirname(rev_dic_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    path_dir = os.path.dirname(noise_token_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(test_raw_data_path, 'r') as f3:
        test_raw_data = json.load(f3)
    # token_counter=Counter()

    parser = MySitter(lan='bash', lemmatize=False, lower=False, ast_intact=True, seg_attr=True, rev_dic=None, user_words=USER_WORDS)
    all_tokens = []
    for raw_data in [test_raw_data]:
        pbar = tqdm(raw_data)
        pbar.set_description('[Extract tokens]')
        error_item = 0
        count1 = 0
        for item in pbar:
            count1 += 1
            code = item['code'].replace('\\n', ' ').strip()
            code = code.replace('"', "'")
            try:
                parser.parse(code)
                parser.DFG_edges
            except Exception:
                print("error：" + str(count1))
                print("error：" + code)
            if "ERROR" in parser.ast_nodes:
                error_item += 1
                pbar.set_description("[The {}-th erroneously parsed item]".format(error_item))
            text_tokens = tokenize_code_str(item['text'], user_words=USER_WORDS, lemmatize=False, lower=False,
                                            keep_punc=True, rev_dic=None, punc_str=''.join(parser.puncs),
                                            operators=parser.operators, pos_tag=False)
            tokens = [token for token in parser.code_tokens + text_tokens if token not in parser.digits]
            all_tokens.extend(tokens)

    token_counter = Counter(all_tokens)
    token_sum = sum(token_counter.values())
    token2weight = dict((token, token_counter[token] / token_sum) for token in token_counter.keys())
    seg_token_dic = dict()
    token1s, token2s = [], []
    for token in token_counter.keys():
        if re.findall(r'[^A-Z ][A-Z]', token, re.S):
            token1s.append(token)
        else:
            token2s.append(token)
    pbar = tqdm(token1s + token2s)

    pbar.set_description('[Segment tokens]')
    cw_splitter = CompoundWordSplitter(user_words=user_words, exclude_words=exclude_words, word2weight=token2weight)
    all_noise_tokens = []
    benign_lower_tokens = set()
    sec_seg_token_dic = dict()
    for token in pbar:
        seg_tokens, noise_tokens = cw_splitter.split(token, lemmatize=False, need_noise_str=True)
        end_s_flag = False
        if len(seg_tokens) > 1 and len(seg_tokens[-2]) > 1 and token[-2].isalpha() and seg_tokens[-1].lower() == 's':
            seg_tokens = seg_tokens[:-1]
            end_s_flag = True
        seg_token = ' '.join(seg_tokens)
        all_noise_tokens.extend(noise_tokens)
        if seg_token != token:
            seg_token_dic[token] = seg_token
            if end_s_flag and len(seg_tokens[-1]) > 1:
                sec_seg_token_dic[seg_tokens[-1] + 's'] = seg_tokens[-1]
                sec_seg_token_dic[seg_tokens[-1].upper() + 'S'] = seg_tokens[-1].upper()
                sec_seg_token_dic[seg_tokens[-1].upper() + 's'] = seg_tokens[-1].upper()
                sec_seg_token_dic[seg_tokens[-1][0].upper() + seg_tokens[-1][1:].lower() + 's'] = seg_tokens[-1][0].upper() + seg_tokens[-1][1:].lower()
            benign_lower_tokens |= set(seg_token.lower().split()) - set(' '.join(noise_tokens).lower().split(' '))
            pbar.set_description('[Segment tokens: {}-th segmented: {}:::{}]'.format(len(seg_token_dic), token, seg_token))
        else:
            benign_lower_tokens.add(token.lower())

    noise_counter = Counter(all_noise_tokens)

    en_checker = EnWordCheck(user_words=user_words, exclude_words=exclude_words)
    pbar = tqdm(list(seg_token_dic.values()))
    pbar.set_description('[Segment additional tokens]')
    add_num = 0
    for seg_token in pbar:
        if ' ' in seg_token:
            sub_tokens = seg_token.split(' ')
            en_num = sum([en_checker.check(sub_token) for sub_token in sub_tokens])
            if en_num == len(sub_tokens):
                min_sub_token_len = min([len(sub_token) for sub_token in sub_tokens])
                if min_sub_token_len > 1:
                    new_tokens = [seg_token.replace(' ', '')]
                    if new_tokens[0][-1].lower() != 's':
                        new_tokens.append(seg_token.replace(' ', '') + 's')
                    for i, new_token in enumerate(new_tokens):
                        if new_token.lower() in benign_lower_tokens and new_token.lower() not in user_words:
                            sec_seg_token_dic[new_token] = seg_token
                            sec_seg_token_dic[new_token.upper()] = seg_token.upper()
                            sec_seg_token_dic[new_token.lower()] = seg_token.lower()
                            sec_seg_token_dic[new_token[0].upper() + new_token[1:].lower()] = seg_token[0].upper() + seg_token[1:].lower()
                            if i == 1:
                                sec_seg_token_dic[new_token[:-1].upper() + new_token[-1].lower()] = seg_token.upper()
                            add_num += 1
                            pbar.set_description(
                                '[Segment additional tokens: {}-th segmented: {}:::{}]'.format(add_num, new_token,
                                                                                               seg_token))

                    if len(sub_tokens) > 2:
                        for i in range(0, len(sub_tokens) - 1):
                            mini_seg_token = ' '.join([sub_tokens[i], sub_tokens[i + 1]])
                            mini_new_tokens = [''.join([sub_tokens[i], sub_tokens[i + 1]])]
                            if mini_new_tokens[0][-1].lower() != 's':
                                mini_new_tokens.append(''.join([sub_tokens[i], sub_tokens[i + 1] + 's']))
                            for mini_new_token in mini_new_tokens:
                                if mini_new_token.lower() in benign_lower_tokens and mini_new_token.lower() not in user_words:
                                    sec_seg_token_dic[mini_new_token] = mini_seg_token
                                    sec_seg_token_dic[mini_new_token.upper()] = mini_seg_token.upper()
                                    sec_seg_token_dic[mini_new_token.lower()] = mini_seg_token.lower()
                                    sec_seg_token_dic[mini_new_token[0].upper() + mini_new_token[1:].lower()] = \
                                        mini_seg_token[0].upper() + mini_seg_token[1:].lower()
                                    if i == 1:
                                        sec_seg_token_dic[mini_new_token[:-1].upper() + mini_new_token[
                                            -1].lower()] = mini_seg_token.upper()
                                    add_num += 1
                                    pbar.set_description(
                                        '[Segment additional tokens: {}-th segmented: {}:::{}]'.format(add_num, mini_new_token, mini_seg_token))
                for sub_token in sub_tokens:
                    if len(sub_token) > 2 and sub_token[-1].lower() != 's':
                        mini_new_token = sub_token + 's'
                        if mini_new_token.lower() in benign_lower_tokens and mini_new_token.lower() not in user_words:
                            sec_seg_token_dic[mini_new_token] = sub_token
                            sec_seg_token_dic[mini_new_token.upper()] = sub_token.upper()
                            sec_seg_token_dic[mini_new_token.lower()] = sub_token.lower()
                            sec_seg_token_dic[mini_new_token[0].upper() + mini_new_token[1:].lower()] = sub_token[0].upper() + sub_token[1:].lower()
                            sec_seg_token_dic[
                                mini_new_token[:-1].upper() + mini_new_token[-1].lower()] = sub_token.upper()
    pbar = tqdm(list(seg_token_dic.items()))
    pbar.set_description('[Modify segmented tokens]')
    mod_num = 0
    for compound_token, seg_token in pbar:
        new_sub_tokens = []
        for sub_token in seg_token.split(' '):
            if sub_token in sec_seg_token_dic.keys():
                new_sub_tokens.append(sec_seg_token_dic[sub_token])
                seg_token_dic[sub_token] = sec_seg_token_dic[sub_token]
            else:
                new_sub_tokens.append(seg_token_dic.get(sub_token, sub_token))
        new_seg_token = ' '.join(new_sub_tokens)
        if new_seg_token != seg_token:
            seg_token_dic[compound_token] = new_seg_token
            mod_num += 1
            pbar.set_description(
                '[Modify segmented tokens: {}-th modified: {}:::{}:::{}]'.format(mod_num, compound_token, seg_token,
                                                                                 new_seg_token))

    with codecs.open(rev_dic_path, 'w', encoding='utf-8') as f:
        json.dump(seg_token_dic, f, indent=4, ensure_ascii=False)
    with codecs.open(noise_token_path, 'w', encoding='utf-8') as f:
        json.dump(noise_counter, f, indent=4, ensure_ascii=False)
    logging.info('Finish making segmented word dictionary.')


def _truncate_ast_by_code_and_layout(parser, start_code_token_id, end_code_token_id, max_graph_size=np.inf,
                                     renew_pos=True, ):
    parser_dict = {'code_tokens': parser.code_tokens,
                   'code_token_poses': parser.code_token_poses,
                   'ast_nodes': parser.ast_nodes,
                   'ast_edges': parser.ast_edges,
                   'ast_sibling_edges': parser.ast_sibling_edges,
                   'ast_node_poses': parser.ast_node_poses,
                   'ast_node_in_code_poses': parser.ast_node_in_code_poses,
                   'code_token_edges': parser.code_token_edges,
                   'code_layout_edges': parser.code_layout_edges,
                   'code_layout_sibling_edges': parser.code_layout_sibling_edges,
                   'DFG_edges': parser.DFG_edges
                   }
    _parser_dict = deepcopy(parser_dict)
    code_token_ids = list(_parser_dict['code_token_edges'][0, :]) + [
        _parser_dict['code_token_edges'][1, -1]]
    keep_ast_node_ids = np.unique(_parser_dict['ast_edges'])
    if start_code_token_id > 0 or end_code_token_id < len(code_token_ids):
        code_token_ids = code_token_ids[start_code_token_id:end_code_token_id]
        keep_ast_node_ids = set(code_token_ids)
        child_node_ids = set(code_token_ids)
        root_node_id = _parser_dict['ast_edges'].min()
        while len(child_node_ids) > 0:
            child_node_ids = np.array(sorted(child_node_ids))
            wids = np.where(
                _parser_dict['ast_edges'][0, :] == child_node_ids[:, None])
            father_node_ids = _parser_dict['ast_edges'][1, :][wids[1]]
            keep_ast_node_ids |= set(father_node_ids)
            child_node_ids = set(father_node_ids)
            child_node_ids.discard(root_node_id)
        keep_ast_node_ids = sorted(keep_ast_node_ids)
        child_wids = np.where(_parser_dict['ast_edges'][0, :] == np.array(keep_ast_node_ids)[:, None])[1]
        father_wids = np.where(_parser_dict['ast_edges'][1, :] == np.array(keep_ast_node_ids)[:, None])[1]
        keep_wids = list(set(child_wids) & set(father_wids))
        parser_dict['ast_edges'] = _parser_dict['ast_edges'][:, keep_wids]
    code_layout_node_ids = np.unique(_parser_dict['code_layout_edges'])
    if len(set(keep_ast_node_ids) & set(code_layout_node_ids)) > max_graph_size:
        assert parser_dict['ast_edges'][0, 0] > parser_dict['ast_edges'][1, 0]
        node_id_stack = [parser_dict['ast_edges'].min()]
        keep_ast_node_ids = []
        keep_code_node_ids = []
        while node_id_stack and len(set(keep_ast_node_ids) & set(code_layout_node_ids)) < max_graph_size:
            cur_node_id = node_id_stack.pop(-1)
            keep_ast_node_ids.append(cur_node_id)
            if cur_node_id in code_token_ids:
                keep_code_node_ids.append(cur_node_id)
            father_wids = np.argwhere(parser_dict['ast_edges'][1, :] == cur_node_id)
            child_ids = [parser_dict['ast_edges'][0, wid[0]] for wid in father_wids]
            node_id_stack.extend(sorted(child_ids, reverse=True))
        assert keep_code_node_ids[-1] in keep_ast_node_ids
        keep_ast_node_ids = keep_ast_node_ids[:keep_ast_node_ids.index(keep_code_node_ids[-1]) + 1]
    if len(keep_ast_node_ids) < len(_parser_dict['ast_nodes']):
        child_wids = np.where(_parser_dict['code_layout_edges'][0, :] == np.array(keep_ast_node_ids)[:, None])[1]
        father_wids = np.where(_parser_dict['code_layout_edges'][1, :] == np.array(keep_ast_node_ids)[:, None])[1]
        keep_wids = list(set(child_wids) & set(father_wids))
        parser_dict['code_layout_edges'] = _parser_dict['code_layout_edges'][:, keep_wids]
    if start_code_token_id > 0:
        tmp_ast_node_num = len(keep_ast_node_ids)
        root_node_id = np.min(parser_dict['code_layout_edges'])
        keep_ast_node_ids = set(keep_ast_node_ids)
        father_wids = np.argwhere(
            parser_dict['code_layout_edges'][1, :] == root_node_id)
        while len(father_wids) == 1:
            keep_ast_node_ids.discard(root_node_id)
            root_node_id = parser_dict['code_layout_edges'][0, father_wids[0][0]]
            father_wids = np.argwhere(
                parser_dict['code_layout_edges'][1, :] == root_node_id)
        keep_ast_node_ids = keep_ast_node_ids - set(range(root_node_id))
        if len(keep_ast_node_ids) < tmp_ast_node_num:
            keep_ast_node_ids = sorted(keep_ast_node_ids)
            child_wids = np.where(parser_dict['ast_edges'][0, :] == np.array(keep_ast_node_ids)[:, None])[1]
            father_wids = np.where(parser_dict['ast_edges'][1, :] == np.array(keep_ast_node_ids)[:, None])[1]
            keep_wids = list(set(child_wids) & set(father_wids))
            parser_dict['ast_edges'] = parser_dict['ast_edges'][:, keep_wids]
    if len(keep_ast_node_ids) < len(_parser_dict['ast_nodes']):
        keep_ast_node_ids = sorted(keep_ast_node_ids)

        child_wids = np.where(parser_dict['ast_edges'][0, :] == np.array(keep_ast_node_ids)[:, None])[1]
        father_wids = np.where(parser_dict['ast_edges'][1, :] == np.array(keep_ast_node_ids)[:, None])[1]
        keep_wids = list(set(child_wids) & set(father_wids))
        parser_dict['ast_edges'] = parser_dict['ast_edges'][:, keep_wids]

        code_len = len(set(parser_dict['ast_edges'][0, :]) - set(parser_dict['ast_edges'][1, :]))
        parser_dict['code_tokens'] = _parser_dict['code_tokens'][start_code_token_id:end_code_token_id][:code_len]
        parser_dict['code_token_poses'] = _parser_dict['code_token_poses'][start_code_token_id:end_code_token_id][
                                          :code_len]

        parser_dict['code_token_edges'] = _parser_dict['code_token_edges'][:,
                                          start_code_token_id:end_code_token_id - 1][:, :code_len - 1]

        child_wids = np.where(_parser_dict['code_layout_edges'][0, :] == np.array(keep_ast_node_ids)[:, None])[1]
        father_wids = np.where(_parser_dict['code_layout_edges'][1, :] == np.array(keep_ast_node_ids)[:, None])[1]
        keep_wids = list(set(child_wids) & set(father_wids))
        parser_dict['code_layout_edges'] = _parser_dict['code_layout_edges'][:, keep_wids]

        prev_wids = np.where(_parser_dict['code_layout_sibling_edges'][0, :] == np.array(keep_ast_node_ids)[:, None])[1]
        next_wids = np.where(_parser_dict['code_layout_sibling_edges'][1, :] == np.array(keep_ast_node_ids)[:, None])[1]
        keep_wids = list(set(prev_wids) & set(next_wids))
        parser_dict['code_layout_sibling_edges'] = _parser_dict['code_layout_sibling_edges'][:, keep_wids]

        prev_wids = np.where(_parser_dict['DFG_edges'][0, :] == np.array(keep_ast_node_ids)[:, None])[1]
        next_wids = np.where(_parser_dict['DFG_edges'][1, :] == np.array(keep_ast_node_ids)[:, None])[1]
        keep_wids = list(set(prev_wids) & set(next_wids))
        parser_dict['DFG_edges'] = _parser_dict['DFG_edges'][:, keep_wids]

        prev_wids = np.where(_parser_dict['ast_sibling_edges'][0, :] == np.array(keep_ast_node_ids)[:, None])[1]
        next_wids = np.where(_parser_dict['ast_sibling_edges'][1, :] == np.array(keep_ast_node_ids)[:, None])[1]
        keep_wids = list(set(prev_wids) & set(next_wids))
        parser_dict['ast_sibling_edges'] = _parser_dict['ast_sibling_edges'][:, keep_wids]
        parser_dict['ast_nodes'] = []
        parser_dict['ast_node_poses'] = []
        parser_dict['ast_node_in_code_poses'] = []
        for i, node_id in enumerate(sorted(keep_ast_node_ids)):
            parser_dict['ast_nodes'].append(_parser_dict['ast_nodes'][node_id])
            parser_dict['ast_edges'][parser_dict['ast_edges'] == node_id] = i
            parser_dict['ast_sibling_edges'][parser_dict['ast_sibling_edges'] == node_id] = i
            parser_dict['ast_node_poses'].append(_parser_dict['ast_node_poses'][node_id])
            parser_dict['ast_node_in_code_poses'].append(_parser_dict['ast_node_in_code_poses'][node_id])
            parser_dict['code_token_edges'][parser_dict['code_token_edges'] == node_id] = i
            parser_dict['code_layout_edges'][parser_dict['code_layout_edges'] == node_id] = i
            parser_dict['code_layout_sibling_edges'][parser_dict['code_layout_sibling_edges'] == node_id] = i
            parser_dict['DFG_edges'][parser_dict['DFG_edges'] == node_id] = i
        if "ERROR" not in _parser_dict['ast_nodes']:
            assert parser_dict['ast_node_in_code_poses'][parser_dict['ast_edges'].min()][0] == \
                   parser_dict['code_token_poses'][0][0]
            assert parser_dict['ast_node_in_code_poses'][parser_dict['ast_edges'].min()][1] <= \
                   parser_dict['code_token_poses'][0][1]
        if renew_pos and (parser_dict['ast_edges'].min() > _parser_dict['ast_edges'].min() or start_code_token_id > 0):
            if parser_dict['code_token_poses'][0][0] > 0 or parser_dict['code_token_poses'][0][1] > 0:
                start_mpos = parser_dict['code_token_poses'][0][0]
                start_npos = parser_dict['code_token_poses'][0][1]
                for i, pos in enumerate(parser_dict['code_token_poses']):
                    if pos[0] == start_mpos:
                        parser_dict['code_token_poses'][i] = (pos[0] - start_mpos, pos[1] - start_npos)
                    else:
                        parser_dict['code_token_poses'][i] = (pos[0] - start_mpos, pos[1])

                for i, pos in enumerate(parser_dict['ast_node_in_code_poses']):
                    if pos[0] < start_mpos:
                        parser_dict['ast_node_in_code_poses'][i] = (0, 0)
                    elif pos[1] == start_mpos:
                        parser_dict['ast_node_in_code_poses'][i] = (pos[0] - start_mpos, pos[1] - start_npos)
                    else:
                        parser_dict['ast_node_in_code_poses'][i] = (pos[0] - start_mpos, pos[1])

            # 更新 ast node position,这里ast node （position）序列按照层序从左至右排列
            start_depth = parser_dict['ast_node_poses'][0][0]
            cur_depth = 0
            width = -1
            parser_dict['ast_node_poses'][0] = (parser_dict['ast_node_poses'][0][0], 0, 0)  # 更新新根节点的ypos和zpos
            for node_id, node_pos in enumerate(parser_dict['ast_node_poses']):
                if node_pos[0] - start_depth > cur_depth:
                    assert node_pos[0] - start_depth - cur_depth == 1
                    cur_depth = node_pos[0] - start_depth
                    width = -1
                if node_pos[2] >= 0:
                    width += 1
                parser_dict['ast_node_poses'][node_id] = (cur_depth, parser_dict['ast_node_poses'][node_id][1],
                                                          parser_dict['ast_node_poses'][node_id][2])  # 更新当前节点的xpos

                father_wids = np.argwhere(parser_dict['ast_edges'][1, :] == node_id)

                plus_zpos_child_ids, minus_zpos_child_ids = [], []
                for wid in father_wids:
                    child_id = parser_dict['ast_edges'][0, wid[0]]
                    # parser_dict['ast_node_poses'][child_id][1]=width #更新子节点的ypos
                    parser_dict['ast_node_poses'][child_id] = (parser_dict['ast_node_poses'][child_id][0], width,
                                                               parser_dict['ast_node_poses'][child_id][2])  # 更新子节点的ypos
                    if parser_dict['ast_node_poses'][child_id][2] >= 0:
                        plus_zpos_child_ids.append(child_id)
                    else:
                        minus_zpos_child_ids.append(child_id)
                plus_zpos_child_ids = sorted(plus_zpos_child_ids)  # 排序
                minus_zpos_child_ids = sorted(minus_zpos_child_ids)  # 排序
                # 根据条件判断是否需要更新子节点的zpos
                if plus_zpos_child_ids and parser_dict['ast_node_poses'][plus_zpos_child_ids[0]][2] > 0:
                    for zpos, child_id in enumerate(plus_zpos_child_ids):
                        parser_dict['ast_node_poses'][child_id] = (
                            parser_dict['ast_node_poses'][child_id][0], parser_dict['ast_node_poses'][child_id][1],
                            zpos)
                if minus_zpos_child_ids and parser_dict['ast_node_poses'][minus_zpos_child_ids[0]][2] < -1:
                    for zpos, child_id in enumerate(sorted(minus_zpos_child_ids)):
                        parser_dict['ast_node_poses'][child_id] = (
                            parser_dict['ast_node_poses'][child_id][0], parser_dict['ast_node_poses'][child_id][1],
                            -1 - zpos)

    assert not set(np.unique(parser_dict['DFG_edges'])) - set(np.unique(parser_dict['code_layout_edges']))
    assert not set(np.unique(parser_dict['code_token_edges'])) - set(np.unique(parser_dict['code_layout_edges']))
    return parser_dict


def tokenize_raw_data(raw_data_path, token_data_path, rev_dic_path,
                      max_code_len=max_code_len,
                      max_graph_size=max_graph_size,
                      max_text_len=max_text_len):
    logging.info('########### Start tokenize data including tokenizing, tree processing, and number-identification transfering ##########')

    token_data_dir = os.path.dirname(token_data_path)
    if not os.path.exists(token_data_dir):
        os.makedirs(token_data_dir)
    with codecs.open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    with codecs.open(rev_dic_path, 'r', encoding='utf-8') as f:
        rev_dic = json.load(f)
    parser = MySitter(lan='bash', lemmatize=True, lower=True, ast_intact=True, seg_attr=True, rev_dic=rev_dic, user_words=USER_WORDS)
    error_item = 0
    pbar = tqdm(raw_data)
    sp_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '16', '32', '64', '128', '256']
    parserExceptionList = []
    parserDictExceptionList = []
    count = 0
    for item in pbar:
        count += 1
        code = item['code'].replace('\\n', ' ').strip()
        code = code.replace('"', "'")
        try:
            parser.parse(code)
            parser.DFG_edges
        except Exception:
            parserExceptionList.append(str(count) + ':' + item['code'])
        assert parser.ast_edges.dtype == np.int64
        if "ERROR" in parser.ast_nodes:
            error_item += 1
            pbar.set_description("[The {}-th erroneously parsed item]".format(error_item))

        try:
            parser_dict = _truncate_ast_by_code_and_layout(parser, 0, min(len(parser.code_tokens), max_code_len),
                                                           max_graph_size=max_graph_size, renew_pos=True)
            assert parser_dict['code_layout_edges'].dtype == np.int64
        except Exception:
            parserDictExceptionList.append(str(count) + ':' + item['code'])

        layout_nodes = []
        layout_edges = parser_dict['code_layout_edges']
        layout_sibling_edges = parser_dict['code_layout_sibling_edges']
        DFG_edges = parser_dict['DFG_edges']
        code_token_edges = parser_dict['code_token_edges']
        layout_node_ids = np.unique(parser_dict['code_layout_edges'])
        for i, layout_node_id in enumerate(sorted(layout_node_ids)):
            layout_nodes.append(parser_dict['ast_nodes'][layout_node_id])
            # layout_node_in_code_poses.append(parser_dict['ast_node_in_code_poses'][layout_node_id])
            layout_edges[layout_edges == layout_node_id] = i
            layout_sibling_edges[layout_sibling_edges == layout_node_id] = i
            DFG_edges[DFG_edges == layout_node_id] = i
            code_token_edges[code_token_edges == layout_node_id] = i
        assert len(set(layout_edges[1, :]) - set(layout_edges[0, :])) == 1
        assert not set(np.unique(code_token_edges)) - set(np.unique(layout_edges))
        layout_nodes = [token if token not in parser.digits and not token.isdigit() else '<number>' for token in
                        layout_nodes]

        item['graph'] = {'nodes': str(layout_nodes),
                         'base_edges': str(layout_edges.tolist()),
                         'sibling_edges': str(layout_sibling_edges.tolist()),
                         'dfg_edges': str(DFG_edges.tolist()),
                         'code_edges': str(code_token_edges.tolist()),
                         }

        text_tokens = tokenize_code_str(item['text'], user_words=USER_WORDS, lemmatize=True, lower=True,
                                        keep_punc=False, rev_dic=rev_dic, punc_str=''.join(parser.puncs),
                                        operators=parser.operators, pos_tag=False)
        text_tokens = [
            token if token in sp_tokens or (token not in parser.digits and not token.isdigit()) else '<number>' for
            token in text_tokens[:max_text_len]]

        text_tokens = ' '.join(text_tokens).split(' ')
        if len(text_tokens) == max_text_len or text_tokens[-1] == '.':
            text_tokens[-1] = '..'
        elif text_tokens[-1] != '.':
            text_tokens.append('..')
        item['text'] = ' '.join(text_tokens)

    logging.info('####parse error bash code#######')
    print(parserExceptionList)
    print(parserDictExceptionList)

    with codecs.open(token_data_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)
    logging.info('########### Finish tokenize data including tokenizing, tree processing, and number-identification transfering ##########')


def build_w2i2w(train_token_data_path,
                io_token_w2i_path,
                io_token_i2w_path,
                io_min_token_count=3,
                unk_aliased=True,
                ):
    logging.info('########### Start building the dictionary of the training set ##########')
    dic_paths = [io_token_w2i_path, io_token_i2w_path]
    for dic_path in dic_paths:
        dic_dir = os.path.dirname(dic_path)
        if not os.path.exists(dic_dir):
            os.makedirs(dic_dir)

    with codecs.open(train_token_data_path, 'r', encoding='utf-8') as f:
        token_data = json.load(f)

    graph_node_counter = Counter()
    text_token_counter = Counter()
    max_code_len = 0
    for item in tqdm(token_data):
        graph_nodes = list(eval(item['graph']['nodes']))
        graph_node_counter += Counter(graph_nodes)
        text_token_counter += Counter(item['text'].split())  # texts是一个列表
        code_len = len(np.unique(np.array(eval(item['graph']['code_edges']))))
        max_code_len = max(max_code_len, code_len)
    general_vocabs = [PAD_TOKEN, UNK_TOKEN]

    io_token_counter = graph_node_counter + text_token_counter
    io_tokens = list(filter(lambda x: io_token_counter[x] >= io_min_token_count, io_token_counter.keys()))
    text_tokens = set(io_tokens) & set(text_token_counter.keys())
    other_tokens = set(io_tokens) - text_tokens

    graph_unk_aliases = []
    if unk_aliased:
        max_alias_num = 0
        for item in token_data:
            aliases = list(filter(lambda x: x not in io_tokens, set(list(eval(item['graph']['nodes'])))))
            max_alias_num = max(max_alias_num, len(aliases))
        graph_unk_aliases = ['<unk-alias-{}>'.format(i) for i in range(max_alias_num)]
    copy_tokens = []
    for i in range(max_code_len):
        COPY_TOKEN = '<copy_{}>'.format(i)
        assert COPY_TOKEN not in io_tokens
        copy_tokens.append(COPY_TOKEN)
    io_tokens = general_vocabs + list(text_tokens) + [OUT_END_TOKEN, OUT_BEGIN_TOKEN] + copy_tokens + list(
        other_tokens) + graph_unk_aliases
    io_token_indices = list(range(len(io_tokens)))
    io_token_w2i = dict(zip(io_tokens, io_token_indices))
    io_token_i2w = dict(zip(io_token_indices, io_tokens))

    dics = [io_token_w2i, io_token_i2w]
    for dic, dic_path in zip(dics, dic_paths):
        with open(dic_path, 'wb') as f:
            pickle.dump(dic, f)
        with codecs.open(dic_path + '.json', 'w') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)
    logging.info('########### Finish building the dictionary of the training set ##########')


def get_ex_tgt_dict(src_tokens, tgt_w2i):
    ex_src_tokens = list(filter(lambda x: x not in tgt_w2i.keys(), src_tokens))
    ex_src_tokens = sorted(list(set(ex_src_tokens)), key=ex_src_tokens.index)  # 去重但保留顺序
    ex_src_token_indices = list(range(len(tgt_w2i), len(tgt_w2i) + len(ex_src_tokens)))
    ex_tgt_w2i = dict(zip(ex_src_tokens, ex_src_token_indices))
    ex_tgt_i2w = dict(zip(ex_src_token_indices, ex_src_tokens))
    return ex_tgt_w2i, ex_tgt_i2w


def get_src2tgt_map_ids(src_tokens, tgt_w2i, ex_tgt_w2i):
    all_tgt_w2i = {**tgt_w2i, **ex_tgt_w2i}
    src_map = [all_tgt_w2i[token] for token in src_tokens]
    return src_map


def get_align_tgt_ids(tgt_tokens, tgt_w2i, ex_tgt_w2i):
    all_tgt_w2i = {**tgt_w2i, **ex_tgt_w2i}
    unk_idx = tgt_w2i[UNK_TOKEN]
    tgt_token_ids = [all_tgt_w2i.get(token, unk_idx) for token in tgt_tokens]
    return tgt_token_ids


def build_avail_data(token_data_path, avail_data_path, io_token_w2i_path, io_token_i2w_path, unk_aliased=True):
    logging.info('########### Start building the train dataset available for the model ##########')
    avail_data_dir = os.path.dirname(avail_data_path)
    if not os.path.exists(avail_data_dir):
        os.makedirs(avail_data_dir)

    w2is = []
    for w2i_path in [io_token_w2i_path, io_token_i2w_path]:
        with open(w2i_path, 'rb') as f:
            w2is.append(pickle.load(f))
    io_token_w2i, io_token_i2w = w2is

    with codecs.open(token_data_path, 'r') as f:
        token_data = json.load(f)

    io_token_unk_idx = io_token_w2i[UNK_TOKEN]
    out_begin_idx = io_token_w2i[OUT_BEGIN_TOKEN]

    text_tokens, text_token_ids = zip(
        *[(io_token_i2w[idx], idx) for idx in range(out_begin_idx + 1)])  # don't forget +1
    text_i2w = dict(zip(text_token_ids, text_tokens))
    text_w2i = dict(zip(text_tokens, text_token_ids))

    avail_data = {'code_graphs': [], 'texts': [], 'ids': [], 'text_dic': {'text_i2w': text_i2w, 'ex_text_i2ws': []}}
    text_token_idx_counter = Counter()

    pbar = tqdm(token_data)
    max_graph_node_id = 0
    for i, item in enumerate(pbar):
        graph_nodes = list(eval(item['graph']['nodes']))

        base_child2father_edges = np.array(eval(item['graph']['base_edges']))
        base_father2child_edges = np.array([base_child2father_edges[1, :], base_child2father_edges[0, :]])
        sibling_prev2next_edges = np.array(eval(item['graph']['sibling_edges']))
        sibling_next2prev_edges = np.array([sibling_prev2next_edges[1, :], sibling_prev2next_edges[0, :]])
        dfg_prev2next_edges = np.array(eval(item['graph']['dfg_edges']))
        dfg_next2prev_edges = np.array([dfg_prev2next_edges[1, :], dfg_prev2next_edges[0, :]])
        code_prev2next_edges = np.array(eval(item['graph']['code_edges']))
        code_next2prev_edges = np.array([code_prev2next_edges[1, :], code_prev2next_edges[0, :]])

        code_node_mask = np.zeros(shape=(len(graph_nodes),), dtype=np.int64)
        code_node_ids = np.unique(code_prev2next_edges)
        code_node_mask[code_node_ids] = 1

        text_tokens = item['text'].split()

        ex_text_w2i, ex_text_i2w = get_ex_tgt_dict(graph_nodes, text_w2i)
        graph_node2text_map_ids = get_src2tgt_map_ids(graph_nodes, text_w2i, ex_text_w2i)
        text_token_ids = get_align_tgt_ids(text_tokens, text_w2i, ex_text_w2i)
        text_token_idx_counter += Counter(text_token_ids)

        if unk_aliased:
            all_unk_aliases = filter(lambda x: x not in io_token_w2i.keys(), graph_nodes)
            unk_aliases = []
            for unk_alias in all_unk_aliases:
                if unk_alias not in unk_aliases:
                    unk_aliases.append(unk_alias)
            graph_nodes = [node if node not in unk_aliases else '<unk-alias-{}>'.format(unk_aliases.index(node)) for
                           node in graph_nodes]
        graph_node_ids = [io_token_w2i.get(node, io_token_unk_idx) for node in graph_nodes]
        max_graph_node_id = max(max_graph_node_id, max(graph_node_ids))
        code_graph = {'nodes': graph_node_ids,
                      'base_child2father_edges': base_child2father_edges,
                      'base_father2child_edges': base_father2child_edges,
                      'sibling_prev2next_edges': sibling_prev2next_edges,
                      'sibling_next2prev_edges': sibling_next2prev_edges,
                      'dfg_prev2next_edges': dfg_prev2next_edges,
                      'dfg_next2prev_edges': dfg_next2prev_edges,
                      'code_prev2next_edges': code_prev2next_edges,
                      'code_next2prev_edges': code_next2prev_edges,
                      'node2text_map_ids': graph_node2text_map_ids,
                      'code_node_mask': code_node_mask}

        avail_data['code_graphs'].append(code_graph)
        avail_data['texts'].append(text_token_ids)
        avail_data['ids'].append(i)
        avail_data['text_dic']['ex_text_i2ws'].append(ex_text_i2w)

    with open(avail_data_path, 'wb') as f:
        pickle.dump(avail_data, f)
    logging.info('########### Finish building the train dataset available for the model ##########')


if __name__ == '__main__':
    make_rev_dic(train_raw_data_path, valid_raw_data_path, test_raw_data_path, tech_term_path, rev_dic_path,
                 noise_token_path)
    tokenize_raw_data(raw_data_path=train_raw_data_path,
                      token_data_path=train_token_data_path,
                      rev_dic_path=rev_dic_path,
                      max_text_len=max_text_len,
                      max_code_len=max_code_len,
                      max_graph_size=max_graph_size)
    tokenize_raw_data(raw_data_path=valid_raw_data_path,
                      token_data_path=valid_token_data_path,
                      rev_dic_path=rev_dic_path,
                      max_text_len=max_text_len,
                      max_code_len=max_code_len,
                      max_graph_size=max_graph_size)
    tokenize_raw_data(raw_data_path=test_raw_data_path,
                      token_data_path=test_token_data_path,
                      rev_dic_path=rev_dic_path,
                      max_text_len=max_text_len,
                      max_code_len=max_code_len,
                      max_graph_size=max_graph_size)
    build_w2i2w(train_token_data_path=train_token_data_path,
                io_token_w2i_path=io_token_w2i_path,
                io_token_i2w_path=io_token_i2w_path,
                io_min_token_count=io_min_token_count,
                unk_aliased=unk_aliased)

    build_avail_data(token_data_path=train_token_data_path,
                     avail_data_path=train_avail_data_path,
                     io_token_w2i_path=io_token_w2i_path,
                     io_token_i2w_path=io_token_i2w_path,
                     unk_aliased=unk_aliased)
