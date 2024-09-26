# coding=utf-8
'''
from https://github.com/tree-sitter/bash-tree-sitter
'''
from copy import deepcopy
import numpy as np
from .code_tokenizer import tokenize_code_str
from tree_sitter import Language, Parser
import os, re
from collections import deque
from ._DFG import *


class SitParser(object):
    def __init__(self, lan: str = 'bash', lemmatize: bool = True, lower: bool = True, seg_attr=True,
                 ast_intact: bool = True,
                 rev_dic=None, user_words=None):

        self.language = lan
        self.lemmatize = lemmatize
        self.lower = lower
        self.seg_attr = seg_attr
        self.ast_intact = ast_intact
        self.puncs = set()  # 标点符号
        self.operators = set()
        self.digits = set()  # 数字
        self.user_words = [] if user_words is None else user_words
        self.user_words += [tuple(operator) for operator in self.operators]
        self.rev_dic = rev_dic

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        lan_path = os.path.join(cur_dir, 'tree_sitter_repo/my-languages.so')
        py_language = Language(lan_path, lan)
        self.parser = Parser()
        self.parser.set_language(py_language)

    def _get_children(self, node):
        '''
        获取生成子节点列表
        :param node:
        :return:
        '''
        if node.child_count > 0:
            func_children = list(filter(self._is_func_node, node.children))
            if len(func_children) == 0:
                return [str(node.text, encoding='utf-8')]
            return func_children
        if node.is_named and str(node.text, encoding='utf-8').lower() != node.type.lower():
            return [str(node.text, encoding='utf-8')]
        return []

    def _get_func_children(self, node):
        children = self._get_children(node)
        func_children = filter(self._is_func_node, children)
        return func_children

    def _get_attr_children(self, node):
        children = self._get_children(node)
        attr_children = filter(self._is_attr_node, children)
        return attr_children

    def _to_str(self, node):
        if self._is_attr_node(node):
            if isinstance(node, str):
                return node
            if node.type == 'escape_sequence':
                return str(node.prev_sibling.text, encoding='utf-8') + str(node.text, encoding='utf-8') + str(
                    node.next_sibling.text, encoding='utf-8')
        elif self._is_func_node(node):
            return node.type

    def _is_func_node(self, node):
        if isinstance(node, str):
            return False
        if 'comment' in node.type:
            return False
        if node.type == '"':
            return False
        if node.type == 'escape_sequence':
            return False
        if not self.ast_intact and \
                not node.is_named and \
                node.parent.named_child_count > 0 and \
                not node.parent.type.endswith('assignment') and \
                not node.parent.type.endswith('operator') and \
                not node.parent.type.endswith('modifiers'):  # 如果有named子节点，说明是功能节点
            return False

        return True

    def _is_redundant_func_node(self, node):
        if self._is_func_node(node) and not node.is_named and node.parent.named_child_count > 0 and \
                not node.parent.type.endswith('assignment') and \
                not node.parent.type.endswith('operator') and \
                not node.parent.type.endswith('modifiers'):  # 如果有named子节点，说明是功能节点
            return True
        return False

    def _is_attr_node(self, node):  # 如果有named子节点或者type
        if isinstance(node, str):
            return True
        if node.type == 'escape_sequence':  # a=='daa'这种string下还会有一个escape_sequence然后才接着'daa'，将escape_sequence节点视为attr node
            return True
        return False

    def _is_operator_func_node(self, node):
        if not node.is_named and (node.parent.type.endswith('assignment') or node.parent.type.endswith('operator')):
            if re.search(r'[A-Za-z0-9]', str(node.text, encoding='utf-8'), flags=re.S) is None:
                return True
        return False

    def _is_digit_func_node(self, node):
        if re.search(r'integer|float', node.type, flags=re.S) is not None:
            if re.search(r'[0-9]', str(node.text, encoding='utf-8'), flags=re.S) is not None:
                return True
        return False

    def _walk(self, node):  # 广度优先遍历所有功能节点
        todo = deque([node])
        while todo:
            node = todo.popleft()
            todo.extend(self._get_func_children(node))
            yield node

    def _get_ast_info(self, tree):
        edge_end_ids, edge_start_ids = [], []
        nodes, node_points = [], []
        depths, subtree_poses, sibling_poses = [0], [0], [0]

        edge_start_id = 0
        edge_end_id_queue = [0]

        edge_depth_queue = [1] * len(list(self._get_func_children(root_node)))  # 边的深度的队列
        node_depth_queue = [0] * 1
        subtree_pos = -1
        attr_depth = -1

        str_node = root_node.type  # caution

        nodes.append(str_node)
        node_points.append((root_node.start_point, root_node.end_point))
        if self.ast_intact:
            redundant_node_tags = [False]
        for node in self._walk(root_node):
            edge_end_id = edge_end_id_queue.pop(0)

            subtree_pos += 1
            if node_depth_queue[0] > attr_depth:
                subtree_pos = 0
            attr_depth = node_depth_queue.pop(0)

            node_depth_queue.extend([attr_depth + 1] * len(list(self._get_func_children(node))))  # 为什么max？
            children = self._get_children(node)
            attr_sibling_pos, func_sibling_pos = 0, 0
            for child in children:
                if self._is_attr_node(child):
                    if isinstance(child, str):
                        str_node = child
                    elif node.type == 'escape_sequence':
                        str_node = str(child.prev_sibling.text, encoding='utf-8') + str(child.text,
                                                                                        encoding='utf-8') + str(child.next_sibling.text, encoding='utf-8')
                    if self.seg_attr and str_node not in self.digits:
                        tokens = tokenize_code_str(str_node, lemmatize=self.lemmatize, lower=self.lower,
                                                   keep_punc=True,
                                                   rev_dic=self.rev_dic, user_words=self.user_words,
                                                   punc_str=''.join(self.puncs), operators=self.operators,
                                                   pos_tag=False)
                    else:
                        tokens = [str_node]

                    for j, token in enumerate(tokens):
                        edge_end_ids.append(edge_end_id)
                        edge_start_id += 1
                        edge_start_ids.append(edge_start_id)
                        nodes.append(token)

                        depths.append(attr_depth + 1)
                        subtree_poses.append(subtree_pos)
                        sibling_poses.append(-(attr_sibling_pos + 1 + j))
                        node_points.append(())  # 属性节点不加points
                        if self.ast_intact:
                            redundant_node_tags.append(False)
                    attr_sibling_pos += 1

                elif self._is_func_node(child):
                    str_node = child.type
                    nodes.append(str_node)

                    node_points.append((child.start_point, child.end_point))
                    if self.ast_intact:
                        if self._is_redundant_func_node(child):
                            redundant_node_tags.append(True)
                        else:
                            redundant_node_tags.append(False)
                    edge_end_ids.append(edge_end_id)
                    edge_start_id += 1
                    edge_start_ids.append(edge_start_id)
                    edge_end_id_queue.append(edge_start_id)

                    depths.append(edge_depth_queue.pop(0))
                    edge_depth_queue.extend([depths[-1] + 1] * len(
                        list(self._get_func_children(child))))
                    subtree_poses.append(subtree_pos)
                    sibling_poses.append(func_sibling_pos)
                    func_sibling_pos += 1

        edges = np.array([edge_start_ids, edge_end_ids])
        node_poses = list(zip(depths, subtree_poses, sibling_poses))
        if self.ast_intact:
            return nodes, edges, node_poses, node_points, redundant_node_tags
        else:
            return nodes, edges, node_poses, node_points

    def _pre_DFS_ids(self, edges):  # 前序深度优先遍历
        assert edges[0, 0] > edges[1, 0]
        node_id_stack = [edges[1, 0]]
        while node_id_stack:
            cur_node_id = node_id_stack.pop(-1)
            edge_ids = np.argwhere(edges[1, :] == cur_node_id)
            child_ids = [edges[0, idx[0]] for idx in edge_ids]
            node_id_stack.extend(reversed(child_ids))
            yield cur_node_id

    def _get_intact_ast_node_in_code_poses(self, intact_ast_edges, intact_ast_node_points):
        node_in_code_poses = [tuple()] * len(intact_ast_node_points)
        line2indent = dict()
        for node_point in intact_ast_node_points:
            if node_point and (
                    node_point[0][0] not in line2indent.keys() or line2indent[node_point[0][0]] > node_point[0][1]):
                line2indent[node_point[0][0]] = node_point[0][1]
        indent2offset = dict([(indent, offset) for offset, indent in enumerate(sorted(set(line2indent.values())))])
        line2spos = dict([(line, indent2offset[indent]) for line, indent in line2indent.items()])
        node_in_code_poses = [tuple()] * len(intact_ast_node_points)
        cur_line = -1
        cur_attr_node_flag = True
        last_node_id = -1
        for node_id in self._pre_DFS_ids(intact_ast_edges):
            if intact_ast_node_points[node_id]:
                cur_attr_node_flag = False
                if intact_ast_node_points[node_id][0][0] > cur_line:

                    cur_line = intact_ast_node_points[node_id][0][0]
                    node_in_code_poses[node_id] = (cur_line, line2spos[cur_line])
                else:
                    offset = 0 if intact_ast_node_points[last_node_id] and \
                                  intact_ast_node_points[node_id][0][1] == intact_ast_node_points[last_node_id][0][
                                      1] else 1
                    node_in_code_poses[node_id] = (cur_line, node_in_code_poses[last_node_id][-1] + offset)
            else:
                offset = 0 if not cur_attr_node_flag else 1
                node_in_code_poses[node_id] = (cur_line, node_in_code_poses[last_node_id][-1] + offset)
                cur_attr_node_flag = True
            last_node_id = node_id

        return node_in_code_poses

    def parse(self, code):
        byte_code = bytes(code, 'utf8')  # caution
        self._tree = self.parser.parse(byte_code)
        root_node = self._tree.root_node

        self._source_code = str(root_node.text, encoding='utf-8')

        self.puncs |= set(re.findall(r'[^0-9A-Za-z\u4e00-\u9fa5\n \t]', code, flags=re.S))
        for node in self._walk(root_node):
            if self._is_operator_func_node(node):
                self.operators.add(str(node.text, encoding='utf-8'))
            if self._is_digit_func_node(node):
                self.digits.add(str(node.text, encoding='utf-8'))
        self.puncs -= self.operators

        ast_intact = self.ast_intact
        self.ast_intact = True
        self._intact_ast_nodes, self._intact_ast_edges, self._intact_ast_node_poses, \
        self._intact_ast_node_points, self._intact_redundant_ast_node_tags = self._get_ast_info(tree=self._tree)
        self._intact_ast_node_in_code_poses = self._get_intact_ast_node_in_code_poses(
            intact_ast_edges=self._intact_ast_edges,
            intact_ast_node_points=self._intact_ast_node_points)

        self.ast_intact = ast_intact
        if not self.ast_intact:
            self._concise_ast_nodes, self._concise_ast_edges, self._concise_ast_node_poses, \
            self._concise_ast_node_points = self._get_ast_info(tree=self._tree)
            self._concise_ast_node_in_code_poses = [pos for pos, tag in zip(self._intact_ast_node_in_code_poses,
                                                                            self._intact_redundant_ast_node_tags) if
                                                    not tag]
            assert len(self._concise_ast_node_in_code_poses) == len(self._concise_ast_nodes)

    @property
    def source_code(self):
        return self._source_code

    @property
    def code_tokens(self):
        code_tokens = []
        for node_id in self._pre_DFS_ids(self._intact_ast_edges):
            if node_id not in self._intact_ast_edges[1, :]:
                code_tokens.append(self._intact_ast_nodes[node_id])
        return code_tokens

    @property
    def code_token_poses(self):
        code_token_poses = []
        for node_id in self._pre_DFS_ids(self._intact_ast_edges):
            if node_id not in self._intact_ast_edges[1, :]:
                code_token_poses.append(self._intact_ast_node_in_code_poses[node_id])
        return code_token_poses

    @property
    def ast_nodes(self):  # ast node
        if self.ast_intact:
            return self._intact_ast_nodes
        return self._concise_ast_nodes

    @property
    def ast_edges(self):  # ast edges
        if self.ast_intact:
            return self._intact_ast_edges
        return self._concise_ast_edges

    @property
    def ast_sibling_edges(self):
        sibling_edges = np.empty(shape=(2, 0), dtype=np.int64)
        for father_id in sorted(set(self.ast_edges[1, :])):
            child_ids = sorted(self.ast_edges[0, :][np.argwhere(self.ast_edges[1, :] == father_id)].reshape((-1,)))
            if len(child_ids) > 1:
                edges = np.array([child_ids[:-1], child_ids[1:]])
                sibling_edges = np.concatenate([sibling_edges, edges], axis=-1)
        return sibling_edges

    @property
    def ast_node_poses(self):  # ast positions
        if self.ast_intact:
            return self._intact_ast_node_poses
        return self._concise_ast_node_poses

    @property
    def ast_node_in_code_poses(self):
        if self.ast_intact:
            return self._intact_ast_node_in_code_poses
        return self._concise_ast_node_in_code_poses

    @property
    def code_token_edges(self):
        code_node_id_path = []
        for node_id in self._pre_DFS_ids(self.ast_edges):
            if node_id not in self.ast_edges[1, :]:
                code_node_id_path.append(node_id)
        return np.array([code_node_id_path[:-1], code_node_id_path[1:]])


    @property
    def code_layout_edges(self):
        code_node_ids = []
        code_token_poses = deepcopy(self.code_token_poses)
        ast_edges = deepcopy(self.ast_edges)
        ast_node_poses = deepcopy(self.ast_node_poses)
        for node_id in self._pre_DFS_ids(ast_edges):
            if node_id not in ast_edges[1, :]:
                code_node_ids.append(node_id)
        if not self.ast_intact:
            code_token_poses = []
            for node_id in self._pre_DFS_ids(self._intact_ast_edges):
                if node_id not in self._intact_ast_edges[1, :] and not self._intact_redundant_ast_node_tags[node_id]:
                    code_token_poses.append(self._intact_ast_node_in_code_poses[node_id])
        assert len(code_node_ids) == len(code_token_poses)

        keep_node_ids = set()
        line_start_node_idss = [set()] * (code_token_poses[-1][0] + 1)
        line_start_pos = (0, 0)
        for code_node_id, code_token_pos in zip(code_node_ids, code_token_poses):
            line_start_node_ids = set()
            if not line_start_node_idss[code_token_pos[0]]:
                line_start_pos = code_token_pos

            child_node_id = code_node_id
            while child_node_id in ast_edges[0, :]:
                father_node_id = ast_edges[1, :][ast_edges[0, :].tolist().index(child_node_id)]
                if self.ast_node_in_code_poses[father_node_id][0] == line_start_pos[0]:
                    if self.ast_node_in_code_poses[father_node_id][1] <= self.ast_node_in_code_poses[child_node_id][1]:
                        if self.ast_node_in_code_poses[father_node_id][1] < self.ast_node_in_code_poses[child_node_id][
                            1]:
                            line_start_node_ids.discard(child_node_id)
                        line_start_node_ids.add(father_node_id)
                else:
                    break
                child_node_id = father_node_id

            if line_start_node_idss[code_token_pos[0]]:
                line_start_node_idss[code_token_pos[0]] &= line_start_node_ids
            else:
                line_start_node_idss[code_token_pos[0]] = line_start_node_ids

            keep_node_ids.add(code_node_id)
            if ast_node_poses[code_node_id][-1] < 0:
                father_node_id = ast_edges[1, :][ast_edges[0, :].tolist().index(code_node_id)]
                keep_node_ids.add(father_node_id)
        for line_start_node_ids in line_start_node_idss:
            if line_start_node_ids:
                keep_node_ids.add(max(line_start_node_ids))

        filter_node_ids = set(np.unique(ast_edges)) - keep_node_ids

        code_layout_edges = deepcopy(ast_edges)
        for node_id in sorted(filter_node_ids):
            if node_id in code_layout_edges[0, :]:
                father_wid = code_layout_edges[0, :].tolist().index(node_id)
                father_id = code_layout_edges[1, :][father_wid]
                code_layout_edges[1, :][code_layout_edges[1, :] == node_id] = father_id
                code_layout_edges = np.delete(code_layout_edges, father_wid, axis=-1)
            else:
                child_wids = np.where(code_layout_edges[1, :] == node_id)[0]
                code_layout_edges = np.delete(code_layout_edges, child_wids, axis=-1)

        line_start_root_node_ids = set(code_layout_edges[1, :]) - set(code_layout_edges[0, :])
        if len(line_start_root_node_ids) > 1:
            root_node_ids = set()
            for line_start_node_id in line_start_root_node_ids:
                tmp_root_node_ids = set()
                child_node_id = line_start_node_id
                while child_node_id in ast_edges[0, :]:
                    father_node_id = ast_edges[1, :][ast_edges[0, :].tolist().index(child_node_id)]
                    tmp_root_node_ids.add(father_node_id)
                    child_node_id = father_node_id
                if root_node_ids:
                    root_node_ids &= tmp_root_node_ids
                else:
                    root_node_ids = tmp_root_node_ids
            root_node_id = max(root_node_ids)

            add_code_layout_edges = np.array(
                [list(line_start_root_node_ids), [root_node_id] * len(line_start_root_node_ids)])
            code_layout_edges = np.concatenate([add_code_layout_edges, code_layout_edges], axis=-1)

        return code_layout_edges

    @property
    def code_layout_sibling_edges(self):
        code_layout_sibling_edges = deepcopy(self.ast_sibling_edges)
        ex_node_ids = sorted(set(np.unique(self.ast_edges)) - set(np.unique(self.code_layout_edges)))
        # print(len(ex_node_ids))
        for ex_node_id in ex_node_ids:
            if ex_node_id in self.ast_sibling_edges[0, :]:
                right_child_id = max(
                    self.ast_edges[0, :][np.argwhere(self.ast_edges[1, :] == ex_node_id)].reshape((-1,)))
                while right_child_id in ex_node_ids:
                    right_child_id = max(
                        self.ast_edges[0, :][np.argwhere(self.ast_edges[1, :] == right_child_id)].reshape((-1,)))
                code_layout_sibling_edges[
                    0, code_layout_sibling_edges[0, :].tolist().index(ex_node_id)] = right_child_id
            if ex_node_id in self.ast_sibling_edges[1, :]:
                left_child_id = min(
                    self.ast_edges[0, :][np.argwhere(self.ast_edges[1, :] == ex_node_id)].reshape((-1,)))
                while left_child_id in ex_node_ids:
                    left_child_id = min(
                        self.ast_edges[0, :][np.argwhere(self.ast_edges[1, :] == left_child_id)].reshape((-1,)))
                code_layout_sibling_edges[1, code_layout_sibling_edges[1, :].tolist().index(ex_node_id)] = left_child_id
        return code_layout_sibling_edges

    @property
    def DFG_edges(self):
        ast_node_point2node_id_and_code = dict()
        for node_id in self._pre_DFS_ids(self._intact_ast_edges):
            if node_id not in self._intact_ast_edges[1, :]:
                if self._intact_ast_node_poses[node_id][-1] >= 0:
                    ast_node_point = self._intact_ast_node_points[node_id]
                    ast_node_point2node_id_and_code[ast_node_point] = [node_id, self._intact_ast_nodes[node_id]]
                else:
                    father_id = self._intact_ast_edges[1, :][self._intact_ast_edges[0, :].tolist().index(node_id)]
                    ast_node_point = self._intact_ast_node_points[father_id]
                    if ast_node_point not in ast_node_point2node_id_and_code.keys():
                        ast_node_point2node_id_and_code[ast_node_point] = [father_id, '']
                    assert ast_node_point2node_id_and_code[ast_node_point][0] == father_id
                    ast_node_point2node_id_and_code[ast_node_point][1] += '' + self._intact_ast_nodes[node_id]
        DFG, DFG_edges = [], [[], []]
        if self.language == 'bash':
            DFG, _ = DFG_bash(root_node=self._tree.root_node, point2code=ast_node_point2node_id_and_code, states={})
        if not self.ast_intact:
            intact_ast_node_id2ast_concise_node_id = dict(zip([node_id for node_id, tags in
                                                               zip(list(range(len(self._intact_ast_nodes))),
                                                                   self._intact_redundant_ast_node_tags) if not tags],
                                                              list(range(len(self._concise_ast_nodes)))))
        for edge_obj in DFG:
            for start_node_id in edge_obj[4]:
                if self.ast_intact:
                    DFG_edges[0].append(start_node_id)
                    DFG_edges[1].append(edge_obj[1])
                elif start_node_id in intact_ast_node_id2ast_concise_node_id.keys() and \
                        edge_obj[1] in intact_ast_node_id2ast_concise_node_id.keys():  # 如果是简树
                    DFG_edges[0].append(intact_ast_node_id2ast_concise_node_id[start_node_id])
                    DFG_edges[1].append(intact_ast_node_id2ast_concise_node_id[edge_obj[1]])

        return np.array(DFG_edges)
