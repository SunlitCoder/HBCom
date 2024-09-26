#coding=utf-8
import numpy as np
import random
from collections import deque

class MyAstor(object):
    def __init__(self,nodes,edges,poses=None,attr_node_ids=None):
        self.nodes=nodes
        self.edges=edges
        self.poses=poses
        self.attr_node_ids=[] if attr_node_ids is None else attr_node_ids

    @property
    def root_id(self):
        root_ids=[idx for idx in self.edges[1,:] if idx not in self.edges[0,:]]
        assert len(set(root_ids))==1
        return root_ids[0]

    def get_child_ids(self,node_id):
        edge_ids=np.argwhere(self.edges[1,:]==node_id)
        child_ids=[self.edges[0,idx[0]] for idx in edge_ids]
        return child_ids

    def get_attr_child_ids(self,node_id):
        edge_ids = np.argwhere(self.edges[1, :] == node_id)
        child_ids = [self.edges[0, idx[0]] for idx in edge_ids if self.edges[0, idx[0]] in self.attr_node_ids]
        return sorted(child_ids)

    def get_func_child_ids(self,node_id):
        edge_ids = np.argwhere(self.edges[1, :] == node_id)
        child_ids = [self.edges[0, idx[0]] for idx in edge_ids if self.edges[0, idx[0]] not in self.attr_node_ids]
        return sorted(child_ids)

    def _breadth_walk_func(self):
        todo = deque([self.root_id])
        while todo:
            node_id = todo.popleft()
            todo.extend(self.get_func_child_ids(node_id))
            yield node_id

    def depth_walk_all_ids(self):
        node_id_stack=[self.root_id]
        while node_id_stack:
            cur_node_id=node_id_stack.pop(-1)
            node_id_stack.extend(reversed(self.get_child_ids(cur_node_id)))
            yield cur_node_id

    def get_poses(self):
        func_depths = [0]
        func_subtree_poses = [0]
        func_subling_poses = [0]

        attr_depths = []
        attr_subtree_poses = []
        attr_subling_poses = []
        edge_depth_queue = [1] * len(self.get_func_child_ids(self.root_id))  # 边的深度的队列
        node_depth_queue = [0] * 1
        subtree_pos = -1
        attr_depth = -1

        func_node_ids=[]
        attr_node_ids=[]
        for node_id in self._breadth_walk_func():
            func_node_ids.append(node_id)
            subtree_pos += 1
            if node_depth_queue[0] > attr_depth:
                subtree_pos = 0
            attr_depth = node_depth_queue.pop(0)

            node_depth_queue.extend([attr_depth + 1] * len(self.get_func_child_ids(node_id)))  # 为什么max？

            attr_child_ids = self.get_attr_child_ids(node_id)

            for subling_pos,child_id in enumerate(attr_child_ids):
                attr_node_ids.append(child_id)
                attr_depths.append(attr_depth + 1)
                attr_subtree_poses.append(subtree_pos)
                attr_subling_poses.append(-(subling_pos + 1))

            func_child_ids = self.get_func_child_ids(node_id)
            for subling_pos,child_id in enumerate(func_child_ids):

                func_depths.append(edge_depth_queue.pop(0))

                edge_depth_queue.extend(
                    [func_depths[-1] + 1] * len(self.get_func_child_ids(child_id)))  ##为什么max？ 如果当前点为树，则将其下所有边的深度值加入深度值队列
                func_subtree_poses.append(subtree_pos)
                func_subling_poses.append(subling_pos)

        func_node_poses = list(zip(func_depths, func_subtree_poses, func_subling_poses))
        attr_node_poses = list(zip(attr_depths, attr_subtree_poses, attr_subling_poses))
        node_poses = func_node_poses + attr_node_poses
        node_ids=func_node_ids+attr_node_ids
        node_id2pos=dict(zip(node_ids,node_poses))

        return [node_id2pos[node_id] for node_id in range(np.max(self.edges)+1)]

    def get_random_subtree(self, max_size=100):
        if len(self.nodes)<=max_size:
            return self.nodes,self.edges,self.poses
        node_ids = [self.root_id]
        while len(node_ids)<max_size:
            node_id=random.choice(node_ids)
            child_ids=self.get_child_ids(node_id)
            while child_ids:
                node_id=random.choice(child_ids)
                if node_id not in node_ids:
                    node_ids.append(node_id)
                child_ids=self.get_child_ids(node_id)
        node_ids=sorted(node_ids[:max_size])
        nodes = []
        poses=[]
        old_id2new_id = dict()
        for i, node_id in enumerate(node_ids):
            old_id2new_id[node_id] = i
            nodes.append(self.nodes[node_id])
            poses.append(self.poses[node_id])
        edges = []
        for i in range(self.edges.shape[1]):
            if self.edges[0, i] in node_ids and self.edges[1, i] in node_ids:
                edges.append(np.array([[old_id2new_id[self.edges[0, i]]],[old_id2new_id[self.edges[1, i]]]]))
        edges = np.concatenate(edges, axis=-1)
        return nodes,edges,poses

    def to_hetero(self,node_hetero=False,add_global_node=False):
        assert self.poses is not None
        global_node='<Glob>'
        if not node_hetero: #如果不异构化节点
            if add_global_node: #如果添加全局节点
                nodes=self.nodes+[global_node]  #把global_node加入nodes
                if isinstance(self.poses[0],tuple):
                    poses=self.poses+[(-1,0,0)] #多加global_node的位置信息,如果position为tuple
                elif isinstance(self.poses[0],str):
                    poses=self.poses+['(-1,0,0)']   #多加global_node的位置信息,如果position为string
                else:
                    raise ValueError("the values of self.poses must be tuples or strings")
                a2g_edges=np.zeros(shape=(2,len(nodes)-1))  #AST中节点到global_node的边
                a2g_edges[0,:]=np.arange(0,len(nodes)-1)    #AST中的节点序号
                a2g_edges[1,:]=len(nodes)-1    #global_node在nodes中的序号
                c2p_edges=np.concatenate([self.edges,a2g_edges],axis=-1)    #合并，成为child to parent边信息
                p2c_edges=np.array([c2p_edges[1,:],c2p_edges[0,:]])
                return nodes,poses,p2c_edges,c2p_edges
            return self.nodes,self.poses,np.array([self.edges[1,:],self.edges[0,:]]),self.edges

        func_nodes,func_poses=[],[]   #功能节点列表
        attr_nodes,attr_poses=[],[]   #属性节点列表

        func_id2id=dict()
        attr_id2id = dict()
        for i,(node,pos) in enumerate(zip(self.nodes,self.poses)):
            if i not in self.attr_node_ids:   #如果是功能节点
                func_id2id[i]=len(func_nodes)
                func_nodes.append(node)
                func_poses.append(pos)
            else:
                attr_id2id[i]=len(attr_nodes)
                attr_nodes.append(node)
                attr_poses.append(pos)

        func_child_func_edges=[]
        func_parent_func_edges=[]
        func_child_attr_edges=[]
        attr_parent_func_edges=[]
        for i in range(self.edges.shape[1]):
            child_id,parent_id=self.edges[0,i],self.edges[1,i]
            if child_id not in self.attr_node_ids:
                func_child_func_edges.append([[func_id2id[parent_id],func_id2id[child_id]]])
                func_parent_func_edges.append([[func_id2id[child_id],func_id2id[parent_id]]])
            else:
                func_child_attr_edges.append([[func_id2id[parent_id],attr_id2id[child_id]]])
                attr_parent_func_edges.append([[attr_id2id[child_id],func_id2id[parent_id]]])

        func_child_func_edges=np.concatenate(func_child_func_edges,axis=0).T
        func_parent_func_edges=np.concatenate(func_parent_func_edges,axis=0).T
        func_child_attr_edges=np.concatenate(func_child_attr_edges,axis=0).T
        attr_parent_func_edges=np.concatenate(attr_parent_func_edges,axis=0).T

        if add_global_node:
            global_nodes=[global_node]
            func_node_num,attr_node_num=len(func_nodes),len(attr_nodes)
            func_node_ids,attr_node_ids=np.arange(func_node_num),np.arange(attr_node_num)
            global_child_func_edges=np.zeros(shape=(2,func_node_num))
            global_child_func_edges[1,:]=func_node_ids
            func_parent_global_edges=np.zeros(shape=(2,func_node_num))
            func_parent_global_edges[0,:]=func_node_ids
            global_child_attr_edges = np.zeros(shape=(2, attr_node_num))
            global_child_attr_edges[1, :] = attr_node_ids
            attr_parent_global_edges = np.zeros(shape=(2, attr_node_num))
            attr_parent_global_edges[0, :] = attr_node_ids

            return func_nodes,\
                   attr_nodes,\
                   func_poses,\
                   attr_poses,\
                   global_nodes,\
                   func_child_func_edges,\
                   func_parent_func_edges,\
                   func_child_attr_edges,\
                   attr_parent_func_edges,\
                   global_child_func_edges,\
                   func_parent_global_edges,\
                   global_child_attr_edges,\
                   attr_parent_global_edges #这里global node不需要position

        return func_nodes, \
               attr_nodes, \
               func_poses, \
               attr_poses, \
               func_child_func_edges, \
               func_parent_func_edges, \
               func_child_attr_edges, \
               attr_parent_func_edges
