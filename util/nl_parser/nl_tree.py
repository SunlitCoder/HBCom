from nltk.tree import Tree

def nltk_tree2edge_info(tree):
    edge_starts=[]
    edge_ends=[]
    depths = []
    global_positions=[]
    local_positions=[]
    tree_queue = [tree]
    depth_queue=[0]*len(tree)
    while tree_queue:
        tree=tree_queue.pop(0)
        if isinstance(tree,Tree):
            tree_label=tree.label()
            for i,subtree in enumerate(tree):
                depths.append(depth_queue.pop(0))
                edge_starts.append(tree_label)
                if isinstance(subtree,Tree):
                    edge_ends.append(subtree.label())
                    depth_queue.extend([depths[-1]+1]*len(subtree))
                elif isinstance(subtree,str):
                    edge_ends.append(subtree)
                if len(depths)==1 or (len(depths)>1 and depths[-1]>depths[-2]):
                    global_position=0
                    global_positions.append(global_position)
                else:
                    global_position+=1
                    global_positions.append(global_position)
                local_positions.append(i)
                tree_queue.append(subtree)
    return edge_starts,edge_ends,depths,global_positions,local_positions