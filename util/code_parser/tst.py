#coding=utf-8

def walk(node):
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(node.children)
        yield node

def is_func_node(node):    #如果有named子节点，说明是功能节点
    if isinstance(node,str):
        return False
    if node.type=='comment':
        return False
    if not node.is_named and not (node.parent.type.endswith('assignment') or node.parent.type.endswith('operator')):
        return False
    return True

from tree_sitter import Language, Parser

py_language = Language('tree_sitter_repo/my-languages.so', 'bash')
parser = Parser()
parser.set_language(py_language)
bcode = bytes(code, 'utf8')
tree = parser.parse(bcode)

i = 0
print(tree.root_node)
for child in walk(tree.root_node):
    print(child)
    print(child.type)
    print(str(child.text,encoding="utf-8"))
    print(child.is_named)
    print("***" * 20)
    i += 1
print(i)