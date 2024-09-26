# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tree_sitter import Language
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info('Tree-sitter language library building...')
Language.build_library(
  'my-languages.so',
  [
    'tree-sitter-bash',
  ]
)