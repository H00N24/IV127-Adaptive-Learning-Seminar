import ast
import itertools
import keyword
import re
from base64 import b64decode

import editdistance
import numpy as np
import pandas as pd
from black import FileMode, format_str
from pyastsim.pyastsim import NormFunctions, NormIdentifiers, astunparse

exc = []


def decode_solution(solution):
    code = eval(solution)[0][1]
    return code


def decode_code(code):
    try:
        code = b64decode(code).decode("utf-8")
        tree = ast.parse(code)

        parsed = astunparse.unparse(tree).strip()
        cleaned = re.sub(r"^\s*\n", "", parsed, flags=re.MULTILINE)
        blacked = format_str(cleaned, mode=FileMode())
        return blacked
    except Exception as e:
        exc.append(type(e))
        return ""


def parse_ast(code):
    try:
        tree = ast.parse(code)
        tree = NormFunctions(func=None).visit(tree)
        tree = NormIdentifiers().visit(tree)

        return astunparse.unparse(tree).strip()
    except Exception as e:
        exc.append(type(e))
        return ""


ip_log_frame = pd.read_csv("data/umimeprogramovatcz-ipython_log.csv", sep=";")
ip_item_frame = pd.read_csv("data/umimeprogramovatcz-ipython_item.csv", sep=";")
ip_item_log_frame = pd.read_csv("data/umimeprogramovatcz-ipython_item_log.csv", sep=";")
ip_log_frame["dec_answer"] = ip_log_frame.answer.apply(decode_code)

ip_log_frame["ast_clean"] = ip_log_frame["dec_answer"].apply(parse_ast)
ip_log_frame = ip_log_frame[
    (ip_log_frame["dec_answer"] != "") & (ip_log_frame["ast_clean"] != "")
]

cols = ["id", "item", "correct", "dec_answer", "ast_clean"]
code_frame = ip_log_frame[cols]
code_frame["dec_len"] = code_frame["dec_answer"].apply(len)
code_frame["clean_len"] = code_frame["ast_clean"].apply(len)

ip_item_frame["dec_solution"] = (
    ip_item_frame["solution"].apply(decode_solution).apply(decode_code)
)
ip_item_frame["ast_clean"] = ip_item_frame["dec_solution"].apply(parse_ast)

solutions_frame = ip_item_frame[["id", "name", "dec_solution", "ast_clean"]]
build_in = "abs(\ndelattr(\nhash(\nmemoryview(\nset(\nall(\ndict(\nhelp(\nmin(\nsetattr(\nany(\ndir(\nhex(\nnext(\nslice(\nascii(\ndivmod(\nid(\nobject(\nsorted(\nbin(\nenumerate(\ninput(\noct(\nstaticmethod(\nbool(\neval(\nint(\nopen(\nstr(\nbreakpoint(\nexec(\nisinstance(\nord(\nsum(\nbytearray(\nfilter(\nissubclass(\npow(\nsuper(\nbytes(\nfloat(\niter(\nprint(\ntuple(\ncallable(\nformat(\nlen(\nproperty(\ntype(\nchr(\nfrozenset(\nlist(\nrange(\nvars(\nclassmethod(\ngetattr(\nlocals(\nrepr(\nzip(\ncompile(\nglobals(\nmap(\nreversed(\ncomplex(\nhasattr(\nmax(\nround(".split(
    "\n"
)
build_in = [f" {x}" for x in build_in]

data_columns = []

for keyw in itertools.chain(keyword.kwlist, build_in):

    res = [code.count(keyw) for code in code_frame["dec_answer"].values]
    if np.any(res):
        data_columns.append(keyw.strip().replace("(", ""))
        code_frame[keyw.strip().replace("(", "")] = res


e_distance = np.ones(code_frame.shape[0])


for solution_row in solutions_frame.itertuples():
    selected_frame = code_frame[code_frame["item"] == solution_row[1]]

    if selected_frame.shape[0] == 0:
        continue

    selected_frame = selected_frame[
        [x for x in selected_frame.columns if np.any(selected_frame[x])]
    ]

    ed = np.array(
        [
            editdistance.eval(code, solution_row[4])
            for code in selected_frame["ast_clean"]
        ]
    )

    norm_len = selected_frame["clean_len"].values
    norm_len[norm_len < len(solution_row[4])] = len(solution_row[4])

    ed = ed / norm_len

    e_distance[code_frame["item"] == solution_row[1]] = ed

code_frame["edit_distance"] = e_distance

code_frame.to_csv("data/code_frame.csv", header=True, index=False)
solutions_frame.to_csv("data/solutions_frame.csv", header=True, index=False)

code_frame = pd.read_csv("data/code_frame.csv", header=0)
solutions_frame = pd.read_csv("data/solutions_frame.csv", header=0)
