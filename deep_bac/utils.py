from typing import Literal

DRUG_SPECIFIC_GENES_DICT = {
    "All": [
        "ahpC",
        "fabG1",
        "inhA",
        "katG",
        "ndh",
        "rpoB",
        "embA",
        "embB",
        "embC",
        "embR",
        "iniA",
        "iniC",
        "manB",
        "rmlD",
        "pncA",
        "rpsA",
        "gyrA",
        "gyrB",
        "rpsL",
        "gid",
        "rrs",
        "tlyA",
        "eis",
    ],
    "INH": [
        "katG",
        "proA",
        "ahpC",
        "fabG1",
        "rpoB",
        "inhA",
        "embB",
        "Rv1139c",
        "Rv1140",
        "Rv1158c",
        "rpsL",
        "Rv1219c",
        "ftsK",
        "Rv2749",
        "gid",
    ],
}


def get_selected_genes(use_drug_specific_genes: Literal["INH"] = None):
    if not use_drug_specific_genes:
        return None
    return DRUG_SPECIFIC_GENES_DICT[use_drug_specific_genes]
