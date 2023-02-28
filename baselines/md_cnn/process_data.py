import logging
import os

from deep_bac.data_preprocessing.run_variants_to_strains_genomes import (
    REF_GENOME_FILE_NAME,
    get_strain_w_phenotype_ids,
    PHENOTYPE_FILE_NAME,
)
from deep_bac.data_preprocessing.utils import (
    read_ref_genome,
    get_gene_data_dict,
)


def run(
    input_dir: str,
    output_dir: str,
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # read ref genome
    ref_genome = read_ref_genome(os.path.join(input_dir, REF_GENOME_FILE_NAME))

    # get unique ids to use
    strain_w_phenotype_ids = get_strain_w_phenotype_ids(
        os.path.join(input_dir, PHENOTYPE_FILE_NAME)
    )

    # genomic_loci_dict = get_genomic_loci_dict(
    #     ref_genome=ref_genome,
    #     gene_data_dict=gene_data_dict,
    #     genomic_loci=MD_CNN_GENOMIC_LOCI,
    # )
    #
    # logging.info("Reading variants")
    # # get variants
    # vaiants_df = get_and_filter_variants_to_loci(
    #     file_path=os.path.join(input_dir, VARIANTS_FILE_NAME),
    #     unique_ids_to_use=strain_w_phenotype_ids,
    #     genomic_loci_dict=genomic_loci_dict,
    # )
