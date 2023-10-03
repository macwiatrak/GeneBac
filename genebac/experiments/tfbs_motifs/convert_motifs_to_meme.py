import os


def convert_psfm_file_to_meme(
    file_path: str = "/Users/maciejwiatrak/Downloads/collectf-export-PSFM-transfac (1).mat",
):
    with open(file_path) as f:
        lines = f.readlines()

    motif_name = lines[0].strip().split()[1]
    pwm = [[int(x) for x in line.strip().split()[1:-1]] for line in lines[3:-2]]

    pwm_sum = [sum(row) for row in pwm]
    pwm_prob = [
        [float(count) / pwm_sum[i] for count in pwm[i]] for i in range(len(pwm))
    ]
    n_sites = pwm_sum[0]

    meme_lines = [
        "\n\nMOTIF {0}\n\n".format(motif_name),
        "letter-probability matrix: alength= {0} w= {1} nsites= {2} E= 0\n".format(
            4, len(pwm), n_sites
        ),
    ]
    for row in pwm_prob:
        meme_lines.append(
            "\t" + "\t".join("{:.6f}".format(prob) for prob in row) + "\n"
        )

    return meme_lines


def run(
    input_dir: str = "/tmp/psfm_motif_files/",
    output_file_path: str = "/tmp/meme_motif_file.meme",
):
    lines = [
        "MEME version 5.4.1 (Tue Mar 1 19:18:48 2022 -0800)\n\n",
        "ALPHABET= ACGT\n\n",
        "strands: + -\n\n",
        "Background letter frequencies (from uniform background):\n",
        "A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n",
    ]

    for file_name in os.listdir(input_dir):
        motif_lines = convert_psfm_file_to_meme(
            os.path.join(input_dir, file_name)
        )
        lines.extend(motif_lines)

    with open(output_file_path, "w") as f:
        f.writelines(lines)
