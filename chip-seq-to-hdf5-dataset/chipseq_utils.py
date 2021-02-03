"""Utilities for ChIP-seq analysis."""

# TODO: figure out how to sample negative data to approximate the distribution of GC
# content in the positive data.
#   One idea is to get a probability function for the GC content in the positive data,
#   apply that to negative data, and sample based on those probabilities.

import gzip
import io
from pathlib import Path
import subprocess
import typing

import numpy as np
import pandas as pd

# Type that can represent a path on the filesystem.
PathType = typing.Union[str, Path]


def filter_bed_by_max_length(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """Remove rows of BED file data frame which contain sequences of
    length greater than `max_length`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of BED file data.
    max_length : int
        Max read length. Sequences greater than this length are removed.

    Returns
    -------
    pandas.DataFrame with modified data.
    """
    # TODO: are start and stop always columns 1 and 2?
    lengths = df.iloc[:, 2] - df.iloc[:, 1]
    bad = lengths > max_length
    return df.loc[~bad, :].copy()


def transform_bed_to_constant_size(df: pd.DataFrame, new_length: int) -> pd.DataFrame:
    """Transform read lengths in BED file dataframe to a constant size.

    This function maintains the center of each read but modifies the start and stop
    points to have length `new_length`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of BED file data.
    new_length : int
        Length to which to transform read lengths.

    Returns
    -------
    pandas.DataFrame with modified data.
    """
    df = df.copy()  # do not modify original
    if new_length < 1:
        raise ValueError("new_length must be positive integer")
    start = df.iloc[:, 1]
    stop = df.iloc[:, 2]
    center = (start + stop) // 2
    half_len = new_length // 2
    new_start = center - half_len
    new_stop = center + half_len
    if (new_start < 0).any():
        raise ValueError("negative start position found and not accounted for")
    df.iloc[:, 1] = new_start
    df.iloc[:, 2] = new_stop
    return df


def twobit_to_fasta(
    twobit_filepath: PathType,
    fasta_filepath: PathType,
    exe_path: PathType = "./twoBitToFa",
) -> subprocess.CompletedProcess:
    """Convert 2bit format to FASTA format. Wrapper around `twoBitToFa` program.

    Parameters
    ----------
    twobit_filepath : Path-like
        Path to 2bit file.
    fasta_filepath : Path-like
        Path to FASTA output file.
    exe_path : Path-like
        Path to the `toBitToFa` executable.

    Returns
    -------
    Instance of `subprocess.CompletedProcess`.
    """
    args = [str(exe_path), str(twobit_filepath), str(fasta_filepath)]
    try:
        return subprocess.run(
            args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise subprocess.SubprocessError(e.stderr.decode()) from e


def bedtools_getfasta(
    *,
    input_fasta: PathType,
    output_fasta: PathType,
    bed_file: PathType,
    use_name=False,
    use_name_coords=False,
    split=False,
    tab_delim=False,
    force_strandedness=False,
    full_header=False,
    bedtools_exe: PathType = "bedtools"
) -> subprocess.CompletedProcess:
    """Extract DNA sequences from a fasta file based on feature coordinates.

    Wrapper around `bedtools getfasta`. This function was made to
    work with bedtools version 2.27.1. It is not guaranteed to work
    with other versions.

    Parameters
    ----------
    input_fasta : str, Path-like
        Input FASTA file.
    output_fasta : str, Path-like
        Output FASTA file.
    bed_file : str, Path-like
        BED/GFF/VCF file of ranges to extract from `input_fasta`.
    use_name : bool
        Use the name field for the FASTA header.
    use_name_coords : bool
        Use the name and coordinates for the FASTA header.
    split : bool
        Given BED12 format, extract and concatenate the sequences
        from the BED "blocks" (e.g., exons).
    tab_delim : bool
        Write output in TAB delimited format.
    force_strandedness : bool
        Force strandedness. If the feature occupies the antisense
        strand, the squence will be reverse complemented.
    full_header : bool
        Use full FASTA header.
    bedtools_exe : Path-like
        The path to the `bedtools` executable. By default, uses `bedtools` in `$PATH`.

    Returns
    -------
    Instance of `subprocess.CompletedProcess`.
    """
    args = [str(bedtools_exe), "getfasta"]
    if use_name:
        args.append("-name")
    if use_name_coords:
        args.append("-name+")
    if split:
        args.append("-split")
    if tab_delim:
        args.append("-tab")
    if force_strandedness:
        args.append("-s")
    if full_header:
        args.append("-fullHeader")
    args.extend(
        ["-fi", str(input_fasta), "-bed", str(bed_file), "-fo", str(output_fasta)]
    )
    try:
        return subprocess.run(
            args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise subprocess.SubprocessError(e.stderr.decode()) from e


def parse_fasta(filepath: PathType) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Parse FASTA file into arrays of descriptions and sequence data.

    Parameters
    ----------
    filepath : Path-like
        FASTA file to parse. Can be gzip-compressed.

    Returns
    -------
    Tuple of two numpy arrays with equal shapes. The first array contains the
    descriptions for each sequence. The second array contains the sequences.
    """
    # FASTA format described here.
    # https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp
    descriptions: typing.List[str] = []
    sequences: typing.List[str] = []
    prev_line_was_sequence = False

    gzipped = _is_gzipped(filepath)
    openfile = gzip.open if gzipped else io.open
    with openfile(filepath, "rt") as f:  # type: ignore
        for line in f:
            line = line.strip()
            # handle blank lines
            if not line:
                continue
            is_description = line.startswith(">")
            if is_description:
                description = line[1:].strip()  # prune ">" char
                descriptions.append(description)
                prev_line_was_sequence = False
            else:  # is sequence data
                sequence = line.upper()
                if prev_line_was_sequence:
                    # This accounts for sequences that span multiple lines.
                    sequences[-1] += sequence
                else:
                    sequences.append(sequence)
                prev_line_was_sequence = True
    return np.array(descriptions), np.array(sequences)


def get_nonsense_sequence_mask(sequences, nonsense_letters="N") -> np.ndarray:
    """Return boolean array, where `True` marks a sequence containing nonsense letters.

    Parameters
    ----------
    sequences : numpy.ndarray
        One-dimensional array of sequences (strings).
    nonsense_letters : str
        Letter(s) that are considered nonsense.

    Returns
    -------
    mask : boolean numpy array with same length as `sequences`.

    Examples
    --------
    >>> import numpy as np
    >>> sequences = np.array(["AGGCCT", "GCTATTAN", "CGCTGC"])
    >>> nonsense = get_nonsense_sequence_mask(sequences, nonsense_letters="N")
    >>> nonsense
    array([False,  True, False])
    >>> sequences[nonsense]
    array(['GCTATTAN'], dtype='<U8')
    >>> sequences[~nonsense]
    array(['AGGCCT', 'CGCTGC'], dtype='<U8')
    """
    sequences = np.asanyarray(sequences)
    if sequences.ndim != 1:
        raise ValueError("array of sequences must be one-dimensional.")
    return np.array(
        [
            any(letter in sequence for letter in nonsense_letters)
            for sequence in sequences
        ]
    )


def one_hot(sequences, alphabet="ACGT") -> np.ndarray:
    """Convert flat array of sequences to one-hot representation.

    Assumes that all sequences have the same length and that all letters in `sequences`
    are contained in `alphabet`.

    Parameters
    ----------
    sequences : numpy.ndarray of strings
        The array of strings. Should be one-dimensional.
    alphabet : str
        The alphabet of the sequences.

    Returns
    -------
    Numpy array of sequences in one-hot representation. The shape of this array is
    `(len(sequences), len(sequences[0]), len(alphabet))`.

    Examples
    --------
    >>> chipseq_utils.one_hot(["TGCA"], alphabet="ACGT")
    array([[[0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]]])
    """
    sequences = np.asanyarray(sequences)
    if sequences.ndim != 1:
        raise ValueError("array of sequences must be one-dimensional.")
    n_sequences = sequences.shape[0]
    sequence_len = len(sequences[0])

    # Unpack strings into 2D array, where each point has one character.
    s = np.zeros((n_sequences, sequence_len), dtype="U1")
    for i in range(n_sequences):
        s[i] = list(sequences[i])

    # Make an integer array from the string array.
    pre_onehot = np.zeros(s.shape, dtype=np.uint8)
    for i, letter in enumerate(alphabet):
        if i:
            pre_onehot[s == letter] = i

    # create one-hot representation
    n_classes = len(alphabet)
    return np.eye(n_classes)[pre_onehot]


def _is_gzipped(filepath: PathType) -> bool:
    """Return `True` if the file is gzip-compressed."""
    with open(filepath, "rb") as f:
        return f.read(2) == b"\x1f\x8b"
