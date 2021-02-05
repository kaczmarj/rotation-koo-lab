"""Utilities for ChIP-seq analysis."""

# TODO: figure out how to sample negative data to approximate the distribution of GC
# content in the positive data.
#   One idea is to get a probability function for the GC content in the positive data,
#   apply that to negative data, and sample based on those probabilities.

import gzip
import http.client
import io
from os import PathLike
from pathlib import Path
import shutil
import subprocess
import typing
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

# Type that can represent a path on the filesystem.
PathType = typing.Union[str, Path]


def download(
    url: typing.Union[str, urllib.request.Request], output_path: PathLike, force=False
) -> http.client.HTTPResponse:
    """Download a file from `url` and save to `output`.

    Parameters
    ----------
    url : str or `urllib.request.Request`
        URL from which to download.
    output_path : Path-like
        Path in which to save downloaded data.
    force : bool
        If `True`, overwrite output file if it exists.

    Returns
    -------
    Instance of `http.client.HTTPResponse` (though it is closed).
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"File exists: '{output_path}'. To overwrite, use `force=True`"
        )
    with urllib.request.urlopen(url=url) as response:
        with output_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        return response


def add_str_before_suffixes(filepath: PathType, string: str) -> Path:
    """Append a string to a filename immediately before extension(s).

    Parameters
    ----------
    filepath : Path-like
        Path to modify. Can contain multiple extensions like `.bed.gz`.
    string : str
        String to append to filename.

    Returns
    -------
    Instance of `pathlib.Path`.

    Examples
    --------
    >>> add_str_before_suffixes("foo", "_baz")
    PosixPath('foo_baz')
    >>> add_str_before_suffixes("foo.bed", "_baz")
    PosixPath('foo_baz.bed')
    >>> add_str_before_suffixes("foo.bed.gz", "_baz")
    PosixPath('foo_baz.bed.gz')
    """
    filepath = Path(filepath)
    suffix = "".join(filepath.suffixes)
    orig_name = filepath.name.replace(suffix, "")
    new_name = f"{orig_name}{string}{suffix}"
    return filepath.with_name(new_name)


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
    bedtools_exe: PathType = "bedtools",
) -> subprocess.CompletedProcess:
    """Extract DNA sequences from a fasta file based on feature coordinates.

    Wrapper around `bedtools getfasta`. This function was made to
    work with bedtools version 2.27.1. It is not guaranteed to work
    with other versions. It is not even guaranteed to work with version 2.27.1, but
    it could and probably will.

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
    >>> one_hot(["TGCA"], alphabet="ACGT")
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
        # do nothing on 0 because array is initialized with zeros.
        if i:
            pre_onehot[s == letter] = i

    # create one-hot representation
    n_classes = len(alphabet)
    return np.eye(n_classes)[pre_onehot]


def bedtools_intersect(
    a: PathType,
    b: PathType,
    *,
    output_bedfile: PathType,
    write_a=False,
    invert_match=False,
    bedtools_exe: PathType = "bedtools",
) -> subprocess.CompletedProcess:
    """Report overlaps between two feature files.

    This is an incomplete wrapper around `bedtools intersect` version 2.27.1.
    The set of arguments here does not include all of the command-line arguments.

    Parameters
    ----------
    a : Path-like
        First feature file <bed/gff/vcf/bam>.
    b : Path-like
        Second feature file <bed/gff/vcf/bam>.
    output_bedfile : Path-like
        Name of output file. Can be compressed (`.bed.gz`).
    write_a : bool
        Write the original entry in `a` for each overlap.
    write_b : bool
        Write the original entry in `b` for each overlap.
    invert_match : bool
        Only report those entries in `a` that have no overlaps with `b`.
    bedtools_exe : Path-like
        The path to the `bedtools` executable. By default, uses `bedtools` in `$PATH`.

    Returns
    -------
    Instance of `subprocess.CompletedProcess`.
    """
    args = [str(bedtools_exe), "intersect"]
    if write_a:
        args.append("-wa")
    if invert_match:
        args.append("-v")
    args.extend(["-a", str(a), "-b", str(b)])

    output_bedfile = Path(output_bedfile)
    gzipped_output = output_bedfile.suffix == ".gz"
    openfile = gzip.open if gzipped_output else io.open
    try:
        # We cannot write stdout directly to a gzip file.
        # See https://stackoverflow.com/a/2853396/5666087
        process = subprocess.run(
            args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if not process.stdout:
            raise subprocess.SubprocessError(
                f"empty stdout, aborting. stderr is {process.stderr.decode()}"
            )
        with openfile(output_bedfile, mode="wb") as f:  # type: ignore
            f.write(process.stdout)
        return process
    except subprocess.CalledProcessError as e:
        raise subprocess.SubprocessError(e.stderr.decode()) from e


def _is_gzipped(filepath: PathType) -> bool:
    """Return `True` if the file is gzip-compressed."""
    with open(filepath, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def sample_b_matched_to_a(
    a: np.ndarray, b: np.ndarray, size: int = None, seed: int = None
) -> np.ndarray:
    """Return indices of `b` that are distributed similarly to `a`.

    Parameters
    ----------
    a : array
        One-dimensional array of samples.
    b : array
        One-dimensional array of samples from which to sample.
    size : int
        Number of samples to take from `b`. Default is `len(a)`.

    Returns
    -------
    Numpy array of indices with shape `(size,)`. If `size` is `None`, the shape is
    `(len(a),)`.

    Examples
    --------
    >>> a = np.array([1, 1, 2])
    >>> b = np.array([0, 0, 1, 2, 3, 4, 1])
    >>> mask = sample_b_matched_to_a(a, b, seed=42)
    >>> mask
    array([2, 6, 3])
    >>> b[mask]
    array([1, 1, 2])

    In the following example, two normal distributions are made. Despite the second
    distribution being bimodal, this function draws samples that are most similar to the
    first distribution.

    >>> rng = np.random.RandomState(seed=42)
    >>> x = rng.normal(size=1000)
    >>> y = np.concatenate((rng.normal(size=1000), rng.normal(loc=5, size=1000)))
    >>> _ = plt.hist(x, bins=25, range=(-3, 8))
    >>> plt.show()
    >>> _ = plt.hist(y, bins=25, range=(-3, 8))
    >>> plt.show()
    >>> mask = sample_b_matched_to_a(x, y, seed=42)
    >>> _ = plt.hist(y[mask], bins=25, range=(-3, 8))
    >>> plt.show()
    """
    a, b = np.asanyarray(a), np.asanyarray(b)
    if a.ndim != 1:
        raise ValueError("`a` must be one-dimensional")
    if b.ndim != 1:
        raise ValueError("`b` must be one-dimensional")
    kde = scipy.stats.gaussian_kde(a)
    p = kde(b)
    p = p / p.sum()  # scale to sum to 1
    if size is None:
        size = a.shape[0]
    if size > b.shape[0]:
        raise ValueError("size is greater than length of data from which to sample")
    idxs = np.arange(b.shape[0])
    rng = np.random.RandomState(seed=seed)
    return rng.choice(idxs, size=size, replace=False, p=p)


class _ProcessingOutput(typing.NamedTuple):
    """Container for output of processing."""

    # Path to the original BED file (before processing).
    bed_file_original: Path
    # Path to the BED file with filtered data.
    bed_file_filtered: Path
    # Path to image of peak lengths in BED file.
    bed_file_peak_length_img: Path
    # Path to reference genome (FASTA).
    reference_genome_path: Path
    # Path to FASTA file (converted from BED with help of reference genome).
    fasta_file: Path
    # Alphabet in sequences.
    alphabet: str
    # Nonsense letters.
    nonsense_letters: str
    # Boolean array indicating which sequences are nonsense.
    nonsense_mask: np.ndarray
    # Descriptions of each sequence. Same length in first dimension as one-hot array.
    descriptions: np.ndarray
    # One-hot representation of sequences.
    one_hot: np.ndarray


def _bed_to_fasta_to_onehot(
    bed_file: PathType,
    max_read_length: int,
    new_read_length: int,
    reference_genome_fasta: PathType,
    alphabet="ACGT",
    nonsense_letters="N",
    bedtools_exe: PathType = "bedtools",
) -> _ProcessingOutput:
    """Convert a BED file to one-hot representation with processing in between.

    This is a high-level function that uses several other functions within this module.

    Order of operations:
    1. visualize peaks (save output)
    2. filter (save output)
    3. convert filtered bedfile to fasta (with help of reference genome) (save output)
    4. parse fasta
    5. filter sequences with nonsense
    6. one-hot encode

    Parameters
    ----------
    bed_file : Path-like
        Input BED file with ChIP-seq results.
    max_read_length : int
        Upper threshold of read length. Reads greater than this length are thrown out.
    new_read_length : int
        Length to which to transform reads.
    reference_genome_fasta : Path-like
        Path to FASTA file containing reference genome.
    alphabet : str
        Letters contained in sequences.
    nonsense_letters : str
        Characters that indicate nonsense.
    bedtools_exe : Path-like
        Path to the `bedtools` command-line program.

    Returns
    -------
    Instance of `_ProcessingOutput` namedtuple.
    """
    print("reading data...")
    df = pd.read_csv(bed_file, delimiter="\t", header=None)

    # Visualize length of peaks.
    # TODO: are columns 1 and 2 guaranteed to be start and stop?
    print("getting peak length...")
    lengths = df.loc[:, 2] - df.loc[:, 1]
    plt.hist(lengths, bins=20)
    plt.title("Distribution of peak length in ChIP-seq data")
    bed_file_peak_length_img = add_str_before_suffixes(
        bed_file, "peak_length"
    ).with_suffix(".png")
    plt.savefig(bed_file_peak_length_img)
    plt.close()

    # Filter.
    print("filtering...")
    df = filter_bed_by_max_length(df, max_length=max_read_length)
    df = transform_bed_to_constant_size(df, new_length=new_read_length)
    bed_file_filtered = add_str_before_suffixes(bed_file, string="_filtered")
    df.to_csv(bed_file_filtered, sep="\t", index=False, header=False)
    print(f"  saved to '{bed_file_filtered}'")

    # Convert filtered bed file to fasta.
    # TODO: replace variable name with more descriptiven name.
    print("converting chip-seq data to fasta...")
    chipseq_fasta = add_str_before_suffixes(
        bed_file_filtered, "_extracted"
    ).with_suffix(".fa")
    _ = bedtools_getfasta(
        input_fasta=reference_genome_fasta,
        output_fasta=chipseq_fasta,
        bed_file=bed_file_filtered,
        force_strandedness=True,
        bedtools_exe=bedtools_exe,
    )
    print(f"  saved to '{chipseq_fasta}'")

    # Load sequences
    print("loading fasta...")
    descriptions, sequences = parse_fasta(chipseq_fasta)

    # Filter out nonsense
    print("filtering nonsense...")
    nonsense = get_nonsense_sequence_mask(sequences, nonsense_letters=nonsense_letters)
    print(f"  found {nonsense.sum()} sequences with nonsense letters")
    descriptions = descriptions[~nonsense]
    sequences = sequences[~nonsense]

    # One-hot encode
    print("one-hot encoding...")
    onehot = one_hot(sequences, alphabet=alphabet)

    return _ProcessingOutput(
        bed_file_original=Path(bed_file),
        bed_file_filtered=bed_file_filtered,
        bed_file_peak_length_img=bed_file_peak_length_img,
        reference_genome_path=Path(reference_genome_fasta),
        fasta_file=chipseq_fasta,
        alphabet=alphabet,
        nonsense_letters=nonsense_letters,
        nonsense_mask=nonsense,
        descriptions=descriptions,
        one_hot=onehot,
    )
