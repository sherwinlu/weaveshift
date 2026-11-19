"""Build setlist CLI args from the real parser so tests track its defaults."""
from argparse import Namespace

from weaveshift import setlist as weaveshift_setlist


def setlist_args(setlist, output, **overrides) -> Namespace:
    args = weaveshift_setlist.build_parser().parse_args([str(setlist), "-o", str(output)])
    args.no_mps = True
    args.search_a_sec = args.search_b_sec = 1.0
    args.max_trim_a_sec = args.b_cue_max_sec = 2.0
    for key, value in overrides.items():
        setattr(args, key, value)
    return args
