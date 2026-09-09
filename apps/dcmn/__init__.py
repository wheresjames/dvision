"""Code shared by more than one dvision2 client.

`dvision2_common.py` owns the protocol -- status keys, command encoding, map
loading, report paths -- and is imported by headless processes, so it stays
free of any display dependency. This package is for the layer above it: the
things a *window* shares, starting with the top-down map every client draws.
"""
