#!/usr/bin/env python3
"""Build the byte-safe pymembus binding locally, leaving its source untouched.

Usage: python scripts/build_sensor_pymembus.py --source /path/to/pymembus
Requires an already configured pymembus build (headers and libmembus).
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sysconfig

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
a = p.parse_args()
out = root / '.cache/sensor-pymembus'
out.mkdir(parents=True, exist_ok=True)
source = (a.source / 'src/py/cpp/main.cpp').read_text()
anchor = '    py::class_<PyMemcmd>(m, "memcmd")'
# Extend the existing generic broadcast ring; its C++ std::string payload is
# already length-delimited and binary-safe. Only Python conversion was missing.
addition = '''    m.attr("memmsg").attr("__doc__") = "Broadcast records with text and byte-safe APIs";
'''
needle = '        .def("write", &PyMemmsg::write, py::arg("sMsg"))'
assert needle in source, 'unsupported pymembus source revision'
source = source.replace(needle, needle + '''
        .def("write_bytes", [](PyMemmsg &r, py::bytes data) {
            return r.write(static_cast<std::string>(data));
        })
        .def("read_bytes_with_overrun", [](PyMemmsg &r, uint64_t wait) {
            bool overrun = false;
            auto data = r.read(wait, &overrun);
            return py::make_tuple(py::bytes(data), overrun);
        }, py::arg("wait") = 0)
''')
(out / 'main.cpp').write_text(source)
shutil.copy2(a.source / 'bld/lib/liblibmembus.so', out)
cmd = ['c++', '-shared', '-fPIC', '-std=c++20', '-O2',
       '-DAPPNAME="pymembus"', '-DAPPNAMERAW=pymembus', '-DAPPVER="2.1.0+records"',
       '-DAPPBUILD="sensor-v1"', '-DAPPDESC="Python shared memory library"']
for inc in [Path(sysconfig.get_path('include')), a.source / 'src/py/headers',
            a.source / 'bld/_deps/pybind11-src/include',
            a.source / 'bld/_deps/libmembus-src/include']:
    cmd += ['-I', str(inc)]
cmd += [str(out / 'main.cpp'), '-L', str(out), '-llibmembus', '-pthread', '-lrt',
        '-Wl,-rpath,$ORIGIN', '-o', str(out / ('pymembus' + sysconfig.get_config_var('EXT_SUFFIX')))]
subprocess.run(cmd, check=True)
print(out)
