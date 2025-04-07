#pragma once

#include <pybind11/pybind11.h>
#include "cif_dict_lib.h"

namespace py = pybind11;

namespace dnaflex {
namespace parsers {

void init_cif_dict(py::module_& m);

} // namespace parsers
} // namespace dnaflex