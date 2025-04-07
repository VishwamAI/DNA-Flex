#include "cif_dict_pybind.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace dnaflex {
namespace parsers {

void init_cif_dict(py::module_& m) {
    py::class_<CifEntry>(m, "CifEntry")
        .def(py::init<>())
        .def_readwrite("id", &CifEntry::id)
        .def_readwrite("type", &CifEntry::type)
        .def_readwrite("values", &CifEntry::values);

    py::class_<CifDictionary>(m, "CifDictionary")
        .def(py::init<>())
        .def("parse", &CifDictionary::parse)
        .def("get_entry", &CifDictionary::get_entry)
        .def("get_categories", &CifDictionary::get_categories);
}

} // namespace parsers
} // namespace dnaflex